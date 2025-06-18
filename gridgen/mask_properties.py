from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table
from gridgen.logger import get_logger
from functools import wraps
import os
import time
import logging
# todo change to receive the logger from the main module
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


@dataclass
class MaskDefinition:
    """
    Definition of a mask to analyze.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask array.
    mask_name : str
        Name identifier for the mask.
    analysis_type : str, optional
        Type of analysis ('per_object', 'bulk', or 'grid'), by default "per_object".
    grid_size : int or None, optional
        Grid size for 'grid' analysis type, by default None.
    """
    mask: np.ndarray
    mask_name: str
    analysis_type: str = "per_object"  # "per_object", "bulk", "grid"
    grid_size: Optional[int] = None

@dataclass
class MaskAnalysisResult:
    """
    Container for the results of mask analysis.

    Parameters
    ----------
    mask_name : str
        Name of the analyzed mask.
    analysis_type : str
        Analysis type performed.
    features : list of dict
        List of extracted features per object.
    """
    mask_name: str
    analysis_type: str
    features: List[Dict[str, Any]]

class MorphologyExtractor:
    """
    Extracts morphological features from labeled masks.
    """

    def extract_per_object_features(self, labeled_mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract per-object morphological features from a labeled mask.

        Parameters
        ----------
        labeled_mask : np.ndarray
            Mask where each object is labeled with an integer.

        Returns
        -------
        list of dict
            List of dictionaries containing features per object.
        """
        properties = [
            'label',
            'area',
            'perimeter',
            'eccentricity',
            'solidity',
            'centroid',
            'bbox',
        ]
        props = regionprops_table(labeled_mask, properties=properties)

        rows = []
        for i in range(len(props['label'])):
            row = {
                'object_id': props['label'][i],
                'area': props['area'][i],
                'perimeter': props['perimeter'][i],
                'eccentricity': props['eccentricity'][i],
                'solidity': props['solidity'][i],
                'centroid_y': props['centroid-0'][i],
                'centroid_x': props['centroid-1'][i],
                'min_row': props['bbox-0'][i],
                'min_col': props['bbox-1'][i],
                'max_row': props['bbox-2'][i],
                'max_col': props['bbox-3'][i],
            }
            rows.append(row)

        return rows

    def extract_bulk_features(self, mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract bulk features for a whole mask (e.g., total area).

        Parameters
        ----------
        mask : np.ndarray
            Binary mask.

        Returns
        -------
        list of dict
            Single-item list with total area and object_id='bulk'.
        """
        total_area = int(np.sum(mask))
        return [{'area': total_area, 'object_id': 'bulk'}]

    def extract_grid_features(self, mask: np.ndarray, grid_size: int, parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract features by dividing the mask into grid tiles.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask.
        grid_size : int
            Size of grid tiles.
        parent_id : str or None, optional
            Optional parent ID prefix for tiles, by default None.

        Returns
        -------
        list of dict
            List of features per grid tile.
        """
        h, w = mask.shape
        results = []
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                tile = mask[y:y+grid_size, x:x+grid_size]
                tile_area = int(np.sum(tile))
                object_id = f'{parent_id}_grid_{x}_{y}' if parent_id else f'grid_{x}_{y}'
                results.append({
                    'x': x, 'y': y,
                    'area': tile_area,
                    'object_id': object_id
                })
        return results

class GeneCounter:
    """
    Counts gene expression values within masks.
    """

    def count_genes_per_object(self, labeled_mask: np.ndarray, array_counts: np.ndarray, target_dict: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Count genes per labeled object.

        Parameters
        ----------
        labeled_mask : np.ndarray
            Labeled mask array.
        array_counts : np.ndarray
            3D array of gene counts per pixel.
        target_dict : dict
            Mapping from gene names to indices in array_counts.

        Returns
        -------
        list of dict
            List of gene counts per object.
        """
        results = []
        for obj_id in np.unique(labeled_mask):
            if obj_id == 0:
                continue
            mask = labeled_mask == obj_id
            # counts = np.einsum('ijk,ij->k', array_counts.astype(np.int16), mask.astype(np.int8))
            counts = array_counts[mask].sum(axis=0, dtype=np.int64)

            counts_dict = {gene: counts[i] for gene, i in target_dict.items()}
            counts_dict['object_id'] = obj_id
            results.append(counts_dict)
        return results

    def count_genes_bulk(self, mask: np.ndarray, array_counts: np.ndarray, target_dict: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Count genes in a bulk mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask.
        array_counts : np.ndarray
            3D array of gene counts.
        target_dict : dict
            Mapping from gene names to indices.

        Returns
        -------
        list of dict
            Single-item list with gene counts.
        """
        mask = mask.astype(bool)
        # counts = np.einsum('ijk,ij->k', array_counts.astype(np.int64), mask.astype(np.int64))
        counts = array_counts[mask].sum(axis=0, dtype=np.int64)

        counts_dict = {gene: counts[i] for gene, i in target_dict.items()}
        counts_dict['object_id'] = 'bulk'
        return [counts_dict]

    def count_genes_grid(self, mask: np.ndarray, array_counts: np.ndarray, target_dict: Dict[str, int], grid_size: int) -> List[Dict[str, Any]]:
        """
        Count genes in grid tiles.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask.
        array_counts : np.ndarray
            3D gene counts.
        target_dict : dict
            Mapping gene to index.
        grid_size : int
            Size of tiles.

        Returns
        -------
        list of dict
            List of gene counts per grid tile.
        """
        h, w = mask.shape
        results = []
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                sub_mask = mask[y:y+grid_size, x:x+grid_size].astype(bool)
                if not np.any(sub_mask):
                    continue
                # counts = np.einsum('ijk,ij->k', array_counts[y:y + grid_size, x:x + grid_size].astype(np.int64),
                #                    sub_mask.astype(np.int64))
                counts = array_counts[y:y + grid_size, x:x + grid_size][sub_mask].sum(axis=0, dtype=np.int64)
                counts_dict = {gene: counts[i] for gene, i in target_dict.items()}
                counts_dict['object_id'] = f'grid_{x}_{y}'
                results.append(counts_dict)
        return results

class HierarchyMapper:
    """
    Maps child objects to parent objects based on label overlaps.
    """

    def map_hierarchy(self, source_labels: np.ndarray, target_labels: np.ndarray) -> Dict[int, List[int]]:
        """
        Map each source object ID to a list of parent object IDs from target mask.

        Parameters
        ----------
        source_labels : np.ndarray
            Labeled source mask.
        target_labels : np.ndarray
            Labeled target mask (parent).

        Returns
        -------
        dict
            Mapping from source object ID to list of parent IDs.
        """
        mapping = {}
        for src_id in np.unique(source_labels):
            if src_id == 0:
                continue
            overlap = target_labels[source_labels == src_id]
            mapping[src_id] = list(np.unique(overlap[overlap > 0]))
        return mapping

class MaskAnalysisPipeline:
    """
    Main pipeline for analyzing masks with gene counts and morphology.
    """

    def __init__(self, mask_definitions: List[MaskDefinition], array_counts: np.ndarray, target_dict: Dict[str, int],
                 logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the pipeline.

        Parameters
        ----------
        mask_definitions : list of MaskDefinition
            List of mask definitions.
        array_counts : np.ndarray
            3D gene counts array.
        target_dict : dict
            Mapping gene names to indices in array_counts.
        logger : logging.Logger or None, optional
            Logger instance, by default None.
        """
        self.mask_definitions = mask_definitions
        self.array_counts = array_counts
        self.target_dict = target_dict
        self.extractor = MorphologyExtractor()
        self.counter = GeneCounter()
        self.results: List[MaskAnalysisResult] = []
        self.labeled_masks: Dict[str, np.ndarray] = {}  # Store labeled versions of masks
        self.logger = logger or get_logger(f'{__name__}.{"GetMasks"}')
        self.logger.info(f"Initialized MaskAnalysisPipeline with {len(mask_definitions)} masks.")

    @timeit
    def run(self) -> List[MaskAnalysisResult]:
        """
        Run the full analysis pipeline on all mask definitions.

        Returns
        -------
        list of MaskAnalysisResult
            List of results per mask.
        """
        self.results.clear()

        for defn in self.mask_definitions:
            self.logger.info(f"Processing mask: {defn.mask_name} ({defn.analysis_type})")

            # if defn.analysis_type == 'per_object':
            #     labeled = label(defn.mask)
            #     self.labeled_masks[defn.mask_name] = labeled
            if defn.analysis_type == "per_object" and defn.mask_name not in self.labeled_masks:
                self.labeled_masks[defn.mask_name] = label(defn.mask)
                morpho = self.extractor.extract_per_object_features(self.labeled_masks[defn.mask_name])
                counts = self.counter.count_genes_per_object(self.labeled_masks[defn.mask_name], self.array_counts.astype(np.int16), self.target_dict)
                merged = self._merge_dicts_by_key(morpho, counts, 'object_id')

            elif defn.analysis_type == 'bulk':
                morpho = self.extractor.extract_bulk_features(defn.mask)
                counts = self.counter.count_genes_bulk(defn.mask, self.array_counts.astype(np.int16), self.target_dict)
                merged = self._merge_dicts_by_key(morpho, counts, 'object_id')

            elif defn.analysis_type == 'grid':
                if defn.grid_size is None:
                    raise ValueError("Grid size required for grid analysis.")
                morpho = self.extractor.extract_grid_features(defn.mask, defn.grid_size)
                counts = self.counter.count_genes_grid(defn.mask, self.array_counts.astype(np.int16), self.target_dict, defn.grid_size)
                merged = self._merge_dicts_by_key(morpho, counts, 'object_id') if counts else morpho

            else:
                raise ValueError(f"Unsupported analysis type: {defn.analysis_type}, should be one of 'per_object', 'bulk', or 'grid'.")

            # Check for negative gene counts
            for c in counts:
                for gene, value in c.items():
                    if gene != 'object_id' and value < 0:
                        print(f"Warning: Negative count for gene '{gene}' in object '{c.get('object_id')}'")

            for item in merged:
                item['mask_name'] = defn.mask_name
                item['analysis_type'] = defn.analysis_type

            self.results.append(MaskAnalysisResult(defn.mask_name, defn.analysis_type, merged))
            # self.logger.info(f"Finished {defn.mask_name} in {elapsed:.2f} sec")

        return self.results

    def get_results_df(self) -> pd.DataFrame:
        """
        Get all results concatenated into a single pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with all extracted features.
        """
        if not self.results:
            self.run()
        all_features = [item for r in self.results for item in r.features]
        return pd.DataFrame(all_features)

    def _merge_dicts_by_key(self, list1: List[Dict[str, Any]], list2: List[Dict[str, Any]], key: str) -> List[
        Dict[str, Any]]:
        """
        Merge two lists of dictionaries by matching values of a specified key.

        Parameters
        ----------
        list1 : list of dict
            First list of dictionaries.
        list2 : list of dict
            Second list of dictionaries.
        key : str
            Key to merge on.

        Returns
        -------
        list of dict
            Merged list of dictionaries.
        """
        if not list1:
            return list2
        if not list2:
            return list1
        index2 = {d[key]: d for d in list2}
        return [{**d1, **index2.get(d1[key], {})} for d1 in list1]

    @timeit
    def map_hierarchies(self, hierarchy_definitions: Dict[str, Dict[str, Any]], save_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Map child objects to their parent objects using reference labeled masks.

        Parameters
        ----------
        hierarchy_definitions : dict
            Dictionary defining the hierarchy relationships.
        save_dir : str or None, optional
            Directory to save labeled masks, by default None.

        Returns
        -------
        pandas.DataFrame
            DataFrame with mapping of child to parent objects.
        """
        records = []

        for child_name, definition in hierarchy_definitions.items():
            reference_labels = definition["labels"]
            parent_name = definition["level_hierarchy"]

            # Make sure both masks are labeled
            if parent_name not in self.labeled_masks:
                self.labeled_masks[parent_name] = label(
                    next(d.mask for d in self.mask_definitions if d.mask_name == parent_name)
                )
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(os.path.join(save_dir, f"{parent_name}_labeled.npy"), self.labeled_masks[parent_name])

            parent_labels = self.labeled_masks[parent_name]

            mapper = HierarchyMapper()
            hierarchy_map = mapper.map_hierarchy(reference_labels, parent_labels)

            # Update results
            for result in self.results:
                if result.mask_name == child_name:
                    for row in result.features:
                        obj_id = row.get("object_id")
                        try:
                            int_obj_id = int(obj_id)
                        except Exception:
                            continue
                        row["parent_ids"] = hierarchy_map.get(int_obj_id, [])

            # Collect for output
            for obj_id, parent_ids in hierarchy_map.items():
                records.append({
                    "mask_name": child_name,
                    "object_id": obj_id,
                    "parent_mask": parent_name,
                    "parent_ids": parent_ids
                })

            # Optionally save reference label
            if save_dir:
                np.save(os.path.join(save_dir, f"{child_name}_ref_labels.npy"), reference_labels)

        return pd.DataFrame(records)