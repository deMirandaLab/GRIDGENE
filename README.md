# GRIDGEN

![CI](https://github.com/deMirandaLab/GRIDGEN/actions/workflows/ci.yml/badge.svg)
[![Docs](https://img.shields.io/badge/docs-GitHub--Pages-blue.svg)](https://demirandalab.github.io/GRIDGEN/)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)


**GRIDGEN** (Guided Region Identification based on Density of GENes) is a Python package designed for defining  
regions of interest based on transcript density. It enables the identification of biologically relevant tissue compartments, including interfaces between regions  
(e.g., cancer vs. stroma), phenotype-enriched areas, and zones defined by specific gene signatures.

---

## Table of Contents
- [Features](#features)
- [Documentation](#documentation)
- [Credits](#credits)
- [License](#license)
- [Contributing](#contributing)



## Features

- Generate masks and expansions for spatial transcriptomics based on density of genes  
  - Contour identification  
  - Tissue mask mapping and expansion  
  - Mask information retrieval  
  - Cell segmentation overlay  

General view: 
![plot](docs/figures/gridgen.png)

GRIDGEN contains the following Case Studies: 
  - tumor microenvironment analysis  
  - Definition of population-specific objects in CRC in CosMx (single and multiclass)  
  - Integration with cell segmentation pipelines  
  - Alternative masking strategies using KD-Trees and Self-Organizing Maps (SOM)

---

## Documentation

Documentation is available at [GRIDGEN Documentation](https://demirandalab-gridgen.readthedocs.io/en/latest/).

---

## Credits
If you find this repository useful in your research or for educational purposes please refer to:



## License

Developed at the Leiden University Medical Centre, The Netherlands and 
Centre of Biological Engineering, University of Minho, Portugal

Released under the GNU Public License (version 3.0).


[//]: # (.. |License| image:: https://img.shields.io/badge/license-GPL%20v3.0-blue.svg)

[//]: # (   :target: https://opensource.org/licenses/GPL-3.0)

[//]: # (.. |PyPI version| image:: https://badge.fury.io/py/propythia.svg)

[//]: # (   :target: https://badge.fury.io/py/propythia)

[//]: # (.. |RTD version| image:: https://readthedocs.org/projects/propythia/badge/?version=latest&style=plastic)

[//]: # (   :target: https://propythia.readthedocs.io/)



