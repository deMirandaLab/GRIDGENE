# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
project_root = os.path.abspath('../../')
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.join(project_root, 'gridgene'))  # Add the 'src' directory to the path
# ONLY add the 'src' directory to the Python path for autodoc to find your 'src' package

project = 'GRIDGENE'
copyright = '2025, AM Sequeira'
author = 'AM Sequeira'
release = '0.1'

nb_execution_mode = "off"
nbsphinx_allow_errors = True
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'myst_nb',  # Jupyter Notebook support
    "sphinx_autodoc_typehints",  # optional: must be installed
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
set_type_checking_flag = False
always_document_param_types = False


autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
autodoc_member_order = 'bysource'
html_theme = 'pydata_sphinx_theme'
# html_static_path = ['_static']
# html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "show_toc_level": 2, # Show current TOC up to 2 levels deep
    "navbar_end": ["theme-switcher", "navbar-icon-links"], # PyData theme default, often includes search/index
    # You might want to customize these:
    # "secondary_sidebar_items": ["page-toc", "searchbox"], # Controls right sidebar items
    # "header_links_before_dropdown": ["Home", "About"], # Example custom links
}

# Source suffix for Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add the following at the end or in a suitable section for autodoc configuration
autodoc_mock_imports = [
    'numpy',
    'pandas',
    'skimage',  # scikit-image is imported as 'skimage'
    'matplotlib',
    'tqdm',
    'cv2',      # opencv-python is imported as 'cv2'
    'shapely',
    'scipy',
    'PIL',      # Pillow is imported as 'PIL'
    'tifffile',
    'natsort',
    'alphashape',
    'jupyter',  # Although jupyter/ipykernel are usually build dependencies, including them doesn't hurt.
    'ipykernel',
    'anndata',
    'scanpy',
    'openpyxl',
    'minisom',
    'squidpy',
    'spatialdata',
    'mpl_toolkits',
    'sklearn',
    'scikit-learn',  # scikit-learn is imported as 'sklearn'
    'xarray'
]