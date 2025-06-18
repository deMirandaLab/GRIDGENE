from setuptools import setup, find_packages

setup(
    name="gridgen",
    version="0.1.0",
    author="AM Sequeira",
    description="GRIDGEN project",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "scikit-image",
        "tqdm",
        "opencv-python",
        "shapely",
        "pillow",
        "openpyxl",
        "xarray",
    ],
    python_requires=">=3.10",
)