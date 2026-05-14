from setuptools import setup, find_packages

setup(
    name="ecotrace",
    version="1.0.0",
    description="Automated Greenwashing Detection Pipeline",
    author="EcoTrace Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.2",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "spacy>=3.6.0",
        "PyMuPDF>=1.23.0",
        "deep-translator>=1.11.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": ["black>=23.0.0", "isort>=5.12.0", "pytest>=7.4.0"],
    },
)
