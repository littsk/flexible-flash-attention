from pathlib import Path
from setuptools import setup, find_packages

# setup.py for the flash_attn_cute package
# This is a pure-Python package with source files located in the local "cute/" directory.

# Read long description from README if present
here = Path(__file__).resolve().parent
readme_path = here / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="flash_attn_cute",
    version="0.0.1",
    description="A monkey-patch flash attention 4 package supporting arbitary mask for magi_attention.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Project URL and author metadata can be adjusted as needed
    url="https://github.com/demonatic/flash-attention/tree/magi_attn_blackwell_support",
    author="",
    author_email="",
    license="MIT",
    # Tell setuptools to look for packages under the "cute" directory
    package_dir={"flash_attn_cute": "cute"},
    packages=["flash_attn_cute"] + [
        f"flash_attn_cute.{pkg}" for pkg in find_packages(
            where="cute",
            exclude=["losses*", "layers*"]
        )
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # List runtime dependencies here, e.g. "numpy>=1.24"
    ],
)