#!/usr/bin/env python3
"""
Setup script for imgcom
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="imgcom",
    version="1.0.0",
    author="Mehmet T. AKALIN",
    author_email="info@dv.com.tr",
    description="Professional Image Combiner & Stitcher CLI Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/makalin/imgcom",
    py_modules=["imgcom"],
    python_requires=">=3.6",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "full": [
            "Pillow>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "imgcom=imgcom:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="image stitching panorama combine merge photos cli opencv computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/makalin/imgcom/issues",
        "Source": "https://github.com/makalin/imgcom",
        "Documentation": "https://github.com/makalin/imgcom#readme",
    },
)

