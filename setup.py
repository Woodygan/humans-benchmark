from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="HUMANS-Benchmark",
    version="0.1.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="HUMANS: Efficient benchmark for evaluating Large Audio Models with human preference prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EfficientAudioBench/HUMANS_Benchmark",
    project_urls={
        "Bug Tracker": "https://github.com/EfficientAudioBench/HUMANS_Benchmark/issues",
        "Documentation": "https://github.com/EfficientAudioBench/HUMANS_Benchmark",
        "Source Code": "https://github.com/EfficientAudioBench/HUMANS_Benchmark",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=6.0.0",
        ],
        "audio": [
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
        ],
        "metrics": [
            "jiwer>=3.0.0",
        ],
        "all": [
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "jiwer>=3.0.0",
            "openai>=1.0.0",
        ],
    },
)