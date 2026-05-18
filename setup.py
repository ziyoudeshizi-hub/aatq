from setuptools import setup, find_packages

setup(
    name="aatq",
    version="0.1.0",
    description="Activation-Aware Ternary Quantization",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "safetensors",
        "tqdm",
        "numpy",
    ],
)
