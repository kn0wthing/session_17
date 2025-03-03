from setuptools import setup, find_packages

setup(
    name="ai_style_transfer_studio",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.19.0",
        "transformers>=4.30.0",
        "accelerate>=0.21.0",
        "streamlit>=1.24.0",
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
        "pathlib>=1.0.1",
        "tqdm>=4.65.0",
        "huggingface-hub>=0.16.0"
    ],
    python_requires=">=3.7",
    author="AI Style Transfer Studio Team",
    author_email="example@example.com",
    description="A Stable Diffusion based application for artistic style transfer",
    keywords="stable-diffusion, ai, style-transfer, art, machine-learning",
    url="https://github.com/yourusername/ai-style-transfer-studio",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Artists",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Artistic Software",
        "Topic :: Multimedia :: Graphics",
    ],
) 