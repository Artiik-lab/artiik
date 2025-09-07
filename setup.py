"""
Setup script for ContextManager.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    # Fallback requirements if requirements.txt is not available
    requirements = [
        "openai>=1.0.0",
        "anthropic>=0.7.0", 
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "sentence-transformers>=2.2.0",
        "python-dotenv>=1.0.0",
        "tiktoken>=0.5.0",
        "loguru>=0.7.0",
    ]

setup(
    name="artiik",
    version="0.1.0",
    author="Boualam Hamza",
    author_email="boualamhamza@outlook.fr",
    description="A modular, plug-and-play memory and context management layer for AI agents made by Artiik.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Artiik-lab/artiik",
    packages=find_packages(include=["context_manager*", "artiik*"]),
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "artiik-demo=context_manager.examples.agent_example:demo_agent",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 