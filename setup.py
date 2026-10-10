from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

loaders_requirements = [
    "langchain-community>=0.3",
    "jq",
    "pypdf",
    "beautifulsoup4",
    "unstructured[md]",
]

setup(
    name="mantiq",
    version="0.2.0",
    author="Tactful AI team",
    author_email="ahmed.osama@tactful.ai",
    description="A package for Cache-Augmented Generation (CAG).",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tactful-ai/mantiq",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,  # Use requirements from the file
    extras_require={
        "loaders": loaders_requirements,
        "azure": loaders_requirements + ["azure-ai-documentintelligence"],
        "test": ["pytest"],
    },
    include_package_data=True,
)
