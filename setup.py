import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="unet",
    version="0.0.0",
    author="Vadim Zinchenko",
    author_email="vadim.zinchenko@hereon.de",
    description="Pytorch implementation of unet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m-dml/unet-pytorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
