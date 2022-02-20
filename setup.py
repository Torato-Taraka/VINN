import setuptools
with open("README.md", "r") as f:
    long_description = f.read()
setuptools.setup(
    name="vinn",
    version="1.0",
    author="qianzhihao",
    author_email="2018302170027@whu.edu.cn",
    description="Visual Interpretation of Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Torato-Taraka/VINN",
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)