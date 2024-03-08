import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Support and Resistance Algorithm",
    version="1.0.0",
    author="Leandro Vallejos",
    description="Package that gathers data from the Yahoo API and plots and returns historical swing highs and swing lows",
    long_description=long_description,
    url="https://github.com/leoi137/Support-and-Resistance-Algorithm/edit/master/README.md",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: >3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)