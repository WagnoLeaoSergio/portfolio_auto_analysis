"""Python setup.py for portfolio_auto_analysis package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("portfolio_auto_analysis", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="portfolio_auto_analysis",
    version=read("portfolio_auto_analysis", "VERSION"),
    description="Awesome portfolio_auto_analysis created by WagnoLeaoSergio",
    url="https://github.com/WagnoLeaoSergio/portfolio_auto_analysis/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="WagnoLeaoSergio",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["portfolio_auto_analysis = portfolio_auto_analysis.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
