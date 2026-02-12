"""OpenG2G: GPU-to-Grid framework for distribution-level voltage regulation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("openg2g")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
