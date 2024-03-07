try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

PACKAGE_DATA = files(__name__)
"""Data stored in pyLIMA.data module"""
