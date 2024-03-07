"""Provides access to data files stored in pyLIMA.data package"""
try:
    from importlib.resources import files
except ImportError:
    # Backport for Python < 3.9
    from importlib_resources import files

PACKAGE_DATA = files(__name__)
"""Represents data stored in pyLIMA.data package. This object is like a folder
that returns files when using ``/`` or PACKAGE_DATA.joinpath(),

Examples:

    >>> from pyLIMA.data import PACKAGE_DATA
    >>> template = PACKAGE_DATA / "Yoo_B0B1.dat"
    >>> print(template.read_text())
    0.00100000 0.00200000 -0.00035619 -0.00018505
    0.00200000 0.00400000 -0.00071238 -0.00037009
    ...

    # Get true path of the file
    >>> from importlib.resources import as_file
    >>> print(as_file(PACKAGE_DATA / "Yoo_B0B1.dat"))
    /path/to/site-packages/pyLIMA/data/Yoo_B0B1.dat

See Python docs on `importlib.resources`.
"""
