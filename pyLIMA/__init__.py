from importlib.metadata import version

try:

    __version__ = version('pyLIMA')

except ImportError:

    pass
