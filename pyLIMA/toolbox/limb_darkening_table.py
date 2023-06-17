import collections
import contextlib

try:
    import StringIO

except ImportError:

    import io as StringIO

_CLARET_COLUMNS = 'log_g, Teff, metallicity, microturbulent_velocity, ' \
                  'linear_limb_darkening, ' \
                  'filter, method, model'
_ClaretType = collections.namedtuple('ClaretType', _CLARET_COLUMNS)


def read_claret_data(file_name, camera_filter):
    """
    Read in claret data from file.

    :param file_name: Path and name of data file.
    :param camera_filter: Retrieve data for supplied filter.
    :return: Generator of claret table.
    """

    try:
        resource = open(file_name)
    except IOError:
        resource = contextlib.closing(StringIO.StringIO(file_name))

    with resource as file_socket:
        for line in file_socket.readlines():
            data = [_convert_datum(x) for x in line.strip().split()]

            claret_datum = _ClaretType(*data)

            if claret_datum.filter == camera_filter:
                yield claret_datum
            elif camera_filter == 'all':
                yield claret_datum


def _convert_datum(datum):
    """
    Convert a datum to a float (if possible).
    :param datum: Datum to convert.
    :return: Float or original if not convertible
    """
    try:
        return float(datum)
    except ValueError:
        return datum
