# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:39:32 2015

@author: ebachelet
"""
import collections
import contextlib
import StringIO

_claret_columns = 'logg, Teff, metallicity, microturbulent_velocity, linear_limb_darkening, ' \
                  'filter, method, model'
_claret_type = collections.namedtuple('ClaretType', _claret_columns)


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

            claret_datum = _claret_type(*data)

            if claret_datum.filter == camera_filter:
                yield claret_datum
            elif camera_filter == 'all':
                yield claret_datum


def _convert_datum(datum):
    try:
        return float(datum)
    except ValueError:
        return datum
