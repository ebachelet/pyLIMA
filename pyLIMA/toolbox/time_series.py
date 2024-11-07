import numpy as np
from astropy.table import QTable


def clean_time_series(data):
    """
    Check an array of non-finite and duplicates values

    Parameters
    ----------
    data : array, the array to clean

    Returns
    -------
    good_lines : list, the list of index containing correct values
    non_finite_lines : list, the list of index containing non-finite values
    non_unique_lines : list, the list of index containing duplicate values
    """

    dataset = [data[key].value for key in data.columns.keys()]
    dataset = np.c_[dataset].T

    finite_data = np.isfinite(dataset)
    finite_lines = np.all(finite_data, axis=1)

    for key in data.columns.keys():

        if 'err' in key:
            mask = data[key].value != 0

            finite_lines = finite_lines & mask

    unique_values, unique_index = np.unique(data['time'].value, return_index=True)

    lines = np.arange(0, len(data))

    good_lines = [line for line in lines if
                  (line in unique_index) & (finite_lines[line])]
    non_finite_lines = lines[~finite_lines].tolist()
    non_unique_lines = [i for i in lines if i not in unique_index]

    return good_lines, non_finite_lines, non_unique_lines


def construct_time_series(data, columns_names, column_units):
    """
    Construct an astropy table based on data, column names and columns units

    Parameters
    ----------
    data : array, the array containing data
    columns_names : array, the columns names
    columns_units : array,the columns units

    Returns
    -------
    table : array, the astropy table
    """

    table = QTable(data, names=columns_names, units=column_units)
    time_sorted_table = table[table['time'].argsort()]

    return time_sorted_table
