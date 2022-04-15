from astropy.table import QTable




def construct_time_series(data, columns_names, column_units):

    table = QTable(data, names=columns_names,units=column_units)

    return table
