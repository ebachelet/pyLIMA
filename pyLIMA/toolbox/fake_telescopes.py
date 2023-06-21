from pyLIMA import telescopes


def create_a_fake_telescope(light_curve=None, astrometry_curve=None,
                            name='A Fake Telescope', astrometry_unit='deg'):
    """
    Create a telescope for plots

    Parameters
    ----------
    light_curve : array, the lightcurves in magnitude
    astrometry_curve : array, the astrometric time series
    name : str, the telescope name
    astrometry_unit : str, the unit of astrometry

    Returns
    -------
    telescope : object, a telescope object
    """

    telescope = telescopes.Telescope(name=name, light_curve=light_curve,
                                     light_curve_names=['time', 'mag', 'err_mag'],
                                     light_curve_units=['JD', 'mag', 'mag'],
                                     astrometry=astrometry_curve,
                                     astrometry_names=['time', 'ra', 'err_ra', 'dec',
                                                       'err_dec'],
                                     astrometry_units=['JD', astrometry_unit,
                                                       astrometry_unit, astrometry_unit,
                                                       astrometry_unit])

    return telescope
