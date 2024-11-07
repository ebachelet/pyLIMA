import numpy as np
from pyLIMA import telescopes

def create_a_fake_telescope(lightcurve=None, astrometry=None,
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

    fake_telescope = telescopes.Telescope(name=name, lightcurve=lightcurve,
                                     lightcurve_names=['time', 'mag', 'err_mag'],
                                     lightcurve_units=['JD', 'mag', 'mag'],
                                     astrometry=astrometry,
                                     astrometry_names=['time', 'ra', 'err_ra', 'dec',
                                                       'err_dec'],
                                     astrometry_units=['JD', astrometry_unit,
                                                       astrometry_unit, astrometry_unit,
                                                       astrometry_unit])

    return fake_telescope


def replicate_a_telescope(microlensing_model, telescope_index, lightcurve_time=None,
                          astrometry_time=None):

    original_telescope = microlensing_model.event.telescopes[telescope_index]

    if lightcurve_time is not None:

        model_time = lightcurve_time
        model_lightcurve = np.c_[
            model_time, [0] * len(model_time), [0.1] * len(model_time)]
    else:

        model_lightcurve = None

    if astrometry_time is not None:

        model_time = astrometry_time
        model_astrometry = np.c_[
            model_time, [0] * len(model_time), [0.1] * len(model_time), [
                0] * len(model_time), [0.1] * len(model_time)]

        unit_astrometry = original_telescope.astrometry['ra'].unit
    else:

        model_astrometry = None
        unit_astrometry='deg'

    model_telescope = create_a_fake_telescope(lightcurve=model_lightcurve,
                            astrometry=model_astrometry,
                            name=original_telescope.name,
                            astrometry_unit=unit_astrometry)

    attributes_to_copy = ['name','filter','location','ld_gamma','ld_sigma','ld_a1',
                          'ld_a2', 'ld_gamma1','ld_gamma2','location','spacecraft_name',
                          'pixel_scale']

    for key in attributes_to_copy:

        try:
            setattr(model_telescope, key, getattr(original_telescope, key))
        except AttributeError:
            pass
    if microlensing_model.parallax_model[0] != 'None':
        model_telescope.initialize_positions()
        model_telescope.compute_parallax(microlensing_model.parallax_model,
                                     microlensing_model.event.North,
                                     microlensing_model.event.East)  # ,
    return model_telescope