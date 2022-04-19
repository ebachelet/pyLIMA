import numpy as np
import pyLIMA.toolbox.brightness_transformation


def photometric_residuals(telescope, model, pyLIMA_parameters):
    """ Compute the residuals of a telescope lightcurve according to the model.

    :param object telescope: a telescope object. More details in telescopes module.
    :param object pyLIMA_parameters: object containing the model parameters, see microlmodels for more details

    :return: the residuals in flux, the priors
    :rtype: array_like, float
    """
    try:
        lightcurve = telescope.lightcurve_flux

        flux = lightcurve['flux'].value

        microlensing_model = model.compute_the_microlensing_model(telescope, pyLIMA_parameters)

        residuals = flux - microlensing_model['flux']

        return residuals

    except:

        return []

def norm_photometric_residuals(telescope, model, pyLIMA_parameters):
    """ Compute the residuals of a telescope lightcurve according to the model.

    :param object telescope: a telescope object. More details in telescopes module.
    :param object pyLIMA_parameters: object containing the model parameters, see microlmodels for more details

    :return: the residuals in flux, the priors
    :rtype: array_like, float
    """
    try:
        lightcurve = telescope.lightcurve_flux

        inv_err_flux = lightcurve['inv_err_flux'].value

        residuals = photometric_residuals(telescope, model, pyLIMA_parameters)

        norm_residuals = residuals*inv_err_flux

        return norm_residuals

    except:

        return []



def all_telescope_photometric_residuals(model, pyLIMA_parameters):
    """ Compute the residuals of all telescopes according to the model.

    :param object pyLIMA_parameters: object containing the model parameters, see microlmodels for more details

    :return: the residuals in flux,
    :rtype: list, a list of array of residuals in flux
    """

    residuals = []
    for telescope in model.event.telescopes:
        # Find the residuals of telescope observation regarding the parameters and model
        residus = photometric_residuals(telescope, model, pyLIMA_parameters)

        residuals = np.append(residuals, residus)

    return residuals

def all_telescope_norm_photometric_residuals(model, pyLIMA_parameters):
    """ Compute the residuals of all telescopes according to the model.

    :param object pyLIMA_parameters: object containing the model parameters, see microlmodels for more details

    :return: the residuals in flux,
    :rtype: list, a list of array of residuals in flux
    """

    residuals = []
    for telescope in model.event.telescopes:
        # Find the residuals of telescope observation regarding the parameters and model
        residus = norm_photometric_residuals(telescope, model, pyLIMA_parameters)

        residuals = np.append(residuals, residus)

    return residuals




def photometric_chi2(telescope, model, pyLIMA_parameters):

    try:
        residuals = norm_photometric_residuals(telescope, model, pyLIMA_parameters)
        chi2 = np.sum(residuals**2)

        return chi2

    except:

        return 0

def all_telescope_photometric_chi2(model, pyLIMA_parameters):

    CHI2 = 0
    for telescope in model.event.telescopes:

        chi2 = photometric_chi2(telescope, model, pyLIMA_parameters)
        CHI2 += chi2

    return CHI2


def photometric_residuals_in_magnitude(telescope, model, pyLIMA_parameters):
    """ Compute the residuals of a telescope lightcurve according to the model.

    :param object telescope: a telescope object. More details in telescopes module.
    :param object pyLIMA_parameters: object containing the model parameters, see microlmodels for more details

    :return: the residuals in flux, the priors
    :rtype: array_like, float
    """
    try:
        lightcurve = telescope.lightcurve_magnitude

        mag = lightcurve['mag'].value

        microlensing_model = model.compute_the_microlensing_model(telescope, pyLIMA_parameters)

        microlensing_model = pyLIMA.toolbox.brightness_transformation.ZERO_POINT-2.5*np.log10(microlensing_model['flux'])

        residuals = mag - microlensing_model

        return residuals

    except:

        return []