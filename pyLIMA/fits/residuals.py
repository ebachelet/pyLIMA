import numpy as np

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
        errflux = lightcurve['err_flux'].value

        microlensing_model = model.compute_the_microlensing_model(telescope, pyLIMA_parameters)

        residuals = (flux - microlensing_model[0]) / errflux

        return residuals

    except:

        return []

def all_telescope_photometric_residuals(event, model, pyLIMA_parameters):
    """ Compute the residuals of all telescopes according to the model.

    :param object pyLIMA_parameters: object containing the model parameters, see microlmodels for more details

    :return: the residuals in flux,
    :rtype: list, a list of array of residuals in flux
    """

    residuals = []
    for telescope in event.telescopes:
        # Find the residuals of telescope observation regarding the parameters and model
        residus = photometric_residuals(telescope, model, pyLIMA_parameters)

        residuals = np.append(residuals,residus)

    return residuals

def photometric_chi2(telescope, model, pyLIMA_parameters):

    try:
        residuals = photometric_residuals(telescope, model, pyLIMA_parameters)
        chi2 = np.sum(residuals**2)

        return chi2

    except:

        return 0

def all_telescope_photometric_chi2(event, model, pyLIMA_parameters):

    CHI2 = 0
    for telescope in event.telescopes:

        chi2 = photometric_chi2(telescope, model, pyLIMA_parameters)
        CHI2 += chi2

    return CHI2