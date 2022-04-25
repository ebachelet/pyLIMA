import numpy as np
import pyLIMA.toolbox.brightness_transformation


def photometric_residuals(flux, photometric_model):
    """ Compute the residuals of a telescope lightcurve according to the model.

    :param object telescope: a telescope object. More details in telescopes module.
    :param object pyLIMA_parameters: object containing the model parameters, see microlmodels for more details

    :return: the residuals in flux, the priors
    :rtype: array_like, float
    """

    residuals = flux - photometric_model

    return residuals


def all_telescope_photometric_residuals(model, pyLIMA_parameters, norm=False, rescaling_photometry_parameters=None):
    """ Compute the residuals of all telescopes according to the model.

    :param object pyLIMA_parameters: object containing the model parameters, see microlmodels for more details

    :return: the residuals in flux,
    :rtype: list, a list of array of residuals in flux
    """

    residuals = []
    errfluxes = []

    for ind,telescope in enumerate(model.event.telescopes):
        # Find the residuals of telescope observation regarding the parameters and model
        lightcurve = telescope.lightcurve_flux

        flux = lightcurve['flux'].value

        microlensing_model = model.compute_the_microlensing_model(telescope, pyLIMA_parameters)

        residus = photometric_residuals(flux, microlensing_model['flux'])

        if norm:

            if rescaling_photometry_parameters is not None:

                err_flux = (lightcurve['err_flux'].value**2+rescaling_photometry_parameters[ind]**2*microlensing_model['flux']**2)**0.5
                residus /= err_flux
                errfluxes = np.append(errfluxes, err_flux)

            else:

                residus *= lightcurve['inv_err_flux'].value
                errfluxes = np.append(errfluxes, lightcurve['err_flux'].value)

        residuals = np.append(residuals, residus)


    return residuals, errfluxes


def photometric_chi2(telescope, model, pyLIMA_parameters):

    pass

def all_telescope_photometric_chi2(model, pyLIMA_parameters,rescaling_parameters=None):

    pass


def all_telescope_photometric_likelihood(model, pyLIMA_parameters, rescaling_photometry_parameters=None):

    #CHI2 = 0
    #for telescope in model.event.telescopes:

    #    chi2 = photometric_chi2(telescope, model, pyLIMA_parameters)
    #    CHI2 += chi2

    #return CHI2

    residus, errflux = all_telescope_photometric_residuals(model, pyLIMA_parameters, norm=True,
                                                  rescaling_photometry_parameters=rescaling_photometry_parameters)

    chi2 = np.sum(residus**2)+2*np.sum(np.log(errflux))+len(errflux)*np.log(2*np.pi)

    return chi2


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