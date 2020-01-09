# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:18:15 2016

@author: ebachelet
"""

import numpy as np
import copy

# magnitude reference
MAGNITUDE_CONSTANT = 27.4


def chichi(residuals_fn, fit_process_parameters):
    """Return the chi^2 .


    :param func residuals_fn: a function which compute the residuals
    :param list fit_process_parameters: the model parameters ingested by the correpsonding
                                        fitting routine.

    :returns: the chi^2

    :rtype: float
    """

    residuals = residuals_fn(fit_process_parameters)
    _chichi = np.sum(residuals ** 2)

    return _chichi


def magnitude_to_flux(magnitude):
    """ Transform the injected magnitude to the the corresponding flux.

    :param array_like magnitude: the magnitude you want to transform.

    :return: the transformed magnitude in flux unit
    :rtype: array_like
    """

    flux = 10 ** ((MAGNITUDE_CONSTANT - magnitude) / 2.5)

    return flux


def flux_to_magnitude(flux):
    """ Transform the injected flux to the the corresponding magnitude.

    :param array_like flux: the flux you want to transform.

    :return: the transformed magnitude
    :rtype: array_like
    """

    mag = MAGNITUDE_CONSTANT - 2.5 * np.log10(flux)

    return mag


def error_magnitude_to_error_flux(error_magnitude, flux):
    """ Transform the injected magnitude error to the the corresponding error in flux.

    :param array_like error_magnitude: the magnitude errors measurements you want to transform.
    :param array_like flux: the fluxes corresponding to these errors

    :return: the transformed errors in flux units
    :rtype: array_like
    """

    error_flux = np.abs(-error_magnitude * flux * np.log(10) / 2.5)

    return error_flux


def error_flux_to_error_magnitude(error_flux, flux):
    """ Transform the injected flux error to the the corresponding error in magnitude.

    :param array_like error_flux: the flux errors measurements you want to transform.
    :param array_like flux: the fluxes corresponding to these errors

    :return: the transformed errors in magnitude
    :rtype: array_like
    """
    error_magnitude = np.abs(-2.5 * error_flux / (flux * np.log(10)))

    return error_magnitude


def MCMC_compute_fs_g(fit, mcmc_chains):
    """ Compute the corresponding source flux fs and blending factor g corresponding to each mcmc
    chain.

    :param fit: a fit object. See the microlfits for more details.
    :param mcmc_chains: a numpy array representing the mcmc chains.
    :return: a numpy array containing the corresponding fluxes parameters
    :rtype: array_type

    """
    fluxes_chains = np.zeros((len(mcmc_chains), 2 * len(fit.event.telescopes)))
    for i in range(len(mcmc_chains)):
        fluxes = fit.find_fluxes(mcmc_chains[i], fit.model)
        fluxes_chains[i] = fluxes

    return fluxes_chains


def align_the_data_to_the_reference_telescope(fit, telescope_index = 0, parameters = None) :
    """Align data to the telescope_index. Used to plot fit results. Ugly microlensing alignement....

    :param object fit: a microlfits object
    :param int telescope_index: the reference telescope



    :return: the aligned to survey lightcurve in magnitude
    :rtype: array_like
    """
    if parameters is not None:

        pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(parameters)
    else:
        pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(fit.fit_results)


    reference_telescope = fit.event.telescopes[telescope_index]
    fs_ref = getattr(pyLIMA_parameters, 'fs_' + reference_telescope.name)

    if fit.model.blend_flux_ratio == True:

        g_ref = getattr(pyLIMA_parameters, 'g_' + reference_telescope.name)

    else:

        fb_ref = getattr(pyLIMA_parameters, 'fb_' + reference_telescope.name)



    normalised_lightcurve = []
    for telescope in fit.event.telescopes:

        flux = telescope.lightcurve_flux[:, 1]

        flux_model = fit.model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[0]

        residuals = 2.5 * np.log10(flux_model / flux)

        if telescope.name == reference_telescope.name:

            lightcurve = telescope.lightcurve_magnitude

        else:

            fs = getattr(pyLIMA_parameters, 'fs_' + telescope.name)

            if fit.model.blend_flux_ratio == True:

                g = getattr(pyLIMA_parameters, 'g_' + telescope.name)

                amp = fit.model.model_magnification(telescope, pyLIMA_parameters)


                flux_normalised = fs_ref*(amp+g_ref)

            else:

                fb = getattr(pyLIMA_parameters, 'fb_' + telescope.name)
                amp = fit.model.model_magnification(telescope, pyLIMA_parameters)

                flux_normalised = fs_ref * amp + fb_ref

            magnitude_normalised = flux_to_magnitude(flux_normalised)+residuals

            time = telescope.lightcurve_magnitude[:,0]
            err_mag = telescope.lightcurve_magnitude[:,2]

            lightcurve_normalised = [time, magnitude_normalised, err_mag]

            lightcurve = np.array(lightcurve_normalised).T



        normalised_lightcurve.append(lightcurve)


    return normalised_lightcurve


