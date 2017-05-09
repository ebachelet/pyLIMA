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


def align_the_data_to_the_reference_telescope(fit):
    """Align data to the survey telescope (i.e telescope 0). Used to plot fit results. Ugly microlensing alignement....

    :param object fit: a microlfits object

    :return: the aligned to survey lightcurve in magnitude
    :rtype: array_like
    """

    reference_telescope = fit.event.telescopes[0]
    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(fit.fit_results)

    normalised_lightcurve = []
    for telescope in fit.event.telescopes:

        if telescope.name == reference_telescope.name:

            lightcurve = telescope.lightcurve_magnitude

        else:

            telescope_ghost = copy.copy(telescope)
            telescope_ghost.name = reference_telescope.name
            telescope_ghost.filter = reference_telescope.filter
            # import pdb;
            # pdb.set_trace()
            model_ghost = fit.model.compute_the_microlensing_model(telescope_ghost, pyLIMA_parameters)[0]
            model_telescope = fit.model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[0]

            time = telescope.lightcurve_flux[:, 0]
            flux = telescope.lightcurve_flux[:, 1]
            error_flux = telescope.lightcurve_flux[:, 2]
            err_mag = error_flux_to_error_magnitude(error_flux, flux)

            residuals = 2.5 * np.log10(model_telescope / flux)

            magnitude_normalised = flux_to_magnitude(model_ghost) + residuals

            lightcurve_normalised = [time, magnitude_normalised, err_mag]

            lightcurve_mag_normalised = np.array(lightcurve_normalised).T

            lightcurve = lightcurve_mag_normalised

        normalised_lightcurve.append(lightcurve)

    return normalised_lightcurve
