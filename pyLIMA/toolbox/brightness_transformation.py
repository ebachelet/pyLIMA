import numpy as np

# ZERO POINT AND EXPOSURE TIME MATCH ~Roman telescope by default, should match
#https://iopscience.iop.org/article/10.3847/1538-4365/aafb69/pdf
ZERO_POINT = 27.4
EXPOSURE_TIME = 50  # s


def magnitude_to_flux(magnitude):
    """
    Return flux from magnitude

    Parameters
    ----------
    magnitude : array, an array of magnitudes

    Returns
    -------
    flux : array, the corresponding fluxes
    """

    flux = 10 ** ((ZERO_POINT - magnitude) / 2.5)

    return flux


def flux_to_magnitude(flux):
    """
    Return magnitude from fluxes

    Parameters
    ----------
    flux : array, the corresponding fluxes

    Returns
    -------
    magnitude : array, an array of magnitudes
    """
    mag = ZERO_POINT - 2.5 * np.log10(flux)

    return mag


def error_magnitude_to_error_flux(error_magnitude, flux):
    """
    Return error in fluxes from error in magnitudes and fluxes

    Parameters
    ----------
    error_magnitude: array, the error in magnitudes
    flux : array, the corresponding fluxes

    Returns
    -------
    error_flux : array, an array of errors in flux
    """

    error_flux = np.abs(error_magnitude * flux * np.log(10) / 2.5)

    return error_flux


def error_flux_to_error_magnitude(error_flux, flux):
    """
    Return error in magnitudes from error in flux and fluxes

    Parameters
    ----------
    error_flux : array, an array of errors in flux
    flux : array, the corresponding fluxes

    Returns
    -------
    error_magnitude: array, the error in magnitudes
    """
    error_magnitude = np.abs(2.5 * error_flux / (flux * np.log(10)))

    return error_magnitude


def noisy_observations(flux, exp_time=None,efficiency=None):
    """
    Add Poisson noise to observations

    Parameters
    ----------
    flux : array, the corresponding fluxes
    exp_time : float, the exposure time in seconds

    Returns
    -------
    flux_observed : array, the observed flux
    err_flux_observed : array, the corresponding uncertainties
    """

    if exp_time is not None:

        exposure_time = exp_time

    else:

        exposure_time = np.copy(EXPOSURE_TIME)

    if efficiency is not None:

        exposure_time *= efficiency

    photons = flux * exposure_time

    photons_observed = np.random.poisson(photons)
    err_photons_observed = photons ** 0.5

    flux_observed = photons_observed / exposure_time
    err_flux_observed = err_photons_observed / exposure_time

    return flux_observed, err_flux_observed
