import numpy as np
from pyLIMA.toolbox import brightness_transformation


def test_magnitude_to_flux():
    flux = brightness_transformation.magnitude_to_flux(18.76)

    assert flux == 2857.5905433749376


def test_flux_to_magnitude():
    mag = brightness_transformation.flux_to_magnitude(18.76)

    assert mag == 24.216917914892385


def test_error_magnitude_to_error_flux():
    err_mag = 0.189
    flux = 27.9

    err_flux = brightness_transformation.error_magnitude_to_error_flux(err_mag, flux)

    assert np.allclose(err_flux, 4.856704581546761)


def test_error_flux_to_error_magnitude():
    flux = 27.9
    error_flux = 4.856704581546761
    err_mag = brightness_transformation.error_flux_to_error_magnitude(error_flux, flux)

    assert np.allclose(err_mag, 0.189)


def test_noisy_observations():
    flux = 10

    flux_obs = brightness_transformation.noisy_observations(flux, exp_time=None)

    assert flux_obs != flux
