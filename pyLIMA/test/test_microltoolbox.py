import os.path

import numpy as np

from pyLIMA import microltoolbox


def test_magnitude_to_flux():

	magnitude = np.array([ 18.0,  19.0,  17.0])
	flux = microltoolbox.magnitude_to_flux(magnitude)
	
	assert len(magnitude) == len(flux)
	assert flux[0]>flux[1]
        assert flux[2]>flux[1]
	assert flux[2]>flux[0]                                                   

def test_flux_to_magnitude():

	flux = np.array([5000,  7000,  15000])
	magnitude = microltoolbox.flux_to_magnitude(flux)
	
	assert len(magnitude) == len(flux)
	assert magnitude[0] > magnitude[1]
        assert magnitude[1] > magnitude[2]

def test_error_magnitude_to_error_flux():

	flux = np.array([1000,  3000,  10000])
	error_magnitude = np.array([0.1,  0.01,  0.001])

	error_flux = microltoolbox.error_magnitude_to_error_flux(error_magnitude, flux)

	assert len(error_magnitude) == len(error_flux)
	assert np.allclose(error_magnitude, -2.5 * error_flux / (flux * np.log(10)))

def test_error_flux_to_error_magnitude():

	flux = np.array([1000,  3000,  10000])
	error_flux = np.array([100,  100,  100])

	error_magnitude = microltoolbox.error_flux_to_error_magnitude(error_flux, flux)

	assert len(error_magnitude) == len(error_flux)
	assert np.allclose(error_magnitude, np.array([-0.10857362, -0.03619121, -0.01085736]))
