import os.path

import numpy as np
import mock
from pyLIMA import microltoolbox


def test_chichi():

	magic_residuals = mock.MagicMock()
	
	parameters = []
	magic_residuals.residuals_fn.return_value = np.array([0,1,50])
 
        
	parameters2 = [0,6,5]                                 
	chichi = microltoolbox.chichi(magic_residuals.residuals_fn, parameters2)
	
	assert chichi == 2501

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
	assert np.allclose(error_magnitude, np.abs(-2.5 * error_flux / (flux * np.log(10))))

def test_error_flux_to_error_magnitude():

	flux = np.array([1000,  3000,  10000])
	error_flux = np.array([100,  100,  100])

	error_magnitude = microltoolbox.error_flux_to_error_magnitude(error_flux, flux)

	assert len(error_magnitude) == len(error_flux)
	assert np.allclose(error_magnitude, np.abs(np.array([-0.10857362, -0.03619121, -0.01085736])))


def test_align_the_data_to_the_reference_telescope():

	fit = mock.MagicMock()
	event = mock.MagicMock()
	model = mock.MagicMock()
	model.compute_the_microlensing_model.return_value = [1,0,0]
	event.telescopes = []


	telescope_0 = mock.MagicMock()
	telescope_0.name = 'Survey'
	telescope_0.lightcurve_flux = np.random.random((100, 3))
	telescope_0.lightcurve_magnitude = np.random.random((100, 3))
	event.telescopes.append(telescope_0)

	telescope_1 = mock.MagicMock()
	telescope_1.name = 'Followup'
	telescope_1.lightcurve_flux = np.random.random((100, 3))*2
	telescope_1.lightcurve_magnitude = np.random.random((100, 3))*2
	event.telescopes.append(telescope_1)



	fit.event = event
	fit.model = model
	expected_lightcurves = microltoolbox.align_the_data_to_the_reference_telescope(fit)

	residuals = 2.5*np.log10(1/telescope_1.lightcurve_flux[:,1])
	lightcurve = np.c_[telescope_1.lightcurve_flux[:,0], 27.4 + residuals, np.abs(2.5/np.log(10)*
																		   telescope_1.lightcurve_flux[:,2]/
																		   telescope_1.lightcurve_flux[:,1])]

	assert np.allclose(expected_lightcurves[0], event.telescopes[0].lightcurve_magnitude)
	assert np.allclose(expected_lightcurves[1], lightcurve)