import os.path

import numpy as np
import mock


from pyLIMA import microlguess

def _create_event():
    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].name = 'Test'
    event.telescopes[0].lightcurve_magnitude = np.array([[0, 1, 1], [42, 6, 6]])
    event.telescopes[0].lightcurve_flux = np.array([[0, 1, 1], [42, 6, 6]])
    event.telescopes[0].gamma = 0.5
    return event

def test_initial_guess_PSPL():

	event = _create_event()
	
	guesses = microlguess.initial_guess_PSPL(event)
	
	assert len(guesses) == 2
	assert len(guesses[0]) == 3
	

	
def test_initial_guess_FSPL():

	event = _create_event()
	
	guesses = microlguess.initial_guess_FSPL(event)
	
	assert len(guesses) == 2
	assert len(guesses[0]) == 4

def test_differential_evolution_parameters_boundaries():

	event = _create_event()
	model = mock.MagicMock()
	model.paczynski_model = 'PSPL'
	model.parallax_model = ['None']
	model.xallarap_model = ['None']
	model.parallax_model = ['None']
	parameters_boundaries = microlguess.differential_evolution_parameters_boundaries(event, model)

	assert len(parameters_boundaries) == 3
	
