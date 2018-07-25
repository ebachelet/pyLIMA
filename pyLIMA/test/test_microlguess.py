import numpy as np
import mock

from pyLIMA import microlguess


def _create_event():
    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].name = 'Test'
    event.telescopes[0].lightcurve_magnitude = np.array([[0, 1, 1], [42, 6, 6],[43, 5, 1], [54, 8, 6]])
    event.telescopes[0].lightcurve_flux = np.array([[0, 1, 1], [42, 6, 6], [43, 5, 1], [54, 8, 6]])
    event.telescopes[0].gamma = 0.5
    event.telescopes[0].filter = 'I'
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


def test_initial_guess_DSPL():
    event = _create_event()

    guesses = microlguess.initial_guess_DSPL(event)

    assert len(guesses) == 2
    assert len(guesses[0]) == 6


def test_differential_evolution_parameters_boundaries_PSPL():
    event = _create_event()
    model = mock.MagicMock()
    model.event = event
    model.model_type = 'PSPL'
    model.parallax_model = ['None']
    model.xallarap_model = ['None']
    model.parallax_model = ['None']
    model.orbital_motion_model = ['None']
    parameters_boundaries = microlguess.differential_evolution_parameters_boundaries(model)

    assert len(parameters_boundaries) == 3

def test_differential_evolution_parameters_boundaries_FSPL():
    event = _create_event()
    model = mock.MagicMock()
    model.event = event
    model.model_type = 'FSPL'
    model.parallax_model = ['None']
    model.xallarap_model = ['None']
    model.parallax_model = ['None']
    model.orbital_motion_model = ['None']
    parameters_boundaries = microlguess.differential_evolution_parameters_boundaries(model)

    assert len(parameters_boundaries) == 4

def test_differential_evolution_parameters_boundaries_DSPL():
    event = _create_event()
    model = mock.MagicMock()
    model.event = event
    model.model_type = 'DSPL'
    model.parallax_model = ['None']
    model.xallarap_model = ['None']
    model.parallax_model = ['None']
    model.orbital_motion_model = ['None']
    parameters_boundaries = microlguess.differential_evolution_parameters_boundaries(model)

    assert len(parameters_boundaries) == 6

def test_MCMC_parameters_initialization():
    parameters = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]
    parameters_dictionnary = {'to' : 0, 'uo' : 1, 'tE' : 2, 'rho' : 3, 'fs_LCOGT' : 4, 'g_LCOGT' : 5}

    parameter_key_0 = 'to'
    trial_0 = microlguess.MCMC_parameters_initialization(parameter_key_0, parameters_dictionnary, parameters)[0]
    assert (trial_0>-1.0) & (trial_0<1.0) 


    parameter_key_1 = 'uo'
    trial_1 = microlguess.MCMC_parameters_initialization(parameter_key_1, parameters_dictionnary, parameters)[0]
    assert (trial_1>0.9*1.1) & (trial_1<1.1*1.1) 

    parameter_key_2 = 'tE'
    trial_2 = microlguess.MCMC_parameters_initialization(parameter_key_2, parameters_dictionnary, parameters)[0]
    assert (trial_2>0.9*2.2) & (trial_2<1.1*2.2) 

    parameter_key_3 = 'rho'
    trial_3 = microlguess.MCMC_parameters_initialization(parameter_key_3, parameters_dictionnary, parameters)[0]
    assert ( trial_3>0.1*3.3) & ( trial_3<10*3.3)
	

    parameter_key_4 = 'fs_LCOGT'
    trial_4 = microlguess.MCMC_parameters_initialization(parameter_key_4, parameters_dictionnary, parameters)
    assert len(trial_4) == 2
    assert (trial_4[0]>0.9*4.4) & (trial_4[0]<1.1*4.4)  
    assert 4.4*(1+5.5) == np.round(trial_4[0]*(1+trial_4[1]),5) 	
 
    parameter_key_5 = 'g_LCOGT'
    trial_5 = microlguess.MCMC_parameters_initialization(parameter_key_5, parameters_dictionnary, parameters)
    assert trial_5 == None




