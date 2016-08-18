# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:46:17 2015

@author: ebachelet
"""
import mock
import numpy as np
import collections

from pyLIMA import microlfits


def _create_event():
    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].name = 'Test'
    event.telescopes[0].lightcurve_flux = np.random.random((100, 3))

    event.telescopes[0].lightcurve_magnitude = np.random.random((100, 3))

    event.telescopes[0].gamma = 0.5
    return event


def _create_model(kind):
    model = mock.MagicMock()
    model.parameters_guess = []
    model.parameters_boundaries = [[0, 100], [0, 1], [0, 300]]
    model.compute_pyLIMA_parameters.return_value = []
    model.Jacobian_flag = 'Not OK'
    model.model_dictionnary = {'to': 0, 'uo': 1, 'tE': 2, 'fs_Test': 3, 'g_Test': 4}
    model.pyLIMA_standards_dictionnary = model.model_dictionnary
    model.compute_the_microlensing_model.return_value = np.random.random(100).tolist(), 0.0
    model.model_magnification.return_value = np.random.random(100).tolist(), 0.0
    fancy_namedtuple = collections.namedtuple('Parameters', model.model_dictionnary.keys())
    model.pyLIMA_standard_parameters_to_fancy_parameters.return_value = fancy_namedtuple(10.0, 0.1, 20, 10, 5)
    model.model_type = kind

    return model


def test_mlfit_PSPL_LM_without_guess() :
	current_event = _create_event()
	model = _create_model('PSPL')
	fit = microlfits.MLFits(current_event)
	fit.mlfit(model,'LM')

	assert fit.fit_covariance.shape == (3+2*len(current_event.telescopes),3+2*len(current_event.telescopes))
	assert len(fit.fit_results) == 3+2*len(current_event.telescopes)+1

def test_mlfit_FSPL_LM_without_guess():

    current_event = _create_event()
    model = _create_model('FSPL')
    model.model_dictionnary = {'to': 0, 'uo': 1, 'tE': 2, 'rho': 3, 'fs_Test': 4, 'g_Test': 5}
    fancy_namedtuple = collections.namedtuple('Parameters', model.model_dictionnary.keys())
    model.pyLIMA_standard_parameters_to_fancy_parameters.return_value = fancy_namedtuple(10.0, 0.1, 20, 0.05, 10, 5)


    model.parameters_boundaries = [[0, 100], [0, 1], [0, 300], [0, 1]]
    fit = microlfits.MLFits(current_event)
    fit.mlfit(model, 'LM')

    assert fit.fit_covariance.shape == (4 + 2 * len(current_event.telescopes), 4 + 2 * len(current_event.telescopes))
    assert len(fit.fit_results) == 4 + 2 * len(current_event.telescopes) + 1


def test_mlfit_DSPL_LM_without_guess():
    current_event = _create_event()
    model = _create_model('DSPL')
    model.model_dictionnary = {'to1': 0, 'uo1': 1, 'delta_to': 2, 'uo2': 3, 'tE': 4, 'q_F_I': 5, 'fs_Test': 6,
                               'g_Test': 7}
    fancy_namedtuple = collections.namedtuple('Parameters', model.model_dictionnary.keys())
    model.pyLIMA_standard_parameters_to_fancy_parameters.return_value = fancy_namedtuple(10.0, 0.1, 20, 0.05, 20, 0.1,
                                                                                         10, 5)
    model.parameters_boundaries = [[0, 100], [0, 1], [0, 100], [0, 1], [0, 300], [0, 1]]
    fit = microlfits.MLFits(current_event)
    fit.mlfit(model, 'LM')

    assert fit.fit_covariance.shape == (6 + 2 * len(current_event.telescopes), 6 + 2 * len(current_event.telescopes))
    assert len(fit.fit_results) == 6 + 2 * len(current_event.telescopes) + 1


def test_mlfit_PSPL_LM_with_guess():
    current_event = _create_event()
    model = _create_model('PSPL')
    model.parameters_guess = [10, 0.1, 20]
    fit = microlfits.MLFits(current_event)
    fit.mlfit(model, 'LM')

    assert fit.fit_covariance.shape == (3 + 2 * len(current_event.telescopes), 3 + 2 * len(current_event.telescopes))
    assert len(fit.fit_results) == 3 + 2 * len(current_event.telescopes) + 1


def test_mlfit_PSPL_MCMC_with_guess():
    current_event = _create_event()
    model = _create_model('PSPL')
    model.parameters_guess = [10, 0.1, 20]
    fit = microlfits.MLFits(current_event)
    fit.mlfit(model, 'MCMC')

    assert fit.MCMC_chains.shape == (200, 100, 5)


def test_check_fit_bad_covariance():
    current_event = _create_event()
    model = _create_model('PSPL')
    fit = microlfits.MLFits(current_event)
    fit.fit_covariance = np.array([[-1.0, 0.0], [0.0, 0.0]])

    flag = fit.check_fit()

    assert flag == 'Bad Fit'


def test_check_fit_bad_rho():
    current_event = _create_event()
    model = _create_model('FSPL')
    model.model_dictionnary = {'to': 0, 'uo': 1, 'tE': 2, 'rho': 3, 'fs_Test': 4, 'g_Test': 5}
    fit = microlfits.MLFits(current_event)
    fit.model = model
    fit.fit_results = [0.0, 0.0, 0.0, -1.0, 1.0, 0.0]

    flag = fit.check_fit()

    assert flag == 'Bad Fit'

    fit.fit_results = [0.0, 0.0, 0.0, 0.8, 1.0, 0.0]

    flag = fit.check_fit()

    assert flag == 'Bad Fit'

    fit.fit_results = [0.0, 0.0, 0.0, 0.05, 1.0, 0.0]

    flag = fit.check_fit()

    assert flag == 'Good Fit'


def test_check_fit_source_flux():
    current_event = _create_event()
    model = _create_model('FSPL')
    model.model_dictionnary = {'to': 0, 'uo': 1, 'tE': 2, 'rho': 3, 'fs_Test': 4, 'g_Test': 5}
    fit = microlfits.MLFits(current_event)
    fit.model = model
    fit.fit_results = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    flag = fit.check_fit()

    assert flag == 'Good Fit'

    fit.fit_results = [0.0, 0.0, 0.0, 0.8, -1.0, 0.0]

    flag = fit.check_fit()

    assert flag == 'Bad Fit'
