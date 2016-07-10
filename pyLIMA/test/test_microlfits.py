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
    event.telescopes[0].lightcurve_flux = np.array([[0, 1, 1], [42, 6, 6], [43, 7, 1],[44, 8, 1],[45, 7, 1]])
    event.telescopes[0].lightcurve_magnitude = np.array([[0, 1, 1], [42, 6, 6],[43, 7, 1],[44, 8, 1],[45, 7, 1]])
    event.telescopes[0].gamma = 0.5
    return event

def _create_model(kind):
    model = mock.MagicMock()
    model.parameters_guess = []
    model.parameters_boundaries = [[0,100],[0,1],[0,300]]
    model.compute_pyLIMA_parameters.return_value = []
    model.Jacobian_flag = 'Not OK'
    model.model_dictionnary = {'to':0,'uo':1,'tE':2,'fs_Test':3, 'g_Test':4}
    model.pyLIMA_standards_dictionnary =  model.model_dictionnary
    model.compute_the_microlensing_model.return_value = 5*[0.1],0.0
    model.model_magnification.return_value = [1.2,1.3,1.4,1.5,1.6],0.0
    fancy_namedtuple = collections.namedtuple('Parameters', model.model_dictionnary.keys())
    model.pyLIMA_standard_parameters_to_fancy_parameters.return_value = fancy_namedtuple(10.0,0.1,20,10,5)
    model.model_type = kind
   
    return model

def test_mlfit_PSPL_LM_without_guess() :

	current_event = _create_event()
	model = _create_model('PSPL')
	fit = microlfits.MLFits(current_event)
	fit.mlfit(model,'LM')
	
	assert fit.fit_covariance.shape == (3+2*len(current_event.telescopes),3+2*len(current_event.telescopes))	
	assert len(fit.fit_results) == 3+2*len(current_event.telescopes)+1

def test_mlfit_PSPL_LM_with_guess() :

	current_event = _create_event()
	model = _create_model('PSPL')
	model.parameters_guess = [10,0.1,20]
	fit = microlfits.MLFits(current_event)
	fit.mlfit(model,'LM')
	
	assert fit.fit_covariance.shape == (3+2*len(current_event.telescopes),3+2*len(current_event.telescopes))	
	assert len(fit.fit_results) == 3+2*len(current_event.telescopes)+1


	



