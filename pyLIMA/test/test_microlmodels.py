# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:08:11 2016

@author: ebachelet
"""
import collections

import mock
import numpy as np

from pyLIMA import microlmodels


def _create_event():
    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].name = 'Test'
    event.telescopes[0].lightcurve_flux = np.array([[0, 1, 1], [42, 6, 6]])

    return event


def test_define_parameters_model_dictionnary():
    event = _create_event()

    Model = microlmodels.MLModels(event, model='FSPL')

    assert Model.model_dictionnary.keys() == ['to', 'uo', 'tE', 'rho', 'fs_Test', 'g_Test']
    assert Model.model_dictionnary.values() == [0, 1, 2, 3, 4, 5]


def test_define_parameters_parameters_boundaries():
    event = _create_event()

    Model = microlmodels.MLModels(event, model='FSPL')

    assert Model.parameters_boundaries == [(-300, 342), (-2.0, 2.0), (1.0, 300), (1e-5, 0.05)]


def test_magnification_FSPL_computation():
    event = _create_event()

    Model = microlmodels.MLModels(event, model='FSPL')
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'rho'])
    parameters = Parameters(0, 0.1, 1, 5e-2)

    amplification, impact_parameter = Model.model_magnification(event.telescopes[0], parameters)

    # CHECK THIS!!!
    assert np.allclose(amplification, np.array([0.0, 1.00]))
    assert np.allclose(impact_parameter, np.array([0.1, 42.0]))

def test_magnification_PSPL_computation():
    event = _create_event()

    Model = microlmodels.MLModels(event, model='PSPL')
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE'])
    parameters = Parameters(0, 0.1, 1)

    amplification, impact_parameter = Model.model_magnification(event.telescopes[0], parameters)
    assert np.allclose(amplification,np.array([10.03746101, 1.00]))
    assert np.allclose(impact_parameter, np.array([0.1, 42.0]))


def test_compute_parallax():

    #NEED TO DO PARALLAX FIRST
    return

def test_compute_parallax_curvature():

    #NEED TO DO PARALLAX FIRST
    return