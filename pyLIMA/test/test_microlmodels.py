# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:08:11 2016

@author: ebachelet
"""
import collections

import mock
import numpy as np
import pytest

from pyLIMA import microlmodels


def _create_event():
    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].name = 'Test'
    event.telescopes[0].lightcurve_flux = np.array([[0, 1, 1], [42, 6, 6]])
    event.telescopes[0].gamma = 0.5
    return event


def test_create_PSPL_model():
    pspl_model = microlmodels.create_model('PSPL')

    assert isinstance(pspl_model, microlmodels.ModelPSPL)


def test_create_FSPL_model():
    pspl_model = microlmodels.create_model('FSPL')

    assert isinstance(pspl_model, microlmodels.ModelFSPL)


def test_create_bad_model():
    # Both tests are equivalent

    # Using a context manager
    with pytest.raises(microlmodels.ModelException) as model_exception:
        microlmodels.create_model('BAD')
    assert 'Unknown model "BAD"' in str(model_exception)

    # Manually checking for an exception and error message
    try:
        microlmodels.create_model('BAD')
        pytest.fail()
    except microlmodels.ModelException as model_exception:
        assert 'Unknown model "BAD"' in str(model_exception)




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

    assert np.allclose(amplification, np.array([ 10.34817883,   1.00000064]))
    assert np.allclose(impact_parameter, np.array([0.1, 42.0]))

def test_magnification_PSPL_computation():
    event = _create_event()

    Model = microlmodels.MLModels(event, model='PSPL')
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE'])
    parameters = Parameters(0, 0.1, 1)

    amplification, impact_parameter = Model.model_magnification(event.telescopes[0], parameters)
    assert np.allclose(amplification,np.array([10.03746101, 1.00]))
    assert np.allclose(impact_parameter, np.array([0.1, 42.0]))


def test_PSPL_computate_microlensing_model():
    event = _create_event()

    Model = microlmodels.MLModels(event, model='PSPL')
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'fs_Test', 'g_Test'])
    parameters = Parameters(0, 0.1, 1, 10, 1)

    model, _ = Model.compute_the_microlensing_model(event.telescopes[0], parameters)
    assert np.allclose(model, np.array([10*(10.03746101+1), 10*(1.00+1)]))


def test_FSPL_computate_microlensing_model():
    event = _create_event()

    Model = microlmodels.MLModels(event, model='FSPL')
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'rho', 'fs_Test',
                                                       'g_Test'])
    parameters = Parameters(0, 0.1, 1, 5e-2, 10, 1)

    model, _ = Model.compute_the_microlensing_model(event.telescopes[0], parameters)
    assert np.allclose(model, np.array([10*(10.34817832+1), 10*(1.00+1)]))


def test_compute_parallax():

    #NEED TO DO PARALLAX FIRST
    return

def test_compute_parallax_curvature():

    #NEED TO DO PARALLAX FIRST
    return
