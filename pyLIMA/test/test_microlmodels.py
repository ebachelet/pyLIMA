# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:08:11 2016

@author: ebachelet
"""
import mock
import numpy as np

from pyLIMA import microlmodels


def test_define_parameters_model_dictionnary():

    event = mock.MagicMock()

    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].name = 'Test'
    event.telescopes[0].lightcurve = np.array([[0,1,1],[42,6,6]])


    Model = microlmodels.MLModels(event,model='FSPL')
    assert Model.model_dictionnary.keys() == ['to', 'uo', 'tE', 'rho', 'fs_Test', 'g_Test']
    assert Model.model_dictionnary.values() == [0, 1, 2, 3, 4, 5]

def test_define_parameters_parameters_boundaries():

    event = mock.MagicMock()

    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].lightcurve = np.array([[0,1,1],[42,6,6]])



    Model = microlmodels.MLModels(event,model='FSPL')
    assert Model.parameters_boundaries == [(-300, 342), (1e-05, 2.0), (1.0, 300), (0.0001, 0.05)]



def test_magnification_FSPL_call():

    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].lightcurve = np.array([[0,1,1],[42,6,6]])

    Model = microlmodels.MLModels(event,'FSPL')


    parameters = [0,0,0,0]
    time = np.array([0,0])
    Magnification = mock.MagicMock()
    Magnification.amplification_FSPL = mock.MagicMock()

    Model.magnification(parameters,time)
    import pdb; pdb.set_trace()

    Magnification.amplification_FSPL.assert_called_with(np.array([0,0]), 0, 0, 0, Model.yoo_table)


def test_magnification_FSPL_computation():

    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].lightcurve = np.array([[0,1,1],[42,6,6]])

    Model = microlmodels.MLModels(event,model='FSPL')
    parameters = [0,0.1,1,0.02]
    time = np.array([0,1])

    Ampli, U = Model.magnification(parameters,time)

    assert np.allclose(Ampli,np.array([ 10.08841727,   1.33816124]))
    assert np.allclose(U, np.array([ 0.1,   1.00498756]))


def test_magnification_PSPL_computation():

    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].lightcurve = np.array([[0,1,1],[42,6,6]])


    Model = microlmodels.MLModels(event,model='PSPL')
    parameters = [0,0.1,1,0.02]
    time = np.array([0,1])

    Ampli, U = Model.magnification(parameters,time)
    assert np.allclose(Ampli,np.array([ 10.03746101,   1.33809499]))
    assert np.allclose(U, np.array([ 0.1,   1.00498756]))


def test_compute_parallax():

    #NEED TO DO PARALLAX FIRST
    return

def test_compute_parallax_curvature():

    #NEED TO DO PARALLAX FIRST
    return