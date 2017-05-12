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
from pyLIMA import microlparallax

def _create_event():
    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].name = 'Test'
    event.telescopes[0].lightcurve_flux = np.array([[0, 1, 1], [42, 6, 6]])
    event.telescopes[0].filter = 'I'
    event.telescopes[0].gamma = 0.5
    return event


def test_create_PSPL_model():
    event = _create_event()
    pspl_model = microlmodels.create_model('PSPL', event)

    assert isinstance(pspl_model, microlmodels.ModelPSPL)


def test_create_FSPL_model():
    event = _create_event()
    fspl_model = microlmodels.create_model('FSPL', event)

    assert isinstance(fspl_model, microlmodels.ModelFSPL)


def test_create_DSPL_model():
    event = _create_event()
    dspl_model = microlmodels.create_model('DSPL', event)

    assert isinstance(dspl_model, microlmodels.ModelDSPL)

def test_create_USBL_model():
    event = _create_event()
    usbl_model = microlmodels.create_model('USBL', event)

    assert isinstance(usbl_model, microlmodels.ModelUSBL)

def test_create_PSBL_model():
    event = _create_event()
    psbl_model = microlmodels.create_model('PSBL', event)

    assert isinstance(psbl_model, microlmodels.ModelPSBL)

def test_create_RRLyraePL_model():
    event = _create_event()
    rrpl_model = microlmodels.create_model('RRLyraePL', event,model_arguments=[1])

    assert isinstance(rrpl_model, microlmodels.ModelRRLyraePL)

def test_create_RRLyraeFS_model():
    event = _create_event()
    rrfs_model = microlmodels.create_model('RRLyraeFS', event,model_arguments=[1])

    assert isinstance(rrfs_model, microlmodels.ModelRRLyraeFS)

def test_create_bad_model():
    # Both tests are equivalent
    event = _create_event()
    # Using a context manager
    with pytest.raises(microlmodels.ModelException) as model_exception:
        microlmodels.create_model('BAD', event)
    assert 'Unknown model "BAD"' in str(model_exception)

    # Manually checking for an exception and error message
    try:
        microlmodels.create_model('BAD', event)
        pytest.fail()
    except microlmodels.ModelException as model_exception:
        assert 'Unknown model "BAD"' in str(model_exception)


def test_define_parameters_model_dictionnary():
    event = _create_event()

    Model = microlmodels.create_model('FSPL', event)
    Model.define_model_parameters()
    assert Model.model_dictionnary.keys() == ['to', 'uo', 'tE', 'rho', 'fs_Test', 'g_Test']
    assert Model.model_dictionnary.values() == [0, 1, 2, 3, 4, 5]


def test_define_parameters_boundaries():
    event = _create_event()

    Model = microlmodels.create_model('FSPL', event)

    assert Model.parameters_boundaries == [(0,42), (0.0, 2.0), (1.0, 300), (5*10**-5, 0.05)]

def test_magnification_USBL_computation():
    event = _create_event()
    print event.telescopes[0].lightcurve_flux
    Model = microlmodels.create_model('USBL', event)
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'rho', 'logs', 'logq','alpha'])


    to = 0
    uo = 0.70710678
    tE = 1
    rho = 0.0033
    s = np.log10(1.0)
    q = np.log10(0.02)
    alpha = -np.pi/4
    parameters = Parameters(to, uo, tE, rho, s, q,alpha)

    magnification = Model.model_magnification(event.telescopes[0], parameters)
    assert np.allclose(magnification, np.array([1.6311724868,  1.00005927]))

    Model.USBL_windows = [min(event.telescopes[0].lightcurve_flux[:,0]), max(event.telescopes[0].lightcurve_flux[:,0])]
    assert np.allclose(magnification, np.array([1.6311724868,  1.00005927]))


def test_magnification_PSBL_computation():
    event = _create_event()
    print event.telescopes[0].lightcurve_flux
    Model = microlmodels.create_model('PSBL', event)
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'logs', 'logq','alpha'])


    to = 0
    uo = 0.70710678
    tE = 1
    s = np.log10(1.0)
    q = np.log10(0.02)
    alpha = -np.pi/4
    parameters = Parameters(to, uo, tE, s, q,alpha)

    magnification = Model.model_magnification(event.telescopes[0], parameters)


    assert np.allclose(magnification, np.array([1.63109244,  1.00000063]))



def test_magnification_DSPL_computation():
    event = _create_event()

    Model = microlmodels.create_model('DSPL', event)
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'delta_to', 'delta_uo', 'tE', 'q_flux_I'])
    to = 0.0
    uo = 0.1
    delta_to = 42.0
    delta_uo = -0.05
    tE = 5.0
    q_flux_I = 0.1

    parameters = Parameters(to, uo, delta_to, delta_uo, tE, q_flux_I)

    amplification = Model.model_magnification(event.telescopes[0], parameters)

    assert np.allclose(amplification, np.array([9.21590819, 2.72932227]))


def test_magnification_FSPL_computation():
    event = _create_event()

    Model = microlmodels.create_model('FSPL', event)
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'rho'])

    to = 0.0
    uo = 0.1
    tE = 1.0
    rho = 0.05
    parameters = Parameters(to, uo, tE, rho)

    amplification = Model.model_magnification(event.telescopes[0], parameters)

    assert np.allclose(amplification, np.array([10.34817883, 1.00000064]))


def test_magnification_PSPL_computation():
    event = _create_event()

    Model = microlmodels.create_model('PSPL', event)
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE'])

    to = 0.0
    uo = 0.1
    tE = 1.0

    parameters = Parameters(to, uo, tE)

    amplification = Model.model_magnification(event.telescopes[0], parameters)
    assert np.allclose(amplification, np.array([10.03746101, 1.00]))

def test_PSPL_computate_microlensing_model():
    event = _create_event()

    Model = microlmodels.create_model('PSPL', event)
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'fs_Test', 'g_Test'])

    to = 0.0
    uo = 0.1
    tE = 1.0
    fs = 10
    g = 1

    parameters = Parameters(to, uo, tE, fs, g)

    model,fs,fb = Model.compute_the_microlensing_model(event.telescopes[0], parameters)
    assert np.allclose(model, np.array([fs * (10.03746101 + g), fs * (1.00 + g)]))


def test_FSPL_computate_microlensing_model():
    event = _create_event()

    Model = microlmodels.create_model('FSPL', event)
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'rho', 'fs_Test',
                                                       'g_Test'])

    to = 0.0
    uo = 0.1
    tE = 1.0
    rho = 0.05
    fs = 10
    g = 1

    parameters = Parameters(to, uo, tE, rho, fs, g)

    model, fs, fb = Model.compute_the_microlensing_model(event.telescopes[0], parameters)
    assert np.allclose(model, np.array([fs * (10.34817832 + g), fs * (1.00 + g)]))


def test_DSPL_computate_microlensing_model():
    event = _create_event()
    Model = microlmodels.create_model('DSPL', event)
    Parameters = collections.namedtuple('parameters',
                                        ['to', 'uo', 'delta_to', 'delta_uo', 'tE', 'q_flux_I', 'fs_Test', 'g_Test'])
    to = 0.0
    uo = 0.1
    delta_to = 42.0
    delta_uo = -0.05
    tE = 5.0
    q_flux_I = 0.1
    fs = 10
    g = 1
    parameters = Parameters(to, uo, delta_to, delta_uo, tE, q_flux_I, fs, g)

    model, fs, fb= Model.compute_the_microlensing_model(event.telescopes[0], parameters)
    assert np.allclose(model, np.array([fs * (9.21590819 + g), fs * (2.72932227 + g)]))


def test_no_fancy_parameters_to_pyLIMA_standard_parameters():
    event = _create_event()
    Model = microlmodels.create_model('PSPL', event)
    parameters = [42, 51]
    fancy = Model.fancy_parameters_to_pyLIMA_standard_parameters(parameters)

    assert parameters == fancy


def test_one_fancy_parameters_to_pyLIMA_standard_parameters():
    event = _create_event()
    Model = microlmodels.create_model('FSPL', event)

    Model.fancy_to_pyLIMA_dictionnary = {'logrho': 'rho'}
    Model.pyLIMA_to_fancy = {'logrho': lambda parameters: np.log10(parameters.rho)}
    Model.fancy_to_pyLIMA = {'rho': lambda parameters: 10 ** parameters.logrho}
    Model.define_model_parameters()

    Parameters = [0.28, 0.1, 35.6, -1.30102]
    pyLIMA_parameters = Model.compute_pyLIMA_parameters(Parameters)

    assert pyLIMA_parameters.to == 0.28
    assert pyLIMA_parameters.uo == 0.1
    assert pyLIMA_parameters.tE == 35.6
    assert pyLIMA_parameters.logrho == -1.30102
    assert np.allclose(pyLIMA_parameters.rho, 0.05, rtol=0.001, atol=0.001)


def test_mixing_fancy_parameters_to_pyLIMA_standard_parameters():
    event = _create_event()
    Model = microlmodels.create_model('FSPL', event)

    Model.fancy_to_pyLIMA_dictionnary = {'tstar': 'tE', 'logrho': 'rho'}
    Model.pyLIMA_to_fancy = {'logrho': lambda parameters: np.log10(parameters.rho),
                             'tstar': lambda parameters: (parameters.uo * parameters.tE)}
    Model.fancy_to_pyLIMA = {'rho': lambda parameters: 10 ** parameters.logrho,
                             'tE': lambda parameters: (parameters.tstar / parameters.uo)}
    Model.define_model_parameters()

    tE = 35.6
    uo = 0.1

    Parameters = [0.28, uo, uo * tE, -1.30102]
    pyLIMA_parameters = Model.compute_pyLIMA_parameters(Parameters)

    assert pyLIMA_parameters.to == 0.28
    assert pyLIMA_parameters.uo == uo
    assert pyLIMA_parameters.tE == tE
    assert pyLIMA_parameters.tstar == uo * tE
    assert pyLIMA_parameters.logrho == -1.30102
    assert np.allclose(pyLIMA_parameters.rho, 0.05, rtol=0.001, atol=0.001)


def test_complicated_mixing_fancy_parameters_to_pyLIMA_standard_parameters():
    event = _create_event()
    Model = microlmodels.create_model('FSPL', event)

    Model.fancy_to_pyLIMA_dictionnary = {'tstar': 'tE', 'logrho': 'rho'}
    Model.pyLIMA_to_fancy = {'logrho': lambda parameters: np.log10(parameters.rho),
                             'tstar': lambda parameters: (parameters.logrho * parameters.tE)}
    Model.fancy_to_pyLIMA = {'rho': lambda parameters: 10 ** parameters.logrho,
                             'tE': lambda parameters: (parameters.tstar / parameters.logrho)}
    Model.define_model_parameters()

    tE = 35.6
    uo = 0.1
    logrho = -1.30102
    Parameters = [0.28, uo, tE * logrho, logrho]
    pyLIMA_parameters = Model.compute_pyLIMA_parameters(Parameters)

    assert pyLIMA_parameters.to == 0.28
    assert pyLIMA_parameters.uo == uo
    assert pyLIMA_parameters.tE == tE
    assert pyLIMA_parameters.tstar == np.log10(pyLIMA_parameters.rho) * tE
    assert pyLIMA_parameters.logrho == logrho
    assert np.allclose(pyLIMA_parameters.rho, 0.05, rtol=0.001, atol=0.001)


def test_compute_parallax_curvature():
    delta_positions = np.array([[1, 2], [3, 4]])
    piE = np.array([0.1, 1.8])

    delta_tau, delta_beta = microlparallax.compute_parallax_curvature(piE, delta_positions)

    assert len(delta_tau) == 2
    assert len(delta_beta) == 2
    assert np.allclose(delta_tau, -(
    piE[0] * delta_positions[0] + piE[1] * delta_positions[1]))  # scalar product, geocentric point of view (-)
    assert np.allclose(delta_beta, -(
    piE[0] * delta_positions[1] - piE[1] * delta_positions[0]))  # vectorial product, geocentric point of view (-)


def test_source_trajectory_with_parallax():
    to = 0.28
    tE = 35.6
    uo = 0.1
    piEN = 0.1
    piEE = -0.97

    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'piEN', 'piEE'])
    parameters = Parameters(to, uo, tE, piEN, piEE)

    telescope = mock.MagicMock()
    telescope.lightcurve_flux = np.array([[0, 1, 1], [42, 6, 6]])
    telescope.deltas_positions = np.array([[1, 2], [3, 4]])
    event = _create_event()
    event.telescopes[0] = telescope
    Model = microlmodels.create_model('PSPL', event,parallax=['Annual',0])


    source_X, source_Y = Model.source_trajectory(telescope, to, uo, tE, parameters)

    assert len(source_X) == len(source_Y) == 2

    tau = (telescope.lightcurve_flux[:, 0] - to) / tE
    piE = np.array([piEN, piEE])

    delta_tau, delta_beta = microlparallax.compute_parallax_curvature(piE, telescope.deltas_positions)

    tau += delta_tau
    beta = uo + delta_beta

    assert np.allclose(source_X, tau)  # rotation of alpha
    assert np.allclose(source_Y, beta)


def test_source_trajectory_with_alpha_non_negative():
    to = 0.28
    tE = 35.6
    uo = 0.1
    alpha = np.pi / 2.0

    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'alpha'])
    parameters = Parameters(to, uo, tE, alpha)
    telescope = mock.MagicMock()
    telescope.lightcurve_flux = np.array([[0, 1, 1], [42, 6, 6]])
    event = _create_event()
    event.telescopes[0] = telescope
    Model = microlmodels.create_model('PSPL', event, parallax=['Annual', 0])
    source_X, source_Y = Model.source_trajectory(telescope, to, uo, tE, parameters)

    assert len(source_X) == len(source_Y) == 2

    tau = (telescope.lightcurve_flux[:, 0] - to) / tE
    assert np.allclose(source_X, tau * np.cos(alpha) - uo * np.sin(alpha))  # rotation of alpha
    assert np.allclose(source_Y, tau * np.sin(alpha) + uo * np.cos(alpha))


def test_define_pyLIMA_standard_parameters_with_parallax():
    event = _create_event()
    Model = microlmodels.create_model('FSPL', event, parallax=['Annual', 598.9])

    assert Model.Jacobian_flag == 'No way'
    assert Model.pyLIMA_standards_dictionnary['piEN'] == 4
    assert Model.pyLIMA_standards_dictionnary['piEE'] == 5


def test_define_pyLIMA_standard_parameters_with_xallarap():
    event = _create_event()
    Model = microlmodels.create_model('FSPL', event, xallarap=['Yes', 598.9])

    assert Model.Jacobian_flag == 'No way'
    assert Model.pyLIMA_standards_dictionnary['XiEN'] == 4
    assert Model.pyLIMA_standards_dictionnary['XiEE'] == 5


def test_define_pyLIMA_standard_parameters_with_orbital_motion():
    event = _create_event()
    Model = microlmodels.create_model('FSPL', event, orbital_motion=['2D', 598.9])

    assert Model.Jacobian_flag == 'No way'
    assert Model.pyLIMA_standards_dictionnary['dsdt'] == 4
    assert Model.pyLIMA_standards_dictionnary['dalphadt'] == 5


def test_define_pyLIMA_standard_parameters_with_source_spots():
    event = _create_event()
    Model = microlmodels.create_model('FSPL', event, source_spots='Yes')

    assert Model.Jacobian_flag == 'No way'
    assert Model.pyLIMA_standards_dictionnary['spot'] == 4


def test_pyLIMA_standard_parameters_to_fancy_parameters():
    event = _create_event()
    Model = microlmodels.create_model('FSPL', event, source_spots='Yes')

    Model.fancy_to_pyLIMA_dictionnary = {'logrho': 'rho'}
    Model.pyLIMA_to_fancy = {'logrho': lambda parameters: np.log10(parameters.rho)}
    Model.fancy_to_pyLIMA = {'rho': lambda parameters: 10 ** parameters.logrho}
    Model.define_model_parameters()

    # to, uo ,tE, log10(0.001), spot
    Parameters = [42.0, 56.9, 2.89, -3.0, 0.0]

    pyLIMA_parameters = Model.compute_pyLIMA_parameters(Parameters)

    fancy_parameters = Model.pyLIMA_standard_parameters_to_fancy_parameters(pyLIMA_parameters)
    assert pyLIMA_parameters.rho == 0.001
    assert fancy_parameters.logrho == -3.0


def test_default_microlensing_model_no_fluxes():
    event = _create_event()

    Model = microlmodels.create_model('PSPL', event)
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'fs_Test', 'g_Test'])

    to = 0.0
    uo = 0.1
    tE = 1.0

    parameters = Parameters(to, uo, tE, 'ihih', None)

    microlensing_model,  fs, fb = Model.compute_the_microlensing_model(event.telescopes[0], parameters)

    assert np.allclose(microlensing_model, np.array([1, 6]))


def test_PSPL_Jacobian():
    event = _create_event()

    Model = microlmodels.create_model('PSPL', event)
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'fs_Test', 'g_Test'])

    to = 0.0
    uo = 0.1
    tE = 1.0
    fs = 10
    g = 1.0

    parameters = Parameters(to, uo, tE, fs, g)

    Jacobian = Model.model_Jacobian(event.telescopes[0], parameters)

    assert Jacobian.shape == (5, len(event.telescopes[0].lightcurve_flux))


def test_FSPL_Jacobian():
    event = _create_event()

    Model = microlmodels.create_model('FSPL', event)
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'rho', 'fs_Test', 'g_Test'])

    to = 0.0
    uo = 0.1
    tE = 1.0
    rho = 0.26
    fs = 10
    g = 1.0

    parameters = Parameters(to, uo, tE, rho, fs, g)

    Jacobian = Model.model_Jacobian(event.telescopes[0], parameters)

    assert Jacobian.shape == (6, len(event.telescopes[0].lightcurve_flux))

def test_FSPL_Jacobian_model_magnification():
    event = _create_event()

    Model = microlmodels.create_model('FSPL', event)
    Parameters = collections.namedtuple('parameters', ['to', 'uo', 'tE', 'rho', 'fs_Test', 'g_Test'])

    to = 0.0
    uo = 0.1
    tE = 1.0
    rho = 0.26
    fs = 10
    g = 1.0

    parameters = Parameters(to, uo, tE, rho, fs, g)

    Jacobian = Model.Jacobian_model_magnification(event.telescopes[0], parameters)

    assert np.allclose(Jacobian[0], np.array([ 7.92527951,  1.00000064]))
    assert np.allclose(Jacobian[1], np.array([  0.1       ,  42.00011905]))