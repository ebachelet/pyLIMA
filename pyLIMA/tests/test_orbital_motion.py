import collections

import numpy as np
from pyLIMA.orbitalmotion import orbital_motion


def test_orbital_motion2D():
    time = np.array([12, 13])
    t0_om = 11
    dalpha_dt = 2  # mas/yr

    dalpha = orbital_motion.orbital_motion_2D.orbital_motion_2D_trajectory_shift(time,
                                                                                 t0_om,
                                                                                 dalpha_dt)

    assert np.allclose(dalpha, [0.0054757, 0.0109514])

    time = np.array([12, 13])
    t0_om = 11
    ds_dt = 2.4  # mas/yr

    dseparation = orbital_motion.orbital_motion_2D.orbital_motion_2D_separation_shift(
        time, t0_om, ds_dt)

    assert np.allclose(dseparation, [0.00657084, 0.01314168])


def test_orbital_motion_circular():
    time = np.array([2458555, 2459855])

    pyLIMA_parameters = collections.OrderedDict()


    pyLIMA_parameters['Rmatrix'] = np.eye(2)
    pyLIMA_parameters['orbital_velocity'] = -2.6
    pyLIMA_parameters['a_true']  = 1.2
    pyLIMA_parameters['t_periastron'] =  2459856
    pyLIMA_parameters['separation']  =  0.8
    pyLIMA_parameters['alpha'] =  0.8

    dsep, dalpha = orbital_motion.orbital_motion_3D.orbital_motion_keplerian(time,
                                                                             pyLIMA_parameters,
                                                                             [
                                                                                 'Circular',
                                                                                 2458555])

    assert np.allclose(dsep, [0.4, 0.4])
    assert np.allclose(dalpha, [2.97786877, 0.00711841])


def test_orbital_motion_keplerian():
    time = np.array([2458555, 2459855])

    pyLIMA_parameters = collections.OrderedDict()


    pyLIMA_parameters['Rmatrix'] = np.eye(2)
    pyLIMA_parameters['orbital_velocity'] = -2.6
    pyLIMA_parameters['a_true'] = 1.2
    pyLIMA_parameters['t_periastron'] = 2459856
    pyLIMA_parameters['separation'] = 0.8
    pyLIMA_parameters['alpha'] = 0.8
    pyLIMA_parameters['eccentricity'] = 0.56

    dsep, dalpha = orbital_motion.orbital_motion_3D.orbital_motion_keplerian(time,
                                                                             pyLIMA_parameters,
                                                                             [
                                                                                 'Keplerian',
                                                                                 2458555])

    assert np.allclose(dsep, [1.06829756, -0.27191207])
    assert np.allclose(dalpha, [3.08578103, 0.03045918])


def test_orbital_parameters_from_position_and_velocities():
    separation_0 = 1.25
    r_s = 0.98
    a_s = 1.2
    v_para = 0.25
    v_perp = -0.33
    v_radial = 1.45
    t0_om = 2459855

    outputs = orbital_motion.orbital_motion_3D \
        .orbital_parameters_from_position_and_velocities(
        separation_0, r_s, a_s, v_para, v_perp, v_radial, t0_om)
    assert np.allclose(outputs[:-3], (
        0.2622029190714234, 1.8293736686171367, -1.5463618954109095, 2.1002142747824566,
        0.7585218053347751, 0.7979704741126448, 2.3559427181074657, 2459575.707195252))

    assert np.allclose(outputs[-3], [-0.04266437, 0.25322504, -0.96646616])
    assert np.allclose(outputs[-2], [0.96715238, 0.25309756, 0.02361975])
    assert np.allclose(outputs[-1], [0.25059134, -0.93371232, -0.25570545])


def test_state_orbital_elements():
    separation_0 = 1.25
    r_s = 0.98
    a_s = 1.2
    v_para = 0.25
    v_perp = -0.33
    v_radial = 1.45

    outputs = orbital_motion.orbital_motion_3D.state_orbital_elements(separation_0, r_s,
                                                                      a_s, v_para,
                                                                      v_perp, v_radial)

    assert np.allclose(outputs[0], [-0.03404491, 0.20206611, -0.77121146])
    assert np.allclose(outputs[1], [0.5053125, -1.8828125, -0.515625])
    assert np.allclose(outputs[2], [1.25, 0., 1.225])
    assert np.allclose(outputs[3], [0.3125, -0.4125, 1.8125])
    assert np.allclose(outputs[4:], (
        1.7501785623187138, 1.225, 2.1002142747824566, 5.329996919004271,
        0.7585218053347751))


def test_eccentric_anomaly_function():
    ecc = orbital_motion.orbital_motion_3D.eccentric_anomaly_function([2458955], 0.27,
                                                                      2456589, 3.2)

    assert ecc[0] == 6.2307350891533675
