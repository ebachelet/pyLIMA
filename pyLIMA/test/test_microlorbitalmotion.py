# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:46:13 2015

@author: ebachelet
"""

import numpy as np
import unittest.mock as mock
from pyslalib import slalib
import collections
from pyLIMA import microlorbitalmotion


def test_orbital_motion_shifts():
    model = ['2D', 2458000]

    pyLIMA_parameters = collections.namedtuple('params', ['dsdt', 'dalphadt'])
    pyLIMA_parameters.dsdt = 0.0001
    pyLIMA_parameters.dalphadt = 0.0001

    time = np.arange(2457950., 2458050., 10)

    ds, dalpha = microlorbitalmotion.orbital_motion_shifts(model, time, pyLIMA_parameters)

    expected = np.array([[-0.005, -0.004, -0.003, -0.002, -0.001, 0., 0.001, 0.002,
                          0.003, 0.004],
                         [-0.005, -0.004, -0.003, -0.002, -0.001, 0., 0.001, 0.002,
                          0.003, 0.004]])

    assert np.allclose(expected, np.array([ds, dalpha]))


def test_orbital_motion_cicular():
    model = ['Circular', 2458000]

    pyLIMA_parameters = collections.namedtuple('params', ['v_para', 'v_perp', 'v_radial', 'logs'])
    pyLIMA_parameters.v_para = 0.000
    pyLIMA_parameters.v_perp = 0.0001
    pyLIMA_parameters.v_radial = 0.000
    pyLIMA_parameters.logs = 0

    time = np.arange(2457950., 2458050., 10)

    ds, dalpha = microlorbitalmotion.orbital_motion_shifts(model, time, pyLIMA_parameters)

    expected = np.array([[0] * 10,
                         [-0.005, -0.004, -0.003, -0.002, -0.001, 0., 0.001, 0.002,
                          0.003, 0.004]])

    assert np.allclose(expected, np.array([ds, dalpha]))

    pyLIMA_parameters.v_para = 0.0001
    pyLIMA_parameters.v_perp = 0.000
    pyLIMA_parameters.v_radial = 0.000001
    pyLIMA_parameters.logs = 0

    time = np.arange(2457950., 2458050., 10)

    ds, dalpha = microlorbitalmotion.orbital_motion_shifts(model, time, pyLIMA_parameters)

    expected = np.array([[-0.005, -0.004, -0.003, -0.002, -0.001, 0., 0.001, 0.002,
                          0.003, 0.004],
                         [0] * 10])

    assert np.allclose(expected, np.array([ds, dalpha]))

    pyLIMA_parameters.v_para = 0.0001
    pyLIMA_parameters.v_perp = 0.0001
    pyLIMA_parameters.v_radial = -0.0001
    pyLIMA_parameters.logs = 0

    time = np.arange(2457950., 2458050., 10)

    ds, dalpha = microlorbitalmotion.orbital_motion_shifts(model, time, pyLIMA_parameters)

    expected = np.array([[-0.00500616, -0.00400395, -0.00300223, -0.00200099, -0.00100025,
                          0., 0.00099975, 0.00199899, 0.00299773, 0.00399595],
                         [-0.00502515, -0.00401607, -0.00300903, -0.00200401, -0.001001,
                          0., 0.000999, 0.00199601, 0.00299103, 0.00398407]])

    assert np.allclose(expected, np.array([ds, dalpha]))
