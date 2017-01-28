# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:02:33 2016

@author: ebachelet
"""

import numpy as np

from pyLIMA import microlmagnification
from pyLIMA import microlmodels

def test_impact_parameter():
    tau = np.array([1])
    uo = np.array([2])


    impact_param = microlmagnification.impact_parameter(tau, uo)
    assert impact_param == np.sqrt(5)
def test_amplification_PSPL():
    tau = np.array([1])
    uo = np.array([0])

    magnification = microlmagnification.amplification_PSPL(tau, uo)

    assert np.allclose(magnification, np.array([1.34164079]))



def test_amplification_FSPL_one_point():
    tau = np.array([0.001])
    uo = np.array([0] * len(tau))
    rho = 0.01
    gamma = 0.5
    yoo_table = microlmodels.yoo_table
    magnification  = microlmagnification.amplification_FSPL(tau, uo, rho, gamma, yoo_table)

    assert np.allclose(magnification, np.array([216.97028636]))



def test_amplification_FSPL_two_points():
    tau = np.array([0.001, 0.002])
    uo = np.array([0] * len(tau))
    rho = 0.01
    gamma = 0.5
    yoo_table = microlmodels.yoo_table
    magnification = microlmagnification.amplification_FSPL(tau, uo, rho, gamma, yoo_table)

    assert np.allclose(magnification, np.array([216.97028636, 214.44622417]))
