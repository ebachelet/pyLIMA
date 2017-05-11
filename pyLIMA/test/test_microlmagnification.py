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

def test_Jacobian_amplification_PSPL():
    tau = np.array([1])
    uo = np.array([0])

    magnification = microlmagnification.Jacobian_amplification_PSPL(tau, uo)[0]
    impact_parameter = microlmagnification.Jacobian_amplification_PSPL(tau, uo)[1]

    assert np.allclose(magnification, np.array([1.34164079]))
    assert np.allclose(impact_parameter, 1)


def test_amplification_FSPL_one_point():
    tau = np.array([0.001])
    uo = np.array([0] * len(tau))
    rho = 0.01
    gamma = 0.5
    yoo_table = microlmodels.yoo_table
    magnification  = microlmagnification.amplification_FSPL(tau, uo, rho, gamma, yoo_table)

    assert np.allclose(magnification, np.array([216.97028636]))


def test_Jacobian_amplification_FSPL():
    tau = np.array([0.001])
    uo = np.array([0] * len(tau))
    rho = 0.01
    gamma = 0.5
    yoo_table = microlmodels.yoo_table
    magnification = microlmagnification.Jacobian_amplification_FSPL(tau, uo, rho, gamma, yoo_table)[0]
    impact_parameter = microlmagnification.Jacobian_amplification_FSPL(tau, uo, rho, gamma, yoo_table)[1]

    assert np.allclose(magnification, np.array([216.97028636]))
    assert np.allclose(impact_parameter, 0.001)




def test_amplification_FSPL_two_points():
    tau = np.array([0.001, 0.002])
    uo = np.array([0] * len(tau))
    rho = 0.01
    gamma = 0.5
    yoo_table = microlmodels.yoo_table
    magnification = microlmagnification.amplification_FSPL(tau, uo, rho, gamma, yoo_table)

    assert np.allclose(magnification, np.array([216.97028636, 214.44622417]))

def test_amplification_USBL():
    s = np.array([1])
    q = 0.02
    xs = np.array([0.5])
    ys = np.array([0.5])
    rho = 0.0033
    tol = 0.001


    magnification = microlmagnification.amplification_USBL(s,q,xs,ys,rho,tol)

    assert np.allclose(magnification, np.array([1.6311724868]))


def test_amplification_PSBL():
    s = np.array([1])
    q = 0.02
    xs = np.array([0.5])
    ys = np.array([0.5])



    magnification = microlmagnification.amplification_PSBL(s,q,xs,ys)

    assert np.allclose(magnification, np.array([1.63109244]))

def test_amplification_FSPL_for_Lyrae():
    tau = np.array([0.001])
    uo = np.array([0] * len(tau))
    rho = 0.01
    gamma = 0.5
    yoo_table = microlmodels.yoo_table
    magnification  = microlmagnification.amplification_FSPL(tau, uo, rho, gamma, yoo_table)

    assert np.allclose(magnification, np.array([216.97028636]))