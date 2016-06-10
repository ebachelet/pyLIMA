# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:02:33 2016

@author: ebachelet
"""

import numpy as np

from pyLIMA import microlmagnification
from pyLIMA import microlmodels

def test_amplification_PSPL():

    tau = np.array([1])
    uo = np.array([0])

    magnification, u = microlmagnification.amplification_PSPL(tau, uo)


    EPSILON = 0.001

    assert np.abs(magnification-1.3416)<=EPSILON
    assert u == 1

def test_amplification_FSPL_one_point():

     tau = np.array([0.001])
     uo = np.array([0]*len(tau))
     rho = 0.01
     gamma = 0.5
     yoo_table = microlmodels.yoo_table
     magnification, u = microlmagnification.amplification_FSPL(tau, uo, rho, gamma, yoo_table)

     assert np.allclose(magnification,np.array([ 216.97028636]))
     assert np.allclose(u,np.array([ 0.001]))

def test_amplification_FSPL_two_points():

     tau = np.array([0.001,0.002])
     uo = np.array([0]*len(tau))
     rho = 0.01
     gamma = 0.5
     yoo_table = microlmodels.yoo_table
     magnification, u = microlmagnification.amplification_FSPL(tau, uo, rho, gamma, yoo_table)

     assert np.allclose(magnification,np.array([ 216.97028636,214.44622417]))
     assert np.allclose(u,np.array([ 0.001,0.002]))




