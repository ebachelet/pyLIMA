# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:37:33 2015

@author: ebachelet
"""

from __future__ import division
import numpy as np

def amplification_PSPL(tau, u):
    """ The Paczynski magnification.
        "Gravitational microlensing by the galactic halo",Paczynski, B. 1986
        http://adsabs.harvard.edu/abs/1986ApJ...304....1P
    
        :param array_like tau: the tau define for example in http://adsabs.harvard.edu/abs/2015ApJ...804...20C
        :param array_like u: the u define for example in http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    
        :return: the PSPL magnification A_PSPL(t) and the impact parameter U(t)
        :rtype: array_like,array_like
    """
    # For notations, check for example : http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    U = (tau ** 2 + u ** 2) ** 0.5
    U_square = U ** 2
    amplification = (U_square + 2) / (U * (U_square + 4) ** 0.5)
    
    #return both magnification and U, required by some methods
    return amplification, U


def amplification_FSPL(tau, u, rho, gamma, yoo_table):
    """ The Yoo FSPL magnification.
        "OGLE-2003-BLG-262: Finite-Source Effects from a Point-Mass Lens",Yoo, J. et al 2004
        http://adsabs.harvard.edu/abs/2004ApJ...603..139Y

        :param array_like tau: the tau define for example in http://adsabs.harvard.edu/abs/2015ApJ...804...20C
        :param array_like u: the u define for example in http://adsabs.harvard.edu/abs/2015ApJ...804...20C
        :param float rho: the normalised (to :math:`\\theta_E') angular source star radius
        :param array_like yoo_table: the interpolated Yoo et al table.
    
        :return: the FSPL magnification A_FSPL(t) and the impact parameter U(t)
        :rtype: array_like,array_like
    """
    U = (tau ** 2 + u ** 2) ** 0.5
    U_square = U ** 2
    amplification_PSPL = (U_square + 2) / (U * (U_square + 4) ** 0.5)
   
    z_yoo = U / rho
    
    amplification_FSPL = np.zeros(len(amplification_PSPL))

    # Far from the lens (z_yoo>>1), then PSPL.    
    indexes_PSPL = np.where((z_yoo > yoo_table[0][-1]))[0]
    amplification_FSPL[indexes_PSPL] = amplification_PSPL[indexes_PSPL]

    # Very close to the lens (z_yoo<<1), then Witt&Mao limit.
    indexes_WM = np.where((z_yoo < yoo_table[0][0]))[0]
    amplification_FSPL[indexes_WM] = amplification_PSPL[indexes_WM] * (2 * z_yoo[indexes_WM] - gamma * (2 - 3 * np.pi / 4) * z_yoo[indexes_WM])
   
    # FSPL regime (z_yoo~1), then Yoo et al derivatives
    indexes_FSPL = np.where((z_yoo <= yoo_table[0][-1]) & (z_yoo >= yoo_table[0][0]))[0]
    amplification_FSPL[indexes_FSPL] = amplification_PSPL[indexes_FSPL] * (yoo_table[1](z_yoo[indexes_FSPL]) - gamma * yoo_table[2](z_yoo[indexes_FSPL]))
    
    amplification = amplification_FSPL
   
    return amplification, U
    

#### TO DO : the following probably depreciated ####


def source_trajectory(model, t, parameters):
    """Not working yet """
    tau = (t - parameters[model.model_dictionnary['to']]) / parameters[
        model.model_dictionnary['tE']]

    if model.paczynski_model is not 'Binary':

        alpha = 0.0

    x = tau * np.cos(alpha) - np.sin(alpha) * parameters[model.model_dictionnary['uo']]
    y = tau * np.sin(alpha) + np.cos(alpha) * parameters[model.model_dictionnary['uo']]

    return x, y


def function_LEE(r, v, u, rho, gamma):
    """Not working yet"""
    if r == 0:
        LEE = 1.0
    else:
        LEE = (r ** 2 + 2) / ((r ** 2 + 4) ** 0.5) * (
            1 - gamma * (
            1 - 1.5 * (1 - (r ** 2 - 2 * u * r * np.cos(v) + u ** 2) / rho ** 2) ** 0.5))

    return LEE


def LEE_limit(v, u, rho, gamma):
    """Not working yet"""
    if u <= rho:
        limit_1 = 0.0
    else:
        if v <= np.arcsin(rho / u):

            limit_1 = u * np.cos(v) - (rho ** 2 - u ** 2 * np.sin(v) ** 2) ** 0.5
        else:
            limit_1 = 0.0

    if u <= rho:
        limit_2 = u * np.cos(v) + (rho ** 2 - u ** 2 * np.sin(v) ** 2) ** 0.5
    else:
        if v <= np.arcsin(rho / u):

            limit_2 = u * np.cos(v) + (rho ** 2 - u ** 2 * np.sin(v) ** 2) ** 0.5
        else:
            limit_2 = 0.0

    return [limit_1, limit_2]
