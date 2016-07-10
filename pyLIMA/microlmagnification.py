# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:37:33 2015

@author: ebachelet
"""

from __future__ import division
import numpy as np


def amplification_PSPL(tau, uo):
    """ The Paczynski magnification.
        "Gravitational microlensing by the galactic halo",Paczynski, B. 1986
        http://adsabs.harvard.edu/abs/1986ApJ...304....1P

        :param array_like tau: the tau define for example in
        http://adsabs.harvard.edu/abs/2015ApJ...804...20C
        :param array_like u: the u define for example in
        http://adsabs.harvard.edu/abs/2015ApJ...804...20C

        :return: the PSPL magnification A_PSPL(t) and the impact parameter U(t)
        :rtype: array_like,array_like
    """
    # For notations, check for example : http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    impact_parameter = (tau ** 2 + uo ** 2) ** 0.5  #u(t)
    impact_parameter_square = impact_parameter ** 2 #u(t)^2

    amplification = (impact_parameter_square + 2) / (impact_parameter * (impact_parameter_square + 4) ** 0.5)

    # return both magnification and U, required by some methods
    return amplification, impact_parameter


def amplification_FSPL(tau, uo, rho, gamma, yoo_table):
    """ The Yoo FSPL magnification.
        "OGLE-2003-BLG-262: Finite-Source Effects from a Point-Mass Lens",Yoo, J. et al 2004
        http://adsabs.harvard.edu/abs/2004ApJ...603..139Y

        :param array_like tau: the tau define for example in
        http://adsabs.harvard.edu/abs/2015ApJ...804...20C
        :param array_like u: the u define for example in
        http://adsabs.harvard.edu/abs/2015ApJ...804...20C
        :param float rho: the normalised (to :math:`\\theta_E') angular source star radius
        :param array_like yoo_table: the interpolated Yoo et al table.

        :return: the FSPL magnification A_FSPL(t) and the impact parameter U(t)
        :rtype: array_like,array_like
    """
    impact_parameter = (tau ** 2 + uo ** 2) ** 0.5  #u(t)
    impact_parameter_square = impact_parameter ** 2 #u(t)^2

    amplification_pspl = (impact_parameter_square + 2) / (impact_parameter * (impact_parameter_square + 4) ** 0.5)

    z_yoo = impact_parameter / rho

    amplification_fspl = np.zeros(len(amplification_pspl))

    # Far from the lens (z_yoo>>1), then PSPL.
    indexes_PSPL = np.where((z_yoo > yoo_table[0][-1]))[0]

    amplification_fspl[indexes_PSPL] = amplification_pspl[indexes_PSPL]

    # Very close to the lens (z_yoo<<1), then Witt&Mao limit.
    indexes_WM = np.where((z_yoo < yoo_table[0][0]))[0]

    amplification_fspl[indexes_WM] = amplification_pspl[indexes_WM] * \
    (2 * z_yoo[indexes_WM] - gamma * (2 - 3 * np.pi / 4) * z_yoo[indexes_WM])

    # FSPL regime (z_yoo~1), then Yoo et al derivatives
    indexes_FSPL = np.where((z_yoo <= yoo_table[0][-1]) & (z_yoo >= yoo_table[0][0]))[0]

    amplification_fspl[indexes_FSPL] = amplification_pspl[indexes_FSPL] * \
    (yoo_table[1](z_yoo[indexes_FSPL]) - gamma * yoo_table[2](z_yoo[indexes_FSPL]))

    amplification = amplification_fspl

    return amplification, impact_parameter


#### TO DO : the following is row development# ###


#def function_LEE(r, v, u, rho, gamma):
#    """Not working yet"""
#    if r == 0:
#        LEE = 1.0
#    else:
#        LEE = (r ** 2 + 2) / ((r ** 2 + 4) ** 0.5) * (
#            1 - gamma * (
#                1 - 1.5 * (1 - (r ** 2 - 2 * u * r * np.cos(v) + u ** 2) / rho ** 2) ** 0.5))

#    return LEE


#def LEE_limit(v, u, rho, gamma):
#    """Not working yet"""
#    if u <= rho:
#        limit_1 = 0.0
#    else:
#        if v <= np.arcsin(rho / u):

#            limit_1 = u * np.cos(v) - (rho ** 2 - u ** 2 * np.sin(v) ** 2) ** 0.5
#        else:
#            limit_1 = 0.0

#    if u <= rho:
#        limit_2 = u * np.cos(v) + (rho ** 2 - u ** 2 * np.sin(v) ** 2) ** 0.5
#    else:
#        if v <= np.arcsin(rho / u):

#            limit_2 = u * np.cos(v) + (rho ** 2 - u ** 2 * np.sin(v) ** 2) ** 0.5
#        else:
#            limit_2 = 0.0

#    return [limit_1, limit_2]
