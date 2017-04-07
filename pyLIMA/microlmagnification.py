# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:37:33 2015

@author: ebachelet
"""

from __future__ import division
import numpy as np
from multiprocessing import Process, Queue
import time

import sys

sys.path.insert(0, '/path/to/application/app/folder')

from subroutines.VBBinaryLensingLibrary import VBBinaryLensingLibrary as VBBinary

COUNT_VBB_CALL = 0

def impact_parameter(tau, uo):

    impact_param = (tau**2+uo**2)**0.5 #u(t)

    return impact_param

def amplification_PSPL(tau, uo):
    """ The Paczynski magnification.
        "Gravitational microlensing by the galactic halo",Paczynski, B. 1986
        http://adsabs.harvard.edu/abs/1986ApJ...304....1P

        :param array_like tau: the tau define for example in
        http://adsabs.harvard.edu/abs/2015ApJ...804...20C
        :param array_like uo: the u define for example in
        http://adsabs.harvard.edu/abs/2015ApJ...804...20C

        :return: the PSPL magnification A_PSPL(t)
        :rtype: array_like
    """
    # For notations, check for example : http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    impact_param = impact_parameter(tau, uo)  # u(t)
    impact_param_square = impact_param ** 2  # u(t)^2

    amplification_pspl = (impact_param_square + 2) / (impact_param * (impact_param_square + 4) ** 0.5)

    # return both magnification and U, required by some methods
    return amplification_pspl

def Jacobian_amplification_PSPL(tau, uo):
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

    impact_param = impact_parameter(tau, uo)  # u(t)
    impact_param_square = impact_param ** 2  # u(t)^2

    amplification_pspl = (impact_param_square + 2) / (impact_param * (impact_param_square + 4) ** 0.5)

    # return both magnification and U, required by some methods
    return amplification_pspl, impact_param
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

        :return: the FSPL magnification A_FSPL(t)
        :rtype: array_like
    """
    impact_param = impact_parameter(tau, uo)  # u(t)
    impact_param_square = impact_param ** 2  # u(t)^2

    amplification_pspl = (impact_param_square + 2) / (impact_param * (impact_param_square + 4) ** 0.5)

    z_yoo = impact_param / rho

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

    return amplification_fspl

def Jacobian_amplification_FSPL(tau, uo, rho, gamma, yoo_table):
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
    impact_param = impact_parameter(tau, uo)  # u(t)
    impact_param_square = impact_param ** 2  # u(t)^2

    amplification_pspl = (impact_param_square + 2) / (impact_parameter * (impact_param_square + 4) ** 0.5)

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

    return amplification_fspl, impact_param
def amplification_USBL(separation, q, Xs, Ys, rho, tolerance=0.001):
    """ The uniform source binary lens amplification, based on the work of Valerio Bozza, thanks :)
        "Microlensing with an advanced contour integration algorithm: Green's theorem to third order, error control,
         optimal sampling and limb darkening ",Bozza, Valerio 2010. Please cite the paper if you used this.
        http://mnras.oxfordjournals.org/content/408/4/2188

        :param array_like s: the projected normalised angular distance between the two bodies
        :param float q: the mass ratio of the two bodies
        :param array_like Xs: the horizontal positions of the source center in the source plane
        :param array_like Ys: the vertical positions of the source center in the source plane
        :param float rho: the normalised (to :math:`\\theta_E') angular source star radius
        :param float tolerance: the relative precision desired in the magnification

        :return: the USBL magnification A_USBL(t)
        :rtype: array_like
    """

    amplification_usbl = []
    #print ''
    #print separation[0],q,rho
   # start = time.time()

    for xs, ys, s in zip(Xs, Ys, separation):

        #print index,len(Xs)
       # print s,q,xs,ys,rho,tolerance
        magnification_VBB = VBBinary.VBBinaryLensing().BinaryMag(s,q,xs,ys,rho,tolerance)

        amplification_usbl.append(magnification_VBB)


   # print time.time()-start
    #import pdb;
    #pdb.set_trace()
    #print amplification_usbl
    return np.array(amplification_usbl)

def amplification_PSBL(separation, q, Xs, Ys):
    """ The point source binary lens amplification, based on the work of Valerio Bozza, thanks :)
        "Microlensing with an advanced contour integration algorithm: Green's theorem to third order, error control,
         optimal sampling and limb darkening ",Bozza, Valerio 2010. Please cite the paper if you used this.
        http://mnras.oxfordjournals.org/content/408/4/2188

        :param array_like s: the projected normalised angular distance between the two bodies
        :param float q: the mass ratio of the two bodies
        :param array_like Xs: the horizontal positions of the source center in the source plane
        :param array_like Ys: the vertical positions of the source center in the source plane

        :return: the PSBL magnification A_PSBL(t)
        :rtype: array_like
    """

    amplification_psbl = []
    #print ''
    #print separation[0],q,rho
    #start = time.time()

    for xs, ys, s in zip(Xs, Ys, separation):

        #print index,len(Xs)
       # print s,q,xs,ys,rho,tolerance
        magnification_VBB = VBBinary.VBBinaryLensing().BinaryMag0(s,q,xs,ys)

        amplification_psbl.append(magnification_VBB)


    return np.array(amplification_psbl)


def USBL_queue_process(queue, args):

    queue.put(VBBinary.VBBinaryLensing().BinaryMag(args[0], args[1], args[2],
                                                   args[3], args[4], args[5]))

def amplification_FSPL_for_Lyrae(tau, uo, rho, gamma, yoo_table):
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
    impact_param = impact_parameter(tau, uo)  # u(t)
    impact_param_square = impact_param ** 2  # u(t)^2

    amplification_pspl = (impact_param_square + 2) / (impact_param * (impact_param_square + 4) ** 0.5)

    z_yoo = impact_param / rho

    amplification_fspl = np.zeros(len(amplification_pspl))

    # Far from the lens (z_yoo>>1), then PSPL.
    indexes_PSPL = np.where((z_yoo > yoo_table[0][-1]))[0]

    amplification_fspl[indexes_PSPL] = amplification_pspl[indexes_PSPL]

    # Very close to the lens (z_yoo<<1), then Witt&Mao limit.
    indexes_WM = np.where((z_yoo < yoo_table[0][0]))[0]

    amplification_fspl[indexes_WM] = amplification_pspl[indexes_WM] * \
                                     (2 * z_yoo[indexes_WM] - gamma[indexes_WM] * (2 - 3 * np.pi / 4) * z_yoo[indexes_WM])

    # FSPL regime (z_yoo~1), then Yoo et al derivatives
    indexes_FSPL = np.where((z_yoo <= yoo_table[0][-1]) & (z_yoo >= yoo_table[0][0]))[0]

    amplification_fspl[indexes_FSPL] = amplification_pspl[indexes_FSPL] * \
                                       (yoo_table[1](z_yoo[indexes_FSPL]) - gamma[indexes_FSPL] * yoo_table[2](z_yoo[indexes_FSPL]))

    return amplification_fspl

#### TO DO : the following is row development# ###


# def function_LEE(r, v, u, rho, gamma):
#    """Not working yet"""
#    if r == 0:
#        LEE = 1.0
#    else:
#        LEE = (r ** 2 + 2) / ((r ** 2 + 4) ** 0.5) * (
#            1 - gamma * (
#                1 - 1.5 * (1 - (r ** 2 - 2 * u * r * np.cos(v) + u ** 2) / rho ** 2) ** 0.5))

#    return LEE


# def LEE_limit(v, u, rho, gamma):
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
