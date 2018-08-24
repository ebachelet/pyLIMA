# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:37:33 2015

@author: ebachelet
"""

from __future__ import division
import numpy as np


import VBBinaryLensing

VBB = VBBinaryLensing.VBBinaryLensing()
VBB.Tol = 0.001
VBB.RelTol = 0.001


def impact_parameter(tau, uo):
    """
    The impact parameter U(t).
    "Gravitational microlensing by the galactic halo",Paczynski, B. 1986
    http://adsabs.harvard.edu/abs/1986ApJ...304....1P

    :param array_like tau: the tau define for example in
                               http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    :param array_like uo: the uo define for example in
                              http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :return: the impact parameter U(t)
    :rtype: array_like
    """
    impact_param = (tau ** 2 + uo ** 2) ** 0.5  # u(t)

    return impact_param


def amplification_PSPL(tau, uo):
    """
    The Paczynski Point Source Point Lens magnification and the impact parameter U(t).
    "Gravitational microlensing by the galactic halo",Paczynski, B. 1986
    http://adsabs.harvard.edu/abs/1986ApJ...304....1P

    :param array_like tau: the tau define for example in
                           http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param array_like uo: the uo define for example in
                         http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :return: the PSPL magnification A_PSPL(t) and the impact parameter U(t)
    :rtype: tuple, tuple of two array_like
    """
    # For notations, check for example : http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    impact_param = impact_parameter(tau, uo)  # u(t)
    impact_param_square = impact_param ** 2  # u(t)^2

    amplification_pspl = (impact_param_square + 2) / (impact_param * (impact_param_square + 4) ** 0.5)

    # return both magnification and U, required by some methods
    return amplification_pspl


def Jacobian_amplification_PSPL(tau, uo):
    """ Same function as above, just also returns the impact parameter needed for the Jacobian PSPL model.
    The Paczynski Point Source Point Lens magnification and the impact parameter U(t).
    "Gravitational microlensing by the galactic halo",Paczynski, B. 1986
    http://adsabs.harvard.edu/abs/1986ApJ...304....1P

    :param array_like tau: the tau define for example in
                           http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param array_like uo: the uo define for example in
                         http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :return: the PSPL magnification A_PSPL(t) and the impact parameter U(t)
    :rtype: tuple, tuple of two array_like
    """
    # For notations, check for example : http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    impact_param = impact_parameter(tau, uo)  # u(t)
    impact_param_square = impact_param ** 2  # u(t)^2

    amplification_pspl = (impact_param_square + 2) / (impact_param * (impact_param_square + 4) ** 0.5)

    # return both magnification and U, required by some methods
    return amplification_pspl, impact_param


def amplification_FSPL(tau, uo, rho, gamma, yoo_table):
    """
    The Yoo et al. Finite Source Point Lens magnification.
    "OGLE-2003-BLG-262: Finite-Source Effects from a Point-Mass Lens",Yoo, J. et al 2004
    http://adsabs.harvard.edu/abs/2004ApJ...603..139Y

    :param array_like tau: the tau define for example in
                               http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param array_like uo: the uo define for example in
                             http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param float rho: the normalised angular source star radius

    :param float gamma: the microlensing limb darkening coefficient.

    :param array_like yoo_table: the Yoo et al. 2004 table approximation. See microlmodels for more details.

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
    """Same function as above, just also returns the impact parameter needed for the Jacobian FSPL model.
    The Yoo et al. Finite Source Point Lens magnification and the impact parameter U(t).
    "OGLE-2003-BLG-262: Finite-Source Effects from a Point-Mass Lens",Yoo, J. et al 2004
    http://adsabs.harvard.edu/abs/2004ApJ...603..139Y

    :param array_like tau: the tau define for example in
                               http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param array_like uo: the uo define for example in
                             http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param float rho: the normalised angular source star radius

    :param float gamma: the microlensing limb darkening coefficient.

    :param array_like yoo_table: the Yoo et al. 2004 table approximation. See microlmodels for more details.

    :return: the FSPL magnification A_FSPL(t) and the impact parameter U(t)
    :rtype: tuple, tuple of two array_like
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

    return amplification_fspl, impact_param


def amplification_USBL(separation, mass_ratio, x_source, y_source, rho):
    """
    The Uniform Source Binary Lens amplification, based on the work of Valerio Bozza, thanks :)
    "Microlensing with an advanced contour integration algorithm: Green's theorem to third order, error control,
    optimal sampling and limb darkening ",Bozza, Valerio 2010. Please cite the paper if you used this.
    http://mnras.oxfordjournals.org/content/408/4/2188

    :param array_like separation: the projected normalised angular distance between the two bodies
    :param float mass_ratio: the mass ratio of the two bodies
    :param array_like x_source: the horizontal positions of the source center in the source plane
    :param array_like y_source: the vertical positions of the source center in the source plane
    :param float rho: the normalised (to :math:`\\theta_E') angular source star radius
    :param float tolerance: the relative precision desired in the magnification

    :return: the USBL magnification A_USBL(t)
    :rtype: array_like
    """

    amplification_usbl = []


    for xs, ys, s in zip(x_source, y_source, separation):

        magnification_VBB = VBB.BinaryMag2(s, mass_ratio, xs, ys, rho)

        amplification_usbl.append(magnification_VBB)



    return np.array(amplification_usbl)


def amplification_FSBL(separation, mass_ratio, x_source, y_source, rho, limb_darkening_coefficient):
    """
    The Uniform Source Binary Lens amplification, based on the work of Valerio Bozza, thanks :)
    "Microlensing with an advanced contour integration algorithm: Green's theorem to third order, error control,
    optimal sampling and limb darkening ",Bozza, Valerio 2010. Please cite the paper if you used this.
    http://mnras.oxfordjournals.org/content/408/4/2188

    :param array_like separation: the projected normalised angular distance between the two bodies
    :param float mass_ratio: the mass ratio of the two bodies
    :param array_like x_source: the horizontal positions of the source center in the source plane
    :param array_like y_source: the vertical positions of the source center in the source plane
    :param float limb_darkening_coefficient: the linear limb-darkening coefficient
    :param float rho: the normalised (to :math:`\\theta_E') angular source star radius

    :param float tolerance: the relative precision desired in the magnification

    :return: the USBL magnification A_USBL(t)
    :rtype: array_like
    """

    amplification_fsbl = []

    for xs, ys, s in zip(x_source, y_source, separation):
        # print index,len(Xs)
        # print s,q,xs,ys,rho,tolerance
        magnification_VBB = VBB.BinaryMagDark(s, mass_ratio, xs, ys, rho, limb_darkening_coefficient, VBB.Tol)

        amplification_fsbl.append(magnification_VBB)



    return np.array(amplification_fsbl)


def amplification_PSBL(separation, mass_ratio, x_source, y_source):
    """
    The Point Source Binary Lens amplification, based on the work of Valerio Bozza, thanks :)
    "Microlensing with an advanced contour integration algorithm: Green's theorem to third order, error control,
    optimal sampling and limb darkening ",Bozza, Valerio 2010. Please cite the paper if you used this.
    http://mnras.oxfordjournals.org/content/408/4/2188

    :param array_like separation: the projected normalised angular distance between the two bodies
    :param float mass_ratio: the mass ratio of the two bodies
    :param array_like x_source: the horizontal positions of the source center in the source plane
    :param array_like y_source: the vertical positions of the source center in the source plane

    :return: the PSBL magnification A_PSBL(t)
    :rtype: array_like
    """

    amplification_psbl = []


    for xs, ys, s in zip(x_source, y_source, separation):

        magnification_VBB =VBB.BinaryMag0(s, mass_ratio, xs, ys)

        amplification_psbl.append(magnification_VBB)

    return np.array(amplification_psbl)



def amplification_FSPL_for_Lyrae(tau, uo, rho, gamma, yoo_table):
    """
    The Yoo et al Finite Source Point Lens magnification.
    "OGLE-2003-BLG-262: Finite-Source Effects from a Point-Mass Lens",Yoo, J. et al 2004
    http://adsabs.harvard.edu/abs/2004ApJ...603..139Y

    :param array_like tau: the tau define for example in
                                   http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param array_like uo: the uo define for example in
                                 http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param float rho: the normalised angular source star radius

    :param float gamma: the microlensing limb darkening coefficient.

    :param array_like yoo_table: the Yoo et al. 2004 table approximation. See microlmodels for more details.

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
                                     (2 * z_yoo[indexes_WM] - gamma[indexes_WM] * (2 - 3 * np.pi / 4) * z_yoo[
                                         indexes_WM])

    # FSPL regime (z_yoo~1), then Yoo et al derivatives
    indexes_FSPL = np.where((z_yoo <= yoo_table[0][-1]) & (z_yoo >= yoo_table[0][0]))[0]

    amplification_fspl[indexes_FSPL] = amplification_pspl[indexes_FSPL] * \
                                       (yoo_table[1](z_yoo[indexes_FSPL]) - gamma[indexes_FSPL] * yoo_table[2](
                                           z_yoo[indexes_FSPL]))

    return amplification_fspl
    
    
def amplification_PSTL(separation1, separation2, mass_ratio1, mass_ratio2, psis, x_source, y_source):
    """
    The Point Source Triple Lens amplification, follows the definitions in the works of K. Danek
    "CRITICAL CURVES AND CAUSTICS OF TRIPLE-LENS MODELS ",K. Danek and D. Heyrovský 2015. 
    http://iopscience.iop.org/article/10.1088/0004-637X/806/1/99/meta
    and C. Han "THE SECOND MULTIPLE-PLANET SYSTEM DISCOVERED BY MICROLENSING: OGLE-2012-BLG-0026Lb, c—A PAIR OF JOVIAN PLANETS BEYOND THE SNOW LINE", C. Han et al. 2013.
    http://adsabs.harvard.edu/abs/2013ApJ...762L..28H

    :param array_like separation1: the projected normalised angular distance between body1 and host mass
    :param array_like separation2: the projected normalised angular distance between body2 and host mass
    :param float mass_ratio1: the mass ratio of body1 and host mass
    :param float mass_ratio2: the mass ratio of body2 and host mass
    :param float psi: the angle between x-axis and the connecting line of body2 and host mass
    :param array_like x_source: the horizontal positions of the source center in the source plane
    :param array_like y_source: the vertical positions of the source center in the source plane

    :return: the PSTL magnification A_PSTL(t)
    :rtype: array_like
    """

    amplification_pstl = []

    for xs, ys, s1, s2, psi in zip(x_source, y_source, separation1, separation2, psis):

        magnification_triple = TripleMag0(s1, s2, mass_ratio1, mass_ratio2, psi, xs, ys)
        
        amplification_pstl.append(float(magnification_triple))
          
    return np.array(amplification_pstl)
    
    
def TripleMag0(s1, s2, mass_ratio1, mass_ratio2, psi, xs, ys, limit=1e-5):
    
    m1 = 1/(1+mass_ratio1+mass_ratio2)
    m2 = mass_ratio1/(1+mass_ratio1+mass_ratio2)
    zeta = xs+1.j*ys
    z2 = s1
    z3 = s2*(np.cos(psi)+1.j*np.sin(psi))
        
    x0 = complex.conjugate(zeta)
    x1 = x0**2
    x2 = complex.conjugate(z3)
    x3 = x0*x2
    x4 = x2*z2
    x5 = x0*z2
    x6 = x4 - x5
    x7 = 2*x1
    x8 = x0**3
    x9 = x8*zeta
    x10 = 3*x8
    x11 = 2*x3
    x12 = z2**2
    x13 = 3*x1
    x14 = x12*x13
    x15 = m1*x3
    x16 = m2*x3
    x17 = z2*zeta
    x18 = x1*x17
    x19 = x2*zeta
    x20 = x1*x19
    x21 = m1*x4
    x22 = m2*x5
    x23 = 3*x12
    x24 = x23*x3
    x25 = z2*z3
    x26 = x2*z3
    x27 = x13*x26
    x28 = x4*zeta
    x29 = x0*x28
    x30 = x0*z3
    x31 = 3*x4
    x32 = -x2
    x33 = m1*x2
    x34 = m2*x2
    x35 = m2*z2
    x36 = 6*x1
    x37 = x13*z3
    x38 = x13*zeta
    x39 = z2**3
    x40 = x13*x39
    x41 = z3**2
    x42 = x12*x34
    x43 = m2*x12
    x44 = x0*x43
    x45 = x5*z3
    x46 = m1*z3
    x47 = x13*x46
    x48 = m2*z3
    x49 = x41*z2
    x50 = x14*zeta
    x51 = x2*x41
    x52 = 2*z3
    x53 = x5*zeta
    x54 = 2*x53
    x55 = x11*zeta
    x56 = 3*x9
    x57 = 3*x39
    x58 = x3*x57
    x59 = 4*z3
    x60 = x3*x59
    x61 = x0*x4
    x62 = 9*x8
    x63 = x3*x46
    x64 = x4*x48
    x65 = x3*x48
    x66 = 9*x1
    x67 = x66*z3
    x68 = 3*x21
    x69 = x0*x68
    x70 = 2*x46
    x71 = x0*x41
    x72 = x31*x71
    x73 = x23*zeta
    x74 = x3*x73
    x75 = x48*x5
    x76 = x12*z3
    x77 = 9*x76
    x78 = 3*x28
    x79 = x2*x23
    x80 = x0*x23
    x81 = -x1*x77 + x23*x33 - x79 + x80
    x82 = z2**4
    x83 = x1*x82
    x84 = m2**2
    x85 = x12*x84
    x86 = x39*x8
    x87 = z3**3
    x88 = x8*x87
    x89 = 3*x5
    x90 = x0*zeta
    x91 = 3*x90
    x92 = x0*x39
    x93 = 3*x92
    x94 = 2*x43
    x95 = x2*x39
    x96 = 3*x95
    x97 = m2*x39
    x98 = x0*x97
    x99 = x25*x84
    x100 = x4*x84
    x101 = x4*x41
    x102 = x41*x5
    x103 = x1*x87
    x104 = x103*z2
    x105 = m1**2
    x106 = x105*x26
    x107 = x3*x82
    x108 = x36*x43
    x109 = 6*x12
    x110 = x109*x90
    x111 = x109*x3
    x112 = m2*x4
    x113 = 5*x112
    x114 = 4*x30
    x115 = m2*x114
    x116 = x33*x39
    x117 = x23*x30
    x118 = x34*x39
    x119 = x11*x41
    x120 = 3*z3
    x121 = 4*x22
    x122 = x36*x41
    x123 = m1*x122
    x124 = m2*x122
    x125 = z3*zeta
    x126 = 9*x18
    x127 = x39*z3
    x128 = x12*x66
    x129 = x128*x41
    x130 = x35*x46
    x131 = m2*x21
    x132 = x21*x41
    x133 = x15*x41
    x134 = x42*zeta
    x135 = x16*x41
    x136 = x61*x87
    x137 = 12*x30
    x138 = 9*x30
    x139 = 9*x9
    x140 = 9*x3
    x141 = x12*x41
    x142 = x140*x141
    x143 = x22*x41
    x144 = 6*x33
    x145 = m1*x109
    x146 = x145*x30
    x147 = 6*x29
    x148 = 4*m1
    x149 = 4*zeta
    x150 = x39*zeta
    x151 = 3*x150
    x152 = x151*x3
    x153 = 2*x33
    x154 = x153*x48
    x155 = m2*x101
    x156 = 2*zeta
    x157 = x28*x52
    x158 = x17*x41
    x159 = x19*x41
    x160 = x12*x16
    x161 = m1*x25
    x162 = x76*zeta
    x163 = x28*x46
    x164 = x28*x48
    x165 = 3*x41
    x166 = 2*m2
    x167 = x28*z3
    x168 = x161 - x25
    x169 = x19*x23
    x170 = x13*x150
    x171 = x169 + x170
    x172 = -zeta
    x173 = x39*x84
    x174 = x0*x82
    x175 = x2*x82
    x176 = 2*x41
    x177 = x0*x176
    x178 = x39*x9
    x179 = x87*x9
    x180 = x33*x41
    x181 = x35*x41
    x182 = m2*x174
    x183 = x34*x41
    x184 = x17*z3
    x185 = x105*x49
    x186 = x5*x87
    x187 = x26*zeta
    x188 = x83*zeta
    x189 = x128*zeta
    x190 = x39*x41
    x191 = x12*x48
    x192 = x26*x39
    x193 = 3*x103
    x194 = m1*x193
    x195 = m2*x193
    x196 = x31*z3
    x197 = 3*x45
    x198 = 3*x84
    x199 = x19*x39
    x200 = 3*x85
    x201 = x41*x79
    x202 = x41*x80
    x203 = x103*x23
    x204 = x94*zeta
    x205 = 2*x85
    x206 = 3*z2
    x207 = 3*zeta
    x208 = x30*x39
    x209 = 3*x105
    x210 = 6*zeta
    x211 = x210*x92
    x212 = 9*x53
    x213 = x15*x87
    x214 = x16*x87
    x215 = x82*zeta
    x216 = x215*x3
    x217 = x43*x46
    x218 = x28*x41
    x219 = x103*x17
    x220 = m1*x49
    x221 = 18*x1
    x222 = 18*x18
    x223 = x181*x66
    x224 = x210*x22
    x225 = 6*x84
    x226 = 4*x181
    x227 = x90*x97
    x228 = 3*x97
    x229 = x52*zeta
    x230 = m2*x176
    x231 = x230*x33
    x232 = 2*m1
    x233 = x180*x23
    x234 = x22*x87
    x235 = 3*x127
    x236 = x82*z3
    x237 = 3*x3
    x238 = m1*m2
    x239 = 6*x71
    x240 = x210*x30
    x241 = m1*x240
    x242 = 6*m1
    x243 = 6*x0
    x244 = m1*x12
    x245 = x137*zeta
    x246 = 12*x3
    x247 = x41*x44
    x248 = x73*z3
    x249 = x176*zeta
    x250 = m1*x102
    x251 = x41*zeta
    x252 = x150*z3
    x253 = -x73
    x254 = -x230*x28 + x253
    x255 = x23*z3
    x256 = x23*x46
    x257 = 2*x34
    x258 = -x129*zeta + x150*x257 + x255 - x256
    x259 = x23*x41
    x260 = m1*x176
    x261 = x105*x87
    x262 = x261*z2
    x263 = x173*zeta
    x264 = m1*x87
    x265 = x30*x82
    x266 = 9*x12
    x267 = x149*x97
    x268 = x173*z3
    x269 = 3*x0
    x270 = x84*x87
    x271 = 2*x0
    x272 = m2*x87
    x273 = x270*z2
    x274 = 2*x82
    x275 = 2*x173
    x276 = x39*x46
    x277 = x105*x259
    x278 = x41*x93
    x279 = x103*x39
    x280 = 3*x279
    x281 = 4*x41
    x282 = 6*x102
    x283 = x39*x48
    x284 = x105*x158
    x285 = x158*x84
    x286 = x46*x97
    x287 = m1*x41
    x288 = 12*x92
    x289 = x41*x98
    x290 = 9*zeta
    x291 = x102*x105
    x292 = x44*x87
    x293 = m2*x264
    x294 = x41*x90
    x295 = x0*x109*x264
    x296 = 6*x294
    x297 = 4*x127
    x298 = 3*x190
    x299 = x105*x41
    x300 = x41*x84
    x301 = x41*x82
    x302 = x176*x35
    x303 = 2*x87
    x304 = x264*x35
    x305 = x103*x73
    x306 = x46*zeta
    x307 = 6*x35
    x308 = x150*x26
    x309 = 9*m1
    x310 = x150*x41
    x311 = 12*zeta
    x312 = m1*x141
    x313 = x221*x312
    x314 = x21*zeta
    x315 = x312*x90
    x316 = x147*x41
    x317 = x191*x33
    x318 = x151*z3
    x319 = x236*zeta
    x320 = x260*zeta
    x321 = m1*x43
    x322 = x109*zeta
    x323 = 3*x17
    x324 = -x207*x35 + x207*x46 + x207*x48 + x323
    x325 = x176*x85
    x326 = 8*x43
    x327 = x210*x247
    x328 = -x150*x242*x30 + x180*x322 + x240*x97 + x287*x326 + x325 - x327
    x329 = m2**3
    x330 = x329*x39
    x331 = m1**3*x87
    x332 = x329*x87
    x333 = x236*x84
    x334 = x46*x82
    x335 = x174*x41
    x336 = x109*x71
    x337 = x41*x97
    x338 = 5*x337
    x339 = x49*x84
    x340 = 3*m1
    x341 = m2*x261
    x342 = x166*x215
    x343 = x87*x94
    x344 = m1*x181
    x345 = m1*x103*x266
    x346 = x264*x43
    x347 = x150*x46
    x348 = m1*x190
    x349 = 3*x173
    x350 = x30*x97
    x351 = 2*x215
    x352 = x270*x89
    x353 = 6*x264
    x354 = x30*zeta
    x355 = x131*x87
    x356 = x33*x87
    x357 = x200*zeta
    x358 = x19*x190
    x359 = 6*x90
    x360 = x150*x48
    x361 = 12*x22
    x362 = 4*x150
    x363 = x143*x210
    x364 = x210*x43
    x365 = -x125*x307 + x130*x210 + x134*x52 - 9*x17*x46 + x184*x225 - x198*x251 - x207*x299 - x210*x238*x41 - x30*x364 - x357 + 2*x360 + x364 + x41*x42
    x366 = x87*x97
    x367 = x176*x92
    x368 = 3*x263
    x369 = x85*x87
    x370 = x105*x141
    x371 = x150*x52
    

    c0 = x0*(x1 - x3 + x6)
    c1 = -x10*z2 - x10*z3 - x11 + x13*x25 + x13*x4 + x14 + x15 + x16 + x18 + x20 - x21 - x22 - x24 + x27 - x29 - x30*x31 + x6 + x7 - x9
    c2 = -5*m2*x61 + x0 + x10*x12 + x10*x41 + x13*x35 - x13*x48 - x13*x49 - x13*x51 - x14*x2 - x17*x37 + x25*x62 - x26*x38 - x28 + x3*x77 + x30*x78 + x32 + x33 + x34 - x35 - x36*z2 - x37 - x38*x4 - x38 - x4*x52 - x4*x67 + x4*x70 - x40 + x42 + x44 + x45 - x47 + x5*x70 - x50 + x54 + x55 + x56*z2 + x56*z3 + x58 + x60 + 6*x61 - x63 - x64 - x65 - x69 + x72 + x74 + 5*x75 + x81
    c3 = -m1*x114 + x1*x95 + x100 + x101*x66 + x101 - x102*x148 + x102 + x103*x2 + x104 - x106 - x107 - x108 + x109*x26 - x110 - x111 + x112*x138 - x113 - x115 - 3*x116 - x117 - 2*x118 - x119 + x12*x36 + x120*x35 + x121 + x123 + x124 + x125*x36 + x126 - x127*x140 + x127*x66 + x128*x26 + x129 + x13*x158 + x13*x159 + x130 + x131 - x132 - x133 - x134 - x135 - x136 - x137*x4 - x138*x43 - x139*x25 + x14*x19 - x140*x162 - x142 - 7*x143 - x144*x76 - x146 - x147 - x149*x45 + x15*x23 - x152 - x154 + 2*x155 + x156*x44 + x157 + 7*x160 + x161*x66 + x162*x66 + x163 + x164 - x165*x29 + x166*x29 + x167*x66 + x168 + x17 + x171 + x19 - x23*x9 + x25*x66 - x26*x84 + x26 + x30*x68 + x30 + x31 - x35*x38 + x38*x46 + x38*x48 - x41*x56 - x46*x54 - x46*x55 - x48*x54 - x48*x55 - x49*x62 - x60*zeta - x68 - x77*x8 + x83 - x85 - x86 - x88 - x89 - x91 - x93 + x94 + x96 + x98 + x99
    c4 = m1*x177 - m1*x226 + m2*x177 + m2*x240 - m2*x72 + x0*x200 + x100*x52 + x101*x243 + x102*x156 - x103*x19 - x103*x31 + x103 + x105*x196 + x105*x51 + x108*zeta - x109*x187 + x11*x39 + x111*zeta - x112*x87 + x113*x46 + x12*x245 - 15*x12*x65 - x120*x83 + x120*x86 + x121*x251 - x123*zeta - x124*zeta - x126*x46 + x127*x144 - x128*x187 - x128*x46 + x128*x48 - x128*x51 + x13*x97 + x133*x149 + x135*x149 + x137*x28 + x139*x49 + x140*x190 + x140*x252 + x141*x62 + x142*zeta + x146*zeta - x149*x160 + x149*x250 - x15*x39 - x150*x67 + x166*x28 - x17*x70 + x172 + x173 + x174 - x175 - x177 + x178 + x179 - x180 - x181 - x182 - x183*x23 - x183 - x184 - x185 + x186*x232 - x186 - x187 - x188 - x189 - x190*x66 - 8*x191 - 6*x192 - x194 - x195 - x196 - x197 - x198*x49 + x198*x71 - 3*x199 - x2*x200 - x20*x39 - x201 - x202 - x203 + x204 + x205*z3 + x206*x88 + x207*x30 + x208*x242 + 3*x208 + x209*x71 + x21*x240 - x21*x249 + x211 + x212 + x213 + x214 + x216 - x217 - x218*x66 - x218 - x219 - 6*x22*x46 - x220*x221 - x222*z3 - x223 - x224 - x225*x45 - 4*x227 - x228*x3 + x228*x30 - x229*x33 - x229*x34 - x229*x35 + x231 + x233 + 3*x234 + x235*x34 + x236*x237 + x238*x239 - x24*x46 + x24*x87 + x241 + 12*x244*x71 + x246*x76 + 15*x247 - x248*x33 + x254 + x258 - x26*x40 + x29*x87 + x33*x82 - x33*x94 + x34*x82 + x35 - x38*x41 - x39*x7 + x41*x55 + 7*x42 - 8*x44 + 12*x46*x5 - x46 - x48 + x49 + x51*x84 + 4*x64 + x68*x71 + 6*x75 + x77*x9 - x78 + x81 - x97 + z3
    c5 = -m1*x156*x186 - m1*x282 - m2*x136 + m2*x265 - m2*x296 - m2*x316 - x100*x281 + x103*x307 + x103*x79 + x104*x309 - 3*x104 + x105*x159 + x105*x176 - x106*x23 + x109*x306 - x11*x150 - x110*x41 - x111*x41 - x116 + x117 + x120*x188 + x120*x28 - x125*x205 + x126*x41 + x127*x13 + x13*x41*x95 - 4*x130 - 7*x131*x41 - 12*x132*x90 - 4*x134 + x135*x266 - x137*x150 - x137*x244 + x137*x321 - x139*x141 - x140*x310 + 9*x141*x20 - x143*x242 - 6*x143 - x15*x259 + x150*x27 + x151 - x153*x236 + x155 - x156*x174 + x156*x182 + x156*x191 - x156*x234 - x157*x84 + x159*x84 + 3*x161 + x162*x221 - x162*x246 - x163*x166 + 6*x163 + 2*x164 + x165*x83 - x165*x86 + x17*x260 + x171 - x173*x269 + x176*x84 - x179*x206 + x183*x73 + x189*x46 - x189*x48 + x19*x259 + x19*x82 + x19*x85 + x193*x28 + x194*zeta + x195*zeta + x2*x275 + x200*x30 - x204*x46 + x205*x26 + x205 + x210*x45*x84 - x215*x34 + x222*x287 + x223*zeta + x224*x46 + x23*x26 - x23*x88 - x230 + x231*zeta - x232*x265 - x235*x9 - x235 - x236*x257 - x237*x301 - x237*x319 + x238*x281 - x238*x296 - x242*x294 - x243*x293 + x248 + x249*x33 + x249*x34 - 3*x25 - x259 + x26*x274 - x260 - x261*x269 + x262 + x263 + x264*x271 - x264*x55 - x264*z2 - x265 - x266*x90 - x267 - 4*x268 - x269*x270 + x271*x272 + x272*x28 - x272*x55 + 2*x273 + x276*x3 + 3*x276 + x277 + x278 + x280 + x281*x43 + x282 - x283*x36 + 7*x283 + x284 + x285 - x286 - x287*x288 - 9*x289 - x290*x45 - 9*x291 - 7*x292 - x295 - x297*x3 - x298*x33 - x299*x31 - x299*x91 + x300*x89 - x300*x91 + x302*zeta + x303*x42 + 3*x304 + x305 + 6*x308 + x310*x66 + x311*x44 + x313 + x314*x87 - 12*x315 - x316 - 4*x317 + x318*x33 - x318*x34 + x320*x35 - x322*x63 + x322*x65 + x324 + x328 + x33*x97 - x34*x57 + x35*x59 - x35*x87 - x38*x97 + 7*x39*x65 + x40*x46 + x41*x68 + x41*x96 - 8*x42*z3 - 18*x46*x53 - 6*x48*x53 + x55*x97 - x58*x87 - x69*x87 - x74*x87 + x80*x87 - x85*x91 - x92 - x94 + x95 + 4*x98 - 4*x99
    c6 = m1*x251*x288 + m1*x336 + m1*x363 - m2*x203 - 4*m2*x218 + m2*x334 - x103*x151 - x103*x169 - x103*x210*x35 - x103*x95 + x105*x201 + x105*x266*x71 - x105*x298 + x106*x39 + x107*x87 + x11*x190 + x11*x347 + x110*x264 + x111*x251 + x114*x215 - x115*x215 + x116*x48 - x118*x87 + x12*x133*x311 - x12*x214 - x121*x87 + x13*x337 - x13*x358 - x132*x210 - x134*x303 - x144*x162 - x145*x251 + x148*x208 + x148*x335 - x148*x337 + x149*x292 + x149*x317 - x149*x41*x43 + x15*x190 - 6*x150*x180 - x150*x47 + x150*x60 + x152*x87 - x156*x272*x33 - x156*x304 - x165*x188 + x165*x216 + x166*x264 - x17*x261 - x17*x270 + x173*x176 - x173*x19 + x173*x91 - x175*x41 + x179*x23 + x180*x326 + x181*x209 + x182*x41 - 6*x185 - x186*x242 + x187*x85 - x19*x261 - x19*x270 - x19*x298 - 5*x191 - x192 - x199 + x2*x325 - x200*x46 - x200*x87 + x203 - x204*x287 - x208 + x21*x359*x87 + x211*x41 + x213*x23 + x215*x232*x30 - x215*x84 - x215 + 8*x217 + x218*x84 - x219*x309 + 6*x220 + x226 - 6*x227 - x23*x261 + x23*x264 - x230*x314 - x233 + x236 - x239*x85 - x242*x247 + x242*x339 - x242*x350 + 18*x244*x354 - x245*x321 + 6*x247 + 18*x250*zeta + x251*x361 - x251*x84*x89 + x251*x85 - x252*x36 + x253 - x255*x329 + x257*x319 + x258 - x26*x349 - x26*x351 - x26*x73 + 9*x261*x5 + x261*x91 + x261 + x263*z3 + x264*x361 + x266*x354 + x267*x46 - x270*x340 + x270*x4 + x270*x91 + x270 + 4*x272*x29 - x274*x48 - x275 + x290*x291 + x293*x359 + x297*x34 + x298*x9 + x298 - x299*x73 - x299*x78 - x3*x338 + x30*x349 - x30*x357 + x301*x33 + x301*x34 - x313*zeta - x318 - x319*x33 + 3*x329*x49 + x330 - x331 - x332 + x333 - x334 - x335 - x336 - x338 - 4*x339 - 3*x341 + x342 + x343 - 10*x344 - x345 - 7*x346 - 6*x347 - x348*x36 - 2*x350 + x352 + x353*x92 + x355 - x356*x73 + x36*x360 - x362*x65 + x365 - x83*x87 + 5*x85*z3 + x86*x87 - x87*x93 + 5*x87*x98 + x93*zeta + x97
    c7 = -m1*x149*x335 + m1*x207*x270 + m1*x280 + m1*x327 + m2*x305 - x0*x261*x266 + x103*x199 - x105*x181*x207 - x105*x278 + x105*x301 + x109*x299 - x119*x150 + x123*x150 - 6*x125*x85 - x127 - x133*x362 + x134*x176 - x145*x41 + x149*x346 + x149*x355 - x150*x154 + x150*x34*x87 + x150 + x151*x299 - x151*x30 + x151*x356 + x166*x301 + x170*x41 + x173*x187 + x173*x51 + x173*x87 - x174*x249 - 2*x174*x264 + x174*x87 + x176*x215*x33 - x176*x263 - x176*x98 - x178*x87 - x180*x204 - x180*x228 + x182*x249 - x182*x87 - x183*x215 + x188*x87 + x19*x277 + x19*x301 - x19*x325 + x190*x33 - x190*x34 - x2*x369 + x206*x331 + x207*x341 - x207*x97 - x211*x264 - x212*x261 - x213*x322 - x213*x39 + x215*x70 - x216*x87 - x217*x311 - x22*x264*x311 + x241*x97 - x242*x285 + x242*x289 + x242*x310 - x242*x41*x85 + x248*x329 + x249*x97 - x260*x92 + x261*x307 + x261*x57 + x261*x73 + x261*x78 - 3*x262 - x264*x57 + 5*x264*x97 + x266*x306 - x268 + x270*x28 + x273*x340 - x273 + x276 - x279 + 2*x283 + 9*x284 - 4*x286 + 2*x292 + x295 + x296*x85 - 6*x299*x43 - x299*x95 + x3*x366 - x30*x368 + x300*x323 - x301*x84 - x301 - 4*x304 + x308 + x311*x344 - 18*x315 + x319 - x320*x97 - x323*x329*x41 + x328 - x33*x343 + x33*x371 - x330*zeta + x331*zeta + x332*zeta + x333*zeta - x337*x38 + x337*x55 - x34*x371 - x342*x46 + x345*zeta + x349*x46 - x351*x48 - x352*zeta - x353*x44 + x357*x46 - 2*x366*x90 - x366 + x367 + x368 + x369*zeta - 9*x370*x90 - x41*x94 - x43*x55*x87 + x48*x73
    c8 = x161*(m1*x202 + m1*x30*x73 + m2*x256 + x109*x354 - x12*x177 - 2*x12*x187 - x12*x70 + x141*x55 - x162*x33 - x181*x340 - 3*x185 + x190 - 2*x191 - x204*x41 + x250*x290 - x251*x68 + x254 - x287*x73 + x302 - x337 + x340*x49 - x347 - x348 - x358 + x363 + x365 + x367*zeta - x371 - x41*x50 + 2*x76)
    c9 = x370*(x162 + x167 + x168 - x197*zeta + x324)
    c10 = -x150*x331

    #polyroots
    par = [c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0]
    rf = np.polynomial.polynomial.polyroots(par)  
    mu = 0 
    for idx in range(len(rf)):
        x372 = complex.conjugate(rf[idx])
        x373 = rf[idx]**2
        c11 = -m1/x372 + m2/(-x372 + z2) + x172 + rf[idx] + (m1 + m2 - 1)/(x32 + x372)
        if abs(c11)<limit:
            z = rf[idx]
            x373 = z**2
            x374 = 2*z
            x375 = 1/(x373 - x374*z3 + x41)
            x376 = abs(-m1*x375 + m1/x373 - m2*x375 + m2/(x12 + x373 - x374*z2) + x375)
            #print rf[idx].real,rf[idx].imag
            c12 = -(x376 - 1)*(x376 + 1)
            mu += 1./abs(c12)
            
    return mu
