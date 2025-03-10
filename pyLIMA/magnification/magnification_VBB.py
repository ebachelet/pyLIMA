import os
import numpy as np
import VBMicrolensing

#VBB = VBBinaryLensing.VBBinaryLensing()
#VBB.Tol = 0.001
#VBB.RelTol = 0.001
#VBB.minannuli = 2  # stabilizing for rho>>caustics


VBM = VBMicrolensing.VBMicrolensing()
VBM.Tol = 0.001
VBM.RelTol = 0.001
VBM.minannuli = 2  # stabilizing for rho>>caustics



def magnification_FSPL(tau, beta, rho, limb_darkening_coefficient,
                       sqrt_limb_darkening_coefficient=None):
    """
    The VBB FSPL for large source. Faster than the numba implementations...
    Much slower than Yoo et al. but valid for all rho, all u_o

    Parameters
    ----------
    tau : array, (t-t0)/tE
    beta : array, [u0]*len(t)
    rho : float, the normalized angular source radius
    limb_darkening_coefficient: the linear limb-darkening coefficient (a1)
    sqrt_limb_darkening_coefficient: the square-root limb-darkening
    coefficient (a2)

    Returns
    -------
    magnification_fspl : array, A(t) for FSPL
    impact_parameter : array, u(t)
    """
    #VBB.LoadESPLTable(
    #    os.path.dirname(VBBinaryLensing.__file__) + '/data/ESPL.tbl')
    #VBB.a1 = limb_darkening_coefficient

    VBM.LoadESPLTable(
        os.path.dirname(VBMicrolensing.__file__) + '/data/ESPL.tbl')
    VBM.a1 = limb_darkening_coefficient


    if sqrt_limb_darkening_coefficient is not None:
        VBM.SetLDprofile(VBM.LDsquareroot)
        VBM.a2 = sqrt_limb_darkening_coefficient

    magnification_fspl = []

    import pyLIMA.magnification.impact_parameter

    impact_parameter = pyLIMA.magnification.impact_parameter.impact_parameter(tau,
                                                                              beta)  #
    # u(t)

    for ind, u in enumerate(impact_parameter):
        #magnification_VBB = VBB.ESPLMagDark(u, rho)
        magnification_VBM = VBM.ESPLMagDark(u, rho)

        #magnification_fspl.append(magnification_VBB)
        magnification_fspl.append(magnification_VBM)

    return np.array(magnification_fspl)


def magnification_USBL(separation, mass_ratio, x_source, y_source, rho):
    """
    The Uniform Source Binary Lens magnification, based on the work of Valerio Bozza,
    thanks :) Please cite the paper if you used this.
    See http://mnras.oxfordjournals.org/content/408/4/2188

    Parameters
    ----------
    separation : array, the projected normalised angular distance between
    the two bodies
    mass_ratio : float, the mass ratio of the two bodies
    x_source : array, the horizontal positions of the source center in the source plane
    y_source : array, the vertical positions of the source center in the source plane
    rho : float, the normalized angular source radius

    Returns
    -------
    magnification_usbl : array, the USBL magnification
    """

    magnification_usbl = []

    for xs, ys, s in zip(x_source, y_source, separation):
       # print(s, mass_ratio, xs, ys, rho)
        #magnification_vbb = VBB.BinaryMag2(s, mass_ratio, xs, ys, rho)
       magnification_vbb = VBM.BinaryMag2(s, mass_ratio, xs, ys, rho)

       magnification_usbl.append(magnification_vbb)
        #import decimal
        #print(decimal.Decimal.from_float(s))
        #print(decimal.Decimal.from_float(mass_ratio))
        #print(decimal.Decimal.from_float(xs))
        #print(decimal.Decimal.from_float(ys))
        #print(decimal.Decimal.from_float(rho))
        #print(decimal.Decimal.from_float(magnification_vbb))
        #print('####')

        #if magnification_vbb<0:
         #       breakpoint()
    return np.array(magnification_usbl)


def magnification_FSBL(separation, mass_ratio, x_source, y_source, rho,
                       limb_darkening_coefficient):
    """
    The Finite Source Binary Lens magnification, including limb-darkening, based on
    the work of Valerio Bozza, thanks :)  Please cite the paper if you used this.
    See http://mnras.oxfordjournals.org/content/408/4/2188

    Parameters
    ----------
    separation : array, the projected normalised angular distance between
    the two bodies
    mass_ratio : float, the mass ratio of the two bodies
    x_source : array, the horizontal positions of the source center in the source plane
    y_source : array, the vertical positions of the source center in the source plane
    rho : float, the normalized angular source radius
    limb_darkening_coefficient: the linear limb-darkening coefficient (a1)

    Returns
    -------
    magnification_fsbl : array, the FSBL magnification
    """

    magnification_fsbl = []

    for xs, ys, s in zip(x_source, y_source, separation):
        #magnification_VBB = VBB.BinaryMagDark(s, mass_ratio, xs, ys, rho,
        #                                      limb_darkening_coefficient)
        magnification_VBB = VBM.BinaryMagDark(s, mass_ratio, xs, ys, rho,
                                              limb_darkening_coefficient)

        magnification_fsbl.append(magnification_VBB)

    return np.array(magnification_fsbl)


def magnification_PSBL(separation, mass_ratio, x_source, y_source):
    """
    The Point Source Binary Lens magnification,, including limb-darkening, based on
    the work of Valerio Bozza, thanks :)  Please cite the paper if you used this.
    See http://mnras.oxfordjournals.org/content/408/4/2188

    Parameters
    ----------
    separation : array, the projected normalised angular distance between
    the two bodies
    mass_ratio : float, the mass ratio of the two bodies
    x_source : array, the horizontal positions of the source center in the source plane
    y_source : array, the vertical positions of the source center in the  source plane

    Returns
    -------
    magnification_psbl : array, the PSBL magnification
    """

    magnification_psbl = []

    for xs, ys, s in zip(x_source, y_source, separation):
#        magnification_VBB = VBB.BinaryMag0(s, mass_ratio, xs, ys)
        magnification_VBB = VBM.BinaryMag0(s, mass_ratio, xs, ys)

        magnification_psbl.append(magnification_VBB)

    return np.array(magnification_psbl)
