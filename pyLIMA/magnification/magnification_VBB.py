import VBBinaryLensing
import numpy as np
import os

VBB = VBBinaryLensing.VBBinaryLensing()
VBB.Tol = 0.01
VBB.RelTol = 0.01
VBB.minannuli = 2  # stabilizing for rho>>caustics


def magnification_FSPL(tau, uo, rho, limb_darkening_coefficient, sqrt_limb_darkening_coefficient=None):
    """
    The VBB FSPL for large source. Faster than the numba implementations...
    Much slower than Yoo et al. but valid for all rho, all u_o
    :param array_like tau: the tau define for example in
                               http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    :param array_like uo: the uo define for example in
                             http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    :param float rho: the normalised angular source star radius
    :param float limb_darkening_coefficient: the linear limb-darkening coefficient (~gamma)
    :param float sqrt_limb_darkening_coefficient: the square-root limb-darkening coefficient (~sigma)

    :return: the FSPL magnification A_FSPL(t) for large sources
    :rtype: array_like
    """
    VBB.LoadESPLTable(os.path.dirname(VBBinaryLensing.__file__) + '/VBBinaryLensing/data/ESPL.tbl')
    VBB.a1 = limb_darkening_coefficient

    if sqrt_limb_darkening_coefficient:
        VBB.SetLDprofile(VBB.LDsquareroot)
        VBB.a2 = sqrt_limb_darkening_coefficient

    magnification_fspl = []

    import pyLIMA.magnification.impact_parameter

    impact_parameter = pyLIMA.magnification.impact_parameter.impact_parameter(tau, uo)  # u(t)

    for ind, u in enumerate(impact_parameter):
        magnification_VBB = VBB.ESPLMagDark(u, rho, limb_darkening_coefficient)

        magnification_fspl.append(magnification_VBB)

    return np.array(magnification_fspl)

def magnification_USBL(separation, mass_ratio, x_source, y_source, rho):
    """
    The Uniform Source Binary Lens magnification, based on the work of Valerio Bozza, thanks :)
    "Microlensing with an advanced contour integration algorithm: Green's theorem to third order, error control,
    optimal sampling and limb darkening ",Bozza, Valerio 2010. Please cite the paper if you used this.
    http://mnras.oxfordjournals.org/content/408/4/2188

    :param array_like separation: the projected normalised angular distance between the two bodies
    :param float mass_ratio: the mass ratio of the two bodies
    :param array_like x_source: the horizontal positions of the source center in the source plane
    :param array_like y_source: the vertical positions of the source center in the source plane
    :param float rho: the normalised (to :math:`\\theta_E') angular source star radius

    :return: the USBL magnification A_USBL(t)
    :rtype: array_like
    """

    magnification_usbl = []

    for xs, ys, s in zip(x_source, y_source, separation):

        magnification_VBB = VBB.BinaryMag2(s, mass_ratio, xs, ys, rho)

        magnification_usbl.append(magnification_VBB)

    return np.array(magnification_usbl)
    

def magnification_FSBL(separation, mass_ratio, x_source, y_source, rho, limb_darkening_coefficient):
    """
    The Uniform Source Binary Lens magnification, based on the work of Valerio Bozza, thanks :)
    "Microlensing with an advanced contour integration algorithm: Green's theorem to third order, error control,
    optimal sampling and limb darkening ",Bozza, Valerio 2010. Please cite the paper if you used this.
    http://mnras.oxfordjournals.org/content/408/4/2188

    :param array_like separation: the projected normalised angular distance between the two bodies
    :param float mass_ratio: the mass ratio of the two bodies
    :param array_like x_source: the horizontal positions of the source center in the source plane
    :param array_like y_source: the vertical positions of the source center in the source plane
    :param float limb_darkening_coefficient: the linear limb-darkening coefficient
    :param float rho: the normalised (to :math:`\\theta_E') angular source star radius

    :return: the FSBL magnification A_FSBL(t)
    :rtype: array_like
    """

    magnification_fsbl = []

    for xs, ys, s in zip(x_source, y_source, separation):

        magnification_VBB = VBB.BinaryMagDark(s, mass_ratio, xs, ys, rho, limb_darkening_coefficient, VBB.Tol)

        magnification_fsbl.append(magnification_VBB)

    return np.array(magnification_fsbl)


def magnification_PSBL(separation, mass_ratio, x_source, y_source):
    """
    The Point Source Binary Lens magnification, based on the work of Valerio Bozza, thanks :)
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

    magnification_psbl = []

    for xs, ys, s in zip(x_source, y_source, separation):

        magnification_VBB = VBB.BinaryMag0(s, mass_ratio, xs, ys)

        magnification_psbl.append(magnification_VBB)

    return np.array(magnification_psbl)
