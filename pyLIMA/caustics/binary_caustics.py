"""
Created on Thu Apr 20 11:19:38 2017

@author: ebachelet
"""

import numpy as np
import scipy.spatial as ss

def find_2_lenses_caustics_and_critical_curves(separation, mass_ratio, resolution=1000):
    """  Find and sort caustics for a binary lens

        :param float separation: the projected normalised angular distance between the two bodies
        :param float mass_ratio: the mass ratio of the two bodies
        :param int resolution: number of points desired in the caustic computation.

        :return: the caustic regime,  the sorted caustics and the critical curve
        :rtype: str, list of array_like, list of array_like
    """

    close_top_caustic = None
    close_top_cc = None
    close_bottom_caustic = None
    close_bottom_cc = None
    central_caustic = None
    central_cc = None
    wide_caustic = None
    wide_cc = None
    resonant_caustic = None
    resonant_cc = None

    caustics_points, critical_curve_points = compute_2_lenses_caustics_points(separation, mass_ratio,
                                                                              resolution=resolution)

    caustic_regime = find_2_lenses_caustic_regime(separation, mass_ratio)

    if caustic_regime == 'resonant':
        resonant_caustic, resonant_cc = sort_2lenses_resonant_caustic(caustics_points, critical_curve_points)

    if caustic_regime == 'close':
        result = sort_2lenses_close_caustics(caustics_points, critical_curve_points)
        central_caustic, close_top_caustic, close_bottom_caustic, central_cc, close_top_cc, close_bottom_cc = result

    if caustic_regime == 'wide':
        central_caustic, wide_caustic, central_cc, wide_cc = sort_2lenses_wide_caustics(caustics_points,
                                                                                        critical_curve_points)

    caustics = [central_caustic, close_top_caustic, close_bottom_caustic, wide_caustic, resonant_caustic]
    critical_curve = [central_cc, close_top_cc, close_bottom_cc, wide_cc, resonant_cc]

    return caustic_regime, caustics, critical_curve


def sort_2lenses_resonant_caustic(caustic_points, critical_curves_points):
    """ Sort the caustic points to have regular resonant caustic

        :param array caustic_points: the points of the caustics, in complex notation
        :param array critical_curves_points: the points of the critical curve, in complex notation

        :return: the resonant caustic,  the resonant critical curve
        :rtype:  array, array_like
    """

    try:
        medians_y = np.median(caustic_points[:, :].imag, axis=0)

        positive_y_branches = np.where(medians_y > 0)[0]

        first_branch = positive_y_branches[0]
        second_branch = positive_y_branches[1]

        if np.max((caustic_points[:, first_branch]).real) > np.max((caustic_points[:, second_branch]).real):

            resonant_caustic = np.r_[caustic_points[:, second_branch], caustic_points[:, first_branch],
                                     np.conj(caustic_points[:, first_branch][::-1]),
                                     np.conj(caustic_points[:, second_branch][::-1])]

            resonant_cc = np.r_[critical_curves_points[:, second_branch], critical_curves_points[:, first_branch],
                                np.conj(critical_curves_points[:, first_branch][::-1]),
                                np.conj(critical_curves_points[:, second_branch][::-1])]
        else:

            resonant_caustic = np.r_[caustic_points[:, first_branch], caustic_points[:, second_branch],
                                     np.conj(caustic_points[:, second_branch][::-1]),
                                     np.conj(caustic_points[:, first_branch][::-1])]

            resonant_cc = np.r_[critical_curves_points[:, first_branch], critical_curves_points[:, second_branch],
                                np.conj(critical_curves_points[:, second_branch][::-1]),
                                np.conj(critical_curves_points[:, first_branch][::-1])]

    except:

        resonant_caustic = caustic_points

        resonant_cc = critical_curves_points

    return resonant_caustic, resonant_cc


def sort_2lenses_close_caustics(caustic_points, critical_curves_points):
    """ Sort the caustic points to have one central caustic and two "planetary" caustics

        :param array caustic_points: the points of the caustics, in complex notation
        :param array critical_curves_points: the points of the critical curve, in complex notation

        :return: the central caustic,  the top planetary caustic, the bottom planetary caustic,
        the central critical curve, the top critical curve, the bottom critical curve
        :rtype:  array, array, array, array, array, array
    """

    try:
        medians_y = np.median(caustic_points[:, :].imag, axis=0)

        order = np.argsort(medians_y)

        close_bottom_caustic = caustic_points[:, order[0]]
        close_bottom_cc = critical_curves_points[:, order[0]]

        close_top_caustic = caustic_points[:, order[-1]]
        close_top_cc = critical_curves_points[:, order[-1]]

        if np.abs(caustic_points[-1, order[1]].real - caustic_points[0, order[2]].real) < np.abs(
            caustic_points[-1, order[1]].real - caustic_points[:, order[2]][::-1][0].real):

            central_caustic = np.r_[caustic_points[:, order[1]], caustic_points[:, order[2]]]
            central_cc = np.r_[critical_curves_points[:, order[1]], critical_curves_points[:, order[2]]]

        else:

            central_caustic = np.r_[caustic_points[:, order[1]], caustic_points[:, order[2]][::-1]]
            central_cc = np.r_[critical_curves_points[:, order[1]], critical_curves_points[:, order[2]][::-1]]


    except:

        medians_y = np.median(caustic_points[:, :].imag)

        order = np.argsort(medians_y)

        close_bottom_caustic = caustic_points[:, order[0]]
        close_bottom_cc = critical_curves_points[:, order[0]]

        close_top_caustic = caustic_points[:, order[-1]]
        close_top_cc = critical_curves_points[:, order[-1]]

        if np.abs(caustic_points[-1, order[1]].real - caustic_points[0, order[2]].real) < np.abs(
                        caustic_points[-1, order[1]].real - caustic_points[:, order[2]][::-1][0].real):

            central_caustic = np.r_[caustic_points[:, order[1]], caustic_points[:, order[2]]]
            central_cc = np.r_[critical_curves_points[:, order[1]], critical_curves_points[:, order[2]]]

        else:

            central_caustic = np.r_[caustic_points[:, order[1]], caustic_points[:, order[2]][::-1]]
            central_cc = np.r_[critical_curves_points[:, order[1]], critical_curves_points[:, order[2]][::-1]]


    return central_caustic, close_top_caustic, close_bottom_caustic, central_cc, close_top_cc, close_bottom_cc


def sort_2lenses_wide_caustics(caustic_points, critical_curves_points):
    """ Sort the caustic points to have one central caustic and one "planetary" caustics

        :param array caustic_points: the points of the caustics, in complex notation
        :param array critical_curves_points: the points of the critical curve, in complex notation

        :return: the central caustic,  the  planetary caustic, the central critical curve,
        the planetary critical curve
        :rtype:  array, array, array, array
    """
    try:

        medians_y = np.median(caustic_points[:, :].imag, axis=0)

        order = np.argsort(medians_y)

        wide_bottom_caustic = caustic_points[:, order[0]]
        wide_bottom_cc = critical_curves_points[:, order[0]]

        central_bottom_caustic = caustic_points[:, order[1]]
        central_bottom_cc = critical_curves_points[:, order[1]]

        wide_top_caustic = caustic_points[:, order[-1]]
        wide_top_cc = critical_curves_points[:, order[-1]]

        central_top_caustic = caustic_points[:, order[2]]
        central_top_cc = critical_curves_points[:, order[2]]

        central_caustic = np.r_[central_bottom_caustic, central_top_caustic]
        central_cc = np.r_[central_bottom_cc, central_top_cc]

        wide_caustic = np.r_[wide_bottom_caustic, wide_top_caustic]
        wide_cc = np.r_[wide_bottom_cc, wide_top_cc]

    except:

        medians_y = np.median(caustic_points[:, :].imag)

        order = np.argsort(medians_y)

        wide_bottom_caustic = caustic_points[:, order[0]]
        wide_bottom_cc = critical_curves_points[:, order[0]]

        central_bottom_caustic = caustic_points[:, order[1]]
        central_bottom_cc = critical_curves_points[:, order[1]]

        wide_top_caustic = caustic_points[:, order[-1]]
        wide_top_cc = critical_curves_points[:, order[-1]]

        central_top_caustic = caustic_points[:, order[2]]
        central_top_cc = critical_curves_points[:, order[2]]

        central_caustic = np.r_[central_bottom_caustic, central_top_caustic]
        central_cc = np.r_[central_bottom_cc, central_top_cc]

        wide_caustic = np.r_[wide_bottom_caustic, wide_top_caustic]
        wide_cc = np.r_[wide_bottom_cc, wide_top_cc]

    return central_caustic, wide_caustic, central_cc, wide_cc


def compute_2_lenses_caustics_points(separation, mass_ratio, resolution=1000):
    """  Find the critical curve points and caustics points associated to a binary lens. See :
        "On the Minimum Magnification Between Caustic Crossings for Microlensing by Binary and Multiple Stars"
        Witt and Mao 1995 http://adsabs.harvard.edu/abs/1995ApJ...447L.105W
        "Investigation of high amplification events in light curves of gravitationally lensed quasars"
        Witt H.J 1990 http://adsabs.harvard.edu/abs/1990A%26A...236..311W

        :param float separation: the projected normalised angular distance between the two bodies
        :param float mass_ratio: the mass ratio of the two bodies
        :param int resolution: number of points desired in the caustic computation.

        :return: caustic_points, cc_points
        :rtype:  array, array
    """
    caustics = []
    critical_curves = []

    center_of_mass = mass_ratio / (1 + mass_ratio) * separation

    # Witt&Mao magic numbers
    total_mass = 0.5
    mass_1 = 1 / (1 + mass_ratio)
    mass_2 = mass_ratio * mass_1
    delta_mass = (mass_2 - mass_1) / 2
    lens_1 = -separation / 2.0
    lens_2 = separation / 2.0
    lens_1_conjugate = np.conj(lens_1)
    lens_2_conjugate = np.conj(lens_2)

    phi = np.linspace(0.00, 2 * np.pi, resolution)
    roots = []
    wm_1 = 4.0 * lens_1 * delta_mass
    wm_3 = 0.0
    slice_1 = []
    slice_2 = []
    slice_3 = []
    slice_4 = []
    slices = [slice_1, slice_2, slice_3, slice_4]
    for angle in phi:

        e_phi = np.cos(-angle) + 1j * np.sin(-angle)  # See Witt & Mao

        wm_0 = -2.0 * total_mass * lens_1 ** 2 + e_phi * lens_1 ** 4
        wm_2 = -2.0 * total_mass - 2 * e_phi * lens_1 ** 2
        wm_4 = e_phi

        polynomial_coefficients = [wm_4, wm_3, wm_2, wm_1, wm_0]

        polynomial_roots = np.roots(polynomial_coefficients)

        checks = np.polyval(polynomial_coefficients, polynomial_roots)

        if np.max(np.abs(checks)) > 10 ** -10:
            pass
        else:
            # polynomial_roots = np.polynomial.polynomial.polyroots(polynomial_coefficients[::-1])

            if len(roots) == 0:

                pol_roots = polynomial_roots

            else:

                aa = np.c_[polynomial_roots.real, polynomial_roots.imag]
                bb = np.c_[roots[-1].real, roots[-1].imag]

                distances = ss.distance.cdist(aa, bb)
                good_order = [0, 0, 0, 0]

                for i in range(4):
                    line, column = np.where((distances) == np.min(distances))
                    good_order[column[0]] = polynomial_roots[line[0]]
                    distances[line[0], :] += 10 ** 10
                    distances[:, column[0]] += 10 ** 10

                pol_roots = np.array(good_order)

            roots.append(pol_roots)

            images_conjugate = np.conj(pol_roots)
            zeta_caustics = pol_roots + mass_1 / (lens_1_conjugate - images_conjugate) + mass_2 / (
                lens_2_conjugate - images_conjugate)

            if len(caustics) == 0:
                caustics = zeta_caustics
                critical_curves = pol_roots
            else:
                caustics = np.vstack((caustics, zeta_caustics))
                critical_curves = np.vstack((critical_curves, pol_roots))

    # shift into center of mass referentiel

    caustics += -center_of_mass + separation / 2
    critical_curves += -center_of_mass + separation / 2

    return caustics, critical_curves

def find_2_lenses_caustic_regime(separation, mass_ratio):
    """  Find the caustic regime.
        "An alternative parameterisation for binary-lens caustic-crossing events"
        Cassan A.2008 http://adsabs.harvard.edu/abs/2008A%26A...491..587C
        "Speeding up Low-mass Planetary Microlensing Simulations and Modeling: The Caustic Region Of INfluence"
        Penny M.2014 http://adsabs.harvard.edu/abs/2014ApJ...790..142P

       :param float separation: the projected normalised angular distance between the two bodies
       :param float mass_ratio: the mass ratio of the two bodies

       :return: caustic_regime: close, wide or resonant
       :rtype: str
   """

    caustic_regime = 'resonant'

    wide_limit = ((1 + mass_ratio ** (1 / 3.0)) ** 3.0 / (1 + mass_ratio)) ** 0.5

    if (separation > 1.0) & (separation > wide_limit):
        caustic_regime = 'wide'
        return caustic_regime

    close_limit = mass_ratio / (1 + mass_ratio) ** 2.0

    if (separation < 1.0) & (1 / separation ** 8.0 * ((1 - separation ** 4.0) / 3.0) ** 3.0 > close_limit):
        caustic_regime = 'close'
        return caustic_regime

    return caustic_regime

def poly_binary_eiphi_0(separation, mass_ratio):
    """  Build the polynomial associated to phi=0 binary caustics.
        "Investigation of high amplification events in light curves of gravitationally lensed quasars"
        Witt H.J 1990 http://adsabs.harvard.edu/abs/1990A%26A...236..311W

       :param float separation: the projected normalised angular distance between the two bodies
       :param float mass_ratio: the mass ratio of the two bodies

       :return: polynomial coefficients
       :rtype: array
   """

    s = separation
    q = mass_ratio

    polynomial_coefficients = [1.0, 2.0 * s * (q - 1) / (q + 1.0),
                               (s ** 2 * (q ** 2 - 4 * q + 1) - q ** 2 - 2 * q - 1) / (q ** 2 + 2.0 * q + 1.0),
                               2 * (-q ** 3 * s + (s ** 2 + 1) * q * s * (- q + 1) + s) / (q ** 3 + 3.0 * q ** 2 + 3.0 * q + 1.0),
                               (-q ** 2 * s ** 2 * (q ** 2 + q - s ** 2) - s ** 2 * (1 + q)) / (q ** 4 + 4.0 * q ** 3 + 6.0 * q ** 2 + 4.0 * q + 1.0)]

    return polynomial_coefficients

def caustic_points_at_phi_0(separation, mass_ratio):
    """ Find caustics points at phi=0 binary caustics.
           "Investigation of high amplification events in light curves of gravitationally lensed quasars"
           Witt H.J 1990 http://adsabs.harvard.edu/abs/1990A%26A...236..311W

          :param float separation: the projected normalised angular distance between the two bodies
          :param float mass_ratio: the mass ratio of the two bodies

          :return: caustic points
          :rtype: array
    """

    polynomial_coefficients = poly_binary_eiphi_0(separation, mass_ratio)
    solutions = np.roots(polynomial_coefficients)

    m_tot = 1+mass_ratio
    m_1 = 1/m_tot
    m_2 = mass_ratio*m_1

    position_1 = -separation*m_2
    position_2 = position_1+separation

    caustic_points = lens_equation(solutions, [m_1, m_2], [position_1, position_2])

    return caustic_points

def lens_equation(z,lenses_mass,lenses_pos):
    """ The complex lens equation
           "Investigation of high amplification events in light curves of gravitationally lensed quasars"
           Witt H.J 1990 http://adsabs.harvard.edu/abs/1990A%26A...236..311W

          :param array z: the complex position in the image plane
          :param array lenses_mass: the masses of the lenses (scale to 1)
          :param array lenses_pos: the complex position of the lenses

          :return: zeta, the associated complex position in the source plane
          :rtype: array
    """

    zeta = np.array([j-np.sum(np.array([lenses_mass[i]/(np.conj(j)-np.conj(lenses_pos[i]))
                                        for i in range(len(lenses_mass))])) for j in z])

    return zeta
