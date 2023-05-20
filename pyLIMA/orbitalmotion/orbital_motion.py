import numpy as np

from pyLIMA.orbitalmotion import orbital_motion_2D, orbital_motion_3D


def orbital_motion_shifts(orbital_motion_model, time, pyLIMA_parameters):
    """ Compute the trajectory curvature depending on the model.

    :param str orbital_motion_model: the orbital  motion model
    :param array_like time: the time array to compute the trajectory shift
    :param  object pyLIMA_parameters: the namedtuple containing the parameters

    :return: dseparation, dalpha the shifs in slens separation and angle
    :rtype: array_like, array_like
    """

    if orbital_motion_model[0] == '2D':

        ds_dt = pyLIMA_parameters.v_para*pyLIMA_parameters.separation
        dseparation = orbital_motion_2D.orbital_motion_2D_separation_shift(time, orbital_motion_model[1], ds_dt)

        dalpha_dt = pyLIMA_parameters.v_perp
        dalpha = orbital_motion_2D.orbital_motion_2D_trajectory_shift(time, orbital_motion_model[1], dalpha_dt)

    else:

        dseparation, dalpha = orbital_motion_3D.orbital_motion_keplerian(time, pyLIMA_parameters, orbital_motion_model)

    return dseparation, dalpha



### Depreciated
def orbital_motion_circular(time, to_om,  separation_0, v_para, v_perp, v_radial):
    """ Compute the binary separation change induced by the orbital motion of the lens,  circular version of :
    "Binary Microlensing Event OGLE-2009-BLG-020 Gives Verifiable Mass, Distance, and Orbit Predictions",Skowron et al. 2011
    http://adsabs.harvard.edu/abs/2011ApJ...738...87S

    :param float to_om: the reference time for the orbital motion
    :param float v_para: the normalised binary separation change rate, 1/s ds/dt in 1/yr
    :param float v_perp: the binary angle  change rate, dalpha/dt in 1/yr
    :param float v_radial: the normalised radial separation change rate, 1/s ds_r/dt in 1/yr
    :param float separation_0: the binary separation at to_om
    :param array_like time: the time array to compute the trajectory shift


    :return: dseparation, dalpha the binary separation and angle shifts
    :rtype: array_like, array_like
    """

    w1 = v_para
    w2 = v_perp
    w3 = v_radial

    norm_w = (w1 ** 2 + w2 ** 2 + w3 ** 2) ** 0.5
    norm_w13 = (w1 ** 2 + w3 ** 2) ** 0.5

    if norm_w13 > 10 ** -8:

        if np.abs(w3) < 10 ** -8:
            w3 = 10 ** -8

        omega = w3 * norm_w / norm_w13/365.25
        inclination = np.arcsin(w2 * w3 / (norm_w13 * norm_w))
        phi0 = np.arctan(-w1 * norm_w / (norm_w13 * w3))  # omega_N + phi_0 !!!



    else:

        omega = w2/365.25
        inclination = np.pi / 2
        phi0 = 0
    print(omega,inclination,phi0)

    eps0 = (np.cos(phi0) ** 2 + np.sin(inclination) ** 2 * np.sin(phi0) ** 2) ** 0.5
    a_true = separation_0 / eps0

    s_0 = a_true * np.array([np.cos(phi0), np.sin(inclination) * np.sin(phi0)])
    alpha_0 = np.arctan2(s_0[1], s_0[0])

    phi = omega * (time - to_om) + phi0
    separation = a_true * (np.cos(phi) ** 2 + np.sin(inclination) ** 2 * np.sin(phi) ** 2) ** 0.5

    s_t = a_true * np.array([np.cos(phi), np.sin(inclination) * np.sin(phi)])
    alpha = np.arctan2(s_t[1], s_t[0])

    return separation - separation_0, alpha - alpha_0
