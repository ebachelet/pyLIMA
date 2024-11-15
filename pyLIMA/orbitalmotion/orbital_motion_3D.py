import numpy as np


def orbital_motion_keplerian(time, pyLIMA_parameters, om_model):
    """"
    Circular and Keplerian orbital motion of the lens. Use the same parametrization
    from Bozza 2021.
    See https://ui.adsabs.harvard.edu/abs/2021MNRAS.505..126B/abstract
        https://ui.adsabs.harvard.edu/abs/2011ApJ...738...87S/abstract

    Parameters
    ----------
    time : array, containing the time to treat
    pyLIMA_parameters : a pyLIMA_parameters object
    om_model :  list, [str,float] the orbital motion model

    Returns
    -------
    dseparation : array, containts the variation of the lens separation due to the
    motion of the lens
    dalpha : array, containts the variation of the lens trajectory angle due to the
    motion of the lens
    """
    Rmatrix = pyLIMA_parameters['Rmatrix']
    orbital_velocity = pyLIMA_parameters['orbital_velocity']
    a_true = pyLIMA_parameters['a_true']
    t_periastron = pyLIMA_parameters['t_periastron']

    if om_model[0] == 'Circular':  # Circular

        theta = orbital_velocity * (time - t_periastron) / 365.25

        r_prime = a_true * np.array([np.cos(theta), np.sin(theta)])

        r_microlens = np.dot(Rmatrix, r_prime)

    else:  # Keplerian

        eccentricity = pyLIMA_parameters['eccentricity']

        eccentric_anomaly = eccentric_anomaly_function(time, eccentricity, t_periastron,
                                                       orbital_velocity / 365.25)

        r_prime = a_true * np.array([np.cos(eccentric_anomaly) - eccentricity,
                                     (1 - eccentricity ** 2) ** 0.5 * np.sin(
                                         eccentric_anomaly)])

        r_microlens = np.dot(Rmatrix, r_prime)

    sep = (r_microlens[0] ** 2 + r_microlens[1] ** 2) ** 0.5
    angle = np.arctan2(r_microlens[1], r_microlens[0])

    separation0 = pyLIMA_parameters['separation']
    angle_0 = 0
    ### Just to check,
    # to_om = om_model[1]
    # eccentricity = pyLIMA_parameters.eccentricity

    # eccentric_anomaly_0 = eccentric_anomaly_function([to_om], eccentricity,
    # t_periastron, orbital_velocity/365.25)
    # r_prime_0 = np.array([np.cos(eccentric_anomaly_0) - eccentricity,
    # (1 - eccentricity ** 2) ** 0.5 *
    #                      np.sin(eccentric_anomaly_0)]) * a_true# / rE
    # r_microlens_0 = np.dot(Rmatrix, r_prime_0)
    # separation0 = (r_microlens_0[0] ** 2 + r_microlens_0[1] ** 2) ** 0.5
    # angle_0 = np.arctan2(r_microlens_0[1], r_microlens_0[0])
    # breakpoint()
    return sep - separation0, (angle - angle_0)


def orbital_parameters_from_position_and_velocities(separation_0, r_s, a_s, v_para,
                                                    v_perp, v_radial, t0_om):
    """
    Compute the (unscaled!, i.e. GMass is not known) orbital parameters from the
    microlensing parameters
    See https://ui.adsabs.harvard.edu/abs/2021MNRAS.505..126B/abstract
        https://ui.adsabs.harvard.edu/abs/2011ApJ...738...87S/abstract
        https://orbital-mechanics.space/classical-orbital-elements/orbital-elements
        -and-the-state-vector.html

    Parameters
    ----------
    separation_0 : float, the binary separation at t0_om
    r_s : float, the ratio of the radial separation s_z over the projected separation
    s_0
    a_s : float, the ratio of the microlesing separation over the true (unscaled)
    semi-major axis
    v_para : float, the rotation speed along the s_0 axis, i.e. 1/s ds/dt
    v_perp : float, the rotation speed perpendicular to s_0 axis, i.e. dalpha/dt
    v_radial : float, the rotation speed in the z axis, i.e. 1/s ds_z/dt
    t0_om:  float, the time of reference of the orbital _motion model

    Returns
    -------
    longitude_ascending_node : float
    inclination : float
    omega_peri : float
    orbital_velocity : float
    eccentricity : float
    true_anomaly : float
    t_periastron :  float
    x_0 : float, the first column of the Rotation matrix
    y_0 : float, the second column of the Rotation matrix
    z_0 : float, the first column of the Rotation matrix
    """
    e_0, h_0, r_0, v_0, r_norm, separation_z, a_true, GMass, orbital_velocity = \
        state_orbital_elements(
            separation_0, r_s, a_s, v_para, v_perp, v_radial)

    # From Skowron2011, Bozza2020
    # and https://orbital-mechanics.space/classical-orbital-elements/orbital-elements
    # -and-the-state-vector.html

    eccentricity = np.sum(e_0 ** 2) ** 0.5
    h_norm = np.sum(h_0 ** 2) ** 0.5
    z_0 = h_0 / h_norm

    inclination = np.arccos(z_0[2])

    N = np.cross([0, 0, 1], h_0)
    longitude_ascending_node = np.arctan2(N[1], N[0])

    if np.abs(np.dot(r_0, v_0)) < 10 ** -10:  # Circular

        eccentricity = 0
        cosw = separation_0 / a_true * np.cos(longitude_ascending_node)
        sinw = separation_z / np.sin(inclination) / a_true
        omega_peri = np.arctan2(sinw, cosw)

        true_anomaly = omega_peri
        t_periastron = t0_om

        from scipy.spatial.transform import Rotation

        Rmatrix = Rotation.from_euler("ZXZ", [-omega_peri, -inclination,
                                              -longitude_ascending_node])
        x_0, y_0, z_0 = Rmatrix.as_matrix()

    else:  # Keplerian

        x_0 = e_0 / eccentricity
        z_0 = h_0 / h_norm
        y_0 = np.cross(z_0, x_0)
        r_0_norm = r_0 / r_norm

        omega_peri = np.arctan2(x_0[2], y_0[2])

        cos_true_anomaly = np.dot(r_0_norm, x_0)
        sin_true_anomaly = np.dot(r_0_norm, y_0)

        true_anomaly = np.arctan2(sin_true_anomaly, cos_true_anomaly)

        eccentric_anomaly = np.arctan2(
            (1 - eccentricity ** 2) ** 0.5 * np.sin(true_anomaly),
            (np.cos(true_anomaly) + eccentricity))

        t_periastron = t0_om - \
                       (eccentric_anomaly - eccentricity * np.sin(
                           eccentric_anomaly)) / orbital_velocity * 365.25

    return longitude_ascending_node, inclination, omega_peri, a_true, \
        orbital_velocity, eccentricity, true_anomaly, \
        t_periastron, x_0, y_0, z_0


def state_orbital_elements(separation_0, r_s, a_s, v_para, v_perp, v_radial):
    """
    Compute the stated orbital parameters from the microlensing parameters
    See https://ui.adsabs.harvard.edu/abs/2021MNRAS.505..126B/abstract
        https://ui.adsabs.harvard.edu/abs/2011ApJ...738...87S/abstract
        https://orbital-mechanics.space/classical-orbital-elements/orbital-elements
        -and-the-state-vector.html

    Parameters
    ----------
    separation_0 : float, the binary separation at t0_om
    r_s : float, the ratio of the radial separation s_z over the projected separation
    s_0
    a_s : float, the ratio of the microlesing separation over the true (unscaled)
    semi-major axis
    v_para : float, the rotation speed along the s_0 axis, i.e. 1/s ds/dt
    v_perp : float, the rotation speed perpendicular to s_0 axis, i.e. dalpha/dt
    v_radial : float, the rotation speed in the z axis, i.e. 1/s ds_z/dt

    Returns
    -------
    e_0 : array, the eccentricity vector
    h_0 : array, the specific angular momentum vector
    r_0 : array, the microlensing separation vector at time t0_om
    v_0 : array, the microlensing speed vector at time t0_om
    r_norm : float, the norm of the separation vector
    separation_z : float, the radial separation
    a_true : float,the semi-major-axis
    GMass : float, the unscaled mass
    orbital_velocity : float, the orbital velocity of the lens
    """
    separation_z = r_s * separation_0
    r_0 = np.array([separation_0, 0, separation_z])

    r_norm = np.sum(r_0 ** 2) ** 0.5
    a_true = a_s * r_norm

    v_0 = r_0[0] * np.array([v_para, v_perp, v_radial])

    h_0 = np.cross(r_0, v_0)

    GMass = np.sum(v_0 ** 2) / (-1 / a_true + 2 / r_norm)

    e_0 = np.cross(v_0, h_0) / GMass - r_0 / r_norm

    orbital_velocity = (GMass / a_true ** 3) ** 0.5

    return e_0, h_0, r_0, v_0, r_norm, separation_z, a_true, GMass, orbital_velocity


def eccentric_anomaly_function(time, ellipticity, t_periastron, speed):
    """
    Solve the Kepler equation, see https://github.com/dfm/kepler.py

    Parameters
    ----------
    time : array, the time to treat
    ellipticity : float, the eccentricity of the orbit
    t_periastron : float, the time of periastron of the orbit
    speed : float, the orbital velocity

    Returns
    -------
    eccentricities : array, the associated eccentric anomalies at time t
    """
    import kepler

    eccentricities = []

    for t in time:
        phase = speed * (t - t_periastron)
        phase = phase % (2 * np.pi)

        ecc = kepler.solve(phase, ellipticity)

        eccentricities.append(ecc)

    return eccentricities
