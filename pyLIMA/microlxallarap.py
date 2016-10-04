import numpy as np



def compute_xallarap_curvature(XiE, delta_positions):
    """ Compute the curvature induce by the xallarap
    deltas_positions of a telescope.

    :param array_like piE: the microlensing parallax vector. Have a look :
                           http://adsabs.harvard.edu/abs/2004ApJ...606..319G
    :param array_like delta_positions: the delta_positions of the telescope. More details in microlparallax module.
    :return: delta_tau and delta_u, the shift introduce by parallax
    :rtype: array_like,array_like
    """

    delta_tau = np.dot(XiE, delta_positions)
    delta_beta = np.cross(XiE, delta_positions.T)

    return -delta_tau, -delta_beta



def North_East_vectors_target(ra_xallarap, dec_xallarap):
    """This function define the North and East vectors projected on the sky plane
    perpendicular to the line
    of sight (i.e the line define by ra,dec of the center of mass of the source binary).

    :param float ra_xallarap : the right acsension of the source binary center of mass in degree.
    :param float dec_xallarap : the declinaision of the source binary center of mass in degree.
    :return: the North and East vectors projected in the sky plan for this trajectory
    :rtype: array_like, array_like
    """
    target_angles_in_the_sky = [ra_xallarap * np.pi / 180, dec_xallarap * np.pi / 180]
    source_center_of_mass = np.array(
        [np.cos(target_angles_in_the_sky[1]) * np.cos(target_angles_in_the_sky[0]),
         np.cos(target_angles_in_the_sky[1]) * np.sin(target_angles_in_the_sky[0]),
         np.sin(target_angles_in_the_sky[1])])

    East = np.array([-np.sin(target_angles_in_the_sky[0]), np.cos(target_angles_in_the_sky[0]), 0.0])
    North = np.cross(source_center_of_mass, East)

    return North, East


def true_anomaly_from_mean_anomaly(mean_anomaly, eccentricity):
    """Find the true anomaly from the mean anomaly (i.e solve Kepler equation).
    :param float mean_anomaly : the mean_anomaly of your orbit.
    :param float eccentricity: the eccentricity of your ellipse.
    :return: the true anomaly of your orbit
    :rtype: float
    """

    precision = 0.000001
    eccentric_anomaly_approximation = mean_anomaly
    eccentric_anomaly = 10 * mean_anomaly
    last_eccentric_anomaly = 5 * mean_anomaly

    while np.abs((last_eccentric_anomaly - eccentric_anomaly) / eccentric_anomaly) > precision:
        last_eccentric_anomaly = eccentric_anomaly_approximation
        eccentric_anomaly = mean_anomaly + eccentricity * np.sin(eccentric_anomaly_approximation)
        eccentric_anomaly_approximation = eccentric_anomaly

    true_anomaly = 2 * np.sign(eccentric_anomaly) * np.arctan(
        np.tan(eccentric_anomaly / 2.0) * ((1 + eccentricity) / (1 - eccentricity)) ** 0.5)

    return true_anomaly


def xallarap(time_to_treat, ra_xallarap, dec_xallarap, orbital_period, eccentricity, time_periastron):
    """Compute the delta positions induced by the source xallarap. See Miyake et al 2012 :
        http://iopscience.iop.org/article/10.1088/0004-637X/752/2/82/pdf for more details.

    :param array_like time_to_treat : the time you looking at.
    :param float ra_xallarap : the right acsension of the source binary center of mass in degree.
    :param float dec_xallarap : the declinaision of the source binary center of mass in degree.
    :param float oribtal_period : the orbital period of your orbit in days.
    :param float eccentricity: the eccentricity of your ellipse.
    :param float time_periastron: the time passage at periastron.

    :return: deltas_positions, the shift in North and East cooridnates due to the mouvement of the source.
    :rtype: array_like
    """

    North, East = North_East_vectors_target(ra_xallarap, dec_xallarap)


    delta_positions_projected = []
    for time in time_to_treat:
        phase = 2 * np.pi / orbital_period
        mean_anomaly = phase * (time - time_periastron)

        true_anomaly = true_anomaly_from_mean_anomaly(mean_anomaly, eccentricity)
        delta_positions_source = np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0.0])

        delta_positions_projected.append([np.dot(delta_positions_source, North), np.dot(delta_positions_source, East)])

    deltas_positions = np.array(delta_positions_projected)
    return deltas_positions