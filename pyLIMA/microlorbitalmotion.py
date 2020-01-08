import numpy as np
from PyAstronomy import pyasl

def orbital_motion_shifts(orbital_motion_model, time, pyLIMA_parameters):
    """ Compute the trajectory curvature depending on the model.

    :param str orbital_motion_model: the orbital  motion model
    :param array_like time: the time array to compute the trajectory shift
    :param  object pyLIMA_parameters: the namedtuple containing the parameters

    :return: dseparation, dalpha the shifs in slens separation and angle
    :rtype: array_like, array_like
    """

    if orbital_motion_model[0] == '2D':
        ds_dt = pyLIMA_parameters.dsdt
        dseparation = orbital_motion_2D_separation_shift(orbital_motion_model[1], time, ds_dt)

        dalpha_dt = pyLIMA_parameters.dalphadt
        dalpha = orbital_motion_2D_trajectory_shift(orbital_motion_model[1], time, dalpha_dt)

    if orbital_motion_model[0] == 'Circular':
        v_para = pyLIMA_parameters.v_para
        v_perp = pyLIMA_parameters.v_perp
        v_radial = pyLIMA_parameters.v_radial
        separation = 10 ** pyLIMA_parameters.logs

        dseparation, dalpha = orbital_motion_circular(orbital_motion_model[1], v_para, v_perp, v_radial, separation,
                                                      time)

    if orbital_motion_model[0] == 'Keplerian':
        v_para = pyLIMA_parameters.v_para
        v_perp = pyLIMA_parameters.v_perp
        v_radial = pyLIMA_parameters.v_radial
        separation_0 = 10 ** pyLIMA_parameters.logs
        separation_z = 10 ** pyLIMA_parameters.logs_z
        mass_lens = pyLIMA_parameters.mass_lens
        rE = pyLIMA_parameters.rE

        dseparation, dalpha = orbital_motion_keplerian(orbital_motion_model[1], v_para, v_perp, v_radial, separation_0,
                                                       separation_z, mass_lens, rE, time)

        dalpha = -dalpha
        #mass_ratio =  10 ** pyLIMA_parameters.logq


        #center_of_mass = mass_ratio / (1 + mass_ratio) *(separation_0+dseparation)

        #x = (separation_0+dseparation)-center_of_mass*np.cos(dalpha)
        #y = -center_of_mass*np.sin(dalpha)



        #dalpha = np.arctan2(y,x)


    return dseparation, dalpha


def orbital_motion_2D_trajectory_shift(to_om, time, dalpha_dt):
    """ Compute the trajectory curvature induced by the orbital motion of the lens.

    :param float to_om: the reference time for the orbital motion
    :param array_like time: the time array to compute the trajectory shift
    :param float dalpha_dt: the angle change rate, in radian/day

    :return: dalpha, the angle shift
    :rtype: array_like
    """

    dalpha = dalpha_dt * (time - to_om)/365.25

    return dalpha


def orbital_motion_2D_separation_shift(to_om, time, ds_dt):
    """ Compute the binary separation change induced by the orbital motion of the lens.

    :param float to_om: the reference time for the orbital motion
    :param array_like time: the time array to compute the trajectory shift
    :param float ds_dt: the binary separation change rate, in einstein_ring_unit/day

    :return: dseparation, the binary separation shift
    :rtype: array_like
    """
    dseparation = ds_dt * (time - to_om)/365.25

    return dseparation


def orbital_motion_circular(to_om, v_para, v_perp, v_radial, separation_0, time):
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

    eps0 = (np.cos(phi0) ** 2 + np.sin(inclination) ** 2 * np.sin(phi0) ** 2) ** 0.5
    a_true = separation_0 / eps0

    s_0 = a_true * np.array([np.cos(phi0), np.sin(inclination) * np.sin(phi0)])
    alpha_0 = np.arctan2(s_0[1], s_0[0])

    phi = omega * (time - to_om) + phi0
    separation = a_true * (np.cos(phi) ** 2 + np.sin(inclination) ** 2 * np.sin(phi) ** 2) ** 0.5

    s_t = a_true * np.array([np.cos(phi), np.sin(inclination) * np.sin(phi)])
    alpha = np.arctan2(s_t[1], s_t[0])

    return separation - separation_0, alpha - alpha_0


def orbital_motion_keplerian(to_om, v_para, v_perp, v_radial, separation_0, separation_z, mass, rE, time):
    """ Compute the binary separation change induced by the orbital motion of the lens :
    "Binary Microlensing Event OGLE-2009-BLG-020 Gives Verifiable Mass, Distance, and Orbit Predictions",Skowron et al. 2011
    http://adsabs.harvard.edu/abs/2011ApJ...738...87S

    :param float to_om: the reference time for the orbital motion
    :param float v_para: the normalised binary separation change rate, 1/s ds/dt in 1/yr
    :param float v_perp: the binary angle  change rate, dalpha/dt in 1/yr
    :param float v_radial: the normalised radial separation change rate, 1/s ds_r/dt in 1/yr
    :param float separation_0: the binary separation at to_om
    :param float separation_z: the binary separation parallel to line of sight at to_om

    :param array_like time: the time array to compute the trajectory shift


    :return: dseparation, dalpha the binary separation and angle shifts
    :rtype: array_like, array_like
    """

    r_0 = np.array([separation_0, 0, separation_z]) * rE
    v_0 = r_0[0] *np.array([v_para, v_perp, v_radial])

    # speed = (r_0[0]**2*(np.sum(v_0**2)))**0.5/a_true

    eps = np.sum(( v_0) ** 2) / 2 - 4 * np.pi ** 2 * mass / np.sum(r_0 ** 2) ** 0.5
    a_true = -(4 * np.pi ** 2 * mass) / (2 * eps)
    period = (a_true ** 3 / mass) ** 0.5

    N = 2 * np.pi / period * 1 / 365.25
    h_0 = np.cross(r_0, v_0)

    z_0 = h_0 / (np.sum(h_0 ** 2)) ** 0.5

    e_0 = np.cross( v_0, h_0) / (4 * np.pi ** 2 * mass) - r_0 / np.sum(r_0 ** 2) ** 0.5
    ellipticity = np.sum(e_0 ** 2) ** 0.5

    x_0 = e_0 / ellipticity
    y_0 = np.cross(z_0, x_0)

    cos_true_anomaly = np.dot(x_0, r_0) / np.sum(r_0 ** 2) ** 0.5
    sin_true_anomaly = np.dot(y_0, r_0) / np.sum(r_0 ** 2) ** 0.5
    true_anomaly = np.arctan2(sin_true_anomaly, cos_true_anomaly)

    cos_eccentric_anomaly = (cos_true_anomaly + ellipticity) / (1 + ellipticity * cos_true_anomaly)
    eccentric_anomaly = np.arccos(cos_eccentric_anomaly)
    if (true_anomaly > 0) & (true_anomaly < np.pi):
        pass
    else:
        eccentric_anomaly *= -1

    t_periastron = to_om - (eccentric_anomaly - ellipticity * np.sin(eccentric_anomaly)) / N

    Rmatrix = np.c_[x_0, y_0, z_0]

    #inclination = np.arccos(Rmatrix[2, 2])

    #if np.abs(inclination) < 10 ** -5:

    #    omega = 0
    #    phi0 = np.arctan2(Rmatrix[1, 0], Rmatrix[0, 0])

    #else:

    #    cos_phi = -Rmatrix[1, 2]/np.sin(inclination)
    #    sin_phi = Rmatrix[0, 2]/np.sin(inclination)

    #    phi0 = np.arctan2(sin_phi,cos_phi)

    #    cos_omega = Rmatrix[0,0]*cos_phi+Rmatrix[1,0]*sin_phi
    #    sin_omega = (Rmatrix[1,0]*cos_phi-Rmatrix[0,0]*sin_phi)*np.cos(inclination)+Rmatrix[2,0]*np.sin(inclination)


    #    omega = np.arctan2(sin_omega, cos_omega)

    #R_phi = np.array([[1, 0, 0], [0, np.cos(phi0), -np.sin(phi0)], [0, np.sin(phi0), np.cos(phi0)]])

    #R_omega = np.array([[1, 0, 0], [0, np.cos(omega), -np.sin(omega)], [0, np.sin(omega), np.cos(omega)]])
    #R_inc = np.array(
    #    [[np.cos(inclination), -np.sin(inclination), 0], [np.sin(inclination), np.cos(inclination), 0], [0, 0, 1]])

    #R_invariant = np.dot(R_phi, np.dot(R_inc, R_omega))


    eccentric_anomaly = eccentric_anomaly_function(time, ellipticity, t_periastron, N)

    r_prime = np.array(
        [np.cos(eccentric_anomaly) - ellipticity, (1 - ellipticity ** 2) ** 0.5 * np.sin(eccentric_anomaly),
         0]) * a_true
    r_microlens = np.dot(Rmatrix, r_prime)/rE
    separation = (r_microlens[0] ** 2 + r_microlens[1] ** 2) ** 0.5
    angle = np.arctan2(r_microlens[1], r_microlens[0])


    eccentric_anomaly_0 = eccentric_anomaly_function([to_om], ellipticity, t_periastron, N)
    r_prime_0 = np.array([np.cos(eccentric_anomaly_0) - ellipticity, (1 - ellipticity ** 2) ** 0.5 * np.sin(eccentric_anomaly_0),0]) * a_true
    r_microlens_0 = np.dot(Rmatrix, r_prime_0)/rE
    separation0 = (r_microlens_0[0] ** 2 + r_microlens_0[1] ** 2) ** 0.5
    angle_0 = np.arctan2(r_microlens_0[1], r_microlens_0[0])


    return separation - separation0, (angle-angle_0)



def orbital_motion_keplerian_direct(to_om, a_true,period,eccentricity,inclination,omega_node,omega_periastron,t_periastron,
                                  rE, time):
    """ Compute the binary separation change induced by the orbital motion of the lens :
    "Binary Microlensing Event OGLE-2009-BLG-020 Gives Verifiable Mass, Distance, and Orbit Predictions",Skowron et al. 2011
    http://adsabs.harvard.edu/abs/2011ApJ...738...87S

    :param float to_om: the reference time for the orbital motion
    :param float v_para: the normalised binary separation change rate, 1/s ds/dt in 1/yr
    :param float v_perp: the binary angle  change rate, dalpha/dt in 1/yr
    :param float v_radial: the normalised radial separation change rate, 1/s ds_r/dt in 1/yr
    :param float separation_0: the binary separation at to_om
    :param float separation_z: the binary separation parallel to line of sight at to_om

    :param array_like time: the time array to compute the trajectory shift


    :return: dseparation, dalpha the binary separation and angle shifts
    :rtype: array_like, array_like
    """


    R_phi = np.array([[1, 0, 0], [0, np.cos(omega_node), -np.sin(omega_node)], [0, np.sin(omega_node), np.cos(omega_node)]])

    R_omega = np.array([[1, 0, 0], [0, np.cos(omega_periastron), -np.sin(omega_periastron)], [0, np.sin(omega_periastron), np.cos(omega_periastron)]])
    R_inc = np.array(
        [[np.cos(inclination), -np.sin(inclination), 0], [np.sin(inclination), np.cos(inclination), 0], [0, 0, 1]])

    R_invariant = np.dot(R_phi, np.dot(R_inc, R_omega))
    N = 2*np.pi/period/365.25

    eccentric_anomaly = eccentric_anomaly_function(time, eccentricity, t_periastron, N)
    eccentric_anomaly_to_om = eccentric_anomaly_function(to_om, eccentricity, t_periastron, N)

    r_prime = np.array(
        [np.cos(eccentric_anomaly) - ellipticity, (1 - ellipticity ** 2) ** 0.5 * np.sin(eccentric_anomaly),
         0]) * a_true
    r_microlens = np.dot(R_invariant, r_prime)

    r_prime_to_om = np.array(
        [np.cos(eccentric_anomaly_to_om) - ellipticity, (1 - ellipticity ** 2) ** 0.5 * np.sin(eccentric_anomaly_to_om),
         0]) * a_true
    r_microlens_to_om = np.dot(R_invariant, r_prime_to_om)


    r_microlens /= rE
    r_microlens_to_om /= rE

    separation = (r_microlens[0] ** 2 + r_microlens[1] ** 2) ** 0.5
    angle = np.arctan2(r_microlens[1], r_microlens[0])
    angle_0 =  np.arctan2(r_microlens_to_om[1], r_microlens_om[0])

    import pdb;
    pdb.set_trace()
    return separation - separation_0, (angle-angle_0)




def eccentric_anomaly_function(time, ellipticity, t_periastron, speed):
    ks = pyasl.MarkleyKESolver()

    eccentricities = []
    for t in time:
        phase = speed * (t - t_periastron)
        phase = phase%(2*np.pi)
        ecc = ks.getE(phase, ellipticity)


        eccentricities.append(ecc)
    return eccentricities


def mean_to_eccentric_anomaly(params, mean_anomaly, ellipticity):
    if np.abs(params[0])/np.pi>2:
        return np.inf
    return (mean_anomaly - params[0] + ellipticity * np.sin(params[0])) ** 2


def mean_to_eccentric_anomaly_prime(params, mean_anomaly, ellipticity):
    return 2 * (mean_anomaly - params[0] + ellipticity * np.sin(params[0])) * (-1 + ellipticity * np.cos(params[0]))

def mean_to_eccentric_anomaly_prime2(params, mean_anomaly, ellipticity):
    return 2 * ((-1 + ellipticity * np.cos(params[0]))**2+
                (mean_anomaly - params[0] + ellipticity * np.sin(params[0])) * (- ellipticity * np.sin(params[0])))
