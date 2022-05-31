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

        ds_dt = pyLIMA_parameters.v_para*10**pyLIMA_parameters.logs
        dseparation = orbital_motion_2D_separation_shift(time, orbital_motion_model[1], ds_dt)

        dalpha_dt = pyLIMA_parameters.v_perp
        dalpha = orbital_motion_2D_trajectory_shift(time, orbital_motion_model[1], dalpha_dt)

    if orbital_motion_model[0] == 'Circular':

        v_para = pyLIMA_parameters.v_para
        v_perp = pyLIMA_parameters.v_perp
        v_radial = pyLIMA_parameters.v_radial
        separation = 10 ** pyLIMA_parameters.logs

        dseparation, dalpha = orbital_motion_keplerian(time, orbital_motion_model[1], v_para, v_perp, v_radial, separation, rE=None, mass_lens=None, r_s=None, a_s=None)
        ds,da =  orbital_motion_circular(time, orbital_motion_model[1], v_para, v_perp, v_radial,separation)
        import pdb;
        pdb.set_trace()
    if orbital_motion_model[0] == 'Keplerian':

        v_para = pyLIMA_parameters.v_para
        v_perp = pyLIMA_parameters.v_perp
        v_radial = pyLIMA_parameters.v_radial
        separation = 10 ** pyLIMA_parameters.logs
        rE = pyLIMA_parameters.rE
        mass_lens = pyLIMA_parameters.mass_lens

        r_s = pyLIMA_parameters.r_s
        a_s = pyLIMA_parameters.a_s

        dseparation, dalpha = orbital_motion_keplerian(time, orbital_motion_model[1], v_para, v_perp, v_radial,
                                                       separation, rE, mass_lens=mass_lens, r_s=r_s, a_s=a_s)

        dalpha = -dalpha


    return dseparation, dalpha

def orbital_motion_keplerian(time, to_om, v_para, v_perp, v_radial, separation_0, rE=None, mass_lens=None,r_s=None,a_s=None):
    """" https: // arxiv.org / pdf / 2011.04780.pdf"""

    v_para /= 365.25
    v_perp /= 365.25
    v_radial /= 365.25

    if r_s is None:

        r_s = -v_para/v_radial

    if a_s is None:

        a_s = 1

    if (mass_lens is None):

        separation_z = r_s * separation_0
        r_0 = np.array([separation_0, 0, separation_z])
        v_0 = r_0[0] * np.array([v_para, v_perp, v_radial])

        r_norm = np.sum(r_0 ** 2) ** 0.5
        v_norm = np.sum(v_0 ** 2) ** 0.5

        a_true = a_s * r_norm

        h_0 = np.cross(r_0, v_0)
        h_0 /= np.sum(h_0 ** 2) ** 0.5



        orbital_velocity = np.sign(v_radial)*v_norm/r_norm

        inclination = np.arccos(h_0[2])
        longitude_ascending_node = np.arctan2(-h_0[0], h_0[1])
        #longitude_ascending_node =  np.arctan2(-h_0[1] / np.sin(inclination), h_0[0] / np.sin(inclination))
        #longitude_ascending_node = 0
        #N = np.cross([0,0,1],h_0)
        #N /= np.sum(N**2)**0.5

        #long = np.arccos(N[0])

        omega_peri = np.arcsin(r_0[2]/a_true)/2
        omega_peri2 = np.arccos(np.sqrt((v_0[2]/(a_true*orbital_velocity*np.sin(inclination))+1)/2))
        import pdb;
        pdb.set_trace()
        print(orbital_velocity, inclination, omega_peri)

        theta = orbital_velocity * (time - to_om)+omega_peri

        r_prime = a_true*np.c_[np.cos(theta), np.sin(theta), [0] * len(theta)]

       # R_inc = np.array([[np.cos(inclination), -np.sin(inclination), 0],
       #                   [np.sin(inclination), np.cos(inclination), 0],
       #                   [0, 0, 1]])
        R_inc = np.array([[1,0,0],
                          [0,np.cos(inclination), -np.sin(inclination)],
                          [0,np.sin(inclination), np.cos(inclination)]])
       #R_node = np.array([[1, 0, 0],
       #                       [0, np.cos(longitude_ascending_node), -np.sin(longitude_ascending_node)],
       #                       [0, np.sin(longitude_ascending_node), np.cos(longitude_ascending_node)]])

        R_node = np.array([[np.cos(longitude_ascending_node), -np.sin(longitude_ascending_node),0],
                           [np.sin(longitude_ascending_node), np.cos(longitude_ascending_node),0],
                           [0,0,1]])
        #R_node = np.array([[np.cos(longitude_ascending_node), 0, -np.sin(longitude_ascending_node)],
        #                   [0,1,0],
        #                   [np.sin(longitude_ascending_node), 0, np.cos(longitude_ascending_node)]])
        #R_peri = np.array([[1, 0, 0],
        #                         [0, np.cos(omega_peri), -np.sin(omega_peri)],
        #                         [0, np.sin(omega_peri), np.cos(omega_peri)]])

        R_peri = np.array([[np.cos(omega_peri), -np.sin(omega_peri), 0],
                           [np.sin(omega_peri), np.cos(omega_peri), 0],
                           [0, 0, 1]])

        Rmatrix = np.dot(R_node, np.dot(R_inc, R_peri))

        r_microlens = np.dot(Rmatrix, r_prime.T)

        import matplotlib.pyplot as plt

        plt.scatter(r_microlens[0],r_microlens[1])
        plt.xlim(-separation_0, separation_0)
        plt.ylim(-separation_0, separation_0)



        plt.show()

    else:

        separation_z = r_s * separation_0
        r_0 = np.array([separation_0, 0, separation_z]) * rE
        v_0 = r_0[0] * np.array([v_para, v_perp, v_radial])

        r_norm = np.sum(r_0 ** 2) ** 0.5
        v_norm = np.sum(v_0 ** 2) ** 0.5

        a_true = a_s * r_norm

        h_0 = np.cross(r_0, v_0)
        h_0 /= np.sum(h_0 ** 2) ** 0.5


        #eps = v_norm / 2 - 4 * np.pi ** 2 * mass_lens / r_norm

        #Gmass = -a_true * 2 * eps

        GMass = rE**3*separation_0**3*a_s*(1+r_s**2)**0.5/(2*a_s-1)*v_norm**2

        orbital_velocity = (GMass / a_true ** 3) ** 0.5

        e_0 = np.cross(v_0, h_0)/GMass-r_0/r_norm #Laplace_Runge_Lenz
        eccentricity = np.sum(e_0 ** 2) ** 0.5
        #eccentricity = (1-np.sum(h_0**2)**0.5/(Gmass*a_true))**0.5
        x_0 = e_0 / eccentricity
        z_0 = h_0/np.sum(h_0**2)**0.5
        y_0 = np.cross(z_0, x_0)

        cos_true_anomaly = np.dot(r_0/r_norm, x_0)

        cos_eccentric_anomaly = (cos_true_anomaly + eccentricity) / (1 + eccentricity * cos_true_anomaly)
        eccentric_anomaly = np.arccos(cos_eccentric_anomaly)

        if np.dot(r_0, y_0) > 0:

            pass

        else:

            eccentric_anomaly *= -1

        t_periastron = to_om - (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)) / orbital_velocity

        Rmatrix = np.c_[x_0, y_0, z_0]

        eccentric_anomaly = eccentric_anomaly_function(time, eccentricity, t_periastron, orbital_velocity)

        r_prime = np.array(
            [np.cos(eccentric_anomaly) - eccentricity, (1 - eccentricity ** 2) ** 0.5 * np.sin(eccentric_anomaly),
             0]) * a_true

        r_microlens = np.dot(Rmatrix, r_prime.T)/rE

    sep = (r_microlens[0] ** 2 + r_microlens[1] ** 2) ** 0.5
    angle = np.arctan2(r_microlens[1], r_microlens[0])

    separation0 = separation_0
    angle_0 = 0

    return sep - separation0, (angle - angle_0)


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


def eccentric_anomaly_function(time, ellipticity, t_periastron, speed):
    ks = pyasl.MarkleyKESolver()


    eccentricities = []
    for t in time:
        phase = speed * (t - t_periastron)
        phase = phase%(2*np.pi)
        ecc = ks.getE(phase, ellipticity)


        eccentricities.append(ecc)
    return eccentricities


def orbital_motion_circular(time, to_om, v_para, v_perp, v_radial, separation_0):
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
