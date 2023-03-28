import numpy as np
from PyAstronomy import pyasl

def orbital_motion_keplerian(time, to_om,  separation_0, v_para, v_perp, v_radial, r_s=None, a_s=None, rE=None):
    """" https: // arxiv.org / pdf / 2011.04780.pdf"""

    longitude_ascending_node, inclination, omega_peri, a_true, orbital_velocity, eccentricity, true_anomaly, x, y, z = \
        orbital_parameters_from_position_and_velocities_new(separation_0, r_s, a_s, v_para, v_perp, v_radial)

    Rmatrix = np.c_[x[:2], y[:2]]

    if (a_s ==1) & ( r_s == -v_para / v_radial) | (r_s == 0): #Circular

        theta = orbital_velocity * (time - to_om)/365.25

        r_prime = a_true*np.r_[np.cos(theta), np.sin(theta)]

        r_microlens = np.dot(Rmatrix, r_prime)

    else: #Keplerian

        eccentric_anomaly = np.sign(np.sin(true_anomaly))*(np.cos(true_anomaly)+eccentricity)/(1+eccentricity*np.cos(true_anomaly))


        t_periastron = to_om - (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)) / orbital_velocity*365.25



        eccentric_anomaly = eccentric_anomaly_function(time, eccentricity, t_periastron, orbital_velocity/365.25)

        r_prime = np.array([np.cos(eccentric_anomaly) - eccentricity, (1 - eccentricity ** 2) ** 0.5 *
                             np.sin(eccentric_anomaly)]) * a_true

        r_microlens = np.dot(Rmatrix, r_prime)#/rE

    sep = (r_microlens[0] ** 2 + r_microlens[1] ** 2) ** 0.5
    angle = np.arctan2(r_microlens[1], r_microlens[0])

    separation0 = separation_0
    angle_0 = 0

    return sep - separation0, (angle - angle_0)


def eccentric_anomaly_function(time, ellipticity, t_periastron, speed):

    #ks = pyasl.MarkleyKESolver()
    import kepler
    #import time as TIME


    eccentricities = []
    for t in time:

        phase = speed * (t - t_periastron)
        phase = phase%(2*np.pi)
        #start = TIME.time()
        #ecc2 = ks.getE(phase, ellipticity)
        #end1 = TIME.time()
        ecc = kepler.solve(phase, ellipticity)
        #end2 = TIME.time()
        #print(end1-start,end2-end1,ecc2-ecc)
        #import pdb;
        #pdb.set_trace()

        eccentricities.append(ecc)

    return eccentricities

def orbital_parameters_from_position_and_velocities(separation_0, r_s, a_s, v_para, v_perp, v_radial):
    """ Return Euler angles, semi-major axis and orbital velocity"""

    separation_z = r_s * separation_0
    r_0 = np.array([separation_0, 0, separation_z])
    v_0 = r_0[0] * np.array([v_para, v_perp, v_radial])

    r_norm = np.sum(r_0 ** 2) ** 0.5
    a_true = a_s * r_norm

    h_0 = np.cross(r_0, v_0)
    GMass = np.sum(v_0 ** 2) / (-2 / (2 * a_true) + 2 / r_norm)

    e_0 = np.cross(r_0, v_0) /GMass - r_0 / r_norm
    eccentricity = np.sum(e_0 ** 2) ** 0.5
    h_0 /= np.sum(h_0 ** 2) ** 0.5

    longitude_ascending_node = np.arctan2(-h_0[0], h_0[1])

    if (longitude_ascending_node < -np.pi / 2) | (longitude_ascending_node > np.pi / 2):
        longitude_ascending_node += np.pi

    if longitude_ascending_node == 0:

        inclination = np.arctan(-h_0[1] / h_0[2])

    else:

        inclination = np.arctan(h_0[0] / np.sin(longitude_ascending_node) / h_0[2])

    if inclination == 0:

        omega_peri = 0
        orbital_velocity = v_perp

    else:

        omega_peri = np.arcsin(r_0[2] / (a_true * np.sin(inclination))) / 2
        orbital_velocity = v_0[2] / a_true / np.sin(inclination) / np.cos(2 * omega_peri)

    return longitude_ascending_node, inclination, omega_peri, a_true, orbital_velocity, eccentricity,0


def orbital_parameters_from_position_and_velocities_new(separation_0, r_s, a_s, v_para, v_perp, v_radial):
    """ Return Euler angles, semi-major axis and orbital velocity"""

    e_0, h_0, r_0, v_0, r_norm, separation_z, a_true, GMass, orbital_velocity = state_orbital_elements(separation_0, r_s, a_s, v_para, v_perp, v_radial)

    # From Skowron2011, Bozza2020
    # and https://orbital-mechanics.space/classical-orbital-elements/orbital-elements-and-the-state-vector.html

    eccentricity = np.sum(e_0**2)**0.5 #
    h_norm = np.sum(h_0**2)**0.5
    z_0 = h_0 / h_norm

    inclination = np.arccos(z_0[2])

    N = np.cross([0, 0, 1], h_0)
    longitude_ascending_node = np.arctan2(N[1], N[0])

    if eccentricity < 10**-10:

        x_0 = [np.cos(longitude_ascending_node), np.sin(longitude_ascending_node), 0]
        y_0 = [-np.sin(longitude_ascending_node)*z_0[2], z_0[2]*np.cos(longitude_ascending_node), np.sin(inclination)]
        z_0 = [np.sin(inclination)*np.cos(longitude_ascending_node),
               -np.sin(inclination)*np.sin(longitude_ascending_node), z_0[2]]

        omega_peri = 0
        true_anomaly = 0

    else:

        x_0 = e_0 / eccentricity
        z_0 = h_0/h_norm
        y_0 = np.cross(z_0,x_0)
        r_0_norm = r_0/r_norm

        omega_peri = np.arctan2(x_0[2], y_0[2])
        cos_true_anomaly = np.dot(r_0_norm, x_0)
        sin_true_anomaly = np.dot(r_0_norm, y_0)

        true_anomaly = np.arctan2(sin_true_anomaly, cos_true_anomaly)

    return longitude_ascending_node, inclination, omega_peri, a_true, orbital_velocity, eccentricity, true_anomaly,x_0,y_0,z_0



def state_orbital_elements(separation_0, r_s, a_s, v_para, v_perp, v_radial):

    separation_z = r_s * separation_0
    r_0 = np.array([separation_0, 0, separation_z])
    v_0 = r_0[0] * np.array([v_para, v_perp, v_radial])

    r_norm = np.sum(r_0 ** 2) ** 0.5
    a_true = a_s * r_norm

    h_0 = np.cross(r_0, v_0)
    GMass = np.sum(v_0 ** 2) / (-2 / (2 * a_true) + 2 / r_norm)

    e_0 = np.cross(v_0, h_0) / GMass - r_0 / r_norm

    orbital_velocity = (GMass / a_true ** 3) ** 0.5

    return e_0, h_0, r_0, v_0, r_norm, separation_z, a_true, GMass, orbital_velocity
