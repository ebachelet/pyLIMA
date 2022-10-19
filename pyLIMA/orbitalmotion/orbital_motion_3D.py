import numpy as np
from PyAstronomy import pyasl

def orbital_motion_keplerian(time, to_om,  separation_0, v_para, v_perp, v_radial, r_s=None, a_s=None, rE=None):
    """" https: // arxiv.org / pdf / 2011.04780.pdf"""

    v_para /= 365.25
    v_perp /= 365.25
    v_radial /= 365.25

    if (rE is None): #Circular

        try:

            r_s = -v_para/v_radial

        except:

            #v_radial = v_para = 0

            r_s = 0

        a_s = 1

        longitude_ascending_node, inclination, omega_peri, a_true, orbital_velocity = \
            orbital_parameters_from_position_and_velocities(separation_0,r_s,a_s, v_para, v_perp, v_radial)

        theta = orbital_velocity * (time - to_om)+omega_peri

        r_prime = a_true*np.c_[np.cos(theta), np.sin(theta), [0] * len(theta)]

        R_inc = np.array([[1,0,0],
                          [0,np.cos(inclination), -np.sin(inclination)],
                          [0,np.sin(inclination), np.cos(inclination)]])

        R_node = np.array([[np.cos(longitude_ascending_node), -np.sin(longitude_ascending_node), 0],
                           [np.sin(longitude_ascending_node), np.cos(longitude_ascending_node), 0],
                           [0, 0, 1]])

        R_peri = np.array([[np.cos(omega_peri), -np.sin(omega_peri), 0],
                           [np.sin(omega_peri), np.cos(omega_peri), 0],
                           [0, 0, 1]])

        Rmatrix = np.dot(R_node, np.dot(R_inc, R_peri))

        r_microlens = np.dot(Rmatrix, r_prime.T)

    else: #Keplerian

        separation_z = r_s * separation_0
        r_0 = np.array([separation_0, 0, separation_z]) * rE
        v_0 = r_0[0] * np.array([v_para, v_perp, v_radial])

        r_norm = np.sum(r_0 ** 2) ** 0.5
        v_norm = np.sum(v_0 ** 2) ** 0.5

        a_true = a_s * r_norm

        h_0 = np.cross(r_0, v_0)

        #GMass = rE**3*separation_0**2*a_s*(1+r_s**2)**0.5/(2*a_s-1)*v_norm**2
        GMass = v_norm**2/2*(-1.0/2/a_true+1/r_norm)**(-1)

        if GMass<0:

            #print('Parabolic trajectory....')
            return  np.array([separation_0]*len(time)), np.array([0]*len(time))

        orbital_velocity = (GMass / a_true ** 3) ** 0.5

        e_0 = np.cross(v_0, h_0)/GMass-r_0/r_norm #Laplace_Runge_Lenz
        eccentricity = np.sum(e_0 ** 2) ** 0.5
        h_0 /= np.sum(h_0 ** 2) ** 0.5

        #eccentricity = (1-np.sum(h_0**2)**0.5/(Gmass*a_true))**0.5
        x_0 = e_0 / eccentricity
        z_0 = h_0/np.sum(h_0**2)**0.5
        y_0 = np.cross(z_0, x_0)

        cos_true_anomaly = np.dot(r_0/r_norm, x_0)

        if cos_true_anomaly < -1:

            cos_true_anomaly = -1

        if cos_true_anomaly > 1:

            cos_true_anomaly = 1

        cos_eccentric_anomaly = (cos_true_anomaly + eccentricity) / (1 + eccentricity * cos_true_anomaly)
        eccentric_anomaly = np.arccos(cos_eccentric_anomaly)

        if np.dot(r_0, y_0) > 0:

            pass

        else:

            eccentric_anomaly *= -1

        t_periastron = to_om - (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)) / orbital_velocity

        Rmatrix = np.c_[x_0, y_0, z_0]

        eccentric_anomaly = eccentric_anomaly_function(time, eccentricity, t_periastron, orbital_velocity)
        r_prime = np.array( [np.cos(eccentric_anomaly) - eccentricity, (1 - eccentricity ** 2) ** 0.5 *
                             np.sin(eccentric_anomaly), [0] * len(eccentric_anomaly)]) * a_true

        r_microlens = np.dot(Rmatrix, r_prime)/rE

    sep = (r_microlens[0] ** 2 + r_microlens[1] ** 2) ** 0.5
    angle = np.arctan2(r_microlens[1], r_microlens[0])

    separation0 = separation_0
    angle_0 = 0

    return sep - separation0, (angle - angle_0)


def eccentric_anomaly_function(time, ellipticity, t_periastron, speed):

    ks = pyasl.MarkleyKESolver()

    eccentricities = []
    for t in time:

        phase = speed * (t - t_periastron)
        phase = phase%(2*np.pi)
        ecc = ks.getE(phase, ellipticity)

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


    return longitude_ascending_node, inclination, omega_peri, a_true, orbital_velocity

