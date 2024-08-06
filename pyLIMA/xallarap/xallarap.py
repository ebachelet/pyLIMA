import numpy as np

def xallarap_shifts(xallarap_model, time, pyLIMA_parameters, body='primary'):

    if xallarap_model[0] == 'Static':

        #if body != 'primary':

        #    separation_1 = pyLIMA_parameters.delta_t0/pyLIMA_parameters.tE
        #    separation_2 = pyLIMA_parameters.delta_u0

        #else:

        #    separation_1 = 0
        #    separation_2 = 0

        separation_1_1 = 0
        separation_2_1 = 0

        separation_1_2 = -pyLIMA_parameters['delta_t0']/pyLIMA_parameters['tE']
        separation_2_2 = pyLIMA_parameters['delta_u0']

    if xallarap_model[0] == 'Circular':

        xi_angular_velocity = pyLIMA_parameters['xi_angular_velocity']
        xi_phase = pyLIMA_parameters['xi_phase']
        xi_inclination = pyLIMA_parameters['xi_inclination']

        mass_1 = pyLIMA_parameters['xi_mass_ratio']/(1+pyLIMA_parameters['xi_mass_ratio'])
        origin_1 = np.cos(xi_phase)*mass_1
        origin_2 = np.sin(xi_phase)*np.sin(xi_inclination)*mass_1

        xallarap_delta_positions = circular_xallarap(time, xallarap_model[1],
                                                      xi_angular_velocity, xi_phase,
                                                      xi_inclination)

        xallarap_delta_positions[0] *= mass_1
        xallarap_delta_positions[1] *= mass_1

        separation_1_1 = xallarap_delta_positions[0]-origin_1
        separation_2_1 = xallarap_delta_positions[1]-origin_2

        separation_1_2 = -xallarap_delta_positions[0]*1/pyLIMA_parameters['xi_mass_ratio']-origin_1
        separation_2_2 = -xallarap_delta_positions[1]*1/pyLIMA_parameters['xi_mass_ratio']-origin_2

    return separation_1_1, separation_2_1, separation_1_2, separation_2_2


def circular_xallarap(time, t0_xi, xi_angular_velocity, xi_phase,
                      xi_inclination):


    angular_velocity = xi_angular_velocity

    omega = angular_velocity*(time-t0_xi)+xi_phase

    separation_1 = np.cos(omega)#-np.cos(xi_phase)
    separation_2 = np.sin(xi_inclination)*(np.sin(omega))#np.sin(xi_phase))

    return np.array([separation_1, separation_2])


def compute_xallarap_curvature(xiE, delta_positions):
    """
    Compute the curvature induce by the parallax of from deltas_positions of a
    telescope.
    See https://ui.adsabs.harvard.edu/abs/2004ApJ...606..319G/abstract


    Parameters
    ----------
    piE : array, [piEN,piEE] the parallax vector
    delta_positions : array, [d_N,d_E] the projected positions of the telescope

    Returns
    -------
    delta_tau : array, the x shift induced by the parallax
    delta_beta : array, the y shift induced by the parallax
    """
    #delta_tau = np.dot(xiE, delta_positions)
    #delta_beta = np.cross(xiE, delta_positions.T)

    delta_tau = xiE[0] * delta_positions[0] + xiE[1] * delta_positions[1]
    delta_beta = xiE[0] * delta_positions[1] - xiE[1] * delta_positions[0]

    return delta_tau, delta_beta
