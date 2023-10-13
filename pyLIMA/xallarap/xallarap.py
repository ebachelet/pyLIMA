import numpy as np

def xallarap_shifts(xallarap_model, time, pyLIMA_parameters):

    if xallarap_model[0] == 'Circular':

        xi_angular_velocity = pyLIMA_parameters.xi_angular_velocity/pyLIMA_parameters.tE

        xi_phase = pyLIMA_parameters.xi_phase
        xi_inclination = pyLIMA_parameters.xi_inclination
        
        separation_1, separation_2 = circular_xallarap(time, xallarap_model[1],
                                                      xi_angular_velocity, xi_phase,
                                                      xi_inclination)


    return np.array([separation_1, separation_2])


def circular_xallarap(time, t0_xi, xi_angular_velocity, xi_phase,
                      xi_inclination):


    angular_velocity = xi_angular_velocity

    omega = angular_velocity*(time-t0_xi)+xi_phase

    separation_1 = np.cos(omega)#-np.cos(xi_phase)
    separation_2 = np.sin(xi_inclination)*(np.sin(omega))#-np.sin(xi_phase))

    return separation_1, separation_2


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
    delta_tau = np.dot(xiE, delta_positions)
    delta_beta = np.cross(xiE, delta_positions.T)

    return delta_tau, delta_beta
