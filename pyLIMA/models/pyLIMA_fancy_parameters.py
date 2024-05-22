import numpy as np


class StandardFancyParameters(object):

    def __init__(self, fancy_parameters = {'tE': 'log_tE', 'rho': 'log_rho',
                                         'separation': 'log_separation',
                                         'mass_ratio': 'log_mass_ratio'
                                           },
                        fancy_boundaries = {'log_tE':(0,3),'log_rho':(-5,-1.3),
                                             'log_separation':(-1,1),
                                             'log_mass_ratio':(-5,0)}):

        self.fancy_parameters = fancy_parameters

        self.fancy_boundaries = fancy_boundaries


    def tE(self, fancy_params):

        return 10**fancy_params['log_tE']

    def rho(self, fancy_params):

        return 10 **fancy_params['log_rho']

    def separation(self, fancy_params):

        return 10 ** fancy_params['log_separation']

    def mass_ratio(self, fancy_params):

        return 10 ** fancy_params['log_mass_ratio']


    def log_tE(self, standard_params):

        return np.log10(standard_params['tE'])

    def log_rho(self, standard_params):

        return np.log10(standard_params['rho'])

    def log_separation(self, standard_params):

        return np.log10(standard_params['separation'])

    def log_mass_ratio(self, standard_params):

        return np.log10(standard_params['mass_ratio'])


class StandardFancyParameters2(object):

    def __init__(self, fancy_parameters={'tE': 'tEcos', 'rho': 'log_rho',
                                         'separation': 'log_separation',
                                         'mass_ratio': 'log_mass_ratio',
                                         'alpha':'tEsin'},
                 fancy_boundaries={'tEcos': (-2, 2), 'log_rho': (-5, -1),
                                   'log_separation': (-1, 1),
                                   'log_mass_ratio': (-5, 0),
                                   'tEsin':(-2,2)}):
        self.fancy_parameters = fancy_parameters

        self.fancy_boundaries = fancy_boundaries

    def tE(self, fancy_params):

        tE = 1/(fancy_params['tEsin']**2+fancy_params['tEcos']**2)**0.5

        return tE

    def alpha(self, fancy_params):

        alph = np.arctan2(fancy_params['tEsin'], fancy_params['tEcos'])

        return alph
    def rho(self, fancy_params):
        return 10 ** fancy_params['log_rho']

    def separation(self, fancy_params):
        return 10 ** fancy_params['log_separation']

    def mass_ratio(self, fancy_params):
        return 10 ** fancy_params['log_mass_ratio']

def _t_center_to_t0(pyLIMA_parameters, x_center=0, y_center=0):
    #CROIN : https://iopscience.iop.org/article/10.1088/0004-637X/790/2/142/pdf

    try:

        alpha = pyLIMA_parameters['alpha']

    except KeyError:

        alpha = 0

    rotation = np.array([np.cos(alpha), np.sin(alpha)])

    tau_prime = -np.dot(rotation, [x_center, y_center])

    t_0 = float(pyLIMA_parameters['t_center'] - tau_prime * pyLIMA_parameters['tE'])

    return t_0
def _t0_to_t_center(pyLIMA_parameters, x_center=0, y_center=0):

    try:

        alpha = pyLIMA_parameters['alpha']

    except KeyError:

        alpha = 0

    rotation = np.array([np.cos(alpha), np.sin(alpha)])

    tau_prime = -np.dot(rotation, [x_center, y_center])

    t_center = float(pyLIMA_parameters['t0'] + tau_prime * pyLIMA_parameters['tE'])

    return t_center


def _u_center_to_u0(pyLIMA_parameters, x_center=0, y_center=0):
    try:

        alpha = pyLIMA_parameters['alpha']

    except KeyError:

        alpha = 0

    rotation = np.array([-np.sin(alpha), np.cos(alpha)])

    u_prime = np.dot(rotation, [x_center, y_center])

    u_0 = float(pyLIMA_parameters['u_center'] - u_prime)

    return u_0


def _u0_to_u_center(pyLIMA_parameters, x_center=0, y_center=0):
    try:

        alpha = pyLIMA_parameters['alpha']

    except KeyError:

        alpha = 0

    rotation = np.array([-np.sin(alpha), np.cos(alpha)])

    u_prime = np.dot(rotation, [x_center, y_center])

    u_center = float(pyLIMA_parameters['u0'] + u_prime)

    return u_center