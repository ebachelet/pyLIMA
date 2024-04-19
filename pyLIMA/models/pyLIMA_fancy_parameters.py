import numpy as np


class StandardFancyParameters(object):

    def __init__(self, fancy_parameters = {'tE': 'log_tE', 'rho': 'log_rho',
                                         'separation': 'log_separation',
                                         'mass_ratio': 'log_mass_ratio'
                                           },
                        fancy_boundaries = {'log_tE':(0,3),'log_rho':(-5,-1),
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
                 fancy_boundaries={'tEcos': (-1, 1), 'log_rho': (-5, -1),
                                   'log_separation': (-1, 1),
                                   'log_mass_ratio': (-5, 0),
                                   'tEsin':(-1,1)}):
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


