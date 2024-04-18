import numpy as np


class StandardFancyParameters(object):

    def __init__(self, fancy_parameters = {'tE': 'log_tE', 'rho': 'log_rho',
                                         'separation': 'log_separation',
                                         'mass_ratio': 'log_mass_ratio'},
                        fancy_boundaries = {'log_tE':(-3,1),'log_rho':(-5,-1),
                                             'log_separation':(-1,1),
                                             'log_mass_ratio':(-5,0)}):

        self.fancy_parameters = fancy_parameters

        self.fancy_boundaries = fancy_boundaries

    def tE(self, fancy_params):

        return 10**-fancy_params.log_tE

    def rho(self, fancy_params):

        return 10 **fancy_params.log_rho

    def separation(self, fancy_params):

        return 10 ** fancy_params.log_separation

    def mass_ratio(self, fancy_params):

        return 10 ** fancy_params.log_mass_ratio


    def log_tE(self, standard_params):

        return np.log10(1/standard_params.tE)

    def log_rho(self, standard_params):

        return np.log10(standard_params.rho)

    def log_separation(self, standard_params):

        return np.log10(standard_params.separation)

    def log_mass_ratio(self, standard_params):

        return np.log10(standard_params.mass_ratio)