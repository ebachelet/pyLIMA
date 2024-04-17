import numpy as np


class StandardFancyParameters(object):

    def __init__(self, fancy_parameters = {'tE': 'log_tE', 'rho': 'log_rho',
                                         'separation': 'log_separation',
                                         'mass_ratio': 'log_mass_ratio'},
                        fancy_boundaries = {'log_tE':(-1,3),'log_rho':(-5,-1),
                                             'log_separation':(-1,1),
                                             'log_mass_ratio':(-5,0)}):

        self.fancy_parameters = fancy_parameters
        self.fancy_boundaries = fancy_boundaries

    @property
    def _tE(self):

        return 10**self.log_tE

    @property
    def _rho(self):

        return 10 ** self.log_rho

    @property
    def _separation(self):

        return 10 ** self.log_separation

    @property
    def _mass_ratio(self):

        return 10 ** self.log_mass_ratio


    @property
    def _log_tE(self):

        return np.log10(self.tE)

    @property
    def _log_rho(self):
        return np.log10(self.rho)

    @property
    def _log_separation(self):

        return  np.log10(self.separation)

    @property
    def _log_mass_ratio(self):

        return  np.log10(self.mass_ratio)