import numpy as np

### This gives some examples...

def log_t0(x):

    return np.log10(x.t0)

def t0(x):

    return 10 ** x.log_t0


def log_tE(x):

    return np.log10(x.tE)

def tE(x):

    return 10 ** x.log_tE


def log_rho(x):

    return np.log10(x.rho)

def rho(x):

    return 10 ** x.log_rho


def log_separation(x):

    return np.log10(x.separation)

def separation(x):

    return 10 ** x.log_separation


def log_mass_ration(x):

    return np.log10(x.mass_ratio)

def mass_ratio(x):

    return 10 ** x.log_mass_ratio




standard_fancy_parameters = {'log_tE': 'tE', 'log_rho': 'rho', 'log_separation': 'separation',
                             'log_mass_ratio': 'mass_ratio'}



