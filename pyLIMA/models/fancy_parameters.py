import numpy as np


### This gives some examples...

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


def log_mass_ratio(x):
    return np.log10(x.mass_ratio)


def mass_ratio(x):
    return 10 ** x.log_mass_ratio


def t0(x):
    try:

        alpha = x.alpha + np.pi

    except:

        alpha = 0

    rotation = np.array([np.cos(alpha), np.sin(alpha)])

    tau_prime = -np.dot(rotation, [x.x_center, x.y_center])

    t_0 = float(x.t_center + tau_prime * x.tE)
    # breakpoint()
    return t_0


def u0(x):
    try:

        alpha = x.alpha + np.pi

    except:

        alpha = 0

    rotation = np.array([-np.sin(alpha), np.cos(alpha)])

    u_prime = -np.dot(rotation, [x.x_center, x.y_center])

    u_0 = float(x.u_center - u_prime)
    # breakpoint()
    return u_0


def t_center(x):
    try:

        alpha = x.alpha + np.pi

    except:

        alpha = 0

    rotation = np.array([np.cos(alpha), np.sin(alpha)])

    tau_prime = np.dot(rotation, [x.x_center, x.y_center])

    t_center = float(x.t0 + tau_prime * x.tE)

    return t_center


def u_center(x):
    try:

        alpha = x.alpha + np.pi

    except:

        alpha = 0

    rotation = np.array([-np.sin(alpha), np.cos(alpha)])

    u_prime = np.dot(rotation, [x.x_center, x.y_center])

    u_center = float(x.u0 + u_prime)

    return u_center


standard_fancy_parameters = {'log_tE': 'tE', 'log_rho': 'rho',
                             'log_separation': 'separation',
                             'log_mass_ratio': 'mass_ratio'}
