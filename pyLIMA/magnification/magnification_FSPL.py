import numpy as np
import pkg_resources
from scipy import interpolate, misc

resource_path = '/'.join(('data', 'Yoo_B0B1.dat'))
template = pkg_resources.resource_filename('pyLIMA', resource_path)

try:

    yoo_table = np.loadtxt(template)

except ValueError:

    print('ERROR : No Yoo_B0B1.dat file found, please check!')

b0b1 = yoo_table
zz = b0b1[:, 0]
b0 = b0b1[:, 1]
b1 = b0b1[:, 2]

interpol_b0 = interpolate.interp1d(zz, b0, kind='linear')
interpol_b1 = interpolate.interp1d(zz, b1, kind='linear')

dB0 = misc.derivative(lambda x: interpol_b0(x), zz[1:-1], dx=10 ** -4, order=3)
dB1 = misc.derivative(lambda x: interpol_b1(x), zz[1:-1], dx=10 ** -4, order=3)
dB0 = np.append(2.0, dB0)
dB0 = np.concatenate([dB0, [dB0[-1]]])
dB1 = np.append((2.0 - 3 * np.pi / 4), dB1)
dB1 = np.concatenate([dB1, [dB1[-1]]])
interpol_db0 = interpolate.interp1d(zz, dB0, kind='linear')
interpol_db1 = interpolate.interp1d(zz, dB1, kind='linear')
YOO_TABLE = [zz, interpol_b0, interpol_b1, interpol_db0, interpol_db1]


def magnification_FSPL_Yoo(tau, uo, rho, gamma, return_impact_parameter=False):
    """
    The Yoo et al. Finite Source Point Lens magnification.
    "OGLE-2003-BLG-262: Finite-Source Effects from a Point-Mass Lens",Yoo, J. et al 2004
    http://adsabs.harvard.edu/abs/2004ApJ...603..139Y

    :param array_like tau: the tau define for example in
                               http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param array_like uo: the uo define for example in
                             http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param float rho: the normalised angular source star radius

    :param float gamma: the microlensing limb darkening coefficient.

    :param boolean return_impact_parameter: if the impact parameter is needed or not

    :return: the FSPL magnification A_FSPL(t)
    :rtype: array_like
    """

    import pyLIMA.magnification.impact_parameter

    impact_parameter = pyLIMA.magnification.impact_parameter.impact_parameter(tau,
                                                                              uo)  #
    # u(t)
    impact_parameter_square = impact_parameter ** 2  # u(t)^2

    magnification_pspl = (impact_parameter_square + 2) / (
            impact_parameter * (impact_parameter_square + 4) ** 0.5)

    z_yoo = impact_parameter / rho

    magnification_fspl = np.zeros(len(magnification_pspl))

    # Far from the lens (z_yoo>>1), then PSPL.
    indexes_PSPL = np.where((z_yoo > YOO_TABLE[0][-1]))[0]

    magnification_fspl[indexes_PSPL] = magnification_pspl[indexes_PSPL]

    # Very close to the lens (z_yoo<<1), then Witt&Mao limit.
    indexes_WM = np.where((z_yoo < YOO_TABLE[0][0]))[0]

    magnification_fspl[indexes_WM] = magnification_pspl[indexes_WM] * (
            2 * z_yoo[indexes_WM] - gamma *
            (2 - 3 * np.pi / 4) * z_yoo[indexes_WM])

    # FSPL regime (z_yoo~1), then Yoo et al derivatives
    indexes_FSPL = np.where((z_yoo <= YOO_TABLE[0][-1]) & (z_yoo >= YOO_TABLE[0][0]))[0]

    magnification_fspl[indexes_FSPL] = magnification_pspl[indexes_FSPL] * (
            YOO_TABLE[1](z_yoo[indexes_FSPL]) -
            gamma * YOO_TABLE[2](z_yoo[indexes_FSPL]))

    if return_impact_parameter:

        # return both
        return magnification_fspl, impact_parameter

    else:

        # return magnification
        return magnification_fspl


### Deprecated
def magnification_FSPL_Lee(tau, uo, rho, gamma):
    """
    The Lee et al. Finite Source Point Lens magnification.
    https://iopscience.iop.org/article/10.1088/0004-637X/695/1/200/pdf Leet et al.2009

    Much slower than Yoo et al. but valid for all rho, all u_o

    :param array_like tau: the tau define for example in
                               http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param array_like uo: the uo define for example in
                             http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param float rho: the normalised angular source star radius

    :param float gamma: the microlensing limb darkening coefficient.



    :return: the FSPL magnification A_FSPL(t)
    :rtype: array_like
    """
    # Using numba to speed up Lee et al. computation

    import numba
    from numba import cfunc, carray
    from numba.types import intc, CPointer, float64
    from scipy import LowLevelCallable

    def jit_integrand_function(integrand_function):
        jitted_function = numba.jit(integrand_function, nopython=True)

        @cfunc(float64(intc, CPointer(float64)))
        def wrapped(n, xx):
            values = carray(xx, n)
            return jitted_function(values)

        return LowLevelCallable(wrapped.ctypes)

    def Lee_limits(x, impact_parameter, source_radius):

        if x > np.arcsin(source_radius / impact_parameter):

            limit_1 = 0
            limit_2 = 0
            return [limit_1, limit_2]

        else:

            factor = (source_radius ** 2 - impact_parameter ** 2 * np.sin(
                x) ** 2) ** 0.5
            ucos = u * np.cos(x)
            if impact_parameter <= source_radius:

                limit_1 = 0
                limit_2 = ucos + factor
                return [limit_1, limit_2]

            else:

                limit_1 = ucos - factor
                limit_2 = ucos + factor
                return [limit_1, limit_2]

    def Lee_US(x, impact_parameter, source_radius, limb_darkening_coefficient=0):

        limits = Lee_limits(x, impact_parameter, source_radius)
        amp = limits[1] * (limits[1] ** 2 + 4) ** 0.5 - limits[0] * (
                limits[0] ** 2 + 4) ** 0.5

        return amp

    @jit_integrand_function
    def Lee_FS(args):
        x, phi, impact_parameter, source_radius, limb_darkening_coeff = args
        x2 = x ** 2
        u2 = impact_parameter ** 2

        factor = (1 - limb_darkening_coeff * (
                1 - 1.5 * (1 - (x2 - 2 * impact_parameter * x * np.cos(phi) + u2) /
                           source_radius ** 2) ** 0.5))
        if np.isnan(factor):
            factor = 0

        amp = (x2 + 2) / ((x2 + 4) ** 0.5)

        amp *= factor

        return amp

    import pyLIMA.magnification.impact_parameter
    from scipy import integrate
    impact_param = pyLIMA.magnification.impact_parameter.impact_parameter(tau,
                                                                          uo)  # u(t)
    impact_param_square = impact_param ** 2  # u(t)^2

    magnification_pspl = (impact_param_square + 2) / (
            impact_param * (impact_param_square + 4) ** 0.5)

    z_yoo = impact_param / rho

    magnification_fspl = np.zeros(len(magnification_pspl))

    # Far from the lens (z_yoo>>1), then PSPL.
    indexes_PSPL = np.where((z_yoo >= 10))[0]

    magnification_fspl[indexes_PSPL] = magnification_pspl[indexes_PSPL]

    # Close to the lens (z>3), USPL
    indexes_US = np.where((z_yoo > 3) & (z_yoo < 10))[0]

    ampli_US = []

    for idx, u in enumerate(impact_param[indexes_US]):
        ampli_US.append(1 / (np.pi * rho ** 2) *
                        integrate.quad(Lee_US, 0.0, np.pi, args=(u, rho), limit=100,
                                       epsabs=0.001,
                                       epsrel=0.001)[0])
    magnification_fspl[indexes_US] = ampli_US

    # Very Close to the lens (z<=3), FSPL
    indexes_FS = np.where((z_yoo <= 3))[0]

    ampli_FS = []

    for idx, u in enumerate(impact_param[indexes_FS]):
        ampli_FS.append(2 / (np.pi * rho ** 2) *
                        integrate.nquad(Lee_FS, [Lee_limits, [0.0, np.pi]],
                                        args=(u, rho, gamma),
                                        opts=[{'limit': 100, 'epsabs': 0.001,
                                               'epsrel': 0.001},
                                              {'limit': 100, 'epsabs': 0.001,
                                               'epsrel': 0.001}])[0])

    magnification_fspl[indexes_FS] = ampli_FS

    # return magnification
    return magnification_fspl
