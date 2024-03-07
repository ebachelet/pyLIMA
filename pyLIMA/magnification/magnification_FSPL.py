import numpy as np
from scipy import interpolate

from pyLIMA.data import PACKAGE_DATA

template = PACKAGE_DATA / "Yoo_B0B1.dat"

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
#dB0 = misc.derivative(lambda x: interpol_b0(x), zz[1:-1], dx=10 ** -4, order=3)
#dB1 = misc.derivative(lambda x: interpol_b1(x), zz[1:-1], dx=10 ** -4, order=3)
epsilon = 10**-6
dB0 = (interpol_b0(zz[1:-1]+epsilon)-interpol_b0(zz[1:-1]-epsilon))/epsilon/2
dB1 = (interpol_b1(zz[1:-1]+epsilon)-interpol_b1(zz[1:-1]-epsilon))/epsilon/2
dB0 = np.append(2.0, dB0)
dB0 = np.concatenate([dB0, [dB0[-1]]])
dB1 = np.append((2.0 - 3 * np.pi / 4), dB1)
dB1 = np.concatenate([dB1, [dB1[-1]]])
interpol_db0 = interpolate.interp1d(zz, dB0, kind='linear')
interpol_db1 = interpolate.interp1d(zz, dB1, kind='linear')
YOO_TABLE = [zz, interpol_b0, interpol_b1, interpol_db0, interpol_db1]


def magnification_FSPL_Yoo(tau, beta, rho, gamma, return_impact_parameter=False):
    """
    The Yoo et al. Finite Source Point Lens magnification.
    See  http://adsabs.harvard.edu/abs/2004ApJ...603..139Y

    Parameters
    ----------
    tau : array, (t-t0)/tE
    beta : array, [u0]*len(t)
    rho : float, the normalized angular source radius
    gamma : float, the linear microlensing limb darkening coefficient.
    return_impact_parameter : bool, if the impact parameter is needed or not

    Returns
    -------
    magnification_FSPL : array, A(t) for FSPL
    impact_parameter : array, u(t)
    """

    import pyLIMA.magnification.impact_parameter

    impact_parameter = pyLIMA.magnification.impact_parameter.impact_parameter(tau,
                                                                              beta)  #
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
