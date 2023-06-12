import numpy as np
from scipy.optimize._numdiff import approx_derivative


def magnification_PSPL_Jacobian(pspl_model, telescope, pyLIMA_parameters):
    """
    The Jacobian of the PSPL magnification

    :param object pspl_model: a PSPL model object
    :param object telescope: a telescope object
    :param object pyLIMA_parameters: a dictionnary containing the microlensing
    parameters

    :return: the jacobian of the magnification, the magnification, the impact parameter
    :rtype: array_like, array_like, array_like
    """
    time = telescope.lightcurve_flux['time'].value

    # Derivative of A = (u^2+2)/(u(u^2+4)^0.5). Amplification[0] is A(t).
    # Amplification[1] is U(t).
    Amplification = pspl_model.model_magnification(telescope, pyLIMA_parameters,
                                                   return_impact_parameter=True)
    dAmplificationdU = (-8) / (
            Amplification[1] ** 2 * (Amplification[1] ** 2 + 4) ** 1.5)

    # Derivative of U = (uo^2+(t-to)^2/tE^2)^0.5
    dUdt0 = -(time - pyLIMA_parameters.t0) / \
            (pyLIMA_parameters.tE ** 2 * Amplification[1])
    dUdu0 = pyLIMA_parameters.u0 / Amplification[1]
    dUdtE = -(time - pyLIMA_parameters.t0) ** 2 / \
            (pyLIMA_parameters.tE ** 3 * Amplification[1])

    # Derivative of the model

    dAdt0 = dAmplificationdU * dUdt0
    dAdu0 = dAmplificationdU * dUdu0
    dAdtE = dAmplificationdU * dUdtE

    magnification_jacobian = np.array([dAdt0, dAdu0, dAdtE]).T

    return magnification_jacobian, Amplification


def magnification_FSPL_Jacobian(fspl_model, telescope, pyLIMA_parameters):
    """
    The Jacobian of the FSPL magnification

    :param object fspl_model: a FSPL model object
    :param object telescope: a telescope object
    :param object pyLIMA_parameters: a dictionnary containing the microlensing
    parameters

    :return: the jacobian of the magnification
    :rtype: array_like
    """

    from pyLIMA.models import PSPL_model
    from pyLIMA.magnification import magnification_FSPL

    yoo_table = magnification_FSPL.YOO_TABLE.copy()

    time = telescope.lightcurve_flux['time'].value

    fake_model = PSPL_model.PSPLmodel(fspl_model.event)
    # Derivative of A = Yoo et al (2004) method.
    Amplification_PSPL = fake_model.model_magnification(telescope, pyLIMA_parameters,
                                                        return_impact_parameter=True)

    dAmplification_PSPLdU = (-8) / (
            Amplification_PSPL[1] ** 2 * (Amplification_PSPL[1] ** 2 + 4) ** (1.5))

    # z_yoo=U/rho
    z_yoo = Amplification_PSPL[1] / pyLIMA_parameters.rho

    dAdu = np.zeros(len(Amplification_PSPL[0]))
    dAdrho = np.zeros(len(Amplification_PSPL[0]))

    # Far from the lens (z_yoo>>1), then PSPL.
    ind = np.where((z_yoo > yoo_table[0][-1]))[0]
    dAdu[ind] = dAmplification_PSPLdU[ind]
    dAdrho[ind] = -0.0

    # Very close to the lens (z_yoo<<1), then Witt&Mao limit.
    ind = np.where((z_yoo < yoo_table[0][0]))[0]
    dAdu[ind] = dAmplification_PSPLdU[ind] * (
            2 * z_yoo[ind] - telescope.ld_gamma * (2 - 3 * np.pi / 4) * z_yoo[ind])

    dAdrho[ind] = -Amplification_PSPL[0][ind] * Amplification_PSPL[1][
        ind] / pyLIMA_parameters.rho ** 2 * \
                  (2 - telescope.ld_gamma * (2 - 3 * np.pi / 4))

    # FSPL regime (z_yoo~1), then Yoo et al derivatives
    ind = np.where((z_yoo <= yoo_table[0][-1]) & (z_yoo >= yoo_table[0][0]))[0]

    dAdu[ind] = dAmplification_PSPLdU[ind] * (yoo_table[1](z_yoo[ind]) - \
                                              telescope.ld_gamma * yoo_table[2](
                z_yoo[ind])) + \
                Amplification_PSPL[0][ind] * \
                (yoo_table[3](z_yoo[ind]) - \
                 telescope.ld_gamma * yoo_table[4](
                            z_yoo[ind])) * 1 / pyLIMA_parameters.rho

    dAdrho[ind] = -Amplification_PSPL[0][ind] * Amplification_PSPL[1][
        ind] / pyLIMA_parameters.rho ** 2 * \
                  (yoo_table[3](z_yoo[ind]) - telescope.ld_gamma * yoo_table[4](
                      z_yoo[ind]))

    dUdt0 = -(time - pyLIMA_parameters.t0) / (
            pyLIMA_parameters.tE ** 2 * Amplification_PSPL[1])

    dUdu0 = pyLIMA_parameters.u0 / Amplification_PSPL[1]

    dUdtE = -(time - pyLIMA_parameters.t0) ** 2 / (
            pyLIMA_parameters.tE ** 3 * Amplification_PSPL[1])

    # Derivative of the model
    dAdt0 = dAdu * dUdt0
    dAdu0 = dAdu * dUdu0
    dAdtE = dAdu * dUdtE
    dAdrho = dAdrho

    magnification_jacobian = np.array([dAdt0, dAdu0, dAdtE, dAdrho]).T

    return magnification_jacobian


def magnification_numerical_Jacobian(microlensing_model, telescope, pyLIMA_parameters):
    """
    The Jacobian of the any models, based on scipy approx_fprime

    :param objectmicrolensing_model: a microlensing model object
    :param object telescope: a telescope object
    :param object pyLIMA_parameters: a dictionnary containing the microlensing
    parameters

    :return: the jacobian of the magnification,
    :rtype: array_like
    """

    x = [getattr(pyLIMA_parameters, key) for key in pyLIMA_parameters._fields if key
         not in microlensing_model.telescopes_fluxes_model_parameters({}).keys()]

    floors = np.zeros(len(x))
    magnification_jacobian_numerical = approx_derivative(model_magnification_numerical,
                                                         x, method='2-point', args=(
            microlensing_model, telescope, floors))

    return np.array(magnification_jacobian_numerical)


def model_magnification_numerical(parameters, microlensing_model, telescope, floors=0):
    # Trick here to stabilize t0 Jacobians

    params = parameters + floors
    pym = microlensing_model.compute_pyLIMA_parameters(params)

    magnification = microlensing_model.model_magnification(telescope, pym)

    return magnification
