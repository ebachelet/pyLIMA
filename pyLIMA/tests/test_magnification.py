import numpy as np


def test_impact_parameter():
    from pyLIMA.magnification import impact_parameter

    tau = np.array([1])
    uo = np.array([2])

    impact_param = impact_parameter.impact_parameter(tau, uo)
    assert impact_param == np.sqrt(5)


def test_magnification_FSPL_Yoo():
    from pyLIMA.magnification import magnification_FSPL

    tau = np.array([0.001])
    uo = np.array([0] * len(tau))
    rho = 0.01
    gamma = 0.5

    magnification = magnification_FSPL.magnification_FSPL_Yoo(tau, uo, rho, gamma)

    assert np.allclose(magnification, np.array([216.97028636]))


def test_magnification_PSPL_Jacobian():
    from pyLIMA.magnification import magnification_Jacobian
    import pyLIMA.telescopes
    from pyLIMA.models import PSPLmodel
    from pyLIMA import event

    ev = event.Event()
    lightcurve = np.array([[2456789, 12.8, 0.01]])

    telo = pyLIMA.telescopes.Telescope(name='fake', camera_filter='I',
                                       lightcurve=lightcurve,
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'])

    ev.telescopes.append(telo)

    pspl = PSPLmodel(ev)
    params = [2456789.2, 0.1, 34.5]

    pym = pspl.compute_pyLIMA_parameters(params)

    jacobian, magnification = magnification_Jacobian.magnification_PSPL_Jacobian(pspl,
                                                                                 telo,
                                                                                 pym)

    assert np.allclose(jacobian, [-1.66561332e-01, -9.91248125e+01, 9.65572939e-04])
    assert np.allclose(magnification, 10.02076281)


def test_magnification_FSPL_Jacobian():
    from pyLIMA.magnification import magnification_Jacobian
    import pyLIMA.telescopes
    from pyLIMA.models import FSPLmodel
    from pyLIMA import event

    ev = event.Event()
    lightcurve = np.array([[2456789, 12.8, 0.01]])

    telo = pyLIMA.telescopes.Telescope(name='fake', camera_filter='I',
                                       lightcurve=lightcurve,
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'])

    ev.telescopes.append(telo)

    pspl = FSPLmodel(ev)
    params = [2456789.2, 0.1, 34.5, 0.028]
    pym = pspl.compute_pyLIMA_parameters(params)

    jacobian = magnification_Jacobian.magnification_FSPL_Jacobian(pspl, telo, pym)

    assert np.allclose(jacobian, [-1.71720554e-01, -1.02195194e+02, 9.95481471e-04,
                                  7.42711447e+00])


def test_magnification_numerical_Jacobian():
    from pyLIMA.magnification import magnification_Jacobian
    import pyLIMA.telescopes
    from pyLIMA.models import FSPLmodel
    from pyLIMA import event

    ev = event.Event()
    lightcurve = np.array([[2456789, 12.8, 0.01]])

    telo = pyLIMA.telescopes.Telescope(name='fake', camera_filter='I',
                                       lightcurve=lightcurve,
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'])

    ev.telescopes.append(telo)

    pspl = FSPLmodel(ev)
    params = [2456789.2, 0.1, 34.5, 0.028]
    pym = pspl.compute_pyLIMA_parameters(params)

    jacobian = magnification_Jacobian.magnification_numerical_Jacobian(pspl, telo, pym)

    assert np.allclose(jacobian, [-1.87252646e-01, -1.02197457e+02, 9.95504683e-04,
                                  7.43527365e+00])  # not as precise...


def test_magnification_PSPL():
    from pyLIMA.magnification import magnification_PSPL

    tau = np.array([1])
    uo = np.array([0])

    magnification, impact_parameter = magnification_PSPL.magnification_PSPL(tau, uo,
                                                                            return_impact_parameter=True)

    assert np.allclose(magnification, np.array([1.34164079]))
    assert impact_parameter == 1


def test_magnification_FSPLarge():
    from pyLIMA.magnification import magnification_VBB

    tau = np.array([0.1])
    uo = 0.001
    rho = 0.25
    limb_darkening_coefficient = 0.3

    magnification_fspl = magnification_VBB.magnification_FSPL(tau, uo, rho,
                                                              limb_darkening_coefficient,
                                                              sqrt_limb_darkening_coefficient=None)

    assert magnification_fspl == 7.959307223839349


def test_magnification_USBL():
    from pyLIMA.magnification import magnification_VBB

    separation = [1.23]
    mass_ratio = 0.034
    x_source = [0.28]
    y_source = [0.02]
    rho = 0.056

    magnification = magnification_VBB.magnification_USBL(separation, mass_ratio,
                                                         x_source, y_source, rho)

    assert np.allclose(magnification[0],4.396361656443579)


def test_magnification_FSBL():
    from pyLIMA.magnification import magnification_VBB

    separation = [1.23]
    mass_ratio = 0.034
    x_source = [0.28]
    y_source = [0.02]
    rho = 0.056
    limb_darkening_coefficient = 0.3

    magnification = magnification_VBB.magnification_FSBL(separation, mass_ratio,
                                                         x_source, y_source, rho,
                                                         limb_darkening_coefficient)

    assert np.allclose(magnification[0],4.39589349809541)


def test_magnification_PSBL():
    from pyLIMA.magnification import magnification_VBB

    separation = [1.23]
    mass_ratio = 0.034
    x_source = [0.28]
    y_source = [0.02]

    magnification = magnification_VBB.magnification_PSBL(separation, mass_ratio,
                                                         x_source, y_source)

    assert np.allclose(magnification[0],4.264164845939242)
