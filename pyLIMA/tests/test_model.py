import unittest.mock as mock

import numpy as np
from pyLIMA.models import DFSPLmodel, DSPLmodel, FSBLmodel, FSPLmodel, FSPLargemodel, \
    PSBLmodel, PSPLmodel, USBLmodel
from pyLIMA.toolbox import time_series


def _create_event(JD=0, astrometry=False):
    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].name = 'Test'
    event.telescopes[0].lightcurve_flux = time_series.construct_time_series(
        np.array([[JD + 0, 10, 2], [JD + 20, 200, 3]]), ['time', 'flux', 'err_flux'],
        ['JD', 'W/m^2', 'W/m^2'])

    if astrometry:
        event.telescopes[0].astrometry = time_series.construct_time_series(
            np.array([[JD + 0, 10, 2, 10, 2], [JD + 20, 200, 3, 200, 3]]),
            ['time', 'ra', 'err_ra', 'dec', 'err_dec'],
            ['JD', 'deg', 'deg', 'deg', 'deg'])
        dico = {'astrometry': np.array([np.array([0.2, 0.1]), np.array([0.6, 0.98])])}
        event.telescopes[
            0].Earth_positions_projected.__getitem__.side_effect = dico.__getitem__
        event.telescopes[
            0].Earth_positions_projected.__iter__.side_effect = dico.__iter__

    else:
        event.telescopes[0].astrometry = None

    event.telescopes[0].filter = 'I'
    event.telescopes[0].ld_gamma = 0.5

    return event


def test_initialize_model():
    event = _create_event()

    Model = FSPLmodel(event)

    assert Model.event == event
    assert Model.parallax_model == ['None', 0.0]
    assert Model.xallarap_model == ['None']
    assert Model.orbital_motion_model == ['None', 0.0]
    assert Model.blend_flux_parameter == 'fblend'
    assert Model.photometry is True
    assert Model.astrometry is False
    assert dict(Model.model_dictionnary) == {'t0': 0, 'u0': 1, 'tE': 2, 'rho': 3,
                                             'fsource_Test': 4, 'fblend_Test': 5}
    assert dict(Model.pyLIMA_standards_dictionnary) == {'t0': 0, 'u0': 1, 'tE': 2,
                                                        'rho': 3, 'fsource_Test': 4,
                                                        'fblend_Test': 5}
    assert dict(Model.fancy_to_pyLIMA_dictionnary) == {}
    assert dict(Model.pyLIMA_to_fancy_dictionnary) == {}
    assert dict(Model.pyLIMA_to_fancy) == {}
    assert dict(Model.fancy_to_pyLIMA) == {}
    assert Model.standard_parameters_boundaries == [(2400000, 2500000), (0.0, 1.0),
                                                    (0.1, 500), (5e-05, 0.05),
                                                    (0.0, 200.0), (-200.0, 200.0)]
    assert Model.origin == ['center_of_mass', [0, 0]]
    assert Model.model_type() == 'FSPL'


def test_model_magnification():
    event = _create_event()

    Model = FSPLmodel(event)

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)

    magnification = Model.model_magnification(event.telescopes[0], pym)

    assert np.allclose(magnification, [42.16176577, 1.99919488])


def test_model_magnification_Jacobian():
    event = _create_event()

    Model = FSPLmodel(event)

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)

    jacob, magnification = Model.model_magnification_Jacobian(event.telescopes[0], pym)

    assert np.allclose(jacob, np.array(
        [[-5.52086665e+00, -2.70522466e+01, 7.88695235e-02, -7.86812979e+02],
         [8.25429191e-02, -1.03707770e-02, 4.59881978e-02, 7.27955884e-02]]))

    assert np.allclose(magnification, [42.16176577, 1.99919488])


def test_photometric_model_Jacobian():
    event = _create_event()

    Model = FSPLmodel(event)

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)
    jacob = Model.photometric_model_Jacobian(event.telescopes[0], pym)

    assert np.allclose(jacob, np.array([[2.61179660e+01, -3.90491801e-01],
                                        [1.27978034e+02, 4.90617904e-02],
                                        [-3.73113801e-01, -2.17559718e-01],
                                        [3.72223348e+03, -3.44379393e-01],
                                        [4.21617658e+01, 1.99919488e+00],
                                        [1.00000000e+00, 1.00000000e+00]]))


def test_compute_the_microlensing_model():
    event = _create_event()

    Model = FSPLmodel(event, origin=['nothere', [9, 8]])

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)

    model = Model.compute_the_microlensing_model(event.telescopes[0], pym)

    assert np.allclose(model['photometry'], [9.99999999, 200.])
    assert model['astrometry'] is None


def test_derive_telescope_flux():
    event = _create_event()

    Model = FSPLmodel(event, origin=['nothere', [9, 8]])

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)
    magnification = Model.model_magnification(event.telescopes[0], pym)

    Model.derive_telescope_flux(event.telescopes[0], pym, magnification)
    assert np.allclose(pym.fsource_Test, -15723296.651212221)
    assert np.allclose(pym.fblend_Test, 15724766.572776612)


def test_find_telescopes_fluxes():
    event = _create_event()

    Model = FSPLmodel(event, origin=['nothere', [9, 8]])

    params = [0.5, 0.002, 35, 0.05]

    fluxes = Model.find_telescopes_fluxes(params)
    assert np.allclose(fluxes.fsource_Test, -15723296.651212221)
    assert np.allclose(fluxes.fblend_Test, 15724766.572776612)


def test_compute_pyLIMA_parameters():
    event = _create_event()

    Model = FSPLmodel(event, origin=['nothere', [9, 8]])

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)
    values = [getattr(pym, key) for key in pym._fields[:4]]
    assert np.allclose(params, values)


def test_source_trajcetory():
    event = _create_event()

    Model = FSPLmodel(event, origin=['nothere', [9, 8]])

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)

    X, Y, dS, dA = Model.source_trajectory(event.telescopes[0], pym,
                                           data_type='photometry')

    assert np.allclose(X, [-8.98571429, -9.55714286])
    assert np.allclose(Y, [-8.002, -8.002])
    assert np.allclose(dS, [0, 0])
    assert np.allclose(dA, [0, 0])


def t_star(x):
    return x.rho * x.tE


def tE(x):
    return x.t_star / 10 ** (x.log_rho)


def test_fancy_parameters():
    from pyLIMA.models import pyLIMA_fancy_parameters

    setattr(pyLIMA_fancy_parameters, 't_star', t_star)
    setattr(pyLIMA_fancy_parameters, 'tE', tE)

    my_pars2 = {'log_rho': 'rho', 't_star': 'tE'}

    event = _create_event()

    fspl2 = FSPLmodel(event, fancy_parameters=my_pars2)

    assert dict(fspl2.model_dictionnary) == {'t0': 0, 'u0': 1, 't_star': 2,
                                             'log_rho': 3, 'fsource_Test': 4,
                                             'fblend_Test': 5}
    assert dict(fspl2.fancy_to_pyLIMA_dictionnary) == {'log_rho': 'rho', 't_star': 'tE'}
    assert dict(fspl2.pyLIMA_to_fancy_dictionnary) == {'rho': 'log_rho', 'tE': 't_star'}
    assert fspl2.pyLIMA_to_fancy['log_rho'] == pyLIMA_fancy_parameters.log_rho
    assert fspl2.pyLIMA_to_fancy['t_star'] == t_star
    assert fspl2.fancy_to_pyLIMA['rho'] == pyLIMA_fancy_parameters.rho
    assert fspl2.fancy_to_pyLIMA['tE'] == pyLIMA_fancy_parameters.tE


def test_DFSPL():
    event = _create_event()

    Model = DFSPLmodel(event)
    Model.event.telescopes[0].ld_gamma1 = 0.25
    Model.event.telescopes[0].ld_gamma2 = 0.25

    assert dict(Model.model_dictionnary) == {'t0': 0, 'delta_t0': 2, 'u0': 1,
                                             'delta_u0': 3, 'tE': 4, 'rho_1': 5,
                                             'rho_2': 6, 'q_flux_I': 7,
                                             'fsource_Test': 8, 'fblend_Test': 9}

    params = [0.5, 0.002, 0.1, 0.14, 35, 0.05, 0.007, 0.25]

    pym = Model.compute_pyLIMA_parameters(params)

    magi = Model.model_magnification(event.telescopes[0], pym)

    assert np.allclose(magi, [33.93701346, 1.99050956])


def test_DSPL():
    event = _create_event()

    Model = DSPLmodel(event)

    assert dict(Model.model_dictionnary) == {'t0': 0, 'u0': 1, 'delta_t0': 2,
                                             'delta_u0': 3, 'tE': 4,
                                             'q_flux_I': 5, 'fsource_Test': 6,
                                             'fblend_Test': 7}

    params = [0.5, 0.002, 0.1, 0.14, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)
    magi = Model.model_magnification(event.telescopes[0], pym)

    assert np.allclose(magi, [66.36341157, 1.99537759])

    source1x, source1y, source2x, source2y = Model.sources_trajectory(
        event.telescopes[0], pym)

    assert np.allclose(source1x, [0.01428571, -0.55714286])
    assert np.allclose(source1y, [-0.002, -0.002])
    assert np.allclose(source2x, [0.01714286, -0.55428571])
    assert np.allclose(source2y, [-0.142, -0.142])


def test_FSBL():
    event = _create_event()
    event.telescopes[0].a1 = 0.5

    Model = FSBLmodel(event)
    params = [0.5, 0.002, 38, 0.006, 1.14, 0.35, 0.0]

    pym = Model.compute_pyLIMA_parameters(params)
    magi = Model.model_magnification(event.telescopes[0], pym)
    assert np.allclose(magi, [4.73518855, 2.32967011])


def test_FSPL():
    event = _create_event()

    Model = FSPLmodel(event)
    params = [0.5, 0.002, 38, 0.006]

    pym = Model.compute_pyLIMA_parameters(params)
    magi = Model.model_magnification(event.telescopes[0], pym)
    assert np.allclose(magi, [76.99680049, 2.13612409])


def test_FSPLarge():
    event = _create_event()
    event.telescopes[0].ld_a1 = 0.28
    event.telescopes[0].ld_a2 = None

    Model = FSPLargemodel(event)
    params = [0.5, 0.002, 38, 6]

    pym = Model.compute_pyLIMA_parameters(params)
    magi = Model.model_magnification(event.telescopes[0], pym)

    assert np.allclose(magi, [1.05837633, 1.05730364])


def test_PSBL():
    event = _create_event()

    Model = PSBLmodel(event)
    params = [0.5, 0.002, 38, 1.14, 0.35, 0.0]

    pym = Model.compute_pyLIMA_parameters(params)
    magi = Model.model_magnification(event.telescopes[0], pym)

    assert np.allclose(magi, [4.73271858, 2.32814229])


def test_PSPL():
    event = _create_event()

    Model = PSPLmodel(event)
    params = [0.5, 0.002, 38]

    pym = Model.compute_pyLIMA_parameters(params)
    magi = Model.model_magnification(event.telescopes[0], pym)
    assert np.allclose(magi, [75.14196484, 2.13609124])

    jacobi, magi = Model.model_magnification_Jacobian(event.telescopes[0], pym)

    assert np.allclose(jacobi,
                       np.array([[-1.46870667e+02, -8.48324973e+02, 1.93250878e+00],
                                 [9.08176526e-02, -1.34503272e-02, 4.66037954e-02]]))

    event = _create_event(JD=2458925, astrometry=True)  # for parallax

    Model = PSPLmodel(event, parallax=['Full', 2458925])
    params = [0.5, 0.002, 38, 1, 0.1, 4.8, 5.2, 100, 150, 1.25, 0.22]

    pym = Model.compute_pyLIMA_parameters(params)
    shifts = Model.model_astrometry(event.telescopes[0], pym)

    assert np.allclose(shifts, np.array([[149.99999998, 150.00000005],
                                         [99.99999999, 100.00000007]]))


def test_USBL():
    event = _create_event()

    Model = USBLmodel(event)
    params = [0.5, 0.002, 38, 0.025, 1.24, 0.002, 0.01]

    pym = Model.compute_pyLIMA_parameters(params)
    magi = Model.model_magnification(event.telescopes[0], pym)

    assert np.allclose(magi, [73.75028234, 2.12549786])

    event = _create_event()

    Model = USBLmodel(event, origin=['central_caustic', [0, 0]])
    params = [0.5, 0.002, 38, 0.025, 1.24, 0.002, 0.01]
    pym = Model.compute_pyLIMA_parameters(params)
    magi = Model.model_magnification(event.telescopes[0], pym)

    assert np.allclose(magi, [76.16515049, 2.11882843])
