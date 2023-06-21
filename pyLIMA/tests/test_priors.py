import unittest.mock as mock

import numpy as np
from pyLIMA.priors import guess, parameters_boundaries, parameters_priors
from pyLIMA.toolbox import time_series


def _create_event(JD=0, astrometry=False):
    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].name = 'Test'
    event.telescopes[0].lightcurve_flux = time_series.construct_time_series(
        np.array([[JD + 0, 10, 2], [JD + 20, 200, 3]]), ['time', 'flux', 'err_flux'],
        ['JD', 'W/m^2', 'W/m^2'])
    event.telescopes[0].lightcurve_magnitude = time_series.construct_time_series(
        np.array([[JD + 0, 15, 0.2], [JD + 20, 20, 0.3]]), ['time', 'mag', 'err_mag'],
        ['JD', 'mag', 'mag'])
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


def test_initial_guess_PSPL():
    event = _create_event()

    theguess = guess.initial_guess_PSPL(event)

    assert np.allclose(theguess[0], [10.0, 0.5631843966084606, 13.105408879743047])
    assert np.allclose(theguess[1], 46056.54738747334)


def test_initial_guess_FSPL():
    event = _create_event()

    theguess = guess.initial_guess_FSPL(event)

    assert np.allclose(theguess[0],
                       [10.0, 0.5631843966084606, 13.105408879743047, 0.05])
    assert np.allclose(theguess[1], 46056.54738747334)


def test_initial_guess_DSPL():
    event = _create_event()

    theguess = guess.initial_guess_DSPL(event)

    assert np.allclose(theguess[0],
                       [10.0, 0.5631843966084606, 5, 0.01, 13.105408879743047, 0.5])
    assert np.allclose(theguess[1], 46056.54738747334)


def test_all_parameters_boundaries():
    limits = []

    for function in dir(parameters_boundaries):

        if ('boundaries' in function) & ('parameters_boundaries' not in function):

            try:

                bound = eval('parameters_boundaries.' + function + '()')

            except TypeError:

                bound = eval('parameters_boundaries.' + function)(*[np.arange(-10, 10)])

            limits.append(bound)

    assert np.allclose(limits, [(0.500001, 10), (0, 6.283185307179586), (-150, 150),
                                (-1.0, 1.0), (-9, 9), (0.0, 9), (-1.0, 1000),
                                (1e-06, 1.0), (-20, 20), (-20, 20), (-1.0, 1.0),
                                (-1.0, 1.0), (0.01, 10), (-90, 90), (-4096, 4096),
                                (0, 360), (0.001, 1.0), (0.1, 100.0), (-10, 10),
                                (-5, 5), (-5, 5.0), (5e-05, 0.05), (0.1, 10.0),
                                (2400000, 2500000), (0.1, 500), (2400000, 2500000),
                                (0.0, 10.0), (0.0, 1.0), (-1, 1), (-10.0, 10.0),
                                (-10.0, 10.0), (-10.0, 10.0)])


def test_parameters_boundaries():
    event = _create_event()
    dico = {'t0': 0, 'delta_t0': 1, 'piEN': 2}

    limits = parameters_boundaries.parameters_boundaries(event, dico)
    assert np.allclose(limits, [(2400000, 2500000), (-150, 150), (-1.0, 1.0)])


def test_default_priors():
    dico = {'t0': [0, [0, 10]], 'delta_t0': [1, [-1, 1]], 'piEN': [2, [-5, 5]]}

    priors = parameters_priors.default_parameters_priors(dico)

    assert np.allclose(priors['t0'].pdf(5), 0.1)
    assert np.allclose(priors['delta_t0'].pdf(0.5), 0.5)
    assert np.allclose(priors['piEN'].pdf(0.0), 0.1)
