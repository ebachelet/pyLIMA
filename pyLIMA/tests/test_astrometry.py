import collections
import unittest.mock as mock

import numpy as np
from pyLIMA.astrometry import astrometric_positions, astrometric_shifts


def test_xy_shifts_to_NE_shifts():
    x = [0.285, 1.56]
    y = [-0.85, 9.8]
    xy_shifts = np.r_[[x], [y]]
    piEN = -0.85
    piEE = 1.56
    delta_ra, delta_dec = astrometric_positions.xy_shifts_to_NE_shifts(xy_shifts, piEN,
                                                                       piEE)

    assert np.allclose(delta_ra, np.array([0.65695057, -3.31903292]))
    assert np.allclose(delta_dec, np.array([0.61003357, -9.35187791]))


def test_astrometric_position_of_the_source():
    # Without time_ref
    telescope = mock.MagicMock()
    telescope.name = 'test'
    telescope.astrometry['time'].value = np.array([2459857, 2459959])
    telescope.astrometry['ra'].unit = 'deg'
    dico = {'astrometry': np.array([np.array([0.2, 0.1]), np.array([0.6, 0.98])])}
    telescope.Earth_positions.__getitem__.side_effect = dico.__getitem__
    telescope.Earth_positions.__iter__.side_effect = dico.__iter__

    dico = {'astrometry': np.array([np.array([0.2, 0.1]), np.array([0.6, 0.98])])}
    telescope.Earth_positions_projected.__getitem__.side_effect = dico.__getitem__
    telescope.Earth_positions_projected.__iter__.side_effect = dico.__iter__

    pyparams = collections.OrderedDict()
    pyparams['t0'] =  2459875
    pyparams['position_source_N_test'] = 10.11
    pyparams['position_source_E_test'] = 182.5
    pyparams['mu_source_N'] = 8.88
    pyparams['mu_source_E'] = -2.56
    pyparams['pi_source'] = 5.6

    position_ra, position_dec = \
        astrometric_positions.astrometric_positions_of_the_source(
            telescope, pyparams, time_ref=None)

    assert np.allclose(position_ra, np.array([182.4999991, 182.49999831]))
    assert np.allclose(position_dec, np.array([10.10999957, 10.11000041]))

    # With time_ref and pixel_scale
    telescope = mock.MagicMock()
    telescope.name = 'test'
    telescope.astrometry['time'].value = np.array([2459857, 2459959])
    telescope.astrometry['ra'].unit = 'pix'
    dico = {'astrometry': np.array([np.array([0.2, 0.1]), np.array([0.6, 0.98])])}
    telescope.Earth_positions.__getitem__.side_effect = dico.__getitem__
    telescope.Earth_positions.__iter__.side_effect = dico.__iter__
    dico = {'astrometry': np.array([np.array([0.2, 0.1]), np.array([0.6, 0.98])])}
    telescope.Earth_positions_projected.__getitem__.side_effect = dico.__getitem__
    telescope.Earth_positions_projected.__iter__.side_effect = dico.__iter__

    telescope.pixel_scale = 100  # mas/pix

    pyparams = collections.OrderedDict()

    pyparams['t0'] = 2459875
    pyparams['position_source_N_test'] = 1.11
    pyparams['position_source_E_test'] = 12.5
    pyparams['mu_source_N'] = 8.88
    pyparams['mu_source_E'] = -2.56
    pyparams['pi_source'] = 5.6

    time_ref = 2459800
    position_ra, position_dec = \
        astrometric_positions.astrometric_positions_of_the_source(
            telescope, pyparams,
            time_ref=time_ref)

    assert np.allclose(position_ra, np.array([12.06689281, 11.33070522]))
    assert np.allclose(position_dec, np.array([2.48459055, 4.97002628]))


def test_source_astrometric_positions():
    # Without shifts
    telescope = mock.MagicMock()
    telescope.name = 'test'
    telescope.astrometry['time'].value = np.array([2459857, 2459959])
    telescope.astrometry['ra'].unit = 'deg'
    dico = {'astrometry': np.array([np.array([0.2, 0.1]), np.array([0.6, 0.98])])}
    telescope.Earth_positions_projected.__getitem__.side_effect = dico.__getitem__
    telescope.Earth_positions_projected.__iter__.side_effect = dico.__iter__

    pyparams = collections.OrderedDict()

    pyparams['t0'] =  2459875
    pyparams['position_source_N_test'] = 10.11
    pyparams['position_source_E_test'] = 182.5
    pyparams['mu_source_N'] = 8.88
    pyparams['mu_source_E'] = -2.56
    pyparams['pi_source'] = 5.6

    position_ra, position_dec = astrometric_positions.source_astrometric_positions(
        telescope, pyparams,
        shifts=None, time_ref=None)

    assert np.allclose(position_ra, np.array([182.4999991, 182.49999831]))
    assert np.allclose(position_dec, np.array([10.10999957, 10.11000041]))

    # With shifts
    telescope = mock.MagicMock()
    telescope.name = 'test'
    telescope.astrometry['time'].value = np.array([2459857, 2459959])
    telescope.astrometry['ra'].unit = 'deg'
    dico = {'astrometry': np.array([np.array([0.2, 0.1]), np.array([0.6, 0.98])])}
    telescope.Earth_positions_projected.__getitem__.side_effect = dico.__getitem__
    telescope.Earth_positions_projected.__iter__.side_effect = dico.__iter__

    pyparams = collections.OrderedDict()

    pyparams['t0'] =  2459875
    pyparams['position_source_N_test'] = 10.11
    pyparams['position_source_E_test'] = 182.5
    pyparams['mu_source_N'] = 8.88
    pyparams['mu_source_E'] = -2.56
    pyparams['pi_source'] = 5.6

    shifts = [np.array([0.1, 0.89]), np.array([-0.8, -10])]
    position_ra, position_dec = astrometric_positions.source_astrometric_positions(
        telescope, pyparams,
        shifts=shifts, time_ref=None)
    assert np.allclose(position_ra, np.array([182.49999913, 182.49999856]))
    assert np.allclose(position_dec, np.array([10.10999935, 10.10999763]))


def test_lens_astrometric_positions():
    # Without shifts
    telescope = mock.MagicMock()
    telescope.name = 'test'
    telescope.astrometry['time'].value = np.array([2459857, 2459959])
    telescope.astrometry['ra'].unit = 'deg'
    dico = {'astrometry': np.array([np.array([0.2, 0.1]), np.array([0.6, 0.98])])}
    telescope.Earth_positions_projected.__getitem__.side_effect = dico.__getitem__
    telescope.Earth_positions_projected.__iter__.side_effect = dico.__iter__

    model = mock.MagicMock()
    model.sources_trajectory.return_value = [np.array([0.28, 1.36]),
                                             np.array([-0.28, 0.78]),
                                             np.array([-0.28, 0.78]),
                                             np.array([-0.28, 0.78]),
                                             np.array([-0.28, 0.78]),
                                             np.array([-0.28, 0.78])
                                             ]

    pyparams = collections.OrderedDict()

    pyparams['t0'] =  2459875
    pyparams['position_source_N_test'] = 10.11
    pyparams['position_source_E_test'] = 182.5
    pyparams['mu_source_N'] = 8.88
    pyparams['mu_source_E'] = -2.56
    pyparams['pi_source'] = 5.6
    pyparams['piEN'] = -2.56
    pyparams['piEE'] =  5.6
    pyparams['theta_E'] = 5.6

    position_ra, position_dec = astrometric_positions.lens_astrometric_positions(model,
                                                                                 telescope,
                                                                                 pyparams,
                                                                                 time_ref=None)
    assert np.allclose(position_ra, np.array([182.49999852, 182.49999689]))
    assert np.allclose(position_dec, np.array([10.10999935, 10.11000239]))


def test_PSPL_shifts_no_blend():
    source_x = np.array([0.1, -0.8])
    source_y = np.array([2.1, -10.8])
    theta_E = 51  # mas

    shifts = astrometric_shifts.PSPL_shifts_no_blend(source_x, source_y, theta_E)

    assert np.allclose(shifts[0], np.array([0.79439252, -0.34205231]))
    assert np.allclose(shifts[1], np.array([16.68224299, -4.61770624]))


def test_PSPL_shifts_with_blend():
    source_x = np.array([0.1, -0.8])
    source_y = np.array([2.1, -10.8])
    theta_E = 51  # mas
    g_blend = 11.11

    shifts = astrometric_shifts.PSPL_shifts_with_blend(source_x, source_y, theta_E,
                                                       g_blend)

    assert np.allclose(shifts[0], np.array([0.08888225, -0.02868366]))
    assert np.allclose(shifts[1], np.array([1.86652724, -0.38722946]))
