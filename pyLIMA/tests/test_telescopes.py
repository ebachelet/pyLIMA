import numpy as np

from pyLIMA import telescopes


def simulate_telescope(altitude=154, location='Earth', spacecraft_name=None):
    lightcurve = np.array([[2456789, 12.8, 0.01], [2457789, 22.8, 0.21]])
    astrometry = np.array([[2456789, 178.25, 0.01, 165.22, 0.002], [2457789, 178.25,
                                                                    0.01, 165.22,
                                                                    0.002]])

    telo = telescopes.Telescope(name='fake', camera_filter='I',
                                lightcurve=lightcurve,
                                lightcurve_names=['time', 'mag', 'err_mag'],
                                lightcurve_units=['JD', 'mag', 'mag'],
                                astrometry=astrometry,
                                astrometry_names=['time', 'ra', 'err_ra', 'dec',
                                                  'err_dec'],
                                astrometry_units=['JD', 'deg', 'deg', 'deg', 'deg'],
                                altitude=altitude,
                                location=location,
                                spacecraft_name=spacecraft_name)

    return telo


def test_time_data():
    telo = simulate_telescope()

    telo.initialize_positions()
    photometry_mask = [True, False]
    astrometry_mask = [False, False]

    telo.trim_data(photometry_mask=photometry_mask, astrometry_mask=astrometry_mask)

    assert len(telo.lightcurve) == 1
    assert len(telo.lightcurve) == 1
    assert len(telo.astrometry) == 0

    assert np.allclose(telo.Earth_positions['photometry'], np.array([[-0.64031624,
                                                                      -0.71698051,
                                                                      -0.31093908]]))
    assert np.allclose(telo.Earth_speeds['photometry'],
                       np.array([[0.01300459, -0.01008431, -0.00437255]]))


def test_n_data():
    telo = simulate_telescope()

    assert telo.n_data() == 2


def test_initialize_positions():
    telo = simulate_telescope()

    telo.initialize_positions()

    positions = [telo.Earth_positions[key] for key in telo.Earth_positions.keys()]
    assert np.allclose(positions, [np.array([[-0.64031624, -0.71698051, -0.31093908],
                                             [-0.70147152, 0.63638734, 0.27571799]]),
                                   np.array([[-0.64031624, -0.71698051,
                                              -0.31093908],
                                             [-0.70147152, 0.63638734, 0.27571799]])])

    speeds = [telo.Earth_speeds[key] for key in telo.Earth_speeds.keys()]
    assert np.allclose(speeds, [np.array([[0.01300459, -0.01008431, -0.00437255],
                                          [-0.01230816, -0.01134105, -0.00491614]]),
                                np.array([[0.01300459, -0.01008431,
                                           -0.00437255],
                                          [-0.01230816, -0.01134105, -0.00491614]])])
    sidereals = [telo.sidereal_times[key] for key in telo.sidereal_times.keys()]
    assert np.allclose(sidereals, [np.array([0.85860281, 5.49508311]), np.array([
        0.85860281, 5.49508311])])

    telescope_pos = [telo.telescope_positions[key] for key in
                     telo.telescope_positions.keys()]
    assert np.allclose(telescope_pos,
                       [np.array([[-1.83065582e-05, 2.07824887e-05, -3.24158313e-05],
                                  [-1.93343492e-05, -1.98298979e-05, -3.24158313e-05]]),
                        np.array([[-1.83065582e-05, 2.07824887e-05, -3.24158313e-05],
                                  [-1.93343492e-05, -1.98298979e-05, -3.24158313e-05]])]
                       )

    telo2 = simulate_telescope(location='Space', spacecraft_name='Gaia')
    telo2.initialize_positions()

    telescope_pos = [telo2.telescope_positions[key] for key in
                     telo2.telescope_positions.keys()]

    assert np.allclose(telescope_pos, [np.array([[0.00456092, 0.00803218, 0.00385589],
                                                 [0.00528678, -0.00758553,
                                                  -0.00306416]]),
                                       np.array([[0.00456092, 0.00803218, 0.00385589],
                                                 [0.00528678, -0.00758553,
                                                  -0.00306416]])])

    assert np.allclose(telo2.spacecraft_positions['photometry'][1],
                       [2.45778900e+06, 1.24874800e+02, 1.83351900e+01, 9.74060965e-03])

    assert np.allclose(telo2.spacecraft_positions['astrometry'][1],
                       [2.45778900e+06, 1.24874800e+02, 1.83351900e+01, 9.74060965e-03])


def test_compute_parallax():
    telo = simulate_telescope()

    telo.compute_parallax(['Full', 2456790], [0.25, 0.28, 1.26], [-.25, 1.28, 0])

    delta_pos = [telo.deltas_positions[key] for key in telo.deltas_positions.keys()]

    assert np.allclose(delta_pos, [np.array([[-1.47564979e-04, -5.96912587e+00],
                                             [-7.86600785e-05, -1.76877826e+01]]),
                                   np.array([[-1.47564979e-04, -5.96912587e+00],
                                             [-7.86600785e-05, -1.76877826e+01]])])


def test_define_limb_darkening_coefficients():
    telo = simulate_telescope()

    telo.ld_gamma = 0.25
    telo.ld_sigma = 0.78
    telo.define_limb_darkening_coefficients()
    assert np.allclose(telo.ld_a1, 0.28409090909090906)
    assert np.allclose(telo.ld_a2, 0.7386363636363636)

    telo = simulate_telescope()

    telo.ld_a1 = 0.25
    telo.ld_a2 = 0.78
    telo.define_limb_darkening_coefficients()

    assert np.allclose(telo.ld_gamma, 0.21910604732690622)
    assert np.allclose(telo.ld_sigma, 0.8203330411919368)
