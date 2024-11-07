import numpy as np
from pyLIMA.parallax import astropy_ephemerides, JPL_ephemerides, parallax

from pyLIMA import telescopes


def test_astropy_ephemerides():
    times = np.array([258957, 2458966])

    pos_speed = astropy_ephemerides.Earth_ephemerides(times)

    assert np.allclose(pos_speed[0].xyz.value, np.array([[-0.83856162, -0.81442478],
                                                         [0.47912307, -0.54196625],
                                                         [0.21813489, -0.23488379]]))

    assert np.allclose(pos_speed[1].xyz.value, np.array([[-0.0095267, 0.00993693],
                                                         [-0.01323042, -0.01276249],
                                                         [-0.00594259, -0.00553295]]))


def test_horizons_obscodes():
    code = JPL_ephemerides.horizons_obscodes('Geocentric')

    assert code == '500'


def test_horizons_API():
    times = np.array([2459957, 2459966])
    ephemerides = JPL_ephemerides.horizons_API('TESS', times, observatory='Geocentric')

    assert np.allclose(ephemerides[1], np.array([[ 2.45995700e+06,  8.00606500e+01,  1.28280000e-01,
         2.35566386e-03],
       [ 2.45996600e+06,  4.29997200e+01, -2.45541000e+01,
         1.90008195e-03]]))



def test_EN_trajectory_angle():
    angle = parallax.EN_trajectory_angle(0.65, -0.5)

    assert angle == -0.6556956262415362


def test_compute_parallax_curvature():
    pie = [0.22, 0.11]
    delta_positions = np.array([[0.1, 0.2], [5.4, 8.2]])
    projection = parallax.compute_parallax_curvature(pie, delta_positions)

    assert np.allclose(projection, (np.array([0.616, 0.946]), np.array([1.177, 1.782])))


def test_parallax_combination():
    lightcurve = np.array([[2456789, 12.8, 0.01], [2458888, 12, 0.25]])

    telo = telescopes.Telescope(name='fake', camera_filter='I',
                                lightcurve=lightcurve,
                                lightcurve_names=['time', 'mag', 'err_mag'],
                                lightcurve_units=['JD', 'mag', 'mag'])
    telo.initialize_positions()
    North_vector = [0.25, 0.5, 0.88]
    North_vector /= np.sum(np.array(North_vector) ** 2) ** 0.5
    East_vector = [0.125, 0.15, 0.188]
    East_vector /= np.sum(np.array(East_vector) ** 2) ** 0.5

    parallax.parallax_combination(telo, ['Full', 2458988], North_vector, East_vector)

    assert np.allclose(telo.deltas_positions['photometry'],
                       np.array([[8.24926684, -0.74170405],
                                 [1.4884521, -1.0171205]]))


def test_Earth_ephemerides():
    times = np.array([258927, 2458936])

    eph = parallax.Earth_ephemerides(times)

    assert np.allclose(eph, (np.array([[-0.45314827, 0.7931182, 0.35879775],
                                       [-0.99506234, -0.10511616, -0.04551538]]),
                             np.array([[-0.01560426, -0.0071886,
                                        -0.0032035],
                                       [0.00181462, -0.01573381, -0.00682075]])))


def test_Earth_telescope_sidereal_times():
    times = np.array([258927, 2458936])

    eph = parallax.Earth_telescope_sidereal_times(times, sidereal_type='mean')
    assert np.allclose(eph, [3.72810288, 0.09388775])

    eph = parallax.Earth_telescope_sidereal_times(times, sidereal_type='apparent')
    assert np.allclose(eph, [3.73945738, 0.09380926])


def test_space_ephemerides():
    lightcurve = np.array([[2456789, 12.8, 0.01], [2456790, 12, 0.25]])
    time_to_treat = lightcurve[:, 0]

    telo = telescopes.Telescope(name='fake', camera_filter='I',
                                lightcurve=lightcurve,
                                lightcurve_names=['time', 'mag', 'err_mag'],
                                lightcurve_units=['JD', 'mag', 'mag'],
                                spacecraft_name='HST',
                                location='Space')
    eph = parallax.space_ephemerides(telo, time_to_treat, data_type='photometry')

    assert np.allclose(eph[0], np.array([[3.88646541e-05, 1.27035577e-05, 2.17479720e-05],
       [3.10094521e-05, 2.98014030e-05, 1.71719104e-05]])
)

    assert np.allclose(eph[1], np.array([[ 2.45678900e+06,  1.98100840e+02, -2.80080300e+01,
         4.63121582e-05],
       [ 2.45679000e+06,  2.23861930e+02, -2.17653500e+01,
         4.63096560e-05]])
)


def test_annual_parallax():
    times = np.array([258927, 2458936])
    earth = np.array([[-0.45314827, 0.7931182, 0.35879775],
                      [-0.99506234, -0.10511616, -0.04551538]])
    delta_Sun = parallax.annual_parallax(times, earth, 2588936)

    assert np.allclose(delta_Sun, np.array([[17099.44306232, 33704.72719158,
                                             14577.56911154],
                                            [954.15636748, 1881.01933702,
                                             813.55732731]]))


def test_terrestrial_parallax():
    sid = np.array([3.25, 1.28])
    delta = parallax.terrestrial_parallax(sid, 2588, -152, -3.21)

    assert np.allclose(delta, np.array([[-3.95430017e-05, -1.58070399e-05,
                                         2.38834620e-06],
                                        [2.99339070e-05, -3.02898120e-05,
                                         2.38834620e-06]]))
