import numpy as np

from pyLIMA import telescopes, event


def test_event():
    lightcurve = np.array([[2456789, 12.8, 0.01], [2458888, 12, 0.25]])

    telo = telescopes.Telescope(name='fake', camera_filter='I',
                                lightcurve=lightcurve,
                                lightcurve_names=['time', 'mag', 'err_mag'],
                                lightcurve_units=['JD', 'mag', 'mag'])

    telo2 = telescopes.Telescope(name='fake2', camera_filter='I',
                                 lightcurve=lightcurve,
                                 lightcurve_names=['time', 'mag', 'err_mag'],
                                 lightcurve_units=['JD', 'mag', 'mag'])
    ev = event.Event(ra=20, dec=-20)
    ev.telescopes.append(telo)
    ev.telescopes.append(telo2)

    telo.initialize_positions()
    telo2.initialize_positions()

    ev.find_survey("fake2")

    assert ev.ra == 20
    assert ev.dec == -20
    assert ev.survey == "fake2"
    assert ev.telescopes[0] == telo2

    ev.compute_parallax_all_telescopes(['Full', 2456780])

    assert np.allclose(telo.deltas_positions['photometry'],
                       np.array([[-6.68645090e-03, -6.11752451e+00],
                                 [-4.33482159e-03, -3.26412162e+01]]))

    assert np.allclose(telo2.deltas_positions['photometry'],
                       np.array([[-6.68645090e-03, -6.11752451e+00],
                                 [-4.33482159e-03, -3.26412162e+01]]))

    assert ev.total_number_of_data_points() == 4

    assert np.allclose(ev.North, [0.3213938, 0.11697778, 0.93969262])
    assert np.allclose(ev.East, [-0.34202014, 0.93969262, 0.])
