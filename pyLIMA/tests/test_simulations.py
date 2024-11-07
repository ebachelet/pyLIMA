import numpy as np
from pyLIMA.simulations import simulator


def test_simulate_a_telescope():

    telo = simulator.simulate_a_telescope("Fake", time_start=2459800, time_end=2459850,
                                          sampling=0.25,
                                          uniform_sampling=False, timestamps=[],
                                          location='Earth',
                                          camera_filter='I', altitude=0, longitude=0,
                                          latitude=0,
                                          bad_weather_percentage=0.10,
                                          minimum_alt=20, moon_windows_avoidance=20,
                                          maximum_moon_illumination=80.0,
                                          photometry=True,
                                          astrometry=False, pixel_scale=1, ra=270,
                                          dec=-30)
    assert telo.lightcurve is not None
    assert telo.astrometry is None

    times = np.arange(2458679, 2459788, 1.23)

    telo = simulator.simulate_a_telescope("Fake", timestamps=times,
                                          location='Earth',
                                          camera_filter='R', altitude=1650,
                                          longitude=20,
                                          latitude=-50,
                                          bad_weather_percentage=0.10,
                                          minimum_alt=20, moon_windows_avoidance=20,
                                          maximum_moon_illumination=80.0,
                                          photometry=True,
                                          astrometry=True, )

    assert np.allclose(telo.lightcurve['time'].value, times)
    assert np.allclose(telo.astrometry['time'].value, times)

    times = np.arange(2458679, 2458682, 1.23)

    telo = simulator.simulate_a_telescope("Fake", timestamps=times,
                                          location='Space', spacecraft_name='HST',
                                          camera_filter='I',
                                          photometry=True,
                                          astrometry=True, pixel_scale=40, ra=270,
                                          dec=-30)
    telo.initialize_positions()

    assert telo.pixel_scale == 40
    assert telo.lightcurve is not None
    assert telo.astrometry is not None
    assert telo.Earth_positions != {}
    assert telo.Earth_speeds != {}
    assert telo.spacecraft_positions != {}


def test_time_simulation():
    times = simulator.time_simulation(2459876, 2459878, 0.22, 0.0)

    assert np.allclose(times, [2459876., 2459876.00916667, 2459876.01833333,
                               2459876.0275, 2459876.03666667, 2459876.04583333,
                               2459876.055, 2459876.06416667, 2459876.07333333,
                               2459876.0825, 2459876.09166667, 2459876.10083333,
                               2459876.11, 2459876.11916667, 2459876.12833333,
                               2459876.1375, 2459876.14666667, 2459876.15583334,
                               2459876.165, 2459876.17416667, 2459876.18333334,
                               2459876.1925, 2459876.20166667, 2459876.21083334,
                               2459876.22, 2459876.22916667, 2459876.23833334,
                               2459876.2475, 2459876.25666667, 2459876.26583334,
                               2459876.275, 2459876.28416667, 2459876.29333334,
                               2459876.3025, 2459876.31166667, 2459876.32083334,
                               2459876.33, 2459876.33916667, 2459876.34833334,
                               2459876.3575, 2459876.36666667, 2459876.37583334,
                               2459876.385, 2459876.39416667, 2459876.40333334,
                               2459876.4125, 2459876.42166667, 2459876.43083334,
                               2459876.44000001, 2459876.44916667, 2459876.45833334,
                               2459876.46750001, 2459876.47666667, 2459876.48583334,
                               2459876.49500001, 2459876.50416667, 2459876.51333334,
                               2459876.52250001, 2459876.53166667, 2459876.54083334,
                               2459876.55000001, 2459876.55916667, 2459876.56833334,
                               2459876.57750001, 2459876.58666667, 2459876.59583334,
                               2459876.60500001, 2459876.61416667, 2459876.62333334,
                               2459876.63250001, 2459876.64166667, 2459876.65083334,
                               2459876.66000001, 2459876.66916667, 2459876.67833334,
                               2459876.68750001, 2459876.69666667, 2459876.70583334,
                               2459876.71500001, 2459876.72416668, 2459876.73333334,
                               2459876.74250001, 2459876.75166668, 2459876.76083334,
                               2459876.77000001, 2459876.77916668, 2459876.78833334,
                               2459876.79750001, 2459876.80666668, 2459876.81583334,
                               2459876.82500001, 2459876.83416668, 2459876.84333334,
                               2459876.85250001, 2459876.86166668, 2459876.87083334,
                               2459876.88000001, 2459876.88916668, 2459876.89833334,
                               2459876.90750001, 2459876.91666668, 2459876.92583334,
                               2459876.93500001, 2459876.94416668, 2459876.95333334,
                               2459876.96250001, 2459876.97166668, 2459876.98083334,
                               2459876.99000001, 2459876.99916668, 2459877.00833334,
                               2459877.01750001, 2459877.02666668, 2459877.03583335,
                               2459877.04500001, 2459877.05416668, 2459877.06333335,
                               2459877.07250001, 2459877.08166668, 2459877.09083335,
                               2459877.10000001, 2459877.10916668, 2459877.11833335,
                               2459877.12750001, 2459877.13666668, 2459877.14583335,
                               2459877.15500001, 2459877.16416668, 2459877.17333335,
                               2459877.18250001, 2459877.19166668, 2459877.20083335,
                               2459877.21000001, 2459877.21916668, 2459877.22833335,
                               2459877.23750001, 2459877.24666668, 2459877.25583335,
                               2459877.26500001, 2459877.27416668, 2459877.28333335,
                               2459877.29250001, 2459877.30166668, 2459877.31083335,
                               2459877.32000002, 2459877.32916668, 2459877.33833335,
                               2459877.34750002, 2459877.35666668, 2459877.36583335,
                               2459877.37500002, 2459877.38416668, 2459877.39333335,
                               2459877.40250002, 2459877.41166668, 2459877.42083335,
                               2459877.43000002, 2459877.43916668, 2459877.44833335,
                               2459877.45750002, 2459877.46666668, 2459877.47583335,
                               2459877.48500002, 2459877.49416668, 2459877.50333335,
                               2459877.51250002, 2459877.52166668, 2459877.53083335,
                               2459877.54000002, 2459877.54916668, 2459877.55833335,
                               2459877.56750002, 2459877.57666668, 2459877.58583335,
                               2459877.59500002, 2459877.60416669, 2459877.61333335,
                               2459877.62250002, 2459877.63166669, 2459877.64083335,
                               2459877.65000002, 2459877.65916669, 2459877.66833335,
                               2459877.67750002, 2459877.68666669, 2459877.69583335,
                               2459877.70500002, 2459877.71416669, 2459877.72333335,
                               2459877.73250002, 2459877.74166669, 2459877.75083335,
                               2459877.76000002, 2459877.76916669, 2459877.77833335,
                               2459877.78750002, 2459877.79666669, 2459877.80583335,
                               2459877.81500002, 2459877.82416669, 2459877.83333335,
                               2459877.84250002, 2459877.85166669, 2459877.86083335,
                               2459877.87000002, 2459877.87916669, 2459877.88833336,
                               2459877.89750002, 2459877.90666669, 2459877.91583336,
                               2459877.92500002, 2459877.93416669, 2459877.94333336,
                               2459877.95250002, 2459877.96166669, 2459877.97083336,
                               2459877.98000002, 2459877.98916669, 2459877.99833336])

    times = simulator.time_simulation(2459876, 2459878, 0.22, 100.0)
    assert np.allclose(times, [])


def test_moon_illumination():
    from astropy.coordinates import SkyCoord
    sun = SkyCoord(10, 20, unit="deg")
    moon = SkyCoord(110, -25, unit="deg")
    illu = simulator.moon_illumination(sun, moon)
    assert np.allclose(illu.value, 0.9019377373113325)


def test_simulate_microlensing_model_parameters():
    from pyLIMA.models import FSPLmodel

    event = simulator.simulate_a_microlensing_event()

    Model = FSPLmodel(event, origin=['nothere', [9, 8]])

    params = simulator.simulate_microlensing_model_parameters(Model)

    assert len(params) == 4


def test_simulate_fluxes_parameters():
    params = simulator.simulate_fluxes_parameters([1, 2], source_magnitude=[
        10, 20], blend_magnitude=[10, 20])

    assert len(params) == 4
    assert np.all(np.array(params) > 0)


def test_simulate_lightcurve():
    from pyLIMA.models import PSPLmodel

    event = simulator.simulate_a_microlensing_event()
    times = np.arange(2458679, 2458682, 1.23)

    telo = simulator.simulate_a_telescope("Fake", timestamps=times,
                                          location='Space', spacecraft_name='HST',
                                          camera_filter='I',
                                          photometry=True,
                                          astrometry=True, pixel_scale=40)
    event.telescopes.append(telo)

    Model = PSPLmodel(event, parallax=['Full', 2458680])

    params = [2458680.125730561, 0.7404352139471506, 70.9118088601263,
              5.2750230072835835, 5.116615565480899, -14.371514980726516,
              -1.4679288149463403, -27.608116862181845, 75.96297526485144,
              0.32267656031739733, 0.4617331678523575, 34513.70336974816,
              154.19295132685218]

    pym = Model.compute_pyLIMA_parameters(params)

    simulator.simulate_lightcurve(Model, pym, add_noise=False)

    assert np.allclose(telo.lightcurve['time'].value,
                       [2458679., 2458680.23, 2458681.46])

    assert np.allclose(telo.lightcurve['flux'].value,
                       [21326.85145092, 21336.03588373, 21323.47451914])

    assert np.allclose(telo.lightcurve['err_flux'].value,
                       [146.03715777, 146.06859992, 146.02559542])

    simulator.simulate_lightcurve(Model, pym, add_noise=True)
    assert np.allclose(telo.lightcurve['time'].value,
                       [2458679., 2458680.23, 2458681.46])

    assert np.all(telo.lightcurve['flux'].value !=
                  [21326.85145092, 21336.03588373, 21323.47451914])

    assert np.all(telo.lightcurve['err_flux'].value !=
                  [146.03715777, 146.06859992, 146.02559542])


def test_simulate_astrometry():
    from pyLIMA.models import PSPLmodel

    event = simulator.simulate_a_microlensing_event()
    times = np.arange(2458679, 2458682, 1.23)

    telo = simulator.simulate_a_telescope("Fake", timestamps=times,
                                          location='Space', spacecraft_name='HST',
                                          camera_filter='I',
                                          photometry=True,
                                          astrometry=True, pixel_scale=40)

    event.telescopes.append(telo)

    Model = PSPLmodel(event, parallax=['Full', 2458680])

    params = [2458680.125730561, 0.7404352139471506, 70.9118088601263,
              5.2750230072835835, 5.116615565480899, -14.371514980726516,
              -1.4679288149463403, -27.608116862181845, 75.96297526485144,
              0.32267656031739733, 0.4617331678523575, 34513.70336974816,
              154.19295132685218]
    pym = Model.compute_pyLIMA_parameters(params)

    simulator.simulate_astrometry(Model, pym, add_noise=False)

    assert np.allclose(telo.astrometry['time'].value,
                       [2458679., 2458680.23, 2458681.46])

    assert np.allclose(telo.astrometry['ra'].value,
                       [75.9629745, 75.96297447, 75.96297443])

    assert np.allclose(telo.astrometry['err_ra'].value,
                       [0.75962975, 0.75962974, 0.75962974])

    assert np.allclose(telo.astrometry['dec'].value,
                       [-27.60811665, -27.60811667, -27.60811669])

    assert np.allclose(telo.astrometry['err_dec'].value,
                       [-0.27608117, -0.27608117, -0.27608117])

    simulator.simulate_lightcurve(Model, pym, add_noise=True)

    assert np.allclose(telo.astrometry['time'].value,
                       [2458679., 2458680.23, 2458681.46])

    assert np.all(telo.astrometry['ra'].value !=
                  [75.9629745, 75.96297447, 75.96297443])

    assert np.all(telo.astrometry['err_ra'].value !=
                  [0.75962975, 0.75962974, 0.75962974])

    assert np.all(telo.astrometry['dec'].value !=
                  [-27.60811665, -27.60811667, -27.60811669])

    assert np.all(telo.astrometry['err_dec'].value !=
                  [-0.27608117, -0.27608117, -0.27608117])
