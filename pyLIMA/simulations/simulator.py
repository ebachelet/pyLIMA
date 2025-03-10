import astropy
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body
from astropy.time import Time
from pyLIMA.priors import parameters_boundaries
from pyLIMA.toolbox import brightness_transformation

from pyLIMA import event
from pyLIMA import telescopes


def simulate_a_microlensing_event(name='Microlensing pyLIMA simulation', ra=270,
                                  dec=-30):
    """
    Function to find initial DSPL guess

    Parameters
    ----------
    name : str, event name
    ra : float, the event right ascension
    dec : float, the event dec

    Returns
    -------
    fake_event : object, an event object
    """
    fake_event = event.Event(ra=ra, dec=dec)
    fake_event.name = name

    return fake_event


def simulate_a_telescope(name, time_start=2460000, time_end=2460500, sampling=0.25,
                         uniform_sampling=False, timestamps=[], location='Earth',
                         spacecraft_name=None,
                         spacecraft_positions={'astrometry': [], 'photometry': []},
                         camera_filter='I', altitude=0, longitude=0, latitude=0,
                         bad_weather_percentage=0.0,
                         minimum_alt=20, moon_windows_avoidance=20,
                         maximum_moon_illumination=100.0, photometry=True,
                         astrometry=True, pixel_scale=100, ra=270, dec=-30):
    """
    Simulate a telescope. Can mimic real observations (Moon and Sun avoidance,
    bad weather etc...), having uniform sampling or custom timerange.

    Parameters
    ----------
    name : str, event name
    time_start : float, the JD time start of observations
    time_end : float, the JD time end of observations
    sampling : float, the sampling rate (in days)
    uniform_sampling : bool, turn on/off any observational constraints
    timestamps : array, an array of time
    location : str, Earth or Space
    spacecraft_name : str, the name of the satellite
    spacecraft_positions : dict, give the JPL Horizons positions
    camera_filter : str, the filter of observations
    altitude : float, the telescope altitude in m
    longitude : float, the telescope longitude
    latitude : float, the telescope latitde
    bad_weather_percentage : float, fraction of nights lost due to bad weather
    minimum_alt : float, minimum altitude of observations in degrees
    moon_windows_avoidance : float, minimum distance to the Moon in degrees
    maximum_moon_illumination : float, maximum allowed Moon brightness
    photometry : bool, simulate photometric observations
    astrometry : bool, simulate astrometric observations
    pixel_scale : float, the pixel scale of the camera in mas/pix
    ra : float, right ascension of the target in degrees
    dec : float, declination of the target in degrees

    Returns
    -------
    telescope : object, a telescope object
    """
    if len(timestamps) == 0:

        if (uniform_sampling is False) & (location != 'Space'):

            earth_location = EarthLocation(lon=longitude * astropy.units.deg,
                                           lat=latitude * astropy.units.deg,
                                           height=altitude * astropy.units.m)

            target = SkyCoord(ra, dec, unit='deg')

            minimum_sampling = sampling

            time_of_observations = time_simulation(time_start, time_end,
                                                   minimum_sampling,
                                                   bad_weather_percentage)

            time_convertion = Time(time_of_observations, format='jd').isot

            telescope_altaz = target.transform_to(
                AltAz(obstime=time_convertion, location=earth_location))
            altazframe = AltAz(obstime=time_convertion, location=earth_location)
            Sun = get_sun(Time(time_of_observations, format='jd')).transform_to(
                altazframe)
            Moon = get_body("moon",Time(time_of_observations, format='jd')).transform_to(
                altazframe)
            Moon_illumination = moon_illumination(Sun, Moon)
            Moon_separation = target.separation(Moon)

            observing_windows = \
                np.where((telescope_altaz.alt > minimum_alt * astropy.units.deg)
                         & (Sun.alt < -18 * astropy.units.deg)
                         & (
                                 Moon_separation > moon_windows_avoidance *
                                 astropy.units.deg)
                         & (Moon_illumination < maximum_moon_illumination)
                         )[0]

            time_of_observations = time_of_observations[observing_windows]


        else:

            time_of_observations = np.arange(time_start, time_end, sampling / 24.0)

    else:

        time_of_observations = np.array(timestamps)

    if photometry & (len(time_of_observations) > 0):

        lightcurveflux = np.ones((len(time_of_observations), 3)) * 42
        lightcurveflux[:, 0] = time_of_observations

    else:

        lightcurveflux = None

    if astrometry:

        astrometry = np.ones((len(time_of_observations), 5)) * 42
        astrometry[:, 0] = time_of_observations

    else:

        astrometry = None

    telescope = telescopes.Telescope(name=name, camera_filter=camera_filter,
                                     pixel_scale=pixel_scale,
                                     lightcurve=lightcurveflux,
                                     lightcurve_names=['time', 'flux', 'err_flux'],
                                     lightcurve_units=['JD', 'w/m^2', 'w/m^2'],
                                     astrometry=astrometry,
                                     astrometry_names=['time', 'ra', 'err_ra', 'dec',
                                                       'err_dec'],
                                     astrometry_units=['JD', 'deg', 'deg', 'deg',
                                                       'deg'],
                                     location=location, spacecraft_name=spacecraft_name,
                                     spacecraft_positions=spacecraft_positions)
    return telescope


def time_simulation(time_start, time_end, sampling, bad_weather_percentage):
    """
    Simulate the timestamps

    Parameters
    ----------
    time_start : float, the JD time start of observations
    time_end : float, the JD time end of observations
    bad_weather_percentage : float, fraction of nights lost due to bad weather

    Returns
    -------
    time_of_observations : array, an array of time     """

    time_initial = np.arange(time_start, time_end, sampling / 24.)
    total_number_of_days = int(time_end - time_start)

    time_observed = []
    night_begin = time_start

    for i in range(total_number_of_days):

        good_weather = np.random.uniform(0, 1)

        if good_weather > bad_weather_percentage:

            mask = (time_initial >= night_begin) & (time_initial < night_begin + 1)
            time_observed = np.append(time_observed, time_initial[mask])

        else:

            pass

        night_begin += 1

    time_of_observations = np.array(time_observed)

    return time_of_observations


def moon_illumination(sun, moon):
    """
    Compute the Moon illuminations

    Parameters
    ----------
    sun : array, SkyCoord of the Sun
    moono : array, SkyCoord of the Moon


    Returns
    -------
    illumniation : array, the Moon illumination
    """

    geocentric_elongation = sun.separation(moon).rad
    selenocentric_elongation = np.arctan2(sun.distance * np.sin(geocentric_elongation),
                                          moon.distance - sun.distance * np.cos(
                                              geocentric_elongation))

    illumination = (1 + np.cos(selenocentric_elongation)) / 2.0

    return illumination


def simulate_microlensing_model_parameters(model):
    """
    Given a microlensing model, compute a random parameters (uniform distribution in
    the bounds)

    Parameters
    ----------
    model : object, a microlensing model

    Returns
    -------
    fake_parameters : list, a list of simulated parameters
    """

    model.define_model_parameters()
    boundaries = parameters_boundaries.parameters_boundaries(model.event,
                                                             model.model_dictionnary)

    fake_parameters = []

    for ind, key in enumerate(model.model_dictionnary.keys()):

        try:

            if 'fsource' in key:
                break

            fake_parameters.append(
                np.random.uniform(boundaries[ind][0], boundaries[ind][1]))

        except AttributeError:

            pass

    fake_fluxes_parameters = simulate_fluxes_parameters(model.event.telescopes,
                                                        source_magnitude=[10, 20],
                                                        blend_magnitude=[19, 22])
    fake_parameters += fake_fluxes_parameters

    # t_0 limit fix
    mins_time = []
    maxs_time = []

    for telescope in model.event.telescopes:

        if telescope.lightcurve is not None:
            mins_time.append(np.min(telescope.lightcurve['time'].value))
            maxs_time.append(np.max(telescope.lightcurve['time'].value))

        if telescope.astrometry is not None:
            mins_time.append(np.min(telescope.astrometry['time'].value))
            maxs_time.append(np.max(telescope.astrometry['time'].value))

    try:

        fake_parameters[0] = np.random.uniform(np.min(mins_time), np.max(maxs_time))

    except ValueError:

        pass

    if model.parallax_model[0] != 'None':
        fake_parameters[0] = np.random.uniform(model.parallax_model[1] - 1,
                                               model.parallax_model[1] + 1)


    if model.astrometry:

        for telescope in model.event.telescopes:
            if telescope.astrometry is not None:

                fake_parameters[model.model_dictionnary['position_source_E_'+
                                telescope.name]] = model.event.ra
                fake_parameters[model.model_dictionnary['position_source_N_' +
                                telescope.name]] = model.event.dec

    return fake_parameters


def simulate_fluxes_parameters(list_of_telescopes, source_magnitude=[10, 20],
                               blend_magnitude=[10, 20]):
    """
    Compute the source and blend fluxes for a list of telescopes

    Parameters
    ----------
    list_of_telescopes : list, a list of telescope objects
    source_magnitude : list, [mag_min,max_max] range of the source magnitudes
    blend_magnitude : list, [mag_min,max_max] range of the blend magnitudes

    Returns
    -------
    fake_fluxes_telescopes : list, a list of 2*Ntelescopes fluxes
    """
    fake_fluxes_parameters = []

    for telescope in list_of_telescopes:
        magnitude_source = np.random.uniform(source_magnitude[0], source_magnitude[1])
        flux_source = brightness_transformation.magnitude_to_flux(magnitude_source)

        magnitude_blend = np.random.uniform(blend_magnitude[0], blend_magnitude[1])
        flux_blend = brightness_transformation.magnitude_to_flux(magnitude_blend)

        fake_fluxes_parameters.append(flux_source)
        fake_fluxes_parameters.append(flux_blend)

    return fake_fluxes_parameters


def simulate_lightcurve(model, pyLIMA_parameters, add_noise=True,efficiency=None):
    """
    Simulate the fluxes in the telescopes according to the model and parameters

    Parameters
    ----------
    model : object, a microlensing model object
    pyLIMA_parameters : dict, a pyLIMA_parameters object
    add_noise : bool, adding Poisson noise or not
    """

    for ind, telescope in enumerate(model.event.telescopes):

        if telescope.lightcurve is not None:

            theoritical_flux = \
                model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[
                    'photometry']


            if add_noise:

                # if exposure_times is not None:
                #   exp_time = exposure_times[ind]

                observed_flux, err_observed_flux = \
                    brightness_transformation.noisy_observations(
                        theoritical_flux)

            else:

                observed_flux = theoritical_flux
                err_observed_flux = theoritical_flux ** 0.5

            telescope.lightcurve['flux'] = observed_flux
            telescope.lightcurve['err_flux'] = err_observed_flux
            telescope.lightcurve['inv_err_flux'] = 1/err_observed_flux

            lightcurve_magnitude =  telescope.lightcurve_in_magnitude(telescope.lightcurve)
            telescope.lightcurve['mag'] = lightcurve_magnitude[:,1]
            telescope.lightcurve['err_mag'] = lightcurve_magnitude[:,2]


    model.define_pyLIMA_standard_parameters()

def simulate_astrometry(model, pyLIMA_parameters, add_noise=True):
    """
    Simulate the astrometric signal in the telescopes according to the model and
    parameters

    Parameters
    ----------
    model : object, a microlensing model object
    pyLIMA_parameters : dict, a pyLIMA_parameters object
    add_noise : bool, adding Poisson noise or not
    """
    from astropy import units as unit

    for telescope in model.event.telescopes:

        if telescope.astrometry is not None:

            theoritical_model = model.compute_the_microlensing_model(telescope,
                                                                     pyLIMA_parameters)

            theoritical_flux = theoritical_model['photometry']
            theoritical_astrometry = theoritical_model['astrometry']

            if add_noise:

                observed_flux, err_observed_flux = \
                    brightness_transformation.noisy_observations(
                        theoritical_flux)

                SNR = observed_flux / err_observed_flux

                err_ra = 1 / SNR / 3600.  # assuming FWHM=1 as
                err_dec = 1 / SNR / 3600.

                obs_ra = np.random.normal(theoritical_astrometry[0], err_ra)
                obs_dec = np.random.normal(theoritical_astrometry[1], err_dec)

            else:

                obs_ra = theoritical_astrometry[0]
                err_ra = theoritical_astrometry[0] * 0.01
                obs_dec = theoritical_astrometry[1]
                err_dec = theoritical_astrometry[1] * 0.01

            telescope.astrometry['ra'] = obs_ra * unit.deg
            telescope.astrometry['err_ra'] = err_ra * unit.deg
            telescope.astrometry['dec'] = obs_dec * unit.deg
            telescope.astrometry['err_dec'] = err_dec * unit.deg
    model.define_pyLIMA_standard_parameters()
