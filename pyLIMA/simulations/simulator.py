import numpy as np
import astropy
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy.time import Time

from pyLIMA import telescopes
from pyLIMA import event
from pyLIMA.priors import parameters_boundaries
from pyLIMA.toolbox import brightness_transformation

def simulate_a_microlensing_event(name='Microlensing pyLIMA simulation', ra=270, dec=-30):
    """ Simulate a microlensing event. More details in the event module.

        :param str name:  the name of the event. Default is 'Microlensing pyLIMA simulation'
        :param float ra: the right ascension in degrees of your simulation. Default is 270.
        :param float dec: the declination in degrees of your simulation. Default is -30.


        :return: a event object

        :rtype: object
    """

    fake_event = event.Event()
    fake_event.name = name
    fake_event.ra = ra
    fake_event.dec = dec

    return fake_event

def simulate_a_telescope(name, event, time_start, time_end, sampling, location, filter, uniform_sampling=False,
                         altitude=0, longitude=0, latitude=0, spacecraft_name=None, bad_weather_percentage=0.0,
                         minimum_alt=20, moon_windows_avoidance=20, maximum_moon_illumination=100.0, photometry=True,
                         astrometry=True):
    """ Simulate a telescope. More details in the telescopes module. The observations simulation are made for the
        full time windows, then limitation are applied :
            - Sun has to be below horizon : Sun< -18
            - Moon has to be more than the moon_windows_avoidance distance from the target
            - Observations altitude of the target have to be bigger than minimum_alt

    :param str name:  the name of the telescope.
    :param object event: the microlensing event you look at
    :param float time_start: the start of observations in JD
    :param float time_end: the end of observations in JD
    :param float sampling: the hour sampling.
    :param str location: the location of the telescope.
    :param str filter: the filter used for observations
    :param boolean uniform_sampling: set it to True if you want no bad weather, no moon avoidance etc....

    :param float altitude: the altitude in meters if the telescope
    :param float longitude: the longitude in degree of the telescope location
    :param float latitude: the latitude in degree of the telescope location

    :param str spacecraft_name: the name of your satellite according to JPL horizons

    :param float bad_weather_percentage: the percentage of bad nights
    :param float minimum_alt: the minimum altitude ini degrees that your telescope can go to.
    :param float moon_windows_avoidance: the minimum distance in degrees accepted between the target and the Moon
    :param float maximum_moon_illumination: the maximum Moon brightness you allow in percentage
    :return: a telescope object
    :rtype: object
    """


    if (uniform_sampling == False) & (location != 'Space'):

        earth_location = EarthLocation(lon=longitude * astropy.units.deg,
                                       lat=latitude * astropy.units.deg,
                                       height=altitude * astropy.units.m)

        target = SkyCoord(event.ra, event.dec, unit='deg')

        minimum_sampling = min(4.0, sampling)
        ratio_sampling = np.round(sampling / minimum_sampling)

        time_of_observations = time_simulation(time_start, time_end, minimum_sampling,
                                               bad_weather_percentage)

        time_convertion = Time(time_of_observations, format='jd').isot

        telescope_altaz = target.transform_to(AltAz(obstime=time_convertion, location=earth_location))
        altazframe = AltAz(obstime=time_convertion, location=earth_location)
        Sun = get_sun(Time(time_of_observations, format='jd')).transform_to(altazframe)
        Moon = get_moon(Time(time_of_observations, format='jd')).transform_to(altazframe)
        Moon_illumination = moon_illumination(Sun, Moon)
        Moon_separation = target.separation(Moon)
        observing_windows = np.where((telescope_altaz.alt > minimum_alt * astropy.units.deg)
                                     & (Sun.alt < -18 * astropy.units.deg)
                                     & (Moon_separation > moon_windows_avoidance * astropy.units.deg)
                                     & (Moon_illumination < maximum_moon_illumination)
                                     )[0]

        time_of_observations = time_of_observations[observing_windows]


    else:

        time_of_observations = np.arange(time_start, time_end, sampling / (24.0))

    if photometry:

        lightcurveflux = np.ones((len(time_of_observations), 3)) * 42
        lightcurveflux[:, 0] = time_of_observations

    else:

        lightcurveflux = None

    if astrometry:

        astrometry = np.ones((len(time_of_observations), 5)) * 42
        astrometry[:,0] = time_of_observations

    telescope = telescopes.Telescope(name=name, camera_filter=filter, light_curve=lightcurveflux,
                                     light_curve_names=['time','flux','err_flux'], light_curve_units=['JD','w/m^2','w/m^2'],
                                     clean_the_light_curve=False,
                                     astrometry=astrometry, astrometry_names=['time','delta_ra','err_delta_ra', 'delta_dec','err_delta_dec'],
                                     astrometry_units=['JD','mas','mas','mas','mas'],
                                     location=location, spacecraft_name=spacecraft_name)

    return telescope

def time_simulation(time_start, time_end, sampling, bad_weather_percentage):
    """ Simulate observing time during the observing windows, rejecting windows with bad weather.

    :param float time_start: the start of observations in JD
    :param float time_end: the end of observations in JD
    :param float sampling: the number of points observed per hour.
    :param float bad_weather_percentage: the percentage of bad nights

    :return: a numpy array which represents the time of observations

    :rtype: array_like

    """

    total_number_of_days = int(time_end - time_start)
    time_step_observations = sampling / 24.0
    number_of_day_exposure = int(np.floor(
        1.0 / time_step_observations))  # less than expected total, more likely in a telescope :)
    night_begin = time_start

    time_observed = []
    for i in range(total_number_of_days):

        good_weather = np.random.uniform(0, 1)

        if good_weather > bad_weather_percentage:
            random_begin_of_the_night = 0
            night_end = night_begin + 1
            time_observed += np.linspace(night_begin + time_step_observations + random_begin_of_the_night, night_end,
                                         number_of_day_exposure).tolist()

        night_begin += 1

    time_of_observations = np.array(time_observed)

    return time_of_observations

def moon_illumination(sun, moon):
    """The moon illumination expressed as a percentage.

            :param astropy sun: the sun ephemeris
            :param astropy moon: the moon ephemeris

            :return: a numpy array indicated the moon illumination.

            :rtype: array_like

    """

    geocentric_elongation = sun.separation(moon).rad
    selenocentric_elongation = np.arctan2(sun.distance * np.sin(geocentric_elongation),
                                          moon.distance - sun.distance * np.cos(geocentric_elongation))

    illumination = (1 + np.cos(selenocentric_elongation)) / 2.0

    return illumination


def simulate_microlensing_model_parameters(model):
    """ Simulate parameters given the desired model. Parameters are selected in uniform distribution inside
        parameters_boundaries given by the microlguess modules. The exception is 'to' where it is selected
        to enter inside telescopes observations.

        :param object event: the microlensing event you look at. More details in event module


        :return: fake_parameters, a set of parameters
        :rtype: list
    """

    model.define_model_parameters()
    boundaries = parameters_boundaries.parameters_boundaries(model.event, model.model_dictionnary)

    fake_parameters = []

    for ind, key in enumerate(model.model_dictionnary.keys()):

        try:

            fake_parameters.append(np.random.uniform(boundaries[ind][0],boundaries[ind][1]))

        except:

            pass

        # t_0 limit fix
        mins_time = []
        maxs_time = []

        for telescope in model.event.telescopes:

            if telescope.lightcurve_flux is not None:
                mins_time.append(np.min(telescope.lightcurve_flux['time'].value))
                maxs_time.append(np.max(telescope.lightcurve_flux['time'].value))

            if telescope.astrometry is not None:
                mins_time.append(np.min(telescope.astrometry['time'].value))
                maxs_time.append(np.max(telescope.astrometry['time'].value))

        fake_parameters[0] = np.random.uniform(np.min(mins_time), np.max(maxs_time))

    return fake_parameters

def simulate_fluxes_parameters(list_of_telescopes,source_magnitude = [10,20], blend_magnitude = [10,20]):
    """ Simulate flux parameters (magnitude_source , g) for the telescopes. More details in microlmodels module

    :param list list_of_telescopes: a list of telescopes object

    :return: fake_fluxes parameters, a set of fluxes parameters
    :rtype: list

    """

    fake_fluxes_parameters = []

    for telescope in list_of_telescopes:

        magnitude_source = np.random.uniform(source_magnitude[0], source_magnitude[1])
        flux_source = toolbox.brightness_transformation.magnitude_to_flux(magnitude_source)

        magnitude_blend = magnitude_source = np.random.uniform(blend_magnitude[0], blend_magnitude[1])
        flux_blend = toolbox.brightness_transformation.magnitude_to_flux(magnitude_blend)


        fake_fluxes_parameters.append(flux_source)
        fake_fluxes_parameters.append(flux_blend)

    return fake_fluxes_parameters



def simulate_lightcurve_flux(model, pyLIMA_parameters):
    """ Simulate the flux of telescopes given a model and a set of parameters.
    It updates straight the telescopes object inside the given model.

    :param object model: the microlensing model you desire. More detail in microlmodels.
    :param object pyLIMA_parameters: the parameters used to simulate the flux.
    :param str red_noise_apply: to include or not red_noise

    """

    count = 0

    for telescope in model.event.telescopes:

        theoritical_flux = model.compute_the_microlensing_model(telescope, pyLIMA_parameters)['photometry']

        observed_flux,err_observed_flux = brightness_transformation.noisy_observations(theoritical_flux)

        telescope.lightcurve_flux['flux'] = observed_flux
        telescope.lightcurve_flux['err_flux'] = err_observed_flux

        telescope.lightcurve_magnitude = telescope.lightcurve_in_magnitude()

        count += 1

