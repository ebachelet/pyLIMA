import numpy as np
import astropy
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy.time import Time

import matplotlib.pyplot as plt

from pyLIMA import microlmodels
from pyLIMA import microltoolbox
from pyLIMA import telescopes
from pyLIMA import event
from pyLIMA import microlmagnification

RED_NOISE = 'Yes'
SOURCE_MAGNITUDE = [14, 22]
BLEND_LIMITS = [0, 1]
EXPOSURE_TIME = 50 #seconds

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


def poisson_noise(flux):
    """The Poisson noise.

        :param array_like flux: the observed flux

        :return: a numpy array which represents the Poisson noise,

        :rtype: array_like

    """
    error_flux = flux ** 0.5

    return error_flux


def noisy_observations(flux, error_flux):
    """Add Poisson noise to observations.

        :param array_like flux: the observed flux
        :param array_like error_flux: the error on observed flux

        :return: a numpy array which represents the observed noisy flux

        :rtype: array_like

    """
    try:
       
        flux_observed = np.random.poisson(flux)

    except:

        flux_observed = flux

    return flux_observed


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


def red_noise(time):
    """ Simulate red moise as a sum of 10 low amplitudes/period sinusoidals.

        :param array_like time: the time in JD where you simulate red noise

        :return: a numpy array which represents the red noise

        :rtype: array_like

    """

    red_noise_amplitude = np.random.random_sample(10) * 0.5 / 100
    red_noise_period = np.random.random_sample(10)
    red_noise_phase = np.random.random_sample(10) * 2 * np.pi

    red_noise = 0
    for j in range(10):
        red_noise += np.sin(2 * np.pi * time / red_noise_period[j] + red_noise_phase[j]) * red_noise_amplitude[j]

    return red_noise


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
                         minimum_alt=20, moon_windows_avoidance=20, maximum_moon_illumination=100.0):
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
    #import pdb; pdb.set_trace()

    # fake lightcurve
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

    lightcurveflux = np.ones((len(time_of_observations), 3)) * 42
    lightcurveflux[:, 0] = time_of_observations

    telescope = telescopes.Telescope(name=name, camera_filter=filter, light_curve_flux=lightcurveflux,
                                     location=location, spacecraft_name=spacecraft_name)

    return telescope


def simulate_a_microlensing_model(event, model='PSPL', args=(), parallax=['None', 0.0], xallarap=['None'],
                                  orbital_motion=['None', 0.0], source_spots='None'):
    """ Simulate a a microlensing model.

    :param object event: the microlensing event you look at. More details in event module
    :param str model: the microlensing model you want. Default is 'PSPL'. More details in microlmodels module
    :param array_like parallax: the parallax effect you want to add. Default is no parallax.
                                     More details in microlmodels module
    :param array_like xallarap: the xallarap effect you want to add. Default is no parallax.
                                     More details in microlmodels module
    :param str source_spots: If you want to add source spots. Default is no source_spots.
                                     More details in microlmodels module

    :return: a microlmodel object
    :rtype: object
    """

    fake_model = microlmodels.create_model(model, event, args, parallax, xallarap,
                                           orbital_motion, source_spots)
    fake_model.define_model_parameters()

    return fake_model


def simulate_microlensing_model_parameters(model):
    """ Simulate parameters given the desired model. Parameters are selected in uniform distribution inside
        parameters_boundaries given by the microlguess modules. The exception is 'to' where it is selected
        to enter inside telescopes observations.

        :param object event: the microlensing event you look at. More details in event module


        :return: fake_parameters, a set of parameters
        :rtype: list
    """

    fake_parameters = []

    for key in list(model.pyLIMA_standards_dictionnary.keys())[:len(model.parameters_boundaries)]:

        if key == 'to':

            minimum_acceptable_time = max([min(i.lightcurve_flux[:, 0]) for i in model.event.telescopes])
            maximum_acceptable_time = min([max(i.lightcurve_flux[:, 0]) for i in model.event.telescopes])

            fake_parameters.append(np.random.uniform(minimum_acceptable_time, maximum_acceptable_time))

        else:

            boundaries = model.parameters_boundaries[model.pyLIMA_standards_dictionnary[key]]
            fake_parameters.append(np.random.uniform(boundaries[0], boundaries[1]))

    if model.model_type == 'FSPL':
        if np.abs(fake_parameters[1]) > 0.1:
            fake_parameters[1] /= 100
        if np.abs(fake_parameters[1] / fake_parameters[3]) > 10:
            fake_parameters[1] = np.abs(fake_parameters[1]) * np.random.uniform(0, fake_parameters[3])

    if model.model_type == 'DSPL':

        if np.abs(fake_parameters[2]) > 100:
            fake_parameters[2] = np.random.uniform(10, 15)

    return fake_parameters


def simulate_fluxes_parameters(list_of_telescopes):
    """ Simulate flux parameters (magnitude_source , g) for the telescopes. More details in microlmodels module

    :param list list_of_telescopes: a list of telescopes object

    :return: fake_fluxes parameters, a set of fluxes parameters
    :rtype: list

    """

    fake_fluxes_parameters = []

    for telescope in list_of_telescopes:
        magnitude_source = np.random.uniform(SOURCE_MAGNITUDE[0], SOURCE_MAGNITUDE[1])
        flux_source = microltoolbox.magnitude_to_flux(magnitude_source)
        blending_ratio = np.random.uniform(BLEND_LIMITS[0], BLEND_LIMITS[1])

        fake_fluxes_parameters.append(flux_source)
        fake_fluxes_parameters.append(blending_ratio)

    return fake_fluxes_parameters


def simulate_lightcurve_flux(model, pyLIMA_parameters, red_noise_apply='Yes'):
    """ Simulate the flux of telescopes given a model and a set of parameters.
    It updates straight the telescopes object inside the given model.

    :param object model: the microlensing model you desire. More detail in microlmodels.
    :param object pyLIMA_parameters: the parameters used to simulate the flux.
    :param str red_noise_apply: to include or not red_noise

    """

    count = 0

    for telescope in model.event.telescopes:

        theoritical_flux = model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[0]
        if np.min(theoritical_flux > 0):
            pass
        else:
            microlmagnification.VBB.Tol = 0.0005
            microlmagnification.VBB.RelTol = 0.0005
            theoritical_flux = model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[0]

            microlmagnification.VBB.Tol = 0.001
            microlmagnification.VBB.RelTol = 0.001
        flux_error = poisson_noise(theoritical_flux)
        observed_flux = noisy_observations(theoritical_flux*EXPOSURE_TIME, flux_error)

        if red_noise_apply == 'Yes':
            red = red_noise(telescope.lightcurve_flux[:, 0])

            redded_flux = (1 - np.log(10) / 2.5 * red) * observed_flux
            error_on_redded_flux = poisson_noise(redded_flux)

        else:

            redded_flux = observed_flux
            error_on_redded_flux = poisson_noise(redded_flux)

        redded_flux = redded_flux/EXPOSURE_TIME
        error_on_redded_flux = error_on_redded_flux/EXPOSURE_TIME
        telescope.lightcurve_flux[:, 1] = redded_flux
        telescope.lightcurve_flux[:, 2] = error_on_redded_flux

        telescope.lightcurve_magnitude = telescope.lightcurve_in_magnitude()

        count += 1
