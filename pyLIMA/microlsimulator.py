import numpy as np
import astropy
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
import matplotlib.pyplot as plt

import microlmodels
import microltoolbox
import telescopes
import event

MOON_PERIOD = 29  # days
RED_NOISE = 'Yes'
SOURCE_MAGNITUDE = [14, 22]
BLEND_LIMITS = [0, 1]


def gaussian_noise(flux):
    error_flux = flux ** 0.5

    return error_flux


def noisy_observations(flux, error_flux):
    flux_observed = np.random.normal(flux, error_flux)

    return flux_observed


def time_simulation(time_start, time_end, sampling, bad_weather_percentage,
                    moon_windows_avoidance):
    total_number_of_days = int(time_end - time_start)
    time_step_observations = 1.0 / sampling
    night_begin = time_start

    moon_phase = 0
    time_observed = []
    for i in xrange(total_number_of_days):

        good_weather = np.random.uniform(0, 1)

        if good_weather > bad_weather_percentage:

            if np.abs(moon_phase - MOON_PERIOD) > moon_windows_avoidance:
                random_begin_of_the_night = np.random.uniform(0, 1)
                night_end = night_begin + 1.0
                time_observed += np.arange(night_begin + random_begin_of_the_night, night_end,
                                           time_step_observations).tolist()

        night_begin += 1
        moon_phase += 1

        if moon_phase == MOON_PERIOD:
            moon_phase = 0

    time_of_observations = np.array(time_observed)

    return time_of_observations


def red_noise(time):
    # Sum of low period/ low amplitudes sinusoidales. Computed in magnitude, return in flux unit.

    red_noise_amplitude = np.random.random_sample(10) * 0.5 / 100
    red_noise_period = np.random.random_sample(10)
    red_noise_phase = np.random.random_sample(10) * 2 * np.pi

    red_noise = 0
    for j in xrange(10):
        red_noise += np.sin(2 * np.pi * time / red_noise_period[j] + red_noise_phase[j]) * red_noise_amplitude[j]

    return red_noise


def simulate_a_microlensing_event(name, ra=270, dec=-30):
    # fake event

    fake_event = event.Event()
    fake_event.name = name
    fake_event.ra = ra
    fake_event.dec = dec

    return fake_event


def simulate_a_telescope(name, altitude, longitude, latitude, filter, time_start, time_end, sampling,
                         bad_weather_percentage=0.0, moon_windows_avoidance=2, minimum_alt=20):
    # fake lightcurve

    earth_location = EarthLocation(lon=longitude * astropy.units.deg,
                                   lat=latitude * astropy.units.deg,
                                   height=altitude * astropy.units.m)
    target = SkyCoord(270, -30, unit='deg')

    time_of_observations = time_simulation(time_start, time_end, sampling,
                                           bad_weather_percentage,
                                           moon_windows_avoidance)

    time_convertion = Time(time_of_observations, format='jd').isot

    telescope_altaz = target.transform_to(AltAz(obstime=time_convertion, location=earth_location))

    observing_windows = np.where(telescope_altaz.alt > minimum_alt * astropy.units.deg)[0]

    time_of_observations = time_of_observations[observing_windows]

    lightcurveflux = np.zeros((len(time_of_observations), 3))
    lightcurveflux[:, 0] = time_of_observations

    telescope = telescopes.Telescope(name=name, camera_filter=filter, light_curve_flux=lightcurveflux)

    return telescope


def simulate_a_microlensing_model(event, model='PSPL', parallax=['None', 0.0], xallarap=['None', 0.0],
                                  orbital_motion=['None', 0.0], source_spots='None'):
    fake_model = microlmodels.create_model(model, event, parallax, xallarap,
                                           orbital_motion, source_spots)
    fake_model.define_model_parameters()

    return fake_model


def simulate_microlensing_model_parameters(model):
    fake_parameters = []

    for key in model.pyLIMA_standards_dictionnary.keys()[:len(model.parameters_boundaries)]:

        if key == 'to':

            minimum_acceptable_time = max([min(i.lightcurve_flux[:, 0]) for i in model.event.telescopes])
            maximum_acceptable_time = min([max(i.lightcurve_flux[:, 0]) for i in model.event.telescopes])

            fake_parameters.append(np.random.uniform(minimum_acceptable_time, maximum_acceptable_time))

        else:

            boundaries = model.parameters_boundaries[model.pyLIMA_standards_dictionnary[key]]
            fake_parameters.append(np.random.uniform(boundaries[0], boundaries[1]))

    if model.model_type == 'FSPL':

        if np.abs(fake_parameters[1]/fake_parameters[3])>10:

            fake_parameters[1] = np.abs(fake_parameters[1])*np.random.uniform(0, fake_parameters[3])

    if model.model_type == 'DSPL':

        if np.abs(fake_parameters[2])>100 :

            fake_parameters[2] = np.random.uniform(10,15)

        if np.abs(fake_parameters[1]+fake_parameters[3]) > 0.05 :

                fake_parameters[3] = -fake_parameters[1]+0.05
    return fake_parameters


def simulate_fluxes_parameters(telescopes):
    fake_fluxes_parameters = []

    for telescope in telescopes:
        magnitude_source = np.random.uniform(SOURCE_MAGNITUDE[0], SOURCE_MAGNITUDE[1])
        flux_source = microltoolbox.magnitude_to_flux(magnitude_source)
        blending_ratio = np.random.uniform(BLEND_LIMITS[0], BLEND_LIMITS[1])

        fake_fluxes_parameters.append(flux_source)
        fake_fluxes_parameters.append(blending_ratio)


    return fake_fluxes_parameters


def simulate_lightcurve_flux(model, pylima_parameters, red_noise_apply='Yes'):
    count = 0

    for telescope in model.event.telescopes:

        theoritical_flux = model.compute_the_microlensing_model(telescope, pylima_parameters)[0]

        flux_error = gaussian_noise(theoritical_flux)

        observed_flux = noisy_observations(theoritical_flux, flux_error)

        if red_noise_apply == 'Yes':
            red = red_noise(telescope.lightcurve_flux[:, 0])

            redded_flux = (1 - np.log(10) / 2.5 * red) * observed_flux
            error_on_redded_flux = gaussian_noise(redded_flux)

        else:

            redded_flux = observed_flux
            error_on_redded_flux = gaussian_noise(redded_flux)

        telescope.lightcurve_flux[:, 1] = redded_flux
        telescope.lightcurve_flux[:, 2] = error_on_redded_flux

        telescope.lightcurve_magnitude = telescope.lightcurve_in_magnitude()

        count += 1







