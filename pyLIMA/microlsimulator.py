import numpy as np
import microlmodels
import microltoolbox
import telescopes
import event

MOON_PERIOD = 29  # days
RED_NOISE = 'Yes'
def gaussian_noise(flux):

    error_flux = flux**0.5

    return error_flux

def noisy_observations(flux, error_flux):

    flux_observed = np.random.nornal(flux, error_flux)

    return flux_observed

def time_simulation(time_start, time_end, sampling, observing_windows, bad_weather_percentage,
                    moon_windows_avoidance):
    total_number_of_days = int(time_end - time_start)
    time_step_observations = observing_windows / sampling
    night_begin = time_start

    moon_phase = 0
    time_observed = []
    for i in xrange(total_number_of_days):

        good_weather = np.random.uniform(0, 1)

        if good_weather > bad_weather_percentage:

            if np.abs(moon_phase - MOON_PERIOD) > moon_windows_avoidance:
                night_end = night_begin + observing_windows
                time_observed += np.arange(night_begin, night_end, time_step_observations).tolist()

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


def simulate_a_microlensing_event(ra=270, dec=-30):
    # fake event

    fake_event = event.Event()
    fake_event.ra = ra
    fake_event.dec = dec

    return fake_event


def simulate_a_telescope(name, altitude, longitude, latitude, filter, time_start, time_end, sampling, observing_windows,
                         bad_weather_percentage=0.0, moon_windows_avoidance=2):
    # fake lightcurve
    time_of_observations = time_simulation(time_start, time_end, sampling, observing_windows,
                                           bad_weather_percentage,
                                           moon_windows_avoidance)
    lightcurveflux = np.zeros((len(time_of_observations), 3))
    lightcurveflux[:, 0] = time_of_observations

    telescope = telescopes.Telescope(name='SIMULATOR', light_curve_flux=lightcurveflux)

    return telescope


def simulate_a_microlensing_model(event, model='PSPL', parallax=['None', 0.0], xallarap=['None', 0.0],
                                  orbital_motion=['None', 0.0], source_spots='None'):

    fake_model = microlmodels.create_model(model,event , parallax, xallarap,
                                  orbital_motion, source_spots)

    return fake_model


def simulate_microlensing_model_parameters(model, parameters = None):

    fake_parameters = []
    if parameters == None:

        for boundaries in model.parameters_boundaries:

            fake_parameters.append(np.random.uniform(boundaries[0],boundaries[1]))

    else:

        fake_parameters = parameters

    return fake_parameters

def simulate_fluxes_parameters(telescopes, fluxes_parameters):

    fake_fluxes_parameters = []
    if fluxes_parameters == None:

        for telescope in telescopes :

                magnitude_source = np.random.uniform(18,22)
                blending_ratio = np.random.uniform(0.0,100)

    else :

        fake_fluxes_parameters = fluxes_parameters

    return fake_fluxes_parameters
#Create your event

my_own_creation = simulate_a_microlensing_event(ra=270, dec=-30)

# Create some telescopes
my_own_telescope = simulate_a_telescope('MOUAHAHAHAH', 0, 0, 0, 'I', 0.0, 250, 35, 0.33,
                         bad_weather_percentage=50.0, moon_windows_avoidance=6)
#Add them to your event
my_own_creation.telescopes.append(my_own_telescope)
import pdb;

pdb.set_trace()

#What model you want?
my_own_model = simulate_a_microlensing_model( my_own_creation, model='PSPL', parallax=['None', 0.0], xallarap=['None', 0.0],
                                  orbital_motion=['None', 0.0], source_spots='None')

# Find some model parameters
my_own_parameters = simulate_microlensing_model_parameters(my_own_model)

#Transform into pyLIMA standards
pyLIMA_parameters =my_own_model.compute_pyLIMA_parameters(my_own_parameters)


#Which source magnitude? Which blending?

my_own_flux_parameters = simulate_microlensing_model_parameters(event.telescopes)

# update the lightcurves in your event :
count = 0
for telescope in my_own_creation.telescopes:


    amplification = my_own_model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[0]

    flux_source = microltoolbox.magnitude_to_flux(my_own_flux_parameters[count][0])
    blending_ratio = my_own_flux_parameters[count][1]

    theoritical_flux = flux_source*(amplification+blending_ratio)

    flux_error = gaussian_noise(theoritical_flux)


    observed_flux = noisy_observations(theoritical_flux, flux_error)

    if RED_NOISE == 'Yes' :

        red = red_noise(telescope.lightcurve_flux[:,0])

        redded_flux = (1-np.log(10)/2.5*red)*observed_flux
        error_on_redded_flux = gaussian_noise(redded_flux)

        telescope.lightcurve_flux[:,1] = redded_flux
        telescope.lightcurve_flux[:,2] = error_on_redded_flux
    count += 1

import pdb;

pdb.set_trace()