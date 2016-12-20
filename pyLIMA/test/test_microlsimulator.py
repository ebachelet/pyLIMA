import collections

import mock
import numpy as np
import pytest

from pyLIMA import microlsimulator


def test_poisson_noise():
    flux = np.array([150, 10 ** 5.2, 0.1])
    error = flux ** 0.5
    assert np.allclose(microlsimulator.poisson_noise(flux), error)


def test_noisy_observations():
    flux = np.array([150, 10 ** 5.2, 0.1])
    error = flux ** 0.5

    flux_observed = np.random.normal(flux, error)

    assert np.all(
        np.abs(microlsimulator.noisy_observations(flux, error) - flux_observed) < 10 * error)  # 10 sigma diff allowed


def test_time_simulation():
    time_start = 0
    time_end = 10

    sampling = 2.5

    bad_weather = 0.0

    time = microlsimulator.time_simulation(time_start, time_end, sampling, bad_weather)

    assert len(time) == 90


def test_red_noise():
    time = np.linspace(0, 100, 30)

    red_noise = microlsimulator.red_noise(time)

    assert len(red_noise) == 30
    assert max(red_noise) < 10 * 0.5 / 100.0


def test_simulate_a_microlensing_event():
    event = microlsimulator.simulate_a_microlensing_event()

    assert event.ra == 270
    assert event.dec == -30
    assert event.name == 'Microlensing pyLIMA simulation'


def test_simulate_a_telescope():
    event = mock.MagicMock()
    telescope = microlsimulator.simulate_a_telescope('blabli', 51, 51, 51, 'orange', 0, 42, 1.0, event, 'Space')

    assert len(telescope.lightcurve_flux[:, 0]) == 42*24

    event.ra = 270
    event.dec = -30

    telescope = microlsimulator.simulate_a_telescope('blabli', 0, 0, -30, 'orange', 2457850, 2457900, 1.0, event,
                                                     'Earth',0,0,0,100)
    
    histogram, windows = np.histogram(telescope.lightcurve_flux[:,0],np.arange(2457850,2457900))
    assert len(telescope.lightcurve_flux[:, 0]) == 461.0
    assert histogram[0]<histogram[-1]

def test_simulate_a_microlensing_model():
    event = mock.MagicMock()
    telescope = mock.MagicMock()
    telescope.name = 'NDG'
    telescope.lightcurve_flux = np.array([[0, 1], [2, 2]]).T
    event.telescopes = []
    event.telescopes.append(telescope)
    print event.telescopes[0].lightcurve_flux[:, 0]

    model = microlsimulator.simulate_a_microlensing_model(event)

    assert model.model_type == 'PSPL'



def test_simulate_microlensing_model_parameters():
    event = mock.MagicMock()
    telescope = mock.MagicMock()
    telescope.name = 'NDG'
    telescope.lightcurve_flux = np.array([[0, 1], [2, 2]]).T
    event.telescopes = []
    event.telescopes.append(telescope)

    model = microlsimulator.simulate_a_microlensing_model(event)

    parameters = microlsimulator.simulate_microlensing_model_parameters(model)

    assert -300 < parameters[0]
    assert parameters[0] < 300

    assert 0 < parameters[1]
    assert parameters[1] < 2

    assert 1 < parameters[2]
    assert parameters[2] < 300

    model = microlsimulator.simulate_a_microlensing_model(event,'FSPL')

    parameters = microlsimulator.simulate_microlensing_model_parameters(model)

    assert parameters[1] < 0.1
    assert parameters[3] < 0.05

    model = microlsimulator.simulate_a_microlensing_model(event, 'DSPL')

    parameters = microlsimulator.simulate_microlensing_model_parameters(model)

    assert parameters[2] < 100

def test_simulate_fluxes_parameters():
    telescope = mock.MagicMock()
    telescopes = [telescope]

    fake_flux_parameters = microlsimulator.simulate_fluxes_parameters(telescopes)

    assert len(fake_flux_parameters) == 2

    assert fake_flux_parameters[0] > 10 ** ((27.4 - 22) / 2.5)
    assert fake_flux_parameters[0] < 10 ** ((27.4 - 14) / 2.5)

    assert fake_flux_parameters[1] < 1
    assert fake_flux_parameters[1] > 0


def test_simulate_lightcurve_flux():
    event = mock.MagicMock()
    telescope = mock.MagicMock()
    telescope.name = 'NDG'
    telescope.lightcurve_flux = np.array([[0, 51, 69], [2, 42, 28.65]])
    event.telescopes = []
    event.telescopes.append(telescope)

    model = microlsimulator.simulate_a_microlensing_model(event)
    parameters = microlsimulator.simulate_microlensing_model_parameters(model)

    fake_flux_parameters = microlsimulator.simulate_fluxes_parameters(event.telescopes)

    pyLIMA_parameters = model.compute_pyLIMA_parameters(parameters + fake_flux_parameters)

    microlsimulator.simulate_lightcurve_flux(model, pyLIMA_parameters, 'No')

    assert np.all(telescope.lightcurve_flux[:, 1] != [51, 42])
    assert np.all(telescope.lightcurve_flux[:, 2] != [69, 28.65])

    microlsimulator.simulate_lightcurve_flux(model, pyLIMA_parameters, 'Yes')

    assert np.all(telescope.lightcurve_flux[:, 1] != [51, 42])
    assert np.all(telescope.lightcurve_flux[:, 2] != [69, 28.65])