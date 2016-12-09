import collections

import mock
import numpy as np
import pytest

from pyLIMA import microlsimulator

def test_poisson_noise():

    flux = np.array([150,10**5.2,0.1])
    error = flux**0.5
    assert np.allclose(microlsimulator.poisson_noise(flux),error)

def test_noisy_observations():

    flux = np.array([150,10**5.2,0.1])
    error = flux**0.5

    flux_observed = np.random.normal(flux, error)

    assert np.all(np.abs(microlsimulator.noisy_observations(flux,error)-flux_observed)<10*error) #10 sigma diff allowed

def test_time_simulation():

    time_start = 0
    time_end = 10

    sampling = 2.5

    bad_weather = 0.0

    time = microlsimulator.time_simulation(time_start,time_end,sampling,bad_weather)

    assert len(time) == 90

def test_red_noise():

    time = np.linspace(0,100,30)

    red_noise = microlsimulator.red_noise(time)

    assert len(red_noise) == 30
    assert max(red_noise)<10*0.5/100.0

def test_simulate_a_microlensing_event():


    event = microlsimulator.simulate_a_microlensing_event()

    assert event.ra == 270
    assert event.ded == -30
    assert event.name == 'Microlensing pyLIMA simulation'

def test_simulate_a_telescope():

    event = mock.MagicMock()
    telescope = microlsimulator.simulate_a_telescope('blabli',51,51,51,'orange',0,42,1.0,event,'Space')


    assert len(telescope.lightcurve_flux[:,0]) == 1008.0