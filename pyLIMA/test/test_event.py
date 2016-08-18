# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:19:27 2016

@author: ebachelet
"""
import numpy as np
import mock
import pytest

from pyLIMA import event


def test_check_event_bad_name():
    current_event = event.Event()

    current_event.name = 49.49

    with pytest.raises(event.EventException) as event_exception:
        current_event.check_event()
    assert 'The event name (49.49) is not correct, it has to be a string' in str(event_exception)


def test_check_event_bad_ra():
    current_event = event.Event()

    current_event.ra = -49.49

    with pytest.raises(event.EventException):
        current_event.check_event()


def test_check_event_bad_dec():
    current_event = event.Event()

    current_event.dec = -189

    with pytest.raises(event.EventException):
        current_event.check_event()


def test_check_event_no_telescopes():
    current_event = event.Event()

    with pytest.raises(event.EventException):
        current_event.check_event()


def test_check_event_telescopes_without_lightcurves():
    current_event = event.Event()
    telescope = mock.MagicMock()

    current_event.telescopes.append(telescope)

    with pytest.raises(event.EventException):
        current_event.check_event()


def test_check_event_with_one_telescope_with_magnitude_lightcurve():
    current_event = event.Event()
    telescope = mock.MagicMock()
    telescope.name = 'NDG'
    telescope.lightcurve_magnitude = np.array([0, 36307805477.010025, -39420698921.705284])
    current_event.telescopes.append(telescope)

    current_event.check_event()


def test_check_event_with_one_telescope_with_flux_lightcurve():
    current_event = event.Event()
    telescope = mock.MagicMock()
    telescope.name = 'NDG'
    telescope.lightcurve_flux = np.array([0, 36307805477.010025, -39420698921.705284])
    current_event.telescopes.append(telescope)

    current_event.check_event()


def test_telescopes_names():
    current_event = event.Event()

    telescope1 = mock.MagicMock()
    telescope2 = mock.MagicMock()

    telescope1.name = 'telescope1'
    telescope2.name = 'telescope2'

    current_event.telescopes.append(telescope1)
    current_event.telescopes.append(telescope2)

    current_event.telescopes_names()


def test_fit_bad_event_type():

	current_event = event.Event()
	current_event.kind = 'I am not microlensing'

	with pytest.raises(event.EventException) as event_exception:
        	current_event.fit('PSPL', 'LM')
    	assert 'Can not fit this event kind' in str(event_exception)


def test_fit_bad_method():

	current_event = event.Event()
	
	with pytest.raises(event.EventException) as event_exception:
        	current_event.fit('PSPL', 'I am not a fitter :)')
    	assert 'Wrong fit method request' in str(event_exception)

	with pytest.raises(event.EventException) as event_exception:
        	current_event.fit('PSPL', 51)
    	assert 'Wrong fit method request' in str(event_exception)


def test_find_survey():
    current_event = event.Event()

    telescope1 = mock.MagicMock()
    telescope2 = mock.MagicMock()

    telescope1.name = 'telescope1'
    telescope2.name = 'telescope2'

    current_event.telescopes.append(telescope1)
    current_event.telescopes.append(telescope2)
    current_event.find_survey('telescope2')

    assert current_event.telescopes[0].name == 'telescope2'


def test_lightcurves_in_flux_calls_telescope_lightcurve_in_flux_with_default():
    current_event = event.Event()
    telescopes = [mock.MagicMock(), mock.MagicMock()]
    current_event.telescopes.extend(telescopes)

    current_event.lightcurves_in_flux()

    for telescope in telescopes:
        telescope.lightcurve_in_flux.assert_called_with('Yes')


def test_lightcurves_in_flux_calls_telescope_lightcurve_in_flux():
    current_event = event.Event()
    telescopes = [mock.MagicMock(), mock.MagicMock()]
    current_event.telescopes.extend(telescopes)

    current_event.lightcurves_in_flux(choice='No')

    for telescope in telescopes:
        telescope.lightcurve_in_flux.assert_called_with('No')


def test_lightcurves_in_flux_sets_telescope_lightcurve_flux():
    current_event = event.Event()
    telescope1 = mock.MagicMock()
    telescope1.lightcurve_in_flux.return_value = np.array([])
    telescope2 = mock.MagicMock()
    telescope2.lightcurve_magnitude = np.array([0, 1, 1])
    telescope2.lightcurve_in_flux.return_value = np.array(
        [0, 36307805477.010025, -39420698921.705284])
    telescopes = [telescope1, telescope2]
    current_event.telescopes.extend(telescopes)

    current_event.lightcurves_in_flux()
    results = [np.array([]), np.array([0, 36307805477.010025, -39420698921.705284])]
    count = 0
    for telescope in telescopes:
        assert np.allclose(telescope.lightcurve_flux, results[count])
        count += 1

def test_find_survey_no_survey() :
    current_event = event.Event()

    telescope1 = mock.MagicMock()
    telescope2 = mock.MagicMock()

    telescope1.name = 'telescope1'
    telescope2.name = 'telescope2'

    current_event.telescopes.append(telescope1)
    current_event.telescopes.append(telescope2)

    assert current_event.find_survey('NDG') == None

