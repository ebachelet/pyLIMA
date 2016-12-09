# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:46:13 2015

@author: ebachelet
"""

import numpy as np
import mock
from pyslalib import slalib

from pyLIMA import microlparallax

def _create_event():
    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].name = 'Test'
    event.telescopes[0].lightcurve_flux = np.array([[0, 1, 1], [42, 6, 6]])
    event.telescopes[0].filter = 'I'
    event.telescopes[0].gamma = 0.5
    event.ra = 180
    event.dec = 360
    return event


def test_create_parallax():

    event = _create_event()
    parallax_model = ['None', 1664.51]
    parallax = microlparallax.MLParallaxes(event, parallax_model)

    assert parallax.parallax_model == 'None'
    assert parallax.to_par == 1664.51
    np.allclose(parallax.AU ,  149597870700)
    np.allclose(parallax.speed_of_light ,  299792458)
    np.allclose(parallax.Earth_radius ,  6378137000)


    np.allclose(parallax.target_angles_in_the_sky, [np.pi , 2* np.pi ])
    np.allclose(parallax.North, [0 , 1, 0 ])
    np.allclose(parallax.East, [0 , 0, 1])

def test_HJD_to_JD():

    event = _create_event()
    parallax_model = ['None', 1664.51]
    parallax = microlparallax.MLParallaxes(event, parallax_model)

    JD = 2455000
    MJD = JD- 2400000.5

    Sun_angles = slalib.sla_rdplan(MJD,99,0,0)

    HJD = JD - parallax.AU/parallax.speed_of_light*(np.sin(Sun_angles[1])*np.sin(event.dec*np.pi/180)+
                                                    np.cos(Sun_angles[1])*np.cos(event.dec*np.pi/180)*
                                                    np.cos(event.ra*np.pi/180-Sun_angles[0]))

    jd  = parallax.HJD_to_JD([HJD])

    np.allclose(jd,JD)

def test_annual_parallax():

    event = _create_event()
    parallax_model = ['None', 1664.51]
    parallax = microlparallax.MLParallaxes(event, parallax_model)

    positions = parallax.annual_parallax(event.telescopes[0].lightcurve_flux[:,0])

    np.allclose(np.array([[-11.99055572, -11.40251147],[ 26.46632345,  25.16787805]]),positions)


def test_terrestrial_parallax():

    event = _create_event()
    parallax_model = ['None', 1664.51]
    parallax = microlparallax.MLParallaxes(event, parallax_model)

    positions = parallax.terrestrial_parallax(event.telescopes[0].lightcurve_flux[:,0],
                                         0, 0, 0)

    np.allclose(np.array([[ -4.68497968e-21,   2.65705901e-21],[ -3.81036079e-05,  -4.12319764e-05]]),positions)


def test_space_parallax():

    event = _create_event()
    event.telescopes[0].lightcurve_flux[:,0] = 2458000.5 +  event.telescopes[0].lightcurve_flux[:,0]
    parallax_model = ['None', 1664.51]

    parallax = microlparallax.MLParallaxes(event, parallax_model)

    positions = parallax.space_parallax(event.telescopes[0].lightcurve_flux[:,0], 'Kepler')

    np.allclose(np.array([[-0.26914499, -0.40329631],[ 0.63675129,  0.93466804]]),positions)

def test_parallax_combination_on_Earth_annual():

    event = _create_event()
    event.telescopes[0].location = 'Earth'
    parallax_model = ['Annual', 1664.51]
    parallax = microlparallax.MLParallaxes(event, parallax_model)


    positions_annual = parallax.annual_parallax(event.telescopes[0].lightcurve_flux[:,0])
    parallax.parallax_combination(event.telescopes[0])

    np.allclose(positions_annual,event.telescopes[0].deltas_positions)

def test_parallax_combination_on_Earth_full():

    event = _create_event()
    event.telescopes[0].location = 'Earth'
    event.telescopes[0].altitude = 0
    event.telescopes[0].longitude = 0
    event.telescopes[0].latitude = 0

    parallax_model = ['Full', 1664.51]
    parallax = microlparallax.MLParallaxes(event, parallax_model)


    positions_annual = parallax.annual_parallax(event.telescopes[0].lightcurve_flux[:,0])
    positions_terrestrial = parallax.terrestrial_parallax(event.telescopes[0].lightcurve_flux[:,0],
                                                     0, 0, 0)
    parallax.parallax_combination(event.telescopes[0])

    np.allclose(positions_annual+positions_terrestrial,event.telescopes[0].deltas_positions)

def test_parallax_combination_on_Space():

    event = _create_event()
    event.telescopes[0].location = 'Space'
    event.telescopes[0].name = 'Kepler'
    event.telescopes[0].spacecraft_name = 'Kepler'

    event.telescopes[0].lightcurve_flux[:,0] = 2458000.5 +  event.telescopes[0].lightcurve_flux[:,0]
    parallax_model = ['Full', 1664.51]
    parallax = microlparallax.MLParallaxes(event, parallax_model)


    positions_annual = parallax.annual_parallax(event.telescopes[0].lightcurve_flux[:,0])
    positions_space = parallax.space_parallax(event.telescopes[0].lightcurve_flux[:,0], 'Kepler')

    parallax.parallax_combination(event.telescopes[0])

    np.allclose(positions_annual+positions_space,event.telescopes[0].deltas_positions)

def test_compute_parallax_curvature():

    event = _create_event()
    event.telescopes[0].location = 'Earth'
    event.telescopes[0].altitude = 0
    event.telescopes[0].longitude = 0
    event.telescopes[0].latitude = 0


    parallax_model = ['Full', 1664.51]
    parallax = microlparallax.MLParallaxes(event, parallax_model)


    parallax.parallax_combination(event.telescopes[0])
    piE = [0.5,0.5]
    curvature  = microlparallax.compute_parallax_curvature(piE,event.telescopes[0].deltas_positions)

    dtau = piE[0]* event.telescopes[0].deltas_positions[0]+piE[1]* event.telescopes[0].deltas_positions[1]
    du = piE[1]* event.telescopes[0].deltas_positions[0]-piE[0]* event.telescopes[0].deltas_positions[1]
    np.allclose([dtau,du],curvature)
