# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:46:13 2015

@author: ebachelet
"""

import numpy as np

import event
import microlmodels
import telescopes


def test_annual_parallax():
    to = 2456877.07
    uo = 0.527227
    tE = 95.4095
    ftot = 10 ** ((18 - 16.6822) / 2.5)
    eps = -0.286296
    fs = ftot / (1 + eps)
    fb = fs * eps
    pipar = -0.00179068
    piperp = -0.242817
    angle = -0.084038402660557551
    pien = pipar * np.sin(angle)
    piee = np.cos(angle) * piperp
    piE = np.array([pien, piee])

    data_valerio = np.genfromtxt(
        '/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement'
        '/Parallaxes/Valerio_OB140099.txt',
        delimiter=',')
    Event = event.Event()

    tel = telescopes.Telescope(name='OGLE', camera_filter='I', light_curve=[])
    tel.lightcurve = data_valerio[:, [0, 1, 3]]
    tel.lightcurve = tel.lightcurve[tel.lightcurve[:, 0].argsort(),]
    tel.lightcurve[:, 0] = tel.lightcurve[:, 0] + 2450000

    tel.lightcurve_in_flux()
    Event.telescopes.append(tel)

    Event.ra = 269.607333
    Event.dec = -28.279833
    second_order = [['Annual', 2456877.07], ['None', 0], ['None', 0], 'None']
    model = 'PSPL'
    models = microlmodels.MLModels(Event, model, second_order)
    parallax = models.parallax.delta_position

    delta_tau, delta_u = models.parallax.parallax_outputs(piE)

    tau = (tel.lightcurve[:, 0] - to) / tE + delta_tau
    uuo = uo + delta_u

    u = (uuo ** 2 + tau ** 2) ** 0.5
    A = (u ** 2 + 2) / (u * (u ** 2 + 4) ** 0.5)
    res = tel.lightcurve[:, 1] - A

    max(np.abs(res)) < 0.01
