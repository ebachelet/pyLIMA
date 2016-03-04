# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:46:17 2015

@author: ebachelet
"""

import numpy as np

import microlfits
import event
import telescopes


def test_magnification():
    time = np.arange(-100, 100, 0.01)

    uo = 0.001
    to = 0.0
    tE = 30

    u = (uo ** 2 + (time - to) ** 2 / tE ** 2) ** 0.5
    A = (u ** 2 + 2) / (u * (u ** 2 + 4) ** 0.5)

    Event = event.Event()
    tel = telescopes.Telescope(name='OGLE', camera_filter='I', light_curve=[])
    tel.lightcurve = np.array([[0.0, 16, 0.1], [1.0, 16, 0.1]])
    tel.lightcurve_in_flux()
    Event.telescopes.append(tel)
    second_order = [['None', 2456877.07], ['None', 0], ['None', 0], 'None']
    model = 'PSPL'
    fits = microlfits.MLFits(Event, model, 0, second_order)
    parameters = [to, uo, tE]
    ampli = fits.amplification(parameters, time, model, Event.telescopes[0].gamma)[0]

    np.max(np.abs(ampli - A)) < 0.001
    np.max(ampli) == 1000
