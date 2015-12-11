# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:48:29 2015

@author: ebachelet
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

import microlmagnification
import microlmodels

def plot_lightcurves_mag(event, split):


    filters_dictionnary = {'N': 'NIR', 'I':'NIR', 'Red':'NIR', 'V':'V', 'J':'IR', 'H':'IR', 'K':'IR'}
    observed_bands = np.unique([filters_dictionnary[i.filter] for i in event.telescopes])

    plots_dictionnary = {'NIR':1}

    if 'V' in observed_bands :

         plots_dictionnary['V'] = len(plots_dictionnary)+1

    if 'IR' in observed_bands :

         plots_dictionnary['IR'] = len(plots_dictionnary)+1

    for i in observed_bands:

        plot_location = plots_dictionnary[i]
        plt.subplot(len(observed_bands),1, plot_location)
        plt.title(i, fontsize=20)
        plt.gca().invert_yaxis()

    plt.suptitle(event.name, fontsize=30)

    for i in event.telescopes:

        time = i.lightcurve[:, 0]
        if time[0] > 2450000:
            time = time-2450000

        plot_location = plots_dictionnary[filters_dictionnary[i.filter]]
        plt.subplot(len(observed_bands),1, plot_location)
        plt.errorbar(time, i.lightcurve[:, 1], yerr=i.lightcurve[:, 2], linestyle='none',label=i.name)
        plt.legend(numpoints=1)

def plot_lightcurves_flux(event, split):

    for i in event.telescopes:

        time = i.lightcurve[:, 0]
        if time[0] > 2450000:
            time = time-2450000
        plt.errorbar(time, i.lightcurve_flux[:, 1], yerr=i.lightcurve_flux[:, 2], linestyle='none',label=i.name)
        plt.legend(numpoints=1)

def plot_model_mag( event, request, parameters):

    filters_dictionnary = {'N': 'NIR', 'I':'NIR', 'Red':'NIR', 'V':'V', 'J':'IR', 'H':'IR', 'K':'IR'}
    observed_bands = np.unique([filters_dictionnary[i.filter] for i in event.telescopes])

    plots_dictionnary = {'NIR':1}

    if 'V' in observed_bands :

         plots_dictionnary['V'] = len(plots_dictionnary)+1

    if 'IR' in observed_bands :

         plots_dictionnary['IR'] = len(plots_dictionnary)+1

    for i in observed_bands:

        plot_location = plots_dictionnary[i]
        plt.subplot(len(observed_bands),1, plot_location)
        plt.title(i, fontsize=20)

    plt.suptitle(event.name, fontsize=30)

    plot_model = microlmodels.MLModels(event, request[0], request[1])

    t = np.arange(min(event.telescopes[0].lightcurve[:, 0]), max(event.telescopes[0].lightcurve[:, 0]), 0.01)
    parameters = parameters[:-1]

    if event.telescopes[0].lightcurve[0, 0] > 2450000:

            t = t-2450000

    ampli=microlmagnification.amplification( plot_model, t, parameters, event.telescopes[0].gamma)[0]
    fs=parameters[plot_model.model_dictionnary['fs_'+event.telescopes[0].name]]
    fb=parameters[plot_model.model_dictionnary['g_'+event.telescopes[0].name]]*fs


    plt.plot(t, 27.4-2.5*np.log10(fs*ampli+fb),'r', lw=2)
