# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:48:29 2015

@author: ebachelet
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

import microlfits

class MLPlotter(object):
    
    
    def __init__(self, event):

        self.event = event
        
    def plot_lightcurves_mag(self):
        
        for i in self.event.telescopes:
            time = i.lightcurve[:, 0]
            if time[0] > 2450000:
                time = time-2450000
            plt.errorbar(time, i.lightcurve[:, 1], yerr=i.lightcurve[:, 2], linestyle='none',label=i.name)
            plt.legend(numpoints=1)
        plt.gca().invert_yaxis()

    def plot_lightcurves_flux(self):
        
        for i in self.event.telescopes:
            time = i.lightcurve_flux[:, 0]
            if time[0] > 2450000:
                time = time-2450000
            plt.errorbar(time, i.lightcurve_flux[:, 1], yerr=i.lightcurve_flux[:, 2], linestyle='none')


    def plot_model_mag(self, model, parameters, second_order):

        self.model = model
        self.second_order = second_order
        self.parameters = parameters
        self.number_of_params()
        
        t = np.arange(min(self.event.telescopes[0].lightcurve[:, 0]), max(self.event.telescopes[0].lightcurve[:, 0]), 0.01)
        if self.event.telescopes[0].lightcurve[0, 0] > 2450000:
            t = t-2450000
            self.parameters[0] = self.parameters[0]-2450000

        Ampli=microlfits.MLFits(self.event, self.model, 0, self.second_order)
       
        ampli=Ampli.amplification(self.parameters, t, self.model, self.event.telescopes[0].gamma)[0]
        fs=self.parameters[self.number_of_parameters]
        fb=self.parameters[self.number_of_parameters+1]*fs

        plt.plot(t, 27.4-2.5*np.log10(fs*ampli+fb),'r', lw=2)
        

    def number_of_params(self):
        '''Provide the number of parameters on which depend the magnification computation. (Paczynski parameters+binary)
        '''
        model = {'PSPL':3, 'FSPL':4}
       

        parallax = {'None':0, 'Annual':2, 'Terrestrial':2, 'Full':2}
       

        orbital_motion={'None':0, '2D':2, '3D':999999}
    

        source_spots={'None':0}
       
        self.number_of_parameters = model[self.model]+parallax[self.second_order[0][0]]+orbital_motion[
                                    self.second_order[1][0]]+source_spots[self.second_order[2]]

      