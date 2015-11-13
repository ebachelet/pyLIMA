# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:48:29 2015

@author: ebachelet
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


class MLPlotter(object):
    
    
    def __init__(self, event, model, second_order):

        self.event = event
        self.model = model
        self.method = method
        self.second_order = second_order


        self.number_of_parameters()
        
    def plot_lightcurves_mag(self):
        
        for i in self.telescopes:
            time = i.lightcurve[:, 0]
            if time[0] > 2450000:
                time = time-2450000
            plt.errorbar(time, i.lightcurve[:, 1], yerr=i.lightcurve[:, 2], linestyle='none')
        plt.gca().invert_yaxis()

    def plot_lightcurves_flux(self):
        
        for i in self.telescopes:
            time = i.lightcurve_flux[:, 0]
            if time[0] > 2450000:
                time = time-2450000
            plt.errorbar(time, i.lightcurve_flux[:, 1], yerr=i.lightcurve_flux[:, 2], linestyle='none')


    def plot_model_mag(self, model, pp, second_order):
       
        t = np.arange(min(self.telescopes[0].lightcurve[:, 0]), max(self.telescopes[0].lightcurve[:, 0]), 0.01)
          
        if self.telescopes[0].lightcurve[0, 0] > 2450000:
            t = t-2450000
            pp[0] = pp[0]-2450000

        number_of_par=number_of_parameters(model, second_order)
        ampli=amplification(pp, t, model, self.telescopes[0].gamma)[0]
        fs=pp[number_of_par+1]
        fb=pp[number_of_par+2]*fs

        plt.plot(t, 27.4-2.5*np.log10(fs*ampli+fb),'r', lw=2)
        

    def number_of_parameters(self):
        '''Provide the number of parameters on which depend the magnification computation.
        '''
        model = {'PSPL':3, 'FSPL':4}
       

        parallax = {'None':0, 'Annual':2, 'Terrestrial':2, 'Full':2}
       

        orbital_motion={'None':0, '2D':2, '3D':999999}
    

        source_spots={'None':0}
       
        self.number_of_parameters = model[self.model[0]]+parallax[self.second_order[0][0]]+orbital_motion[
                                    self.second_order[1][0]]+source_spots[self.second_order[2]]

      