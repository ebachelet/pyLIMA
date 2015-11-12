# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:48:29 2015

@author: ebachelet
"""

import matplotlib.pyplot as plt
import numpy as np

class ML_plotter(object):
    
    def plot_lightcurve(self):
        
        for i in self.telescopes:
            time = i.lightcurve[:, 0]
            if time[0] > 2450000:
                time = time-2450000
            plt.errorbar(time, i.lightcurve[:, 1], yerr=i.lightcurve[:, 2], linestyle='none')
        plt.gca().invert_yaxis()

        
    def plot_model(self, pp):
        
        
        print pp
        t = np.arange(min(self.telescopes[0].lightcurve[:, 0]), max(self.telescopes[0].lightcurve[:, 0]), 0.01)
        to = pp[0]
        if self.telescopes[0].lightcurve[0, 0] > 2450000:
            t = t-2450000
            to = to-2450000
        uo = pp[1]
        tE = pp[2]
        fs = pp[3]
        fb = pp[4]*fs
        U = np.sqrt(uo**2+(t-to)**2/tE**2)
        A = (U**2+2)/(U*np.sqrt(U**2+4))
        plt.plot(t, 27.4-2.5*np.log10(fs*A+fb))
        