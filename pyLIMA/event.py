# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:39:32 2015

@author: ebachelet
"""

from __future__ import division

import numpy as np

import microlfits
import microlplotter
import microloutputs
import microlparallax


class Event(object):
    """
    ######## Event module ########
    
    This module create an event object with the informations (attributes) needed for the fits.

    Attributes :

         kind : type of event. In general, should be 'Microlensing' (default)
         
         name : name of the event. Should be a string. Default is 'Sagittarius A*'
         
         ra : right ascension of the event (J2000). Should be a float in degree between 0.0 and 360.0. Default is ra of Sagittarius A, :math:`\\alpha` = 266.416792 from Baker & Sramek 1999ApJ...524..805B
    
         dec : declination of the event (J2000). Should be a float in degree between -90 and 90. Default is dec of Sagittarius A, :math:`\\delta` = -29.007806 from Baker & Sramek 1999ApJ...524..805B
    
         Teff : effective temperature of the star in Kelvin. Should be a float. Default is 5000.0 K
         
         logg : surface gravity in log10 cgs unit. Should be a float. Default is 4.5
         
         telescopes : list of telescopes names (strings). Default is an empty list. Have to be fill with some telescopes class instances.
         
         survey : the reference telescope. Has to be a string, default is 'None'.
         
         fits : list of microlfits objects.
    """

    def __init__(self):
        """ Initialization of the attributes described above. """

        self.kind = 'Microlensing'
        self.name = 'Sagittarius A*'
        self.ra = 266.416792
        self.dec = -29.007806
        self.Teff = 5000 # Kelvins
        self.logg = 4.5
        self.telescopes = []
        self.survey = 'None'
        self.fits = []
        

    def fit(self, Model, method):
        """Function to fit the event with a Model and a method.
        

        :param Model: the Model you want to fit. More details in the microlfits module

        :param method: the fitting method you want to use. Has to be a string in the available_methods parameter:
        
            'LM' : Levenberg-Marquardt algorithm
            
            'DE' : Differential Evolution algorithm
            
            'MCMC' : Monte-Carlo Markov Chain algorithm
            
            More details in the microlfits module

        A microlfits object is added in the event.fits list. For example, if you request two fits,
        you will obtain :
            
        event.fits=[fit1,fit2]

        More details in the microlfits module.

        """
        available_kind = ['Microlensing']
        available_methods = ['LM', 'DE', 'MCMC']

        if self.kind not in available_kind:
            print 'ERROR : No possible fit yet for a non microlensing event, sorry :('
            return

        if method not in available_methods:
            print 'ERROR : Wrong method request, has to be an integer selected between ' + \
                  ' or '.join(available_methods) + ''
            return

        fit = microlfits.MLFits(self)
        fit.mlfit(Model, method)

        self.fits.append(fit)

    def telescopes_names(self):
        """Print the the telescope's names contain in the event. 
        """
        print [self.telescopes[i].name for i in xrange(len(self.telescopes))]
   
    def check_event(self):
        """Function to check if everything is correctly set before the fit.
        An ERROR is returned if the check is not successfull
        Should be used before any event_fit function calls

        First check if the event name is a string.
        Then check if the right ascension (event.ra) is between 0 and 360 degrees.
        Then check if the declination (event.dec) is between -90 and 90 degrees.
        Then check if you have any telescopes ingested.
        Finally check if your telescopes have a lightcurve attributes different from None.
        """
        if self.name == 'None':
            print 'ERROR : The event name (' + str(
                self.name) + ') is not correct, it has to be a string'
            return

        if (self.ra > 360) or (self.ra < 0):
            print 'ERROR : The event ra (' + str(
                self.ra) + ') is not correct, it has to be a float between 0 and 360 degrees'
            return

        if (self.dec > 90) or (self.dec < -90):
            print 'ERROR : The event ra (' + str(
                self.dec) + ') is not correct, it has to be between -90 and 90 degrees'
            return

        if len(self.telescopes) == 0:
            print 'ERROR : There is no associated telescopes with this event, add some using ' \
                  'self.telescopes.append'
            return

        else :
            
            for telescope in self.telescopes :
                
                if len(telescope.lightcurve_magnitude) == 0 & len(telescope.lightcurve_flux) == 0 :
                    
                     print 'ERROR : There is no associated lightcurve in magnitude of flux with this telescopes : '\
                            +telescope.name+', add one with telescope.lightcurve = your_data'
                     return
        
        print 'Everything is fine, this event can be treat'

    def find_survey(self, choice=None):
        """Function to find the survey telescope in the telescopes list,
           and put it on the first place (useful for some fits functions).
        
            :param choice: the name of the telescope choosing as the survey. Has to be a string.
                           Default is the first telescope.
        """
        self.survey = choice or self.telescopes[0].name

        names = [telescope.name for telescope in self.telescopes]
        if any(self.survey in name for name in names):

            index = np.where(self.survey == np.array(names))[0]
            sorting = np.arange(0, len(self.telescopes))
            sorting = np.delete(sorting, index)
            sorting = np.insert(sorting, 0, index)
            self.telescopes = [self.telescopes[i] for i in sorting]

        else:

            print 'ERROR : There is no telescope names containing ' + self.survey
            return

    
    def lightcurves_in_flux(self, choice='Yes'):
        """ Transform all telescopes magnitude lightcurves in flux units.
            

            :param choice: to clean your lightcurve or not. Has to be a string 'Yes' or 'No'. Defaul is 'Yes'. More details in the telescope module
            
        """
        for telescope in self.telescopes:

           telescope.lightcurve_flux = telescope.lightcurve_in_flux(choice)

    def initialize_plots(self, choice, observe):
        """ Not working and probably depreciated"""
        self.plotter = microlplotter.MLPlotter(self)
        self.plotter.initialize_plots(choice, observe)

    def produce_outputs(self, choice):
         """ Not working and probably depreciated"""
         self.outputs = microloutputs.MLOutputs(self)
         # self.outputs.cov2corr()
         self.outputs.errors_on_fits(choice)
         # self.outputs.find_observables()
         # Parameters,lightcurve_model,lightcurve_data = self.outputs.K2_C9_outputs()
         # return Parameters,lightcurve_model,lightcurve_data
        
    def plot_data(self, choice, observe, align):
        """ Not working and probably depreciated"""
        if align == 'Yes':

            self.plotter.align_lightcurves(choice)

        if observe is 'Mag':

            self.plotter.plot_lightcurves_mag(align)

        if observe is 'Flux':

            self.plotter.plot_lightcurves_flux(align)

