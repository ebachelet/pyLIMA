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
    @author: Etienne Bachelet

    This module create an event class with the informations (attributes) needed for the fits.

    Keyword arguments:

    kind --> type of event. In general, should be 'Microlensing' (default)
    name --> name of the event. Should be a string. Default is 'Sagittarius A*'
    ra --> Right ascension of the event (J2000). Should be a float in degree between 0.0 and
    360.0. Default is
           ra of Sagittarius A == 266.416792 from Baker & Sramek 1999ApJ...524..805B.
    dec --> Declination of the event (J2000). Should be a float in degree between -90 and 90.
    Default is
            dec of Sagittarius A == -29.007806 from Baker & Sramek 1999ApJ...524..805B.
    Teff --> Effective temperature of the star in Kelvin. Should be a float. Default is 5000.0 K.
    logg --> Surface gravity in log10 cgs unit. Should be a float. Default is 4.5.
    telescopes --> List of telescopes names (strings). Default is an empty list. Have to be fill
    with some
                   telescopes class instances.

    """

    def __init__(self):
        """ Initialization of the attributes described above. """
        self.kind = 'Microlensing'
        self.name = 'Sagittarius A*'
        self.ra = 266.416792
        self.dec = -29.007806
        self.Teff = 5000
        self.logg = 4.5
        self.telescopes = []
        self.survey = 'None'
        self.fits = []
        self.outputs = []

    def fit(self, Model, method):
        """Function to fit the event.

        Keyword arguments:

        model --> The microlensing model you want to fit. Has to be a string in the
        available_models parameter:

            'PSPL' --> Point Source Point Lens
            'FSPL' --> Finite Source Point Lens
            'DSPL' --> Double Source Point Lens
            'Binary' --> not available now
            'Triple' --> not available now

            More details in the microlfits module

        method --> The fitting method you want to use. Has to be a integer in the
        available_methods parameter:.

            0 --> Levenberg-Marquardt algorithm.

            More details in the microlfits module

        second_order --> Second order effect : parallax, orbital_motion and source_spots . A list
        of string as :

            [parallax,orbital_motion,source_spots]
            Example : [['Annual',2456876.2],['2D',2456876.2],'None']

            parallax --> Parallax model you want to use for the Earth types telescopes.
                         Has to be a list containing the model in the available_parallax
                         parameter and
                         the value of topar.

                         'Annual' --> Annual parallax
                         'Terrestrial' --> Terrestrial parallax
                         'Full' --> combination of previous

                         topar --> a time in HJD choosed as the referenced time fot the parallax

                         If you have some Spacecraft types telescopes, the space based parallax
                         is computed

                         More details in the microlparallax module

            orbital_motion --> Orbital motion you want to use. Has to be a list containing the model
                               in the available_orbital_motion parameter and the value of toom:

                'None' --> No orbital motion
                '2D' --> Classical orbital motion
                '3D' --> Full Keplerian orbital motion

                toom --> a time in HJD choosed as the referenced time fot the orbital motion
                        (Often choose equal to topar)

                More details in the microlomotion module

            source_spots --> Consider spots on the source. Has to be a string in the
            available_source_spots parameter :

                'None' --> No source spots

                More details in the microlsspots module

        Return :

            fits_results --> List results of the requested fits in the form:

                            [[model1,method1,parameters1], [model2, method2, parameters2],...]

            fits_covariance --> List results of covariance matrix of the requested fits in the form:

                            [[model1,method1,covariance1], [model2, method2, covariance2],...]

            fits_time --> List of effective computational time (in seconds) of the requested fits
            in the form:

                            [[model1,method1,time1], [model2, method2, time2],...]

            The number of parameters and row in the covariance matrix depends of the selected model,
            the selected second_order effects choose and the number of telescopes selected for
            the fit.

            More details in the microlfits module.

            The function is incremental, which means that each .fit() function call will fill
            fits_results,
            fits_covariance and fits_time.
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
        self.lightcurves_in_flux('Yes')

        fit = microlfits.MLFits(self)
        fit.mlfit(Model, method)

        self.fits.append(fit)

    def telescopes_names(self):
        """Function to list the telescope names for an event. """
        print [self.telescopes[i].name for i in xrange(len(self.telescopes))]

    def check_event(self):
        """Function to check if everything is correctly set before the fit.
        An ERROR is returned if the check is not successfull
        Should be used before any event_fit function calls

        First check if the event name is a string
        Then check if the right ascension (event.ra) is between 0 and 360 degrees
        Then check if the declination (event.dec) is between -90 and 90 degrees
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

        print 'Everything is fine, this event can be treat'

    def find_survey(self, choice=None):
        """Function to find the survey telescope in the telescopes list,
        and put it on the first place (useful for some fits functions).
        """

        self.survey = choice or self.telescopes[0].name

        names = [i.name for i in self.telescopes]
        if any(self.survey in i for i in names):

            index = np.where(self.survey == np.array(names))[0]
            sorting = np.arange(0, len(self.telescopes))
            sorting = np.delete(sorting, index)
            sorting = np.insert(sorting, 0, index)
            self.telescopes = np.array(self.telescopes)[sorting.tolist()].tolist()

        else:

            print 'ERROR : There is no telescope names containing ' + self.survey
            return

    def plot_data(self, choice, observe, align):

        if align == 'Yes':

            self.plotter.align_lightcurves(choice)

        if observe is 'Mag':

            self.plotter.plot_lightcurves_mag(align)

        if observe is 'Flux':

            self.plotter.plot_lightcurves_flux(align)

    def lightcurves_in_flux(self, choice):

        for i in self.telescopes:

            i.lightcurve_in_flux(choice)

    def initialize_plots(self, choice, observe):

        self.plotter = microlplotter.MLPlotter(self)
        self.plotter.initialize_plots(choice, observe)

    def produce_outputs(self, choice):

        self.outputs = microloutputs.MLOutputs(self)
        # self.outputs.cov2corr()
        self.outputs.errors_on_fits(choice)
        # self.outputs.find_observables()
        # Parameters,lightcurve_model,lightcurve_data = self.outputs.K2_C9_outputs()
        # return Parameters,lightcurve_model,lightcurve_data

    def compute_parallax(self, second_order):
        telescopes = []
        self.lightcurves_in_flux('Yes')
        for i in self.telescopes:
  
            if len(i.deltas_positions)==0:
                telescopes.append(i)

        para = microlparallax.MLParallaxes(self, second_order[0])
        para.parallax_combination(telescopes)
