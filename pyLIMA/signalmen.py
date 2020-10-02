# -*- coding: utf-8 -*-
"""
Created on Wed 30 Sep 2020

@author: martindominik 
"""

from __future__ import division

import sys
import numpy as np


class AnomalyStatus(object):
    """
    ######## AnomalyStatus class ########

    This class defines an AnomalyStatus object holding the assessment of an event for an ongoing anomaly 
    and providing methods to carry out the assessment
    
    It contains the complete SIGNALMEN management
    and builds on the Event class for fitting models to event data.

    Attributes :

         filename : specifies master file name for various input/output files to control SIGNALMEN
                      and log assessment

         alldata : numpy array of full photometric data sorted in time sequence

         event_all : instance of Event class with all data (old and new)

 	 event_prev : instance of Event class with previous data and model  

         event_curr : instance of Event class with data and model corresponding to current assessment step

         status : SIGNALMEN status (0 = ordinary, 1 = check, 2 = anomaly) after assessment

         prev_status : SIGNALMEN status (0 = ordinary, 1 = check, 2 = anomaly) before assessment 
 
	 time_prev : epoch of previous assessment (new assessment starts after this epoch)

	 [Some other attributes for management required, maybe bundle in one or several classes, e.g.
             list of anomalous points
             list of specific points to be excluded for modelling]

	 [For specific fits, data has to be copied from "alldata" to the telescopes attribute of "event_curr"
             taking into account the time range and further exclusion criteria]
  
         ["alldata" initially has to be populated with data stored in "event_all" and sorted in time sequence]
         
    """

    def __init__(self,event):
        """ Initialization of the attributes described above. """

        self.event = event
        alldata_list = []
        tel_idx = 0
        for telescope in self.event.telescopes:
            dlen = len(telescope.lightcurve_flux)
            lightcurve = np.c_[telescope.lightcurve_flux,telescope.lightcurve_magnitude[:,1:3],np.full(dlen,tel_idx)]
            alldata_list.append(lightcurve)
            tel_idx += 1
        self.alldata = np.concatenate(alldata_list)

    def fit(self, model, method, DE_population_size=10, flux_estimation_MCMC='MCMC', fix_parameters_dictionnary=None,
            grid_resolution=10, computational_pool=None, binary_regime=None,
            robust=False):
        """Function to fit the event with a model and a method.


        :param model: the model you want to fit. More details in the microlfits module

        :param method: the fitting method you want to use. Has to be a string in the
        available_methods parameter:

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
        available_methods = ['LM', 'DE', 'MCMC', 'GRIDS', 'TRF']

        if self.kind not in available_kind:
            print('ERROR : No possible fit yet for a non microlensing event, sorry :(')
            raise EventException('Can not fit this event kind')

        if method not in available_methods:
            print('ERROR : Wrong method request, has to be a string selected between ' + \
                  ' or '.join(available_methods) + '')
            raise EventException('Wrong fit method request')

        fit = microlfits.MLFits(self)
        fit.mlfit(model, method, DE_population_size=DE_population_size, flux_estimation_MCMC=flux_estimation_MCMC,
                  fix_parameters_dictionnary=fix_parameters_dictionnary,
                  grid_resolution=grid_resolution, computational_pool=computational_pool, binary_regime=binary_regime,robust=robust)
            # MD: new option "robust"
        self.fits.append(fit)

    def telescopes_names(self):
        """Print the the telescope's names contain in the event.
        """
        print([self.telescopes[i].name for i in range(len(self.telescopes))])

    def check_event(self):
        """Function to check if everything is correctly set before the fit.
        An ERROR is returned if the check is not successful
        Should be used before any event_fit function calls

        First check if the event name is a string.
        Then check if the right ascension (event.ra) is between 0 and 360 degrees.
        Then check if the declination (event.dec) is between -90 and 90 degrees.
        Then check if you have any telescopes ingested.
        Finally check if your telescopes have a lightcurve attributes different from None.
        """

        if not isinstance(self.name, str):
            raise EventException('ERROR : The event name (' + str(
                self.name) + ') is not correct, it has to be a string')

        if (self.ra > 360) or (self.ra < 0):
            raise EventException('ERROR : The event ra (' + str(
                self.ra) + ') is not correct, it has to be a float between 0 and 360 degrees')

        if (self.dec > 90) or (self.dec < -90):
            raise EventException('ERROR : The event dec (' + str(
                self.dec) + ') is not correct, it has to be between -90 and 90 degrees')

        if len(self.telescopes) == 0:
            raise EventException('There is no telescope associated to your event, no fit possible!')

        else:

            for telescope in self.telescopes:

                if (len(telescope.lightcurve_magnitude) == 0) & \
                        (len(telescope.lightcurve_flux) == 0):
                    print('ERROR : There is no associated lightcurve in magnitude or flux with ' \
                          'this telescopes : ' \
                          + telescope.name + ', add one with telescope.lightcurve = your_data')
                    raise EventException('There is no lightcurve associated to the  telescope ' + str(
                        telescope.name) + ', no fit possible!')

        print(sys._getframe().f_code.co_name, ' : Everything looks fine, this event can be fitted')

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

            print('ERROR : There is no telescope names containing ' + self.survey)
            return

    def lightcurves_in_flux(self, choice='Yes'):
        """ Transform all telescopes magnitude lightcurves in flux units.


            :param choice: to clean your lightcurve or not. Has to be a string 'Yes' or 'No'.
            Defaul is 'Yes'. More details in the telescope module
        """

        for telescope in self.telescopes:
            telescope.lightcurve_flux = telescope.lightcurve_in_flux(choice)

    def compute_parallax_all_telescopes(self, parallax_model):
        """ Compute the parallax displacement for all the telescopes, if this is desired in
        the second order parameter.
        """

        for telescope in self.telescopes:

            if len(telescope.deltas_positions) == 0:
                telescope.compute_parallax(self, parallax_model)

    def total_number_of_data_points(self):
        """ Compute the parallax displacement for all the telescopes, if this is desired in
            the second order parameter.
            :return: n_data, the total number of points
            :rtype: float
        """
        n_data = 0.0

        for telescope in self.telescopes:
            n_data = n_data + telescope.n_data('flux')

        return n_data
