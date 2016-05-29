# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:39:32 2015

@author: ebachelet
"""
from __future__ import division

import numpy as np
import astropy.io.fits as fits


import microltoolbox
import microlparallax

class Telescope(object):
    """
    ######## Telescope module ########
    
    This module create a telescope object with the informations (attributes) needed for the fits.


    :param name: name of the telescope. Should be a string. Default is 'None'

    :param filter: telescope filter used. Should be a string wich follows the convention of :
               " Gravity and limb-darkening coefficients for the Kepler, CoRoT, Spitzer, uvby,
               UBVRIJHK,
               and Sloan photometric systems"
               Claret, A. and Bloemen, S. 2011A&A...529A..75C. For example, 'I' (default) is
               Johnson_Cousins I filter
               and 'i'' is the SDSS-i' Sloan filter.

    :param light_curve:  a numpy array with time, magnitude in column error in magnitude. Default is an empty list.

    :param lightcurve_dictionnary: a python dictionnary that informs your columns convention. Used to translate to pyLIMA
                                convention [time,mag,err_mag]

    Attributes :
    
        location : The location of your observatory. Should be "Earth" (default) or "Space".
        
        lightcurve_flux : List of time, flux, error in flux. Default is an empty list.
                          WARNING : has to be set before any fits.

        altitude : Altitude in meter of the telescope. Default is 0.0 (sea level).

        longitude : Longitude of the telescope in degrees. Default is 0.0.

        latitude : Latitude in degrees. Default is 0.0.

        gamma :  (Microlensing covention) Limb darkening coefficient associated to the filter
                 The classical (Milne definition) linear limb darkening coefficient can be found using:
                 u=(3*gamma)/(2+gamma).
                 Default is 0.5.
    """

    def __init__(self, name='NDG', camera_filter='I', light_curve=None, light_curve_dictionnary = {'time': 0, 'mag' : 1, 'err_mag' : 2 }):
        """ Initialization of the attributes described above."""
        
        self.name = name      
        self.filter = camera_filter  # Claret2011 convention
        self.lightcurve_dictionnary = light_curve_dictionnary
        if light_curve is None :
                
            self.lightcurve = []
                
        else :
        
            self.lightcurve = light_curve 
            self.lightcurve = self.arrange_the_lightcurve_columns()
    
       
        self.location = 'Earth'
        self.lightcurve_flux = []
        self.altitude = 0.0
        self.longitude = 0.57
        self.latitude = 49.49
        self.gamma = 0.0 # This mean you will fit uniform source brightness
        self.deltas_positions = []

    
       
    def arrange_the_lightcurve_columns(self):
        """ Rearange the lightcurve in magnitude in the pyLIMA convention"
        """
        lightcurve = []
        pyLIMA_convention = ['time','mag','err_mag']
        for good_column in pyLIMA_convention :
            
            lightcurve.append(self.lightcurve[:,self.lightcurve_dictionnary[good_column]])
        
        return np.array(lightcurve).T

    def n_data(self, choice='Mag'):
        """ Return the number of data points in the lightcurve.
        
        :param choice: ['Flux' or 'Mag'] The unit you want to check data for. Lightcurve_flux and Lightcurve (in mag, initial input can diverge depending on the clean_data() function).        
        
        """

        if choice == 'Flux':

            return len(self.lightcurve_flux[:, 0])
            
        if choice == 'Mag':

            return len(self.lightcurve[:, 0])

    def find_gamma(self, Teff, log_g, path='./data/'):
        """
        Return the associated gamma linear limb-darkening coefficient associated to the filter,
        the given effective
        temperature and the given surface gravity in log10 cgs.
        
        WARNING. Two strong assomption are made :
                  - the microturbulent velocity vt is fixed to 2 km/s
                  
                  - the metallicity is fixed equal to the Sun : metal=0.0
        :param Teff: The effective temperature of the source in Kelvin.
        :param log_g: The log10 surface gravity in cgs.
        :param path: path to Claret table. MODIFY THIS PLEASE!      
        :return: the microlensing linear limb-darkening coefficient gamma.       
        
        """
        # assumption   Microturbulent velocity =2km/s, metallicity= 0.0 (Sun value) Claret2011
        # convention
        vt = 2.0
        metal = 0.0

        # TODO: Use read claret generator

        claret = fits.open(path + 'Claret2011.fits')
        claret = np.array(
            [claret[1].data['log g'], claret[1].data['Teff (K)'], claret[1].data['Z (Sun)'],
             claret[1].data['Xi (km/s)'], claret[1].data['u'], claret[1].data['filter']]).T

        index_filter = np.where(claret[:, 5] == self.filter)[0]
        claret_reduce = claret[index_filter, :-1].astype(float)
        coeff_index = np.sqrt(
            (claret_reduce[:, 0] - log_g) ** 2 + (claret_reduce[:, 1] - Teff) ** 2 + (
            claret_reduce[:, 2] - metal) ** 2
            + (claret_reduce[:, 3] - vt) ** 2).argmin()
        uu = claret_reduce[coeff_index, -1]

        self.gamma = 2 * uu / (3 - uu)
    def compute_parallax(self,event,parallax):   
	
	para = microlparallax.MLParallaxes(event, parallax)
	para.parallax_combination([self])
    def clean_data(self):
        """
        Clean outliers of the telescope for the fits. Points are considered as outliers if they
        are 10 mag brighter
        or fainter than the lightcurve median or if nan appears in any columns or errobar higher
        than a 1 mag.
        
        :return: the microlensing linear limb-darkening coefficient gamma. 
        """
        
        maximum_accepted_precision = 1.0
        outliers = 5.0
        
      
        index = np.where((~np.isnan(self.lightcurve).any(axis=1)) & (
            np.abs(self.lightcurve[:, 2]) < maximum_accepted_precision))[0]

        #Should return at least 2 points
        if len(index) > 2:

            lightcurve = self.lightcurve[index]

        else:

            lightcurve = self.lightcurve

        return lightcurve

    def lightcurve_in_flux(self, clean='Yes'):
        """
        Transform magnitude to flux using m=27.4-2.5*log10(flux) convention. Transform error bar
        accordingly.
        
        :param clean: ['Yes' or 'No']. Perform or not a clean_data call to avoid outliers.
        
        :return : the lightcurve in flux, lightcurve_flux.
        """
        if clean is 'Yes':

            lightcurve = self.clean_data()

        else:

            lightcurve = self.lightcurve
        

        flux = microltoolbox.magnitude_to_flux(lightcurve[:,1])
        errflux = -lightcurve[:, 2] * flux / (2.5) * np.log(10)
        
        return np.array([lightcurve[:, 0], flux, errflux]).T
