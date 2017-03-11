# -*- coding: utf-8 -*-
"""
Created on Thu March 03 12:51:19 2017

@author: ebachelet
"""

from __future__ import division

import numpy as np
import os
import astropy.io.fits as fits
import scipy.interpolate as si

claret_path = os.path.abspath(__file__)
CLARET_PATH, filename = os.path.split(claret_path)
CLARET_PATH += '/data/'


class Star(object):
    """
       ######## Star module ########

       This module create a telescope object with the informations (attributes) needed for the fits.

       Attributes :



       """

    def __init__(self):
        self.name = 'Random star'
        self.Teff = 5000  # Kelvins
        self.log_g = 4
        self.metallicity = 0.0  # Sun metallicity by default
        self.turbulent_velocity = 2.0
        self.mass = 1.0  # Solar mass unit
        self.gammas = []  # microlensing limb-darkening coefficient
        claret_path = CLARET_PATH
        claret_table = fits.open(claret_path + 'Claret2011.fits')

        self.claret_table = np.array([claret_table[1].data['log g'], claret_table[1].data['Teff (K)'],
                             claret_table[1].data['Z (Sun)'], claret_table[1].data['Xi (km/s)'],
                             claret_table[1].data['u'], claret_table[1].data['filter']]).T

        self.define_claret_filter_tables()

    def define_claret_filter_tables(self):

        all_filters = np.unique(self.claret_table[:,-1])
        self.filter_claret_table = {}

        for filter in all_filters:

            good_filter  = np.where(self.claret_table[:,-1] == filter)[0]

            subclaret = self.claret_table[good_filter,:-1].astype(float)

            grid_interpolation = si.NearestNDInterpolator(subclaret[:,:-1],subclaret[:,-1])

            self.filter_claret_table[filter] = grid_interpolation

    def find_gamma(self, filter):
        """
        Set the associated :math:`\\Gamma` linear limb-darkening coefficient associated to the filter,
        the given effective
        temperature and the given surface gravity in log10 cgs.

        WARNING. Two strong assomption are made :
                  - the microturbulent velocity turbulent_velocity is fixed to 2 km/s

                  - the metallicity is fixed equal to the Sun : metallicity=0.0
        :param float Teff: The effective temperature of the source in Kelvin.
        :param float log_g: The log10 surface gravity in cgs.
        :param string claret_path: path to the Claret table.
        """

        good_table = self.filter_claret_table[filter]



        linear_limb_darkening_coefficient = good_table(self.log_g, self.Teff, self.metallicity, self.turbulent_velocity)

        gamma = 2 * linear_limb_darkening_coefficient / (3 - linear_limb_darkening_coefficient)

        return gamma
