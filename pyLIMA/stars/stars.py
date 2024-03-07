# -*- coding: utf-8 -*-
"""
Created on Thu March 03 12:51:19 2017

@author: ebachelet
"""

from __future__ import division

import astropy.io.fits as fits
import numpy as np
import scipy.interpolate as si

from pyLIMA.data import PACKAGE_DATA

template = PACKAGE_DATA / "Yoo_B0B1.dat"
template = PACKAGE_DATA / "Claret2011.fits"

CLARET_PATH = template


class Star(object):
    """
       ######## Star module ########

       This class create a star object .

      Attributes :

        name : The name of the star. Default is 'Random star'.

        T_eff : The effective temperature in Kelvin. Default is 5000.

        log_g : The log surface density. Default is 4.

        metallicity : The star metallicity in solar unit. Default is 0.0.

        mass : the mass in solar unit.

        gammas : (Microlensing covention) List of limb darkening coefficient
                 :math:`\\Gamma` associated to all filters.
                 The classical (Milne definition) linear limb darkening coefficient
                 can be found using:
                 u=(3*gamma)/(2+gamma).
                 Default is an empty list.
    """

    def __init__(self):
        self.name = 'Random star'
        self.T_eff = 5000  # Kelvins
        self.log_g = 4
        self.metallicity = 0.0  # Sun metallicity by default
        self.turbulent_velocity = 2.0  # km/s
        self.mass = 1.0  # Solar mass unit
        self.gammas = []  # microlensing limb-darkening coefficient

        claret_path = CLARET_PATH
        claret_table = fits.open(claret_path)

        self.claret_table = np.array(
            [claret_table[1].data['log g'], claret_table[1].data['Teff (K)'],
             claret_table[1].data['Z (Sun)'], claret_table[1].data['Xi (km/s)'],
             claret_table[1].data['u'], claret_table[1].data['filter']]).T

        self.define_claret_filter_tables()

    def define_claret_filter_tables(self):
        """
            Define the filter_claret table. For more details, see " Gravity and
            limb-darkening coefficients for
            the Kepler, CoRoT,
            Spitzer, uvby,   UBVRIJHK,
            and Sloan photometric systems"
            Claret, A. and Bloemen, S. 2011A&A...529A..75C.
        """
        all_filters = np.unique(self.claret_table[:, -1])
        self.filter_claret_table = {}

        for filter in all_filters:
            good_filter = np.where(self.claret_table[:, -1] == filter)[0]

            subclaret = self.claret_table[good_filter, :-1].astype(float)

            grid_interpolation = si.NearestNDInterpolator(subclaret[:, :-1],
                                                          subclaret[:, -1])

            self.filter_claret_table[filter] = grid_interpolation

    def find_gamma(self, filter):
        """
        Set the associated :math:`\\Gamma` linear limb-darkening coefficient
        associated to the filter.

        :param str filter: the observation filter. Need to match Claret definitions.
        :return: the (microlensing) gamma coefficient
        :rtype: float
        """

        good_table = self.filter_claret_table[filter]

        linear_limb_darkening_coefficient = good_table(self.log_g, self.T_eff,
                                                       self.metallicity,
                                                       self.turbulent_velocity)

        gamma = 2 * linear_limb_darkening_coefficient / (
                3 - linear_limb_darkening_coefficient)

        return gamma
