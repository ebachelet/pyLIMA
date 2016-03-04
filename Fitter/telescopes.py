# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:39:32 2015

@author: ebachelet
"""
from __future__ import division

import numpy as np
import astropy.io.fits as fits

import microlparallax


class Telescope(object):
    """
    ######## Telescope module ########
    @author : Etienne Bachelet

    This module create a telescope object with the informations (attributes) needed for the fits.

    Keyword arguments:

    name --> name of the telescope. Should be a string. Default is 'None'

    kind --> 'Earth' or 'Spacecraft'. Default is 'Earth'

    filter --> telescope filter used. Should be a string wich follows the convention of :
               " Gravity and limb-darkening coefficients for the Kepler, CoRoT, Spitzer, uvby,
               UBVRIJHK,
               and Sloan photometric systems"
               Claret, A. and Bloemen, S. 2011A&A...529A..75C. For example, 'I' (default) is
               Johnson_Cousins I filter
               and 'i'' is the SDSS-i' Sloan filter.

    lightcurve --> List of time, magnitude, error in magnitude covention. Default is an empty list.
                   WARNING : Have to exactly follow this convention.

    lightcurve_flux --> List of time, flux, error in flux. Default is an empty list.
                        WARNING : has to be set before any fits.

    altitude --> Altitude in meter of the given observatory. Default is 0.0 (sea level).

    longitude --> Longitude in degrees. Default is 0.0.

    latitude --> Latitude in degrees. Default is 0.0.

    gamma.--> (Microlensing covention) Limb darkening coefficient associated to the filter
              The classical (Milne definition) linear limb darkening coefficient can be found using:
              u=(3*gamma)/(2+gamma).
              Default is 0.5.
    """

    def __init__(self, name='None', camera_filter='I', light_curve=None):
        """ Initialization of the attributes described above."""
        self.name = name
        self.kind = 'Earth'
        self.filter = camera_filter  # Claret2011 convention

        self.lightcurve = light_curve
        if self.lightcurve is None:
            self.lightcurve = np.array()

        self.lightcurve_flux = []
        self.lightcurve_flux_aligned = []
        self.altitude = 0.0
        self.longitude = 0.0
        self.latitude = 0.0
        self.gamma = 0.5
        self.deltas_positions = []

    def compute_parallax(self, event, model):

        para = microlparallax.MLParallaxes(event, model)
        para.parallax_combination()

    def n_data(self, choice):
        """ Return the number of data points in the lightcurve."""

        if choice == 'Flux':

            return len(self.lightcurve_flux[:, 0])
        else:

            return len(self.lightcurve[:, 0])

    def find_gamma(self, Teff, log_g, path):
        """
        Return the associated gamma linear limb-darkening coefficient associated to the filter,
        the given effective
        temperature and the given surface gravity in log10 cgs.

        Keyword arguments:

        Teff --> The effective temperature of the source in Kelvin.
        log_g --> The log10 surface gravity in cgs.
                 WARNING : Two strong assomption are made :
                 the microturbulent velocity vt is fixed to 2 km/s
            -    the metallicity is fixed equal to the Sun : metal=0.0

        Return :

        gamma
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
        self.gamma = 0.5

    def clean_data(self):
        """
        Clean outliers of the telescope for the fits. Points are considered as outliers if they
        are 10 mag brighter
        or fainter than the lightcurve median or if nan appears in any columns or errobar higher
        than a 1 mag.

        Return :

        the lightcurve without outliers.
        """
        # self.lightcurve=self.lightcurve[~np.isnan(self.lightcurve).any(axis=1)]
        precision = 50
        # index = np.where((np.isnan(self.lightcurve).any(axis=1)) | (
        #    np.abs(self.lightcurve[:, 1] - np.median(self.lightcurve[:, 1])) > 5) | (
        #                 np.abs(self.lightcurve[:, 2]) > precision))[
        #    0]
        index = np.where((np.isnan(self.lightcurve).any(axis=1)) | (
            np.abs(self.lightcurve[:, 2]) > precision))[
            0]
        # for i in index:
        # print self.name + ' point ' + str(self.lightcurve[i]) + ' is consider as outlier and
        # will be ' + \
        # 'rejected for the fit'
        # index = np.where((~np.isnan(self.lightcurve).any(axis=1)) & (
        #    np.abs(self.lightcurve[:, 1] - np.median(self.lightcurve[:, 1])) < 5) & (
        #                 np.abs(self.lightcurve[:, 2]) < precision))[
        #    0]
        index = np.where((~np.isnan(self.lightcurve).any(axis=1)) & (
            np.abs(self.lightcurve[:, 2]) < precision))[
            0]

        if len(index) > 2:

            lightcurve = self.lightcurve[index]

        else:

            lightcurve = self.lightcurve

        return lightcurve

    def lightcurve_in_flux(self, clean):
        """
        Transform magnitude to flux using m=27.4-2.5*log10(flux) convention. Transform error bar
        accordingly.
        Perform a clean_data call to avoid outliers.

        Return :

        lighhtcurve_flux : the lightcurve in flux.
        """
        if clean is 'Yes':

            lightcurve = self.clean_data()

        else:

            lightcurve = self.lightcurve

        flux = 10 ** ((27.4 - lightcurve[:, 1]) / 2.5)
        errflux = -lightcurve[:, 2] * flux / (2.5) * np.log(10)
        self.lightcurve_flux = np.array([lightcurve[:, 0], flux, errflux]).T
