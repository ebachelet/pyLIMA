# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:39:32 2015

@author: ebachelet
"""
from __future__ import division
import numpy as np

from pyLIMA.toolbox.time_series import construct_time_series

# Conventions for magnitude and flux lightcurves for all pyLIMA. If the injected lightcurve format differs, please
# indicate this in the correponding lightcurve_magnitude_dictionnary or lightcurve_flux_dictionnary, see below.
PYLIMA_LIGHTCURVE_MAGNITUDE_NAMES = ['time', 'mag', 'err_mag']
PYLIMA_LIGHTCURVE_FLUX_NAMES = ['time', 'flux', 'err_flux', 'inv_err_flux']

class Telescope(object):
    """
    ######## Telescope module ########

    This class create a telescope object.

    Attributes :

        location : The location of your observatory. Should be "Earth" (default) or "Space".

        altitude : Altitude in meter of the telescope. Default is 0.0 (sea level).

        longitude : Longitude of the telescope in degrees. Default is 0.57.

        latitude : Latitude in degrees. Default is 49.49 .

        gamma : (Microlensing covention) Limb darkening coefficient :math:`\\Gamma` associated to the filter.
                 The classical (Milne definition) linear limb darkening coefficient can be found using:
                 u=(3*gamma)/(2+gamma).
                 Default is 0.0 (i.e uniform source brightness)

        lightcurve_magnitude : a numpy array describing your data in magnitude.

        lightcurve_magnitude_dictionnary : a python dictionnary to transform the lightcurve_magnitude
                                          input to pyLIMA convention.

        lightcurve_flux : a numpy array describing your data in flux.

        lightcurve_flux_dictionnary : a python dictionnary to transform the lightcurve_flux input to pyLIMA convention.

        reference_flux : a float used for the transformation of difference fluxes to real fluxes. Default is 10000.0 .

        deltas_positions : a list containing the position shift of this observatory due to parallax.
                           Default is an empty list. More details in microlparallax.

    :param string name: name of the telescope. Default is 'NDG'

    :param string camera_filter: telescope filter used. Should be a string which follows the convention of :
               " Gravity and limb-darkening coefficients for the Kepler, CoRoT, Spitzer, uvby,
               UBVRIJHK,
               and Sloan photometric systems"
               Claret, A. and Bloemen, S. 2011A&A...529A..75C. For example, 'I' (default) is
               Johnson_Cousins I filter
               and 'i'' is the SDSS-i' Sloan filter.


    :param array_like light_curve_magnitude:  a numpy array with time, magnitude and error in magnitude.
                                              Default is an None.

    :param dict light_curve_magnitude_dictionnary: a python dictionnary that informs your columns convention. Used to
                                                   translate to pyLIMA convention [time,mag,err_mag].
                                                   Default is {'time': 0, 'mag' : 1, 'err_mag' : 2 }


    :param array-like light_curve_flux:  a numpy array with time, flux and error in flux. Default is an None.

    :param dict light_curve_flux_dictionnary: a python dictionnary that informs your columns convention. Used to
                                              translate to pyLIMA convention [time,flux,err_flux].
                                              Default is {'time': 0, 'flux' : 1, 'err_flux' : 2 }

    :param float reference_flux: a float used for the transformation of difference fluxes to real fluxes.
                                 Default is 0.0 .

    :param str clean_the_lightcurve : a string indicated if you want pyLIMA to clean your lightcurves.
                                      Highly recommanded!
    """

    def __init__(self, name='NDG', camera_filter='I', light_curve=None,
                 light_curve_names=None, light_curve_units=None, clean_the_light_curve=False,
                 location = 'Earth', spacecraft_name=None,
                 astrometry=None, astrometry_names=None, astrometry_units=None):
        """Initialization of the attributes described above."""

        self.name = name
        self.filter = camera_filter  # Claret2011 convention
        self.lightcurve_magnitude = None
        self.lightcurve_flux = None
        self.astrometry = None

        if light_curve is not None:
            if 'mag' in light_curve_units:

                self.lightcurve_magnitude = construct_time_series(light_curve, light_curve_names, light_curve_units)
                self.lightcurve_flux = self.lightcurve_in_flux()

            if 'flux' in light_curve_units:
                self.lightcurve_magnitude = construct_time_series(light_curve, light_curve_names, light_curve_units)
                self.lightcurve_flux = self.lightcurve_in_magnitude()

        if astrometry is not None:

            self.astrometry = construct_time_series(astrometry, astrometry_names, astrometry_units)

        self.location = location
        self.altitude = 0.0  # meters
        self.longitude = 0.57  # degrees
        self.latitude = 49.49  # degrees
        self.gamma = 0.0  # This mean you will fit uniform source brightness
        self.deltas_positions = []
        self.spacecraft_name = spacecraft_name # give the true name of the satellite, according to JPL horizon
        self.spacecraft_positions = [] #only for space base observatory, should be a list as
                                       # [dates(JD), ra(degree) , dec(degree) , distances(AU) ]
        self.hidden()

    def n_data(self, choice='magnitude'):
        """ Return the number of data points in the lightcurve.

        :param string choice: 'magnitude' (default) or 'flux' The unit you want to check data for.

        :return: the size of the corresponding lightcurve
        :rtype: int
        """
        try:
            if choice == 'flux':
                return len(self.lightcurve_flux['time'])

            if choice == 'magnitude':
                return len(self.lightcurve_magnitude['mag'])
        except:

            return 0
    def find_gamma(self, star):
        """
        Set the associated :math:`\\Gamma` linear limb-darkening coefficient associated to the filter.


        :param object star: a stars object.

        """

        self.gamma = star.find_gamma(self.filter)

    def compute_parallax(self, parallax_obj):
        """ Compute and set the deltas_positions attribute due to the parallax.

        :param object event: a event object. More details in the event module.
        :param list parallax: a list containing the parallax model and to_par. More details in microlparallax module.
        """

        parallax_obj.parallax_combination(self)
        print('Parallax(' + parallax_obj.parallax_model + ') estimated for the telescope ' + self.name + ': SUCCESS')

    def clean_data_magnitude(self):
        """
        Clean outliers of the telescope for the fits. Points are considered as outliers if they
        are 10 mag brighter
        or fainter than the lightcurve median or if nan appears in any columns or errobar higher
        than a 1 mag.

        :return: the cleaned magnitude lightcurve
        :rtype: array_like
        """

        maximum_accepted_precision = 1.0

        index = np.where((~np.isnan(self.lightcurve_magnitude).any(axis=1)) &
                         (np.abs(self.lightcurve_magnitude[:, 2]) <= maximum_accepted_precision))[0]

        lightcurve = self.lightcurve_magnitude[index]

        index = np.where((np.isnan(self.lightcurve_magnitude).any(axis=1)) |
                         (np.abs(self.lightcurve_magnitude[:, 2]) > maximum_accepted_precision))[0]
        if len(index) != 0:
            self.bad_points_magnitude = index
            print('pyLIMA found some bad points in the telescope ' + self.name + ', you can found these in the ' \
                                                                                 'bad_points_magnitude attribute.')

        return lightcurve

    def clean_data_flux(self):
        """
        Clean outliers of the telescope for the fits. Points are considered as outliers if they
        are 10 mag brighter
        or fainter than the lightcurve median or if nan appears in any columns or errobar higher
        than a 1 mag.

        :return: the cleaned magnitude lightcurve
        :rtype: array_like
        """

        maximum_accepted_precision = 1.0
        flux = self.lightcurve_flux[:, 1]
        error_flux = self.lightcurve_flux[:, 2]
        index = np.where(
            (~np.isnan(self.lightcurve_flux).any(axis=1)) & (np.abs(error_flux / flux) <= maximum_accepted_precision) & (flux>0))[
            0]

        lightcurve = self.lightcurve_flux[index]

        index = np.where(
            (np.isnan(self.lightcurve_flux).any(axis=1)) | (np.abs(error_flux / flux) > maximum_accepted_precision) | (flux<=0))[0]
        if len(index) != 0:
            self.bad_points_flux = index
            print('pyLIMA found some bad points in the telescope ' + self.name + ', you can found these in the ' \
                                                                                 'bad_points_flux attribute.')
        return lightcurve

    def lightcurve_in_flux(self):
        """
        Transform magnitude to flux using m=27.4-2.5*log10(flux) convention. Transform error bar
        accordingly. More details in microltoolbox module.

        :return: the lightcurve in flux, lightcurve_flux.
        :rtype: array_like
        """
        import pyLIMA.toolbox.brightness_transformation

        lightcurve = self.lightcurve_magnitude

        time = lightcurve['time'].value
        mag = lightcurve['mag'].value
        err_mag = lightcurve['err_mag'].value

        flux = pyLIMA.toolbox.brightness_transformation.magnitude_to_flux(mag)
        err_flux = pyLIMA.toolbox.brightness_transformation.error_magnitude_to_error_flux(err_mag, flux)
        inv_err_flux = 1.0/err_flux
        lightcurve_in_flux = construct_time_series(np.c_[time, flux, err_flux, inv_err_flux],
                                                   PYLIMA_LIGHTCURVE_FLUX_NAMES,
                                                   [lightcurve['time'].unit, 'w/m^2', 'w/m^2', 'm^2/W'])

        return lightcurve_in_flux

    def lightcurve_in_magnitude(self):
        """
        Transform flux to magnitude using m = 27.4-2.5*log10(flux) convention. Transform error bar
        accordingly. More details in microltoolbox module.

        :return: the lightcurve in magnitude, lightcurve_magnitude.
        :rtype: array_like
        """
        import pyLIMA.toolbox.brightness_transformation

        lightcurve = self.lightcurve_flux

        time = lightcurve['time'].value
        flux = lightcurve['flux'].value
        err_flux = lightcurve['err_flux'].value

        mag = pyLIMA.toolbox.brightness_transformation.flux_to_magnitude(flux)
        err_mag = pyLIMA.toolbox.brightness_transformation.error_flux_to_error_magnitude(err_flux, flux)
        lightcurve_in_mag = construct_time_series(np.c_[time, mag, err_mag], PYLIMA_LIGHTCURVE_MAGNITUDE_NAMES,
                                                   [lightcurve['time'].unit, 'mag', 'mag'])

        return lightcurve_in_mag
    def plot_data(self, choice='Mag'):

        from pyLIMA.toolbox import plots
        import matplotlib.pyplot as plt

        if choice=='Mag':
            plots.plot_light_curve_magnitude(self.lightcurve_magnitude['time'].value,
                                             self.lightcurve_magnitude['mag'].value,
                                             self.lightcurve_magnitude['err_mag'].value,
                                             name=self.name)

            plt.gca().invert_yaxis()


    def hidden(self):
        try:
            import webbrowser
            controller = webbrowser.get()

            if self.name =='Mexicola':
                controller.open("https://www.youtube.com/watch?v=GcQdU2qA7D4&t=1684s")
        except:

            pass
