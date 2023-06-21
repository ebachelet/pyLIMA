# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:39:32 2015

@author: ebachelet
"""
import numpy as np
from astropy import constants as astronomical_constants
from pyLIMA.parallax import parallax
from pyLIMA.toolbox.time_series import construct_time_series, clean_time_series

# Conventions for magnitude and flux lightcurves for all pyLIMA. If the injected
# lightcurve format differs, please
# indicate this in the correponding lightcurve_magnitude_dictionnary or
# lightcurve_flux_dictionnary, see below.
PYLIMA_LIGHTCURVE_MAGNITUDE_NAMES = ['time', 'mag', 'err_mag']
PYLIMA_LIGHTCURVE_FLUX_NAMES = ['time', 'flux', 'err_flux', 'inv_err_flux']


class Telescope(object):
    """
    This class contains all information about a telescope (details, observations,
    Earth ephemerides...)

    Attributes
    ----------
    name : str, the telescope name (needs to be unique!)
    filter : str, the filter used for observations
    lightcurve_magnitude : array, the observed lightcurve in magnitude
    [time,mag,err_mag]
    lightcurve_flux : array, the observed lightcurve in flux
    [time,flux,err_flux]
    astrometry : array, the astrometric time series
    [time,ra,err_ra,dec,err_dec], should be in degree or pixel
    bad_data : dict, a dictionnary containing non-finite data and duplicates
    location : str, 'Earth' or 'Space'
    altitude : float, the telescope altitude in meter
    longitude : float, the telescope longitude in degree
    latitutde : float, the telescope latitude in degree
    deltas_positions : array, the North and East projected into the plane of
    sky positions of a telescope relative to Earth center (see parallax)
    Earth_positions : dict, dictionnary orf array containing the XYZ positions of
    Earth at time of observations
    Earth_speeds : dict, dictionnary of array containing the XYZ speeds of
    Earth at time of observations
    sidereal_timesL dict, dictionnary of array containing the sidereal time (i.e.
    angle) of a telescope on Earth
    Earth_positions_projects : dict, dictionnary of array containing the projected
    positions of Earth at time of observations
    Earth_speeds_projects : dict, dictionnary of array containing the projected
    speeds of Earth at time of observations
    spacecraft_name : str, the name of the satellite for the JPL Horizons ephemrides
    spacecraft_positions : dict, a dictionnary of arrays containing the positions of
    the satellite
    ld_gamma : float, the microlensing linear limb darkening coefficient
    ld_sigma : float, the microlensing sqrt limb darkending coefficient
    ld_a1 : float, the classic linear  limb darkening coefficient
    ld_a2 : float, the classic sqrt  limb darkening coefficient
    """

    def __init__(self, name='NDG', camera_filter='I', pixel_scale=1, light_curve=None,
                 light_curve_names=None, light_curve_units=None,
                 astrometry=None, astrometry_names=None, astrometry_units=None,
                 location='Earth', altitude=-astronomical_constants.R_earth.value,
                 longitude=0.57, latitude=49.49,
                 spacecraft_name=None,
                 spacecraft_positions={'astrometry': [], 'photometry': []}):
        """Initialization of the attributes described above."""

        self.name = name
        self.filter = camera_filter
        self.pixel_scale = pixel_scale  # mas/pix
        self.lightcurve_magnitude = None
        self.lightcurve_flux = None
        self.astrometry = None
        self.bad_data = {}

        self.location = location
        self.altitude = altitude  # default is Earth center
        self.longitude = longitude  # degrees
        self.latitude = latitude  # degrees , default is somewhere...
        self.deltas_positions = {}
        self.Earth_positions = {}
        self.Earth_speeds = {}
        self.sidereal_times = {}
        self.telescope_positions = {}
        self.Earth_positions_projected = {}
        self.Earth_speeds_projected = {}

        self.spacecraft_name = spacecraft_name  # give the true name of the
        # satellite, according to JPL horizon
        self.spacecraft_positions = spacecraft_positions.copy()  # only for space
        # base observatory, should be a list as
        # [dates(JD), ra(degree) , dec(degree) , distances(AU) ]

        # Microlensing LD coefficients
        self.ld_gamma = 0
        self.ld_sigma = 0
        # Classical LD coefficients
        self.ld_a1 = 0
        self.ld_a2 = 0

        if light_curve is not None:

            if 'mag' in light_curve_names:
                data = construct_time_series(light_curve, light_curve_names,
                                             light_curve_units)
                good_lines, non_finite_lines, non_unique_lines = clean_time_series(data)

                self.lightcurve_magnitude = data[good_lines]
                self.lightcurve_flux = self.lightcurve_in_flux()

                bad_data = {}
                bad_data['non_finite_lines'] = non_finite_lines
                bad_data['non_unique_lines'] = non_unique_lines

                self.bad_data['photometry'] = bad_data

            if 'flux' in light_curve_names:
                data = construct_time_series(light_curve, light_curve_names,
                                             light_curve_units)
                good_lines, non_finite_lines, non_unique_lines = clean_time_series(data)

                self.lightcurve_flux = data[good_lines]
                self.lightcurve_magnitude = self.lightcurve_in_magnitude()

                bad_data = {}
                bad_data['non_finite_lines'] = non_finite_lines
                bad_data['non_unique_lines'] = non_unique_lines

                self.bad_data['photometry'] = bad_data

        if astrometry is not None:
            data = construct_time_series(astrometry, astrometry_names, astrometry_units)
            good_lines, non_finite_lines, non_unique_lines = clean_time_series(data)

            self.astrometry = data[good_lines]

            bad_data = {}
            bad_data['non_finite_lines'] = non_finite_lines
            bad_data['non_unique_lines'] = non_unique_lines
            self.bad_data['astrometry'] = bad_data

        for data_type in self.bad_data:

            for key in self.bad_data[data_type]:

                if len(self.bad_data[data_type][key]) != 0:
                    print(
                        'pyLIMA found (and eliminate) some bad_data for telescope ' +
                        self.name + ', please check your_telescope.bad_data')

                    break

        self.hidden()

    def trim_data(self, photometry_mask=None, astrometry_mask=None):
        """
        Prune the telescope observations

        Parameters
        ----------
        photometry_mask : array, a boolean array to mask photometric data
        astrmetry_mask : array, a boolean array to mask astrometric data
        """
        if photometry_mask is not None:
            self.lightcurve_flux = self.lightcurve_flux[photometry_mask]
            self.lightcurve_magnitude = self.lightcurve_magnitude[photometry_mask]

            self.Earth_positions['photometry'] = self.Earth_positions['photometry'][
                photometry_mask]
            self.Earth_speeds['photometry'] = self.Earth_speeds['photometry'][
                photometry_mask]
            self.sidereal_times['photometry'] = self.sidereal_times['photometry'][
                photometry_mask]

            self.telescope_positions['photometry'] = \
                self.telescope_positions['photometry'][photometry_mask]

        if astrometry_mask is not None:
            self.astrometry = self.astrometry[astrometry_mask]

            self.Earth_positions['astrometry'] = self.Earth_positions['astrometry'][
                astrometry_mask]
            self.Earth_speeds['astrometry'] = self.Earth_speeds['astrometry'][
                astrometry_mask]
            self.sidereal_times['astrometry'] = self.sidereal_times['astrometry'][
                astrometry_mask]

            self.telescope_positions['astrometry'] = \
                self.telescope_positions['astrometry'][astrometry_mask]

    def n_data(self, choice='magnitude'):
        """
        Returns the number of photometric data

        Parameters
        ----------
        choice : str, 'flux' or 'magnitude'

        Returns
        -------
        n_data : int, the number of data points
        """
        try:
            if choice == 'flux':
                return len(self.lightcurve_flux['time'])

            if choice == 'magnitude':
                return len(self.lightcurve_magnitude['mag'])

        except ValueError:

            return 0

    def find_gamma(self, star):
        """
        NOT FUNCTIONNAL YET
        """
        self.ld_gamma = star.find_gamma(self.filter)

    def initialize_positions(self):
        """
        Compute the telescope positions relative to Earth center
        """
        self.find_Earth_positions()

        if self.location == 'Space':

            self.find_space_positions()

        else:

            self.find_sidereal_time()
            self.find_Earth_telescope_positions()

    def find_Earth_positions(self):
        """
        Find the Earh positions relative to photometric and astrometric data
        """
        for data_type in ['astrometry', 'photometry']:

            if data_type == 'photometry':

                data = self.lightcurve_flux

            else:

                data = self.astrometry

            if data is not None:
                time = data['time'].value

                earth_positions, earth_speeds = parallax.Earth_ephemerides(time)
                self.Earth_positions[data_type] = earth_positions
                self.Earth_speeds[data_type] = earth_speeds

    def find_sidereal_time(self, sidereal_type='mean'):
        """
        Returns the sidereal time (angle to vernal point) for each observations

        Parameters
        ----------
        sidereal_type : str, 'mean' or 'apparent' (much, much slower!)
        """
        for data_type in ['astrometry', 'photometry']:

            if data_type == 'photometry':

                data = self.lightcurve_flux

            else:

                data = self.astrometry

            if data is not None:
                time = data['time'].value

                sidereal_times = parallax.Earth_telescope_sidereal_times(time,
                                                                         sidereal_type=sidereal_type)

                self.sidereal_times[data_type] = sidereal_times

    def find_Earth_telescope_positions(self):
        """
        Compute the telescope positions relative to Earth center based on altitude,
        longitude and latitude
        """
        for data_type in ['astrometry', 'photometry']:

            if data_type == 'photometry':

                data = self.lightcurve_flux

            else:

                data = self.astrometry

            if data is not None:
                sidereal_times = self.sidereal_times[data_type]

                telescope_positions = parallax.terrestrial_parallax(sidereal_times,
                                                                    self.altitude,
                                                                    self.longitude,
                                                                    self.latitude)

                self.telescope_positions[data_type] = telescope_positions

    def find_space_positions(self):
        """
        Compute the satellite positions relaitve to Earth center
        """
        for data_type in ['astrometry', 'photometry']:

            if data_type == 'photometry':

                data = self.lightcurve_flux

            else:

                data = self.astrometry

            if data is not None:
                time = data['time'].value

                satellite_positions, space_positions = parallax.space_ephemerides(self,
                                                                                  time,
                                                                                  data_type=data_type)

                self.telescope_positions[data_type] = satellite_positions
                self.spacecraft_positions[data_type] = space_positions

    def compute_parallax(self, parallax_model, North_vector, East_vector):
        """
        Compute and set the deltas_positions attributes according to the parallax model.

        Parameters
        ----------

        parallax_model : list, [str,float] the parallax model and t0_par
        North_vector: array, the projected North vector to project delta_position into
        East_vector: array, the projected Eat vector to project delta_position into
        details in microlparallax module.
        """
        self.initialize_positions()
        parallax.parallax_combination(self, parallax_model, North_vector,
                                      East_vector)  # , right_ascension)
        print('Parallax(' + parallax_model[
            0] + ') estimated for the telescope ' + self.name + ': SUCCESS')

    def lightcurve_in_flux(self):
        """
        Transform lightcurve magnitude to lightcurve flux
        """
        import pyLIMA.toolbox.brightness_transformation

        lightcurve = self.lightcurve_magnitude

        time = lightcurve['time'].value
        mag = lightcurve['mag'].value
        err_mag = lightcurve['err_mag'].value

        flux = pyLIMA.toolbox.brightness_transformation.magnitude_to_flux(mag)
        err_flux = pyLIMA.toolbox.brightness_transformation \
            .error_magnitude_to_error_flux(
            err_mag, flux)
        inv_err_flux = 1.0 / err_flux
        lightcurve_in_flux = construct_time_series(
            np.c_[time, flux, err_flux, inv_err_flux],
            PYLIMA_LIGHTCURVE_FLUX_NAMES,
            [lightcurve['time'].unit, 'w/m^2', 'w/m^2', 'm^2/W'])

        return lightcurve_in_flux

    def lightcurve_in_magnitude(self):
        """
        Transform lightcurve flux to lightcurve  magnitude
        """
        import pyLIMA.toolbox.brightness_transformation

        lightcurve = self.lightcurve_flux

        time = lightcurve['time'].value
        flux = lightcurve['flux'].value
        err_flux = lightcurve['err_flux'].value

        mag = pyLIMA.toolbox.brightness_transformation.flux_to_magnitude(flux)
        err_mag = pyLIMA.toolbox.brightness_transformation \
            .error_flux_to_error_magnitude(
            err_flux, flux)
        lightcurve_in_mag = construct_time_series(np.c_[time, mag, err_mag],
                                                  PYLIMA_LIGHTCURVE_MAGNITUDE_NAMES,
                                                  [lightcurve['time'].unit, 'mag',
                                                   'mag'])

        return lightcurve_in_mag

    def plot_data(self, choice='Mag'):
        """
        Plot the photometric data

        Parameters
        ----------

        choice : str, 'Mag' or 'Flux'
        """
        from pyLIMA.toolbox import plots
        import matplotlib.pyplot as plt

        if choice == 'Mag':
            plots.plot_light_curve_magnitude(self.lightcurve_magnitude['time'].value,
                                             self.lightcurve_magnitude['mag'].value,
                                             self.lightcurve_magnitude['err_mag'].value,
                                             name=self.name)

            plt.gca().invert_yaxis()

    def define_limb_darkening_coefficients(self):
        """
        Transform ld_gamma and ld_sigma to ld_a1 and ld_a2,  and/or vice-versa.
        See https://iopscience.iop.org/article/10.1086/378196/pdf
        """

        if self.ld_gamma == 0:
            self.ld_gamma = 10 * self.ld_a1 / (15 - 5 * self.ld_a1 - 3 * self.ld_a2)

        if self.ld_sigma == 0:
            self.ld_sigma = 12 * self.ld_a2 / (15 - 5 * self.ld_a1 - 3 * self.ld_a2)

        if self.ld_a1 == 0:
            self.ld_a1 = 6 * self.ld_gamma / (4 + 2 * self.ld_gamma + self.ld_sigma)

        if self.ld_a2 == 0:
            self.ld_a2 = 5 * self.ld_sigma / (4 + 2 * self.ld_gamma + self.ld_sigma)

    def hidden(self):

        if self.name == 'Mexicola':

            try:

                import webbrowser
                controller = webbrowser.get()

                controller.open("https://www.youtube.com/watch?v=GcQdU2qA7D4&t=1684s")

            except webbrowser.Error:

                pass
