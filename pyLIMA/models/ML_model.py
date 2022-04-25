import numpy as np
import abc
import collections
from collections import OrderedDict
import sys
import pkg_resources
from scipy import interpolate, misc

import pyLIMA.priors.parameters_boundaries
import pyLIMA.parallax.parallax



class MLmodel(object):
    """
       ######## MLmodel module ########

       This class defines the model you want to fit your data to. Model is the parent class, each model is a child
       class (polymorphism), for example ModelPSPL.

       Attributes :

           event : A event class which describe your event that you want to model. See the event module.


           parallax_model : Parallax model you want to use for the Earth types telescopes.
                      Has to be a list containing the model in the available_parallax
                      parameter and the value of topar. Have a look here for more details :
                      http://adsabs.harvard.edu/abs/2011ApJ...738...87S

                       'Annual' --> Annual parallax
                       'Terrestrial' --> Terrestrial parallax
                       'Full' --> combination of previous

                       topar --> a time in HJD choosed as the referenced time fot the parallax

                     If you have some Spacecraft types telescopes, the space based parallax
                     is computed if parallax is different of 'None'
                     More details in the microlparallax module

           xallarap_model : not available yet

           orbital_motion_model : not available yet

                   'None' --> No orbital motion
                   '2D' --> Classical orbital motion
                   '3D' --> Full Keplerian orbital motion

                   toom --> a time in HJD choosed as the referenced time fot the orbital motion
                           (Often choose equal to topar)

                   More details in the microlomotion module

           source_spots_model : not available yet

                   'None' --> No source spots

                    More details in the microlsspots module

            yoo_table : an array which contains the Yoo et al table

            Jacobian_flag : a flag indicated if a Jacobian can be used ('OK') or not.

            model_dictionnary : a python dictionnary which describe the model parameters

            pyLIMA_standards_dictionnary : the standard pyLIMA parameters dictionnary

            fancy_to_pyLIMA_dictionnary : a dictionnary which described which fancy parameters replace a standard pyLIMA
             parameter. For example : {'logrho': 'rho'}

            pyLIMA_to_fancy : a dictionnary which described the function to transform the standard pyLIMA parameter to
            the fancy one. Example :  {'logrho': lambda parameters: np.log10(parameters.rho)}

            fancy_to_pyLIMA : a dictionnary which described the function to transform the fancy parameters to
            pyLIMA standards. Example :  {'rho': lambda parameters: 10 ** parameters.logrho}

            parameters_guess : a list containing guess on pyLIMA parameters.


       :param object event: a event object. More details on the event module.
       :param list parallax: a list of [string,float] indicating the parallax model you want and to_par
       :param list xallarap: a list of [string,float] indicating the xallarap mode.l. NOT WORKING NOW.
       :param list orbital_motion: a list of [string,float] indicating the parallax model you want and to_om.
                                   NOT WORKING NOW.
       :param string source_spots: a string indicated the source_spots you want. NOT WORKING.
       """
    __metaclass__ = abc.ABCMeta

    def __init__(self, event, parallax=['None', 0.0], xallarap=['None'],
                 orbital_motion=['None', 0.0], blend_flux_parameter='fb'):
        """ Initialization of the attributes described above.
        """
        self.event = event
        self.parallax_model = parallax
        self.xallarap_model = xallarap
        self.orbital_motion_model = orbital_motion
        self.blend_flux_parameter = blend_flux_parameter

        self.photometry = False
        self.astrometry = False

        self.model_dictionnary = {}
        self.pyLIMA_standards_dictionnary = {}
        self.fancy_to_pyLIMA_dictionnary = {}
        self.pyLIMA_to_fancy = {}
        self.fancy_to_pyLIMA = {}

        self.parameters_guess = []
        self.Jacobian_flag = 'OK'
        self.parameters_boundaries = []

        self.x_center = 0.0
        self.y_center = 0.0

        self.check_data_in_event()
        self.define_pyLIMA_standard_parameters()


    @abc.abstractmethod
    def model_type(self):
        pass

    @abc.abstractmethod
    def paczynski_model_parameters(self):
        return

    @abc.abstractmethod
    def model_magnification(self, telescope, pyLIMA_parameters):
        return

    def check_data_in_event(self):

        for telescope in self.event.telescopes:

            if telescope.lightcurve_magnitude is not None:

                self.photometry = True

            if telescope.astrometry is not None:

                self.astrometry = True

    def define_pyLIMA_standard_parameters(self):
        """ Define the standard pyLIMA parameters dictionnary."""

        self.pyLIMA_standards_dictionnary = self.paczynski_model_parameters()

        for telescope in self.event.telescopes:

            if telescope.astrometry is not None:

                self.pyLIMA_standards_dictionnary['theta_E'] = len(self.pyLIMA_standards_dictionnary)

                break

        if self.parallax_model[0] != 'None':

            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['piEN'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['piEE'] = len(self.pyLIMA_standards_dictionnary)

            self.event.compute_parallax_all_telescopes(self.parallax_model)

        if self.xallarap_model[0] != 'None':

            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['XiEN'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['XiEE'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['ra_xallarap'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['dec_xallarap'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['period_xallarap'] = len(self.pyLIMA_standards_dictionnary)

            if self.xallarap_model[0] != 'Circular':

                self.pyLIMA_standards_dictionnary['eccentricity_xallarap'] = len(self.pyLIMA_standards_dictionnary)
                self.pyLIMA_standards_dictionnary['t_periastron_xallarap'] = len(self.pyLIMA_standards_dictionnary)

        if self.orbital_motion_model[0] == '2D':

            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['dsdt'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['dalphadt'] = len(self.pyLIMA_standards_dictionnary)

        if self.orbital_motion_model[0] == 'Circular':

            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['v_para'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['v_perp'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['v_radial'] = len(self.pyLIMA_standards_dictionnary)

        if self.orbital_motion_model[0] == 'Keplerian':

            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['logs_z'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['v_para'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['v_perp'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['v_radial'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['mass_lens'] = len(self.pyLIMA_standards_dictionnary)

            if 'theta_E' not in self.pyLIMA_standards_dictionnary.keys():

                self.pyLIMA_standards_dictionnary['theta_E'] = len(self.pyLIMA_standards_dictionnary)

        for telescope in self.event.telescopes:

            self.pyLIMA_standards_dictionnary['fs_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)

            if self.blend_flux_parameter == 'fb':

                self.pyLIMA_standards_dictionnary['fb_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)

            if self.blend_flux_parameter == 'g':

                self.pyLIMA_standards_dictionnary['g_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)


        self.pyLIMA_standards_dictionnary = OrderedDict(sorted(self.pyLIMA_standards_dictionnary.items(), key=lambda x: x[1]))

        self.parameters_boundaries = pyLIMA.priors.parameters_boundaries.parameters_boundaries(self)

    def define_model_parameters(self):
        """ Define the model parameters dictionnary. It is different to the pyLIMA_standards_dictionnary
         if you have some fancy parameters request.
        """

        self.model_dictionnary = self.pyLIMA_standards_dictionnary.copy()

        if len(self.pyLIMA_to_fancy) != 0:

            self.Jacobian_flag = 'No way'

            for key_parameter in self.fancy_to_pyLIMA_dictionnary.keys():

                try:
                    self.model_dictionnary[key_parameter] = self.model_dictionnary.pop(
                        self.fancy_to_pyLIMA_dictionnary[key_parameter])
                except:

                    pass

            self.model_dictionnary = OrderedDict(
                sorted(self.model_dictionnary.items(), key=lambda x: x[1]))

    def print_model_parameters(self):
        """ Define the model parameters dictionnary and print for the users.
        """
        self.define_model_parameters()

        print(self.model_dictionnary)

    def compute_the_microlensing_model(self, telescope, pyLIMA_parameters):
        """ Compute the microlens model according the injected parameters. This is modified by child submodel sublclass,
        if not the default microlensing model is returned.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :returns: the microlensing model
        :rtype: array_like
        """

        return self._default_microlensing_model(telescope, pyLIMA_parameters)

    def _default_microlensing_model(self, telescope, pyLIMA_parameters):
        """ Compute the default microlens model according the injected parameters:

        flux(t) = f_source*magnification(t)+f_blending

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :param array_like amplification: the magnification associated to the model
        :returns: the microlensing model, the microlensing priors
        :rtype: array_like, float
        """

        magnification = self.model_magnification(telescope, pyLIMA_parameters)
        f_source, f_blending = self.derive_telescope_flux(telescope, pyLIMA_parameters, magnification)

        photometric_model = f_source * magnification + f_blending

        astrometric_model = None
        microlensing_model = {'flux': photometric_model, 'astrometry': astrometric_model, 'f_source': f_source,
                              'f_blending': f_blending}
        return microlensing_model

    def derive_telescope_flux(self, telescope, pyLIMA_parameters, magnification):
        """
        Compute the source/blending flux

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :param array_like magnification: an array containing the magnification

        :returns:  the source and the blending flux
        :rtype: tuple
        """
        try:
            # Fluxes parameters are fitted
            f_source = 2 * getattr(pyLIMA_parameters, 'fs_' + telescope.name) / 2

            if self.blend_flux_parameter == 'fb':

                f_blending = 2 * getattr(pyLIMA_parameters, 'fb_' + telescope.name) / 2

            if self.blend_flux_parameter == 'g':

                g_blending = 2 * getattr(pyLIMA_parameters, 'g_' + telescope.name) / 2
                f_blending = f_source * g_blending

        except TypeError:

            # Fluxes parameters are estimated through np.polyfit
            lightcurve = telescope.lightcurve_flux
            flux = lightcurve['flux'].value
            errflux = lightcurve['err_flux'].value

            try:
                f_source, f_blending = np.polyfit(magnification, flux, 1, w=1 / errflux)
            except:

                f_source = 0.0
                f_blending = 0.0

        return f_source, f_blending

    def compute_pyLIMA_parameters(self, fancy_parameters):
        """ Realize the transformation between the fancy parameters to fit to the
        standard pyLIMA parameters needed to compute a model.

        :param list fancy_parameters: the parameters you fit
        :return: pyLIMA parameters
        :rtype: object (namedtuple)
        """

        model_parameters = collections.namedtuple('parameters', self.model_dictionnary.keys())

        for key_parameter in self.model_dictionnary.keys():

            try:

                setattr(model_parameters, key_parameter, fancy_parameters[self.model_dictionnary[key_parameter]])

            except:

                setattr(model_parameters, key_parameter, None)


        pyLIMA_parameters = self.fancy_parameters_to_pyLIMA_standard_parameters(model_parameters)

        return pyLIMA_parameters

    def fancy_parameters_to_pyLIMA_standard_parameters(self, fancy_parameters):
        """ Transform the fancy parameters to the pyLIMA standards. The output got all
        the necessary standard attributes, example to, uo, tE...


        :param object fancy_parameters: the fancy_parameters as namedtuple
        :return: the pyLIMA standards are added to the fancy parameters
        :rtype: object
        """
        if len(self.fancy_to_pyLIMA) != 0:

            for key_parameter in self.fancy_to_pyLIMA.keys():

                setattr(fancy_parameters, key_parameter, self.fancy_to_pyLIMA[key_parameter](fancy_parameters))

        return fancy_parameters

    def pyLIMA_standard_parameters_to_fancy_parameters(self, pyLIMA_parameters):
        """ Transform the  the pyLIMA standards parameters to the fancy parameters. The output got all
            the necessary fancy attributes.


        :param object pyLIMA_parameters: the  standard pyLIMA parameters as namedtuple
        :return: the fancy parameters are added to the fancy parameters
        :rtype: object
        """
        if len(self.pyLIMA_to_fancy) != 0:

            for key_parameter in self.pyLIMA_to_fancy.keys():

                setattr(pyLIMA_parameters, key_parameter, self.pyLIMA_to_fancy[key_parameter](pyLIMA_parameters))

        return pyLIMA_parameters

    def find_origin(self):

        self.x_center = 0
        self.y_center = 0

    def uo_to_from_uc_tc(self, pyLIMA_parameters):

        return pyLIMA_parameters.to, pyLIMA_parameters.uo

    def uc_tc_from_uo_to(self, pyLIMA_parameters):

        return pyLIMA_parameters.to, pyLIMA_parameters.uo

    def source_trajectory(self, telescope, pyLIMA_parameters):
        """ Compute the microlensing source trajectory associated to a telescope for the given parameters.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: source_trajectory_x, source_trajectory_y the x,y compenents of the source trajectory
        :rtype: array_like,array_like
        """
        # Linear basic trajectory

        lightcurve = telescope.lightcurve_flux
        time = lightcurve['time'].value

        tau = (time - pyLIMA_parameters.to) / pyLIMA_parameters.tE
        beta = np.array([pyLIMA_parameters.uo] * len(tau))

        # These following second order induce curvatures in the source trajectory
        # Parallax?
        if 'piEN' in pyLIMA_parameters._fields:

            piE = np.array([pyLIMA_parameters.piEN, pyLIMA_parameters.piEE])
            parallax_delta_tau, parallax_delta_beta = pyLIMA.parallax.parallax.compute_parallax_curvature(piE,
                                                                                                telescope.deltas_positions)

            tau += parallax_delta_tau
            beta += parallax_delta_beta

        # Xallarap?
        if 'XiEN' in pyLIMA_parameters._fields:

            XiE = np.array([pyLIMA_parameters.XiEN, pyLIMA_parameters.XiEE])
            ra = pyLIMA_parameters.ra_xallarap
            dec = pyLIMA_parameters.dec_xallarap
            period = pyLIMA_parameters.period_xallarap

            if 'eccentricity_xallarap' in pyLIMA_parameters._fields:
                eccentricity = pyLIMA_parameters.eccentricity_xallarap
                t_periastron = pyLIMA_parameters.t_periastron_xallarap

                orbital_elements = [telescope.lightcurve_flux[:, 0], ra, dec, period, eccentricity, t_periastron]
                xallarap_delta_tau, xallarap_delta_beta = microlxallarap.compute_xallarap_curvature(XiE,
                                                                                                    orbital_elements,
                                                                                                    mode='elliptic')
            else:

                orbital_elements = [telescope.lightcurve_flux[:, 0], ra, dec, period]
                xallarap_delta_tau, xallarap_delta_beta = microlxallarap.compute_xallarap_curvature(XiE,
                                                                                                    orbital_elements)

            tau += xallarap_delta_tau
            beta += xallarap_delta_beta

        if 'alpha' in pyLIMA_parameters._fields:

            alpha = pyLIMA_parameters.alpha

        else:

            alpha = 0

        # Orbital motion?

        if self.orbital_motion_model[0] != 'None':

            dseparation, dalpha = microlorbitalmotion.orbital_motion_shifts(self.orbital_motion_model,
                                                                            telescope.lightcurve_flux['time'].value,
                                                                            pyLIMA_parameters)
            alpha += dalpha

        else:

            dseparation = np.array([0] * len(tau))

        lens_trajectory_x = tau * np.cos(alpha) - beta * np.sin(alpha)
        lens_trajectory_y = tau * np.sin(alpha) + beta * np.cos(alpha)

        source_trajectory_x = -lens_trajectory_x
        source_trajectory_y = -lens_trajectory_y

        return source_trajectory_x, source_trajectory_y, dseparation
