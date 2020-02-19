# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:32:13 2015

@author: ebachelet
"""

from __future__ import division
from collections import OrderedDict
import os.path
import abc
import sys
from scipy import interpolate, misc
import pkg_resources

thismodule = sys.modules[__name__]

import numpy as np
import collections
import time as python_time

from pyLIMA import microlguess
from pyLIMA import microlmagnification
from pyLIMA import microlpriors
from pyLIMA import microlparallax
from pyLIMA import microlorbitalmotion
from pyLIMA import microlcaustics
from pyLIMA import stars
from pyLIMA import microlxallarap

resource_package = __name__
resource_path = '/'.join(('data', 'Yoo_B0B1.dat'))
template = pkg_resources.resource_filename(resource_package, resource_path)
try:

    # yoo_table = np.loadtxt('b0b1.dat')
    yoo_table = np.loadtxt(template)
except:

    print('ERROR : No Yoo_B0B1.dat file found, please check!')

b0b1 = yoo_table
zz = b0b1[:, 0]
b0 = b0b1[:, 1]
b1 = b0b1[:, 2]
# db0 = b0b1[:,4]
# db1 = b0b1[:, 5]
interpol_b0 = interpolate.interp1d(zz, b0, kind='linear')
interpol_b1 = interpolate.interp1d(zz, b1, kind='linear')
# import pdb; pdb.set_trace()

dB0 = misc.derivative(lambda x: interpol_b0(x), zz[1:-1], dx=10 ** -4, order=3)
dB1 = misc.derivative(lambda x: interpol_b1(x), zz[1:-1], dx=10 ** -4, order=3)
dB0 = np.append(2.0, dB0)
dB0 = np.concatenate([dB0, [dB0[-1]]])
dB1 = np.append((2.0 - 3 * np.pi / 4), dB1)
dB1 = np.concatenate([dB1, [dB1[-1]]])
interpol_db0 = interpolate.interp1d(zz, dB0, kind='linear')
interpol_db1 = interpolate.interp1d(zz, dB1, kind='linear')
yoo_table = [zz, interpol_b0, interpol_b1, interpol_db0, interpol_db1]


class ModelException(Exception):
    pass


def create_model(model_type, event, model_arguments=[], parallax=['None', 0.0], xallarap=['None'],
                 orbital_motion=['None', 0.0], source_spots='None', blend_flux_ratio=True):
    """
    Load a model according to the supplied model_type. Models are expected to be named
    Model<model_type> e.g. ModelPSPL

    :param string model_type: Model type e.g. PSPL
    :return: Model object for given model_type
    """

    try:

        model = getattr(thismodule, 'Model{}'.format(model_type))

    except AttributeError:

        raise ModelException('Unknown model "{}"'.format(model_type))

    return model(event, model_arguments, parallax, xallarap,
                 orbital_motion, source_spots, blend_flux_ratio)


class MLModel(object):
    """
       ######## MLModels module ########

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

    def __init__(self, event, model_arguments=[], parallax=['None', 0.0], xallarap=['None'],
                 orbital_motion=['None', 0.0], source_spots='None', blend_flux_ratio=True):
        """ Initialization of the attributes described above.
        """
        self.event = event
        self.model_arguments = model_arguments
        self.parallax_model = parallax
        self.xallarap_model = xallarap
        self.orbital_motion_model = orbital_motion
        self.source_spots_model = source_spots
        self.yoo_table = yoo_table
        self.blend_flux_ratio = blend_flux_ratio
        self.variable_blend = None

        self.model_dictionnary = {}
        self.pyLIMA_standards_dictionnary = {}

        self.fancy_to_pyLIMA_dictionnary = {}
        self.pyLIMA_to_fancy = {}
        self.fancy_to_pyLIMA = {}

        self.parameters_guess = []
        self.Jacobian_flag = 'OK'

        self.define_pyLIMA_standard_parameters()

        # binary lens model specific
        self.binary_origin = None
        self.x_center = None
        self.y_center = None

    @abc.abstractproperty
    def model_type(self):
        pass

    @abc.abstractmethod
    def paczynski_model_parameters(self):
        return

    @abc.abstractmethod
    def model_magnification(self, telescope, pyLIMA_parameters):
        return

    def define_pyLIMA_standard_parameters(self):
        """ Define the standard pyLIMA parameters dictionnary."""

        self.pyLIMA_standards_dictionnary = self.paczynski_model_parameters()

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
            self.pyLIMA_standards_dictionnary['rE'] = len(self.pyLIMA_standards_dictionnary)

        if self.orbital_motion_model[0] == '3D':
            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['logs_z'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['v_para'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['v_perp'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['v_radial'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['mass_lens'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['rE'] = len(self.pyLIMA_standards_dictionnary)

        if self.orbital_motion_model[0] == 'Keplerian_direct':

            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['a_true'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['period'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['inclination'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['omega_node'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['omega_periastron'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['t_periastron'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['rE'] = len(self.pyLIMA_standards_dictionnary)

        if self.source_spots_model != 'None':
            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['spot'] = len(self.pyLIMA_standards_dictionnary)

        for telescope in self.event.telescopes:
            self.pyLIMA_standards_dictionnary['fs_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)

            if self.blend_flux_ratio:
                self.pyLIMA_standards_dictionnary['g_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)
            else:
                self.pyLIMA_standards_dictionnary['fb_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)

        self.pyLIMA_standards_dictionnary = OrderedDict(
            sorted(self.pyLIMA_standards_dictionnary.items(), key=lambda x: x[1]))

        self.parameters_boundaries = microlguess.differential_evolution_parameters_boundaries(self)

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

        amplification = self.model_magnification(telescope, pyLIMA_parameters)
        return self._default_microlensing_model(telescope, pyLIMA_parameters, amplification)

    def _default_microlensing_model(self, telescope, pyLIMA_parameters, amplification):
        """ Compute the default microlens model according the injected parameters:

        flux(t) = f_source*magnification(t)+f_blending

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :param array_like amplification: the magnification associated to the model
        :returns: the microlensing model, the microlensing priors
        :rtype: array_like, float
        """

        f_source, g_blending = self.derive_telescope_flux(telescope, pyLIMA_parameters, amplification)

        if self.blend_flux_ratio:
            microlensing_model = f_source * (amplification + g_blending)
        else:
            microlensing_model = f_source * amplification + g_blending

        return microlensing_model, f_source, g_blending

    def derive_telescope_flux(self, telescope, pyLIMA_parameters, amplification):
        """
        Compute the source/blending flux

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :param array_like amplification: an array containing the magnification

        :returns:  the source and the blending flux
        :rtype: tuple
        """
        try:
            # Fluxes parameters are fitted
            f_source = 2 * getattr(pyLIMA_parameters, 'fs_' + telescope.name) / 2

            if self.blend_flux_ratio:
                g_blending = 2 * getattr(pyLIMA_parameters, 'g_' + telescope.name) / 2
            else:
                g_blending = 2 * getattr(pyLIMA_parameters, 'fb_' + telescope.name) / 2


        except TypeError:

            # Fluxes parameters are estimated through np.polyfit
            lightcurve = telescope.lightcurve_flux
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]

            try:
                f_source, f_blending = np.polyfit(amplification, flux, 1, w=1 / errflux)

                if self.blend_flux_ratio:
                    g_blending = f_blending / f_source
                else:
                    g_blending = f_blending

            except:

                f_source = 0.0
                g_blending = 0.0

        return f_source, g_blending

    def compute_pyLIMA_parameters(self, fancy_parameters):
        """ Realize the transformation between the fancy parameters to fit to the
        standard pyLIMA parameters needed to compute a model.

        :param list fancy_parameters: the parameters you fit
        :return: pyLIMA parameters
        :rtype: object (namedtuple)
        """
        # start_time = python_time.time()

        model_parameters = collections.namedtuple('parameters', self.model_dictionnary.keys())

        for key_parameter in self.model_dictionnary.keys():

            try:

                setattr(model_parameters, key_parameter, fancy_parameters[self.model_dictionnary[key_parameter]])

            except:
                setattr(model_parameters, key_parameter, None)



        # print 'arange', python_time.time() - start_time

        pyLIMA_parameters = self.fancy_parameters_to_pyLIMA_standard_parameters(model_parameters)

        # print 'conversion', python_time.time() - start_time

        if self.binary_origin:
            self.x_center = None
            self.y_center = None

            self.find_origin(pyLIMA_parameters)
        return pyLIMA_parameters

    def find_origin(self, pyLIMA_parameters):

        self.x_center = 0
        self.y_center = 0

    def uo_to_from_uc_tc(self, pyLIMA_parameters):

        return pyLIMA_parameters.to, pyLIMA_parameters.uo

    def uc_tc_from_uo_to(selfself,pyLIMA_parameters):

        return pyLIMA_parameters.to, pyLIMA_parameters.uo

    def fancy_parameters_to_pyLIMA_standard_parameters(self, fancy_parameters):
        """ Transform the fancy parameters to the pyLIMA standards. The output got all
        the necessary standard attributes, example to, uo, tE...


        :param object fancy_parameters: the fancy_parameters as namedtuple
        :return: the pyLIMA standards are added to the fancy parameters
        :rtype: object
        """
        # start_time = python_time.time()
        if len(self.fancy_to_pyLIMA) != 0:
            #import pdb;
            #pdb.set_trace()
            for key_parameter in self.fancy_to_pyLIMA.keys():
                setattr(fancy_parameters, key_parameter, self.fancy_to_pyLIMA[key_parameter](fancy_parameters))

        # print 'fancy to PYLIMA', python_time.time() - start_time
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

    def source_trajectory(self, telescope, to, uo, tE, pyLIMA_parameters):
        """ Compute the microlensing source trajectory associated to a telescope for the given parameters.

        :param float to: time of maximum magnification
        :param float uo: minimum impact parameter
        :param float tE: angular Einstein ring crossing time
        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: source_trajectory_x, source_trajectory_y the x,y compenents of the source trajectory
        :rtype: array_like,array_like
        """
        # Linear basic trajectory

        lightcurve = telescope.lightcurve_flux
        time = lightcurve[:, 0]

        tau = (time - to) / tE
        beta = np.array([uo] * len(tau))

        # These following second order induce curvatures in the source trajectory
        # Parallax?
        if 'piEN' in pyLIMA_parameters._fields:
            piE = np.array([pyLIMA_parameters.piEN, pyLIMA_parameters.piEE])
            parallax_delta_tau, parallax_delta_beta = microlparallax.compute_parallax_curvature(piE,
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
                                                                                telescope.lightcurve_flux[:, 0],
                                                                                pyLIMA_parameters)
                alpha += dalpha

        else :

           dseparation = np.array([0]*len(tau))
                
        source_trajectory_x = tau * np.cos(alpha) - beta * np.sin(alpha)
        source_trajectory_y = tau * np.sin(alpha) + beta * np.cos(alpha)


    

                
        return source_trajectory_x, source_trajectory_y, dseparation
        

class ModelPSPL(MLModel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: PSPL
        :rtype: string
        """
        return 'PSPL'

    def paczynski_model_parameters(self):
        """ Define the PSPL standard parameters, [to,uo,tE]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2}

        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a PSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification
        :rtype: array_like
        """

        source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        return microlmagnification.amplification_PSPL(source_trajectory_x, source_trajectory_y)

    def Jacobian_model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a PSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification, impact_parameter
        :rtype: array_like,array_like
        """

        source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        return microlmagnification.Jacobian_amplification_PSPL(source_trajectory_x, source_trajectory_y)

    def model_Jacobian(self, telescope, pyLIMA_parameters):
        """ The derivative of a PSPL model

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: jacobi
        :rtype: array_like
        """

        # Derivatives of the residuals_LM objective function, PSPL version

        lightcurve = telescope.lightcurve_flux

        time = lightcurve[:, 0]
        errflux = lightcurve[:, 2]

        # Derivative of A = (u^2+2)/(u(u^2+4)^0.5). Amplification[0] is A(t).
        # Amplification[1] is U(t).
        Amplification = self.Jacobian_model_magnification(telescope, pyLIMA_parameters)
        dAmplificationdU = (-8) / (Amplification[1] ** 2 * (Amplification[1] ** 2 + 4) ** 1.5)

        # Derivative of U = (uo^2+(t-to)^2/tE^2)^0.5
        dUdto = -(time - pyLIMA_parameters.to) / \
                (pyLIMA_parameters.tE ** 2 * Amplification[1])
        dUduo = pyLIMA_parameters.uo / Amplification[1]
        dUdtE = -(time - pyLIMA_parameters.to) ** 2 / \
                (pyLIMA_parameters.tE ** 3 * Amplification[1])

        # Derivative of the model

        dresdto = getattr(pyLIMA_parameters, 'fs_' + telescope.name) * dAmplificationdU * dUdto / errflux
        dresduo = getattr(pyLIMA_parameters, 'fs_' + telescope.name) * dAmplificationdU * dUduo / errflux
        dresdtE = getattr(pyLIMA_parameters, 'fs_' + telescope.name) * dAmplificationdU * dUdtE / errflux

        if self.blend_flux_ratio == True:
            dresdfs = (Amplification[0] + getattr(pyLIMA_parameters, 'g_' + telescope.name)) / errflux
            dresdg = getattr(pyLIMA_parameters, 'fs_' + telescope.name) / errflux
        else:
            dresdfs = (Amplification[0]) / errflux
            dresdg = 1 / errflux

        jacobi = np.array([dresdto, dresduo, dresdtE, dresdfs, dresdg])

        return jacobi


class ModelFSPL(MLModel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: FSPL
        :rtype: string
        """

        return 'FSPL'

    def paczynski_model_parameters(self):
        """ Define the FSPL standard parameters, [to,uo,tE,rho]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'rho': 3}

        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a FSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification, impact_parameter
        :rtype: array_like,array_like
        """

        source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters.to,
                                                                          pyLIMA_parameters.uo,
                                                                          pyLIMA_parameters.tE,
                                                                          pyLIMA_parameters)
        rho = pyLIMA_parameters.rho
        gamma = telescope.gamma

        return microlmagnification.amplification_FSPL(source_trajectory_x, source_trajectory_y, rho,
                                                      gamma, self.yoo_table)

    def Jacobian_model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a FSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification, impact_parameter
        :rtype: array_like,array_like
        """

        source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters.to,
                                                                          pyLIMA_parameters.uo,
                                                                          pyLIMA_parameters.tE,
                                                                          pyLIMA_parameters)
        rho = pyLIMA_parameters.rho
        gamma = telescope.gamma

        return microlmagnification.Jacobian_amplification_FSPL(source_trajectory_x, source_trajectory_y, rho,
                                                               gamma, self.yoo_table)

    def model_Jacobian(self, telescope, pyLIMA_parameters):
        """ The derivative of a FSPL model

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: jacobi
        :rtype: array_like
        """
        # Derivatives of the residuals_LM objective function, FSPL version

        fake_model = ModelPSPL(self.event)
        fake_model.define_model_parameters()
        lightcurve = telescope.lightcurve_flux
        time = lightcurve[:, 0]
        errflux = lightcurve[:, 2]
        gamma = telescope.gamma

        # Derivative of A = Yoo et al (2004) method.
        Amplification_PSPL = fake_model.Jacobian_model_magnification(telescope, pyLIMA_parameters)

        dAmplification_PSPLdU = (-8) / (Amplification_PSPL[1] ** 2 * (Amplification_PSPL[1] ** 2 + 4) ** (1.5))

        # z_yoo=U/rho
        z_yoo = Amplification_PSPL[1] / pyLIMA_parameters.rho

        dadu = np.zeros(len(Amplification_PSPL[0]))
        dadrho = np.zeros(len(Amplification_PSPL[0]))

        # Far from the lens (z_yoo>>1), then PSPL.
        ind = np.where((z_yoo > self.yoo_table[0][-1]))[0]
        dadu[ind] = dAmplification_PSPLdU[ind]
        dadrho[ind] = -0.0

        # Very close to the lens (z_yoo<<1), then Witt&Mao limit.
        ind = np.where((z_yoo < self.yoo_table[0][0]))[0]
        dadu[ind] = dAmplification_PSPLdU[ind] * (2 * z_yoo[ind] - gamma * (2 - 3 * np.pi / 4) * z_yoo[ind])

        dadrho[ind] = -Amplification_PSPL[0][ind] * Amplification_PSPL[1][ind] / pyLIMA_parameters.rho ** 2 * \
                      (2 - gamma * (2 - 3 * np.pi / 4))

        # FSPL regime (z_yoo~1), then Yoo et al derivatives
        ind = np.where((z_yoo <= self.yoo_table[0][-1]) & (z_yoo >= self.yoo_table[0][0]))[0]

        dadu[ind] = dAmplification_PSPLdU[ind] * (self.yoo_table[1](z_yoo[ind]) - \
                                                  gamma * self.yoo_table[2](z_yoo[ind])) + \
                    Amplification_PSPL[0][ind] * \
                    (self.yoo_table[3](z_yoo[ind]) - \
                     gamma * self.yoo_table[4](z_yoo[ind])) * 1 / pyLIMA_parameters.rho

        dadrho[ind] = -Amplification_PSPL[0][ind] * Amplification_PSPL[1][ind] / pyLIMA_parameters.rho ** 2 * \
                      (self.yoo_table[3](z_yoo[ind]) - gamma * self.yoo_table[4](z_yoo[ind]))

        dUdto = -(time - pyLIMA_parameters.to) / (pyLIMA_parameters.tE ** 2 * Amplification_PSPL[1])

        dUduo = pyLIMA_parameters.uo / Amplification_PSPL[1]

        dUdtE = -(time - pyLIMA_parameters.to) ** 2 / (pyLIMA_parameters.tE ** 3 * Amplification_PSPL[1])

        # Derivative of the model
        dresdto = getattr(pyLIMA_parameters, 'fs_' + telescope.name) * dadu * dUdto / errflux

        dresduo = getattr(pyLIMA_parameters, 'fs_' + telescope.name) * dadu * dUduo / errflux

        dresdtE = getattr(pyLIMA_parameters, 'fs_' + telescope.name) * dadu * dUdtE / errflux

        dresdrho = getattr(pyLIMA_parameters, 'fs_' + telescope.name) * dadrho / errflux

        Amplification_FSPL = self.model_magnification(telescope, pyLIMA_parameters)

        if self.blend_flux_ratio == True:
            dresdfs = (Amplification_FSPL + getattr(pyLIMA_parameters, 'g_' + telescope.name)) / errflux
            dresdg = getattr(pyLIMA_parameters, 'fs_' + telescope.name) / errflux
        else:
            dresdfs = (Amplification_FSPL) / errflux
            dresdg = 1 / errflux

        jacobi = np.array([dresdto, dresduo, dresdtE, dresdrho, dresdfs, dresdg])

        return jacobi

class ModelFSPLee(MLModel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: FSPL-Lee method
        :rtype: string
        """

        return 'FSPL'

    def paczynski_model_parameters(self):
        """ Define the FSPL standard parameters, [to,uo,tE,rho]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'rho': 3}
        self.Jacobian_flag = 'No way'
        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a FSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification, impact_parameter
        :rtype: array_like,array_like
        """

        source_trajectory_x, source_trajectory_y,_ = self.source_trajectory(telescope, pyLIMA_parameters.to,
                                                                          pyLIMA_parameters.uo,
                                                                          pyLIMA_parameters.tE,
                                                                          pyLIMA_parameters)
        rho = pyLIMA_parameters.rho
        gamma = telescope.gamma

        return microlmagnification.amplification_FSPLee(source_trajectory_x, source_trajectory_y, rho,
                                                      gamma)


class ModelFSPLarge(MLModel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: FSPL-VBB method
        :rtype: string
        """

        return 'FSPL'

    def paczynski_model_parameters(self):
        """ Define the FSPL standard parameters, [to,uo,tE,rho]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'rho': 3}
        self.Jacobian_flag = 'No way'
        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a FSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification, impact_parameter
        :rtype: array_like,array_like
        """

        source_trajectory_x, source_trajectory_y,_ = self.source_trajectory(telescope, pyLIMA_parameters.to,
                                                                          pyLIMA_parameters.uo,
                                                                          pyLIMA_parameters.tE,
                                                                          pyLIMA_parameters)
        rho = pyLIMA_parameters.rho
        linear_limb_darkening = telescope.gamma * 3 / (2 + telescope.gamma)
        #linear_limb_darkening = telescope.gamma
        #import pdb;
        #pdb.set_trace()
        return microlmagnification.amplification_FSPLarge(source_trajectory_x, source_trajectory_y, rho,
                                                      linear_limb_darkening)


class ModelDSPL(MLModel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: DSPL
        :rtype: string
        """
        return 'DSPL'

    def paczynski_model_parameters(self):
        """ Define the DSPL standard parameters, [to1,uo1,to2,uo2,tE,q_F_filters]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'delta_to': 2, 'delta_uo': 3, 'tE': 4}
        filters = [telescope.filter for telescope in self.event.telescopes]

        unique_filters = np.unique(filters)

        for filter in unique_filters:
            model_dictionary['q_flux_' + filter] = len(model_dictionary)

        self.Jacobian_flag = 'No way'
        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a DSPL model.
        From Hwang et al 2013 : http://iopscience.iop.org/article/10.1088/0004-637X/778/1/55/pdf

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification, impact_parameter
        :rtype: array_like,array_like
        """

        source1_trajectory_x, source1_trajectory_y,_  = self.source_trajectory(telescope, pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        to2 = pyLIMA_parameters.to + pyLIMA_parameters.delta_to
        uo2 = pyLIMA_parameters.delta_uo + pyLIMA_parameters.uo

        source2_trajectory_x, source2_trajectory_y,_ = self.source_trajectory(telescope, to2, uo2,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        source1_magnification = microlmagnification.amplification_PSPL(source1_trajectory_x, source1_trajectory_y)
        source2_magnification = microlmagnification.amplification_PSPL(source2_trajectory_x, source2_trajectory_y)

        blend_magnification_factor = getattr(pyLIMA_parameters, 'q_flux_' + telescope.filter)

        effective_magnification = (source1_magnification + source2_magnification * blend_magnification_factor) / (
            1 + blend_magnification_factor)

        return effective_magnification

class ModelDFSPL(MLModel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: DFSPL
        :rtype: string
        """
        return 'DFSPL'

    def paczynski_model_parameters(self):
        """ Define the DFSPL standard parameters, [to1,uo1,to2,uo2,tE,rho_1,rho_2,q_F_filters]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'delta_to': 2, 'delta_uo': 3, 'tE': 4, 'rho_1':5,'rho_2':6}
        filters = [telescope.filter for telescope in self.event.telescopes]

        unique_filters = np.unique(filters)

        for filter in unique_filters:
            model_dictionary['q_flux_' + filter] = len(model_dictionary)

        self.Jacobian_flag = 'No way'
        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a DFSPL model.
        From Hwang et al 2013 : http://iopscience.iop.org/article/10.1088/0004-637X/778/1/55/pdf

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification, impact_parameter
        :rtype: array_like,array_like
        """

        source1_trajectory_x, source1_trajectory_y,_  = self.source_trajectory(telescope, pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        to2 = pyLIMA_parameters.to + pyLIMA_parameters.delta_to
        uo2 = pyLIMA_parameters.delta_uo + pyLIMA_parameters.uo
        source2_trajectory_x, source2_trajectory_y,_ = self.source_trajectory(telescope, to2, uo2,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        rho1 = pyLIMA_parameters.rho_1
        gamma1 = telescope.gamma
        source1_magnification = microlmagnification.amplification_FSPL(source1_trajectory_x, source1_trajectory_y, rho1,
                                                                       gamma1, self.yoo_table)

        rho2 = pyLIMA_parameters.rho_2
        gamma2 = telescope.gamma1
        source2_magnification = microlmagnification.amplification_FSPL(source2_trajectory_x, source2_trajectory_y, rho2,
                                                                       gamma2, self.yoo_table)

        blend_magnification_factor = getattr(pyLIMA_parameters, 'q_flux_' + telescope.filter)

        effective_magnification = (source1_magnification + source2_magnification * blend_magnification_factor) / (
            1 + blend_magnification_factor)

        return effective_magnification


class ModelFSBL(MLModel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns:FSBL
        :rtype: string
        """
        return 'FSBL'

    def paczynski_model_parameters(self):
        """ Define the USBL standard parameters, [to,uo,tE,rho, s,q,alpha]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'rho': 3, 'logs': 4, 'logq': 5, 'alpha': 6}

        self.Jacobian_flag = 'No way'
        self.USBL_windows = None
        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a FSBL model.
            From Bozza  2010 : http://adsabs.harvard.edu/abs/2010MNRAS.408.2188B

            :param object telescope: a telescope object. More details in telescope module.
            :param object pyLIMA_parameters: a namedtuple which contain the parameters
            :return: magnification,
            :rtype: array_like,
        """

        linear_limb_darkening = telescope.gamma * 3 / (2 + telescope.gamma)

        to, uo = self.uo_to_from_uc_tc(pyLIMA_parameters)
        source_trajectoire = self.source_trajectory(telescope, to, uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        source_trajectoire = self.source_trajectory(telescope, to, uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        separation = source_trajectoire[2]+10**pyLIMA_parameters.logs

        magnification_FSBL = \
            microlmagnification.amplification_FSBL(separation, 10 ** pyLIMA_parameters.logq,
                                                   source_trajectoire[0], source_trajectoire[1],
                                                   pyLIMA_parameters.rho, linear_limb_darkening)

        return magnification_FSBL

    def find_caustics(self, separation, mass_ratio):
        """ The caustics, critical_curves and area of interest associated to the separation and mass ratio

                       :param  float separation: the binary lens component separation
                       :param  float mass ratio: the binary lens component mass ratio

        """
        caustics, critical_curves = microlcaustics.compute_2_lenses_caustics_points(separation,
                                                                                    mass_ratio,
                                                                                    resolution=100)

        area_of_interest = microlcaustics.find_area_of_interest_around_caustics(caustics, secure_factor=0.1)

        self.caustics = caustics
        self.critical_curves = critical_curves
        self.area_of_interest = area_of_interest

    def find_origin(self, pyLIMA_parameters):
        """ Find the origin of the binary system, if needed. Useful for fine modeling and simulations.

                           :param  object pyLIMA_parameters: the object containint all model parameters

        """
        new_origin_x = 0
        new_origin_y = 0

        if self.binary_origin == 'central_caustic':
        
           new_origin_x, new_origin_y = microlcaustics.change_source_trajectory_center_to_central_caustics_center(
                        10 ** pyLIMA_parameters.logs,
                        10 ** pyLIMA_parameters.logq)

        if self.binary_origin == 'planetary_caustic':
         
           new_origin_x, new_origin_y = microlcaustics.change_source_trajectory_center_to_planetary_caustics_center(
                        10 ** pyLIMA_parameters.logs,
                        10 ** pyLIMA_parameters.logq)

        self.x_center = new_origin_x
        self.y_center = new_origin_y


    def uo_to_from_uc_tc(self, pyLIMA_parameters):
        """ Find the associated to,uo from the new origin of the binary system, if needed.
            Useful for fine modeling and simulations.

                   :param  object pyLIMA_parameters: the object containint all model parameters
                   :returns: to,uo the new origin parameters tc, uc (impact parameter associated to the new origin)
                   :rtype: float,float
        """
        new_origin_x = self.x_center
        new_origin_y = self.y_center

        if new_origin_x:

            to = pyLIMA_parameters.to - pyLIMA_parameters.tE * (new_origin_x * np.cos(pyLIMA_parameters.alpha) +
                                                                new_origin_y * np.sin(pyLIMA_parameters.alpha))

            uo = pyLIMA_parameters.uo - (new_origin_x * np.sin(pyLIMA_parameters.alpha) -
                                         new_origin_y * np.cos(pyLIMA_parameters.alpha))
            return to, uo


        else:

            return pyLIMA_parameters.to, pyLIMA_parameters.uo

    def uc_tc_from_uo_to(self, pyLIMA_parameters, to, uo):

        new_origin_x = self.x_center
        new_origin_y = self.y_center

        if new_origin_x:

            tc = to + pyLIMA_parameters.tE * (new_origin_x * np.cos(pyLIMA_parameters.alpha) +
                                              new_origin_y * np.sin(pyLIMA_parameters.alpha))

            uc = uo + (new_origin_x * np.sin(pyLIMA_parameters.alpha) -
                       new_origin_y * np.cos(pyLIMA_parameters.alpha))

            return tc, uc


        else:

            return to, uo

    def find_binary_regime(self, pyLIMA_parameters):

        binary_regime = microlcaustics.find_2_lenses_caustic_regime(10 ** pyLIMA_parameters.logs,
                                                                    10 ** pyLIMA_parameters.logq)
        return binary_regime

class ModelUSBL(ModelFSBL):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: USBL
        :rtype: string
        """
        return 'USBL'

    def paczynski_model_parameters(self):
        """ Define the USBL standard parameters, [to,uo,tE,rho, s,q,alpha]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'rho': 3, 'logs': 4, 'logq': 5, 'alpha': 6}

        self.Jacobian_flag = 'No way'
        self.USBL_windows = None
        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a USBL model.
            From Bozza  2010 : http://adsabs.harvard.edu/abs/2010MNRAS.408.2188B

            :param object telescope: a telescope object. More details in telescope module.
            :param object pyLIMA_parameters: a namedtuple which contain the parameters
            :return: magnification,
            :rtype: array_like,
        """

        to, uo = self.uo_to_from_uc_tc(pyLIMA_parameters)
        source_trajectoire = self.source_trajectory(telescope, to, uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        separation =  source_trajectoire[2]+10**pyLIMA_parameters.logs

        magnification_USBL = \
            microlmagnification.amplification_USBL(separation, 10 ** pyLIMA_parameters.logq,
                                                   source_trajectoire[0], source_trajectoire[1],
                                                   pyLIMA_parameters.rho)
        return magnification_USBL

class ModelPSBL(ModelFSBL):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns:PSBL
        :rtype: string
        """
        return 'PSBL'

    def paczynski_model_parameters(self):
        """ Define the PSBL standard parameters, [to,uo,tE, s,q,alpha]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'logs': 3, 'logq': 4, 'alpha': 5}

        self.Jacobian_flag = 'No way'

        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a PSBL model.
            From Bozza  2010 : http://adsabs.harvard.edu/abs/2010MNRAS.408.2188B

            :param object telescope: a telescope object. More details in telescope module.
            :param object pyLIMA_parameters: a namedtuple which contain the parameters
            :return: magnification,
            :rtype: array_like,
        """
        to, uo = self.uo_to_from_uc_tc(pyLIMA_parameters)

        source_trajectoire = self.source_trajectory(telescope, to, uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        separation =  source_trajectoire[2]+10**pyLIMA_parameters.logs

        magnification = \
            microlmagnification.amplification_PSBL(separation, 10 ** pyLIMA_parameters.logq, source_trajectoire[0], source_trajectoire[1])

        return magnification


class ModelVariablePL(MLModel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: VariablePL
        :rtype: string
        """
        return 'VariablePL'

    
    def paczynski_model_parameters(self):
        """ Define the PSPL standard parameters, [to,uo,tE]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        # model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'period': 3, 'A1': 4, 'A2': 5, 'A3': 6, 'A4': 7, 'A5': 8,
        # 'A6': 9,
        # 'phi_21': 10, 'phi_31': 11, 'phi_41': 12, 'phi_51': 13, 'phi_61': 14}
        
        self.number_of_harmonics = self.model_arguments[0]
        self.blend_flux_ratio = None
        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'rho': 3, 'period': 4}

        filters = [telescope.filter for telescope in self.event.telescopes]

        unique_filters = np.unique(filters)

        for filter in unique_filters:
            # model_dictionary['AO' + '_' + filter] = len(model_dictionary)
            for i in range(self.number_of_harmonics):
                model_dictionary['A' + str(i + 1) + '_' + filter] = len(model_dictionary)
                model_dictionary['phi' + str(i + 1) + '_' + filter] = len(model_dictionary)
                model_dictionary['k' + str(i + 1) + '_' + filter] = len(model_dictionary)

                # if filter != 'I':
                # model_dictionary['phib' + str(i + 1) + '_' + filter] = len(model_dictionary)

        self.Jacobian_flag = 'No way'
        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a PSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification
        :rtype: array_like
        """

        source_trajectory_x, source_trajectory_y,_ = self.source_trajectory(telescope, pyLIMA_parameters.to,
                                                                          pyLIMA_parameters.uo,
                                                                          pyLIMA_parameters.tE, pyLIMA_parameters)

        return microlmagnification.amplification_FSPL(source_trajectory_x, source_trajectory_y,
                                                                pyLIMA_parameters.rho,
                                                                telescope.gamma, self.yoo_table)


    def compute_the_microlensing_model(self, telescope, pyLIMA_parameters):
        """ Compute the microlens model according the injected parameters. This is modified by child submodel sublclass,
        if not the default microlensing model is returned.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :returns: the microlensing model
        :rtype: array_like
        """
        lightcurve = telescope.lightcurve_flux
        time = lightcurve[:, 0]

        amplification = self.model_magnification(telescope, pyLIMA_parameters)

        pulsations = self.compute_pulsations(time, telescope.filter, pyLIMA_parameters)
        f_source, f_blending = self.derive_telescope_flux(telescope, pyLIMA_parameters, amplification, pulsations)

        if self.variable_blend:

            microlensing_model = f_source * amplification + f_blending * pulsations
        else:
            microlensing_model = f_source * amplification * pulsations + f_blending

        return microlensing_model, f_source, f_blending

    def derive_telescope_flux(self, telescope, pyLIMA_parameters, amplification, pulsations):

        try:
            # Fluxes parameters are fitted
            f_source = 2 * getattr(pyLIMA_parameters, 'fs_' + telescope.name) / 2
            f_blending = 2 * getattr(pyLIMA_parameters, 'fb_' + telescope.name) / 2

        except TypeError:

            # Fluxes parameters are estimated through np.polyfit

            if self.variable_blend:

                lightcurve = telescope.lightcurve_flux
                flux = lightcurve[:, 1]
                errflux = lightcurve[:, 2]
                try:
                    f_source, f_blending = np.polyfit(amplification, flux, 1, w=1 / errflux)
                except:

                    import pdb;
                    pdb.set_trace()
            else:

                magnification = amplification*pulsations

                lightcurve = telescope.lightcurve_flux
                flux = lightcurve[:, 1]
                errflux = lightcurve[:, 2]
                try:
                    f_source, f_blending = np.polyfit(magnification, flux, 1, w=1 / errflux)
                except ValueError:

                    import pdb;
                    pdb.set_trace()
        return f_source, f_blending


    def compute_pulsations(self, time, filter, pyLIMA_parameters):

        time = time - pyLIMA_parameters.to

        pulsations = 0
        period = getattr(pyLIMA_parameters, 'period')
        # factor = 0.0
        # pulsations = getattr(pyLIMA_parameters, 'AO'+'_' + telescope.filter)
        for i in range(self.number_of_harmonics):
            amplitude = getattr(pyLIMA_parameters, 'A' + str(i + 1) + '_' + filter)
            phase = getattr(pyLIMA_parameters, 'phi' + str(i + 1) + '_' + filter)
            octave = getattr(pyLIMA_parameters, 'k' + str(i + 1) + '_' + filter)

            period_harmonic = period*octave
            pulsations += amplitude * np.cos(2 * np.pi  /period_harmonic * time + phase)

        pulsations = 10 ** (pulsations / 2.5)

        return pulsations


