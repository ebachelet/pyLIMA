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

thismodule = sys.modules[__name__]

import numpy as np
import collections
import time as python_time

import microlguess
import microlmagnification
import microlpriors
import microlparallax
import microlorbitalmotion
import microlcaustics
import stars

from scipy import interpolate, misc

full_path = os.path.abspath(__file__)
directory, filename = os.path.split(full_path)

try:

    # yoo_table = np.loadtxt('b0b1.dat')
    yoo_table = np.loadtxt(os.path.join(directory, 'data/Yoo_B0B1.dat'))
except:

    print 'ERROR : No b0b1.dat file found, please check!'

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


def create_model(model_type, event, model_arguments=[], parallax=['None', 0.0], xallarap=['None', 0.0],
                 orbital_motion=['None', 0.0], source_spots='None'):
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
                 orbital_motion, source_spots)


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

    def __init__(self, event, model_arguments=[], parallax=['None', 0.0], xallarap=['None', 0.0],
                 orbital_motion=['None', 0.0], source_spots='None'):
        """ Initialization of the attributes described above.
        """
        self.event = event
        self.model_arguments = model_arguments
        self.parallax_model = parallax
        self.xallarap_model = xallarap
        self.orbital_motion_model = orbital_motion
        self.source_spots_model = source_spots
        self.yoo_table = yoo_table

        self.model_dictionnary = {}
        self.pyLIMA_standards_dictionnary = {}

        self.fancy_to_pyLIMA_dictionnary = {}
        self.pyLIMA_to_fancy = {}
        self.fancy_to_pyLIMA = {}

        self.parameters_guess = []
        self.Jacobian_flag = 'OK'

        self.define_pyLIMA_standard_parameters()

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

        if self.orbital_motion_model[0] != 'None':
            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['dsdt'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['dalphadt'] = len(self.pyLIMA_standards_dictionnary)

        if self.source_spots_model != 'None':
            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['spot'] = len(self.pyLIMA_standards_dictionnary)

        for telescope in self.event.telescopes:
            self.pyLIMA_standards_dictionnary['fs_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['g_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)

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

        self.model_parameters = collections.namedtuple('parameters', self.model_dictionnary)

    def compute_the_microlensing_model(self, telescope, pyLIMA_parameters):
        """ Compute the microlens model according the injected parameters. This is modified by child submodel sublclass,
        if not the default microlensing model is returned.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :returns: the microlensing model
        :rtype: array_like
        """

        amplification  = self.model_magnification(telescope, pyLIMA_parameters)
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

        microlensing_model = f_source * (amplification + g_blending)

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

            g_blending = 2 * getattr(pyLIMA_parameters, 'g_' + telescope.name) / 2


        except TypeError:

            # Fluxes parameters are estimated through np.polyfit
            lightcurve = telescope.lightcurve_flux
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]

            try:
                f_source, f_blending = np.polyfit(amplification, flux, 1, w=1 / errflux)
                g_blending = f_blending / f_source
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


        for key_parameter in self.model_dictionnary.keys():

            try:

                setattr(self.model_parameters, key_parameter, fancy_parameters[self.model_dictionnary[key_parameter]])

            except:
                setattr(self.model_parameters, key_parameter, None)

        # print 'arange', python_time.time() - start_time

        pyLIMA_parameters = self.fancy_parameters_to_pyLIMA_standard_parameters(self.model_parameters)

        # print 'conversion', python_time.time() - start_time
        return pyLIMA_parameters


    def fancy_parameters_to_pyLIMA_standard_parameters(self, fancy_parameters):
        """ Transform the fancy parameters to the pyLIMA standards. The output got all
        the necessary standard attributes, example to, uo, tE...


        :param object fancy_parameters: the fancy_parameters as namedtuple
        :return: the pyLIMA standards are added to the fancy parameters
        :rtype: object
        """
        # start_time = python_time.time()
        if len(self.fancy_to_pyLIMA) != 0:

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

        # Orbital motion?
        if 'alpha' in pyLIMA_parameters._fields:

            alpha = pyLIMA_parameters.alpha
            if 'dalphadt' in pyLIMA_parameters._fields:
                alpha += microlorbitalmotion.orbital_motion_2D_trajectory_shift(self.orbital_motion_model[1],
                                                                                telescope.lightcurve_flux[:, 0],
                                                                                pyLIMA_parameters.dalphadt)
            source_trajectory_x = tau * np.cos(alpha) - beta * np.sin(alpha)
            source_trajectory_y = tau * np.sin(alpha) + beta * np.cos(alpha)

        else:

            source_trajectory_x = tau
            source_trajectory_y = beta

        return source_trajectory_x, source_trajectory_y


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

        source_trajectoire = self.source_trajectory(telescope, pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        return microlmagnification.amplification_PSPL(*source_trajectoire)

    def Jacobian_model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a PSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification, impact_parameter
        :rtype: array_like,array_like
        """

        source_trajectoire = self.source_trajectory(telescope, pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        return microlmagnification.Jacobian_amplification_PSPL(*source_trajectoire)

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
        dresdfs = (Amplification[0] + getattr(pyLIMA_parameters, 'g_' + telescope.name)) / errflux
        dresdg = getattr(pyLIMA_parameters, 'fs_' + telescope.name) / errflux

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

        source_trajectory_x, source_trajectory_y = self.source_trajectory(telescope, pyLIMA_parameters.to,
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

        source_trajectory_x, source_trajectory_y = self.source_trajectory(telescope, pyLIMA_parameters.to,
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

        dresdfs = (Amplification_FSPL + getattr(pyLIMA_parameters, 'g_' + telescope.name)) / errflux

        dresdg = getattr(pyLIMA_parameters, 'fs_' + telescope.name) / errflux

        jacobi = np.array([dresdto, dresduo, dresdtE, dresdrho, dresdfs, dresdg])

        return jacobi


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

        source1_trajectory = self.source_trajectory(telescope, pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        to2 = pyLIMA_parameters.to + pyLIMA_parameters.delta_to
        uo2 = pyLIMA_parameters.delta_uo + pyLIMA_parameters.uo
        source2_trajectory = self.source_trajectory(telescope, to2, uo2,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        source1_magnification = microlmagnification.amplification_PSPL(*source1_trajectory)

        source2_magnification = microlmagnification.amplification_PSPL(*source2_trajectory)

        blend_magnification_factor = getattr(pyLIMA_parameters, 'q_flux_' + telescope.filter)

        effective_magnification = (source1_magnification + source2_magnification * blend_magnification_factor) / (
            1 + blend_magnification_factor)

        return effective_magnification


class ModelUSBL(MLModel):
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

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification
        :rtype: array_like
        """

        source_trajectoire = self.source_trajectory(telescope, pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        magnification = np.zeros(len(source_trajectoire[0]))

        # for key in self.model_dictionnary.keys()[:len(self.parameters_boundaries)]:
        #   param = getattr(pyLIMA_parameters, key)
        #   limits = self.parameters_boundaries[self.model_dictionnary[key]]
        #   if (param<limits[0]) | (limits[1]<param):
        #           magnification += 0.1*np.inf
        #           return magnification,magnification
        if 'dsdt' in pyLIMA_parameters._fields:

            separation = 10 ** pyLIMA_parameters.logs + \
                         microlorbitalmotion.orbital_motion_2D_separation_shift(self.orbital_motion_model[1],
                                                                                telescope.lightcurve_flux[:, 0],
                                                                                pyLIMA_parameters.dsdt)

        else:

            separation = np.array([10 ** pyLIMA_parameters.logs] * len(source_trajectoire[0]))

        if self.USBL_windows:

            index_USBL = np.where((telescope.lightcurve_flux[:, 0] <= self.USBL_windows[1]) & (
                telescope.lightcurve_flux[:, 0] >= self.USBL_windows[0]))[0]

            Xs = source_trajectoire[0][index_USBL]
            Ys = source_trajectoire[1][index_USBL]

            magnification_USBL = \
                microlmagnification.amplification_USBL(separation[index_USBL], 10 ** pyLIMA_parameters.logq,
                                                       Xs, Ys, pyLIMA_parameters.rho,
                                                       tolerance=0.001)

            magnification[index_USBL] = magnification_USBL

            index_PSBL = np.where((telescope.lightcurve_flux[:, 0] > self.USBL_windows[1]) | (
                telescope.lightcurve_flux[:, 0] < self.USBL_windows[0]))[0]

            magnification_PSBL = microlmagnification.amplification_PSBL(separation[index_PSBL],
                                                                        10 ** pyLIMA_parameters.logq,
                                                                        source_trajectoire[0][index_PSBL],
                                                                        source_trajectoire[1][index_PSBL])

            magnification[index_PSBL] = magnification_PSBL

        else:

            Xs, Ys = source_trajectoire
            magnification = \
                microlmagnification.amplification_USBL(separation, 10 ** pyLIMA_parameters.logq,
                                                       Xs, Ys, pyLIMA_parameters.rho,
                                                       tolerance=0.001)

        return magnification


    def find_caustics(self, separation, mass_ratio):

        caustics, critical_curves = microlcaustics.compute_2_lenses_caustics_points(separation,
                                                                                              mass_ratio,
                                                                                              resolution=100)

        area_of_interest = microlcaustics.find_area_of_interest_around_caustics(caustics, secure_factor=0.1)

        self.caustics = caustics
        self.critical_curves = critical_curves
        self.area_of_interest = area_of_interest


class ModelPSBL(MLModel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns:PSBL
        :rtype: string
        """
        return 'PSBL'

    def paczynski_model_parameters(self):
        """ Define the USBL standard parameters, [to,uo,tE,rho, s,q,alpha]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2,'logs': 3, 'logq': 4, 'alpha': 5}

        self.Jacobian_flag = 'No way'

        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a USBL model.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification, impact_parameter
        :rtype: array_like,array_like
        """

        source_trajectoire = self.source_trajectory(telescope, pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        magnification = np.zeros(len(source_trajectoire[0]))

        # for key in self.model_dictionnary.keys()[:len(self.parameters_boundaries)]:
        #   param = getattr(pyLIMA_parameters, key)
        #   limits = self.parameters_boundaries[self.model_dictionnary[key]]
        #   if (param<limits[0]) | (limits[1]<param):
        #           magnification += 0.1*np.inf
        #           return magnification,magnification
        if 'dsdt' in pyLIMA_parameters._fields:

            separation = 10 ** pyLIMA_parameters.logs + \
                         microlorbitalmotion.orbital_motion_2D_separation_shift(self.orbital_motion_model[1],
                                                                                telescope.lightcurve_flux[:, 0],
                                                                                pyLIMA_parameters.dsdt)

        else:

            separation = np.array([10 ** pyLIMA_parameters.logs] * len(source_trajectoire[0]))

        Xs, Ys = source_trajectoire
        magnification = \
                microlmagnification.amplification_PSBL(separation, 10 ** pyLIMA_parameters.logq, Xs, Ys)

        return magnification


    def find_caustics(self, separation, mass_ratio):

        caustics, critical_curves = microlcaustics.compute_2_lenses_caustics_points(separation,
                                                                                              mass_ratio,
                                                                                              resolution=100)

        area_of_interest = microlcaustics.find_area_of_interest_around_caustics(caustics, secure_factor=0.1)

        self.caustics = caustics
        self.critical_curves = critical_curves
        self.area_of_interest = area_of_interest


class ModelRRLyraePL(MLModel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: RRLyraePL
        :rtype: string
        """
        return 'RRLyraePL'

    def define_pyLIMA_standard_parameters(self):
        """ Define the standard pyLIMA parameters dictionnary."""
        self.number_of_harmonics = self.model_arguments[0]
        self.pyLIMA_standards_dictionnary = self.paczynski_model_parameters()
        if self.parallax_model[0] != 'None':
            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['piEN'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['piEE'] = len(self.pyLIMA_standards_dictionnary)

            self.event.compute_parallax_all_telescopes(self.parallax_model)
        for telescope in self.event.telescopes:
            self.pyLIMA_standards_dictionnary['fs_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['fb_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)

        self.pyLIMA_standards_dictionnary = OrderedDict(
            sorted(self.pyLIMA_standards_dictionnary.items(), key=lambda x: x[1]))

        self.parameters_boundaries = microlguess.differential_evolution_parameters_boundaries(self)

    def paczynski_model_parameters(self):
        """ Define the PSPL standard parameters, [to,uo,tE]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        # model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'period': 3, 'A1': 4, 'A2': 5, 'A3': 6, 'A4': 7, 'A5': 8,
        # 'A6': 9,
        # 'phi_21': 10, 'phi_31': 11, 'phi_41': 12, 'phi_51': 13, 'phi_61': 14}

        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'period': 3}

        filters = [telescope.filter for telescope in self.event.telescopes]

        unique_filters = np.unique(filters)

        for filter in unique_filters:
            # model_dictionary['AO' + '_' + filter] = len(model_dictionary)
            for i in xrange(self.number_of_harmonics):
                model_dictionary['A' + str(i + 1) + '_' + filter] = len(model_dictionary)
                model_dictionary['phi' + str(i + 1) + '_' + filter] = len(model_dictionary)
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

        source_trajectoire = self.source_trajectory(telescope, pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        return microlmagnification.amplification_PSPL(*source_trajectoire)

    def compute_the_microlensing_model(self, telescope, pyLIMA_parameters):
        """ Compute the microlensing model according the injected parameters. T

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :returns: the microlensing model, the source and the blending flux
        :rtype: tuple, tuple of array like
        """
        lightcurve = telescope.lightcurve_flux
        time = lightcurve[:, 0]

        amplification = self.model_magnification(telescope, pyLIMA_parameters)

        pulsations = self.compute_pulsations( time, telescope.filter, pyLIMA_parameters)
        f_source, f_blending = self.derive_telescope_flux(telescope, pyLIMA_parameters, amplification, pulsations)

        microlensing_model = f_source * amplification* pulsations + f_blending

        return microlensing_model, f_source, f_blending

    def derive_telescope_flux(self, telescope, pyLIMA_parameters, amplification, pulsations):
        """
        Compute the source/blending flux

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :param array_like amplification: an array containing the magnification
        :param array_like pulsations: an array containing the RR Lyrae pulsations


        :returns:  the source and the blending flux
        :rtype: tuple
        """
        try:
            # Fluxes parameters are fitted
            f_source = 2 * getattr(pyLIMA_parameters, 'fs_' + telescope.name) / 2
            f_blending = 2 * getattr(pyLIMA_parameters, 'fb_' + telescope.name) / 2

        except TypeError:

            # Fluxes parameters are estimated through np.polyfit


            lightcurve = telescope.lightcurve_flux
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            f_source, f_blending = np.polyfit(amplification * pulsations, flux , 1, w=1 / errflux)

        return f_source, f_blending

    def compute_radius(self, Teff, time, telescope_V, pyLIMA_parameters):

        pulsations = self.compute_pulsations(time, telescope_V.filter, pyLIMA_parameters)
        f_source = 2 * getattr(pyLIMA_parameters, 'fs_' + telescope_V.name) / 2

        f_source_V = f_source * pulsations
        V_magnitude = 27.4 - 2.5 * np.log10(f_source_V)

        #radius = (0.636*10**-((V_magnitude-2*2.689)/5)/Teff**2)
        radius = 10**(0.2*(-(V_magnitude-2.689+0.1)-10*np.log10(Teff)+37.35))
        return radius
    def compute_Teff(self,color):


        # Casagrande 2010
        color += -1.250
        theta_eff = 0.4033+0.8171*color-0.1987*color**2
        Teff = 5040/theta_eff

        return Teff
    def compute_color(self, time, telescope_V, telescope_I, pyLIMA_parameters ):



        pulsations = self.compute_pulsations(time, telescope_V.filter, pyLIMA_parameters)
        f_source = 2 * getattr(pyLIMA_parameters, 'fs_' + telescope_V.name) / 2

        f_source_V = f_source * pulsations

        pulsations = self.compute_pulsations(time, telescope_I.filter, pyLIMA_parameters)


        f_source = 2 * getattr(pyLIMA_parameters, 'fs_' + telescope_I.name) / 2

        f_source_I = f_source * pulsations


        V_magnitude = 27.4-2.5*np.log10(f_source_V)
        I_magnitude = 27.4-2.5*np.log10(f_source_I)

        return V_magnitude-I_magnitude

    def compute_pulsations(self, time, filter, pyLIMA_parameters):

        time = time - 2456425

        pulsations = 0
        period = getattr(pyLIMA_parameters, 'period')
        # factor = 0.0
        # pulsations = getattr(pyLIMA_parameters, 'AO'+'_' + telescope.filter)
        for i in xrange(self.number_of_harmonics):
            amplitude = getattr(pyLIMA_parameters, 'A' + str(i + 1) + '_' + filter)
            phase = getattr(pyLIMA_parameters, 'phi' + str(i + 1) + '_' + filter)

            pulsations += amplitude * np.cos(2 * np.pi * (i + 1) / period * time + phase)

        pulsations = 10 ** (pulsations / 2.5)

        return pulsations




class ModelRRLyraeFS(MLModel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: RRLyraeFS
        :rtype: string
        """
        return 'RRLyraeFS'

    def define_pyLIMA_standard_parameters(self):
        """ Define the standard pyLIMA parameters dictionnary."""
        self.number_of_harmonics = self.model_arguments[0]

        self.pyLIMA_standards_dictionnary = self.paczynski_model_parameters()
        if self.parallax_model[0] != 'None':
            self.Jacobian_flag = 'No way'
            self.pyLIMA_standards_dictionnary['piEN'] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['piEE'] = len(self.pyLIMA_standards_dictionnary)

            self.event.compute_parallax_all_telescopes(self.parallax_model)
        for telescope in self.event.telescopes:
            self.pyLIMA_standards_dictionnary['fs_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)
            self.pyLIMA_standards_dictionnary['fb_' + telescope.name] = len(self.pyLIMA_standards_dictionnary)

        self.pyLIMA_standards_dictionnary = OrderedDict(
            sorted(self.pyLIMA_standards_dictionnary.items(), key=lambda x: x[1]))

        self.parameters_boundaries = microlguess.differential_evolution_parameters_boundaries(self)

    def paczynski_model_parameters(self):
        """ Define the PSPL standard parameters, [to,uo,tE]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        # model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'period': 3, 'A1': 4, 'A2': 5, 'A3': 6, 'A4': 7, 'A5': 8,
        # 'A6': 9,
        # 'phi_21': 10, 'phi_31': 11, 'phi_41': 12, 'phi_51': 13, 'phi_61': 14}

        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'theta_E':3, 'period': 4}

        filters = [telescope.filter for telescope in self.event.telescopes]

        unique_filters = np.unique(filters)

        for filter in unique_filters:
            # model_dictionary['AO' + '_' + filter] = len(model_dictionary)
            for i in xrange(self.number_of_harmonics):
                model_dictionary['A' + str(i + 1) + '_' + filter] = len(model_dictionary)
                model_dictionary['phi' + str(i + 1) + '_' + filter] = len(model_dictionary)
                # if filter != 'I':
                # model_dictionary['phib' + str(i + 1) + '_' + filter] = len(model_dictionary)

        self.Jacobian_flag = 'No way'
        self.lyrae = stars.Star()
        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters):
        """ The magnification associated to a PSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification
        :rtype: array_like
        """

        source_trajectory_x, source_trajectory_y = self.source_trajectory(telescope, pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                                    pyLIMA_parameters.tE, pyLIMA_parameters)

        telescope_V, telescope_I = self.find_telecopes_V_and_I()

        color = self.compute_color(telescope.lightcurve_flux[:,0], telescope_V, telescope_I, pyLIMA_parameters)
        teff = self.compute_Teff(color)


        gammas = []
        rhos =  self.compute_radius(teff, telescope.lightcurve_flux[:,0], telescope_V, pyLIMA_parameters) * (
                                    0.00456*10**6 / pyLIMA_parameters.theta_E)

        pyLIMA_parameters.rho = rhos
        #import pdb;
        #pdb.set_trace()
        count = 0
        for temperature in teff :
            self.lyrae.Teff = temperature
            self.lyrae.log_g = 2
            gamma = self.find_gamma(telescope, self.lyrae)
            gammas.append(gamma)

            count += 1

        rho =  pyLIMA_parameters.rho
        gamma = np.array(gammas)

        return microlmagnification.amplification_FSPL_for_Lyrae(source_trajectory_x, source_trajectory_y, rho,
                                                               gamma, self.yoo_table)
    def find_gamma(self, telescope, star):

            telescope.find_gamma(star)
            gamma = telescope.gamma
            return gamma
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

        pulsations = self.compute_pulsations( time, telescope.filter, pyLIMA_parameters)
        f_source, f_blending = self.derive_telescope_flux(telescope, pyLIMA_parameters, amplification, pulsations)

        microlensing_model = f_source * amplification * pulsations + f_blending

        return microlensing_model, f_source, f_blending

    def derive_telescope_flux(self, telescope, pyLIMA_parameters, amplification, pulsations):

        try:
            # Fluxes parameters are fitted
            f_source = 2 * getattr(pyLIMA_parameters, 'fs_' + telescope.name) / 2
            f_blending = 2 * getattr(pyLIMA_parameters, 'fb_' + telescope.name) / 2

        except TypeError:

            # Fluxes parameters are estimated through np.polyfit


            lightcurve = telescope.lightcurve_flux
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            f_source, f_blending = np.polyfit(amplification * pulsations, flux, 1, w=1 / errflux)

        return f_source, f_blending

    def compute_radius(self, Teff, time, telescope_V, pyLIMA_parameters, magic_table = None):

        pulsations = self.compute_pulsations(time, telescope_V.filter, pyLIMA_parameters)
        f_source = 2 * getattr(pyLIMA_parameters, 'fs_' + telescope_V.name) / 2

        f_source_V = f_source * pulsations
        V_magnitude = 27.4 - 2.5 * np.log10(f_source_V)

        #radius = (0.636*10**-((V_magnitude-2*2.689)/5)/Teff**2)
        if magic_table:
            BC = magic_table(Teff)
        else:
            BC=0
        radius = 10**(0.2*(-(V_magnitude-2.689+BC)-10*np.log10(Teff)+37.35))
        return radius
    def compute_Teff(self,color):


        # Casagrande 2010
        color += -1.250
        theta_eff = 0.4033+0.8171*color-0.1987*color**2
        Teff = 5040/theta_eff

        return Teff
    def compute_color(self, time, telescope_V, telescope_I, pyLIMA_parameters ):



        pulsations = self.compute_pulsations(time, telescope_V.filter, pyLIMA_parameters)
        f_source = 2 * getattr(pyLIMA_parameters, 'fs_' + telescope_V.name) / 2

        f_source_V = f_source * pulsations

        pulsations = self.compute_pulsations(time, telescope_I.filter, pyLIMA_parameters)


        f_source = 2 * getattr(pyLIMA_parameters, 'fs_' + telescope_I.name) / 2

        f_source_I = f_source * pulsations


        V_magnitude = 27.4-2.5*np.log10(f_source_V)
        I_magnitude = 27.4-2.5*np.log10(f_source_I)

        return V_magnitude-I_magnitude

    def compute_pulsations(self, time, filter, pyLIMA_parameters):

        time = time - 2456425

        pulsations = 0
        period = getattr(pyLIMA_parameters, 'period')
        # factor = 0.0
        # pulsations = getattr(pyLIMA_parameters, 'AO'+'_' + telescope.filter)
        for i in xrange(self.number_of_harmonics):
            amplitude = getattr(pyLIMA_parameters, 'A' + str(i + 1) + '_' + filter)
            phase = getattr(pyLIMA_parameters, 'phi' + str(i + 1) + '_' + filter)

            pulsations += amplitude * np.cos(2 * np.pi * (i + 1) / period * time + phase)

        pulsations = 10 ** (pulsations / 2.5)

        return pulsations


    def find_telecopes_V_and_I(self):

        telescope_V = [i for i in self.event.telescopes if 'survey2' in i.name][0]
        telescope_I = [i for i in self.event.telescopes if 'survey1' in i.name][0]

        return telescope_V, telescope_I