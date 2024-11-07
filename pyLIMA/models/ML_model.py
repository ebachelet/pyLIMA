import abc
import collections
from collections import OrderedDict

import numpy as np

import pyLIMA.parallax.parallax
import pyLIMA.priors.parameters_boundaries
import pyLIMA.xallarap.xallarap
from pyLIMA.magnification import magnification_Jacobian
from pyLIMA.models import pyLIMA_fancy_parameters
from pyLIMA.orbitalmotion import orbital_motion
from pyLIMA.orbitalmotion import orbital_motion_3D


class MLmodel(object):
    """
    This class is mother of all other microlensing models and define all
    attributes/functions.

    Attributes
    ----------

    event : an Event object
    parallax_model : list[str,float], the parallax model type ('Annual', 'Terrestrial
    or 'Full') and t0,par
    double_source_model : list[str], the double source model, include ('Circular'
    with t0,xal) or not ('Static') xallarap
    yet)
    orbital_motion_model : list[str,float], the orbital motion model type ('2D',
    'Circular' or 'Keplerian') and t0,kep
    blend_flux_parameter : str, the blend flux parameter type ('fblend',
    'gblend=fblend/fsource' or 'noblend')
    photometry : bool, True if any telescopes in event object contains photometric data
    astrometry : bool, True if any telescopes in event object contains astrometric data
    model_dictionnary : dict, that represents the model parameters, including fancy
    parameters
    pyLIMA_standards_dictionnary : dict, that represents the standard model parameters
    fancy_to_pyLIMA_dictionnary : dict, that contains the names to transforms fancy
    parameters to pyLIMA standards
    pyLIMA_to_fancy_dictionnary = dict, that contains the names to transforms pyLIMA
    standards to fancy parameters
    pyLIMA_to_fancy : dict, that contains the functions to transforms fancy
    parameters to pyLIMA standards
    fancy_to_pyLIMA : dict, that contains the functions to transforms pyLIMA
    standards to fancy parameters
    Jacobian_flag : str, indicates if an analytical Jacobian is available for this model
    standard_parameters_boundaries : list[[float,float]], a list of list containing
    the lower and upper limits of standards parameters
    origin : list [str,[float,float]], a list containing the choice of the system
    origin, the floats indicating the X,Y origin
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, event, parallax=['None', 0.0], double_source=['None',0],
                 orbital_motion=['None', 0.0], blend_flux_parameter='ftotal',
                 origin=['center_of_mass', [0.0, 0.0]], fancy_parameters=None):
        """ Initialization of the attributes described above.
        """

        self.event = event
        self.parallax_model = parallax
        self.double_source_model = double_source
        self.orbital_motion_model = orbital_motion
        self.blend_flux_parameter = blend_flux_parameter

        self.photometry = False
        self.astrometry = False

        self.model_dictionnary = {}
        self.pyLIMA_standards_dictionnary = {}

        self.fancy_parameters = fancy_parameters
        self.Jacobian_flag = 'Numerical'
        self.standard_parameters_boundaries = []

        self.origin = origin

        self.check_data_in_event()
        self.define_pyLIMA_standard_parameters()
        self.define_model_parameters()

    @abc.abstractmethod
    def model_type(self):
        """
        Returns
        -------
        str the model type
        """

        pass

    @abc.abstractmethod
    def paczynski_model_parameters(self):
        """
        Returns
        -------
        model_dictionary: dict, the Paczynski parameters dictionnary
        """
        return

    @abc.abstractmethod
    def model_magnification(self, telescope, pyLIMA_parameters):
        """
        Parameters
        ----------
        telescope : a telescope object
        pyLIMA_parameters : a pyLIMA_parameters object

        Returns
        -------
        magnification: array, the corresponding model magnification A(t)
        """
        return

    @abc.abstractmethod
    def model_magnification_Jacobian(self, telescope, pyLIMA_parameters):
        """
        Parameters
        ----------
        telescope : a telescope object
        pyLIMA_parameters : a pyLIMA_parameters object

        Returns
        -------
        magnification_Jacobian : array, the magnification Jacobian, i.e. [dA(t)/dt0,
        dA(t)/du0....]
        amplification : array, the magnification

        """
        magnification_jacobian = \
            magnification_Jacobian.magnification_numerical_Jacobian(
                self, telescope, pyLIMA_parameters)
        amplification = self.model_magnification(telescope, pyLIMA_parameters,
                                                 return_impact_parameter=False)

        return magnification_jacobian, amplification

    @abc.abstractmethod
    def photometric_model_Jacobian(self, telescope, pyLIMA_parameters):
        """
        Parameters
        ----------
        telescope : a telescope object
        pyLIMA_parameters : a pyLIMA_parameters object

        Returns
        -------
        jacobi : array, the model Jacobian, including the magnification and
        telescopes fluxes Jacobians
        """

        magnification_jacobian, amplification = self.model_magnification_Jacobian(
            telescope, pyLIMA_parameters)
        # fsource, fblend = self.derive_telescope_flux(telescope, pyLIMA_parameters,
        # amplification[0])

        self.derive_telescope_flux(telescope, pyLIMA_parameters, amplification)
        fsource = pyLIMA_parameters['fsource_' + telescope.name]

        magnification_jacobian *= fsource

        if self.blend_flux_parameter == 'gblend':
            dfluxdfs = (amplification + pyLIMA_parameters['gblend_' + telescope.name])
            dfluxdg = [pyLIMA_parameters['fsource_' + telescope.name]] * len(
                amplification)

            jacobi = np.c_[magnification_jacobian, dfluxdfs, dfluxdg].T

        if self.blend_flux_parameter == 'fblend':
            dfluxdfs = (amplification)
            dfluxdfblend = [1] * len(amplification)

            jacobi = np.c_[magnification_jacobian, dfluxdfs, dfluxdfblend].T

        if self.blend_flux_parameter == 'ftotal':
            dfluxdfs = (amplification-1)
            dfluxdftot = [1] * len(amplification)

            jacobi = np.c_[magnification_jacobian, dfluxdfs, dfluxdftot].T

        if self.blend_flux_parameter == 'noblend':
            dfluxdfs = (amplification)

            jacobi = np.c_[magnification_jacobian, dfluxdfs].T

        return jacobi

    @abc.abstractmethod
    def new_origin(self, pyLIMA_parameters=None):
        """

        """

        x_center = self.origin[1][0]
        y_center = self.origin[1][1]

        return x_center, y_center

    def change_origin(self, pyLIMA_parameters):
        """
        Change the origin of the model, by modifying x_center and y_center in the
        pyLIMA_parameters.
        Depending of the model.origin[0]. Could be set to caustics, then it will
        compute the origin close
        to the central, wide of close caustics. Could be also primary or secondary,
        the position of the primay and
        secondary body.

        Parameters
        ----------
         pyLIMA_parameters : a pyLIMA_parameters object
        """

        new_x_center, new_y_center = self.new_origin(pyLIMA_parameters)

        t_0 = pyLIMA_fancy_parameters._t_center_to_t0(pyLIMA_parameters,
                                                      x_center=new_x_center,
                                                      y_center=new_y_center)

        u_0 = pyLIMA_fancy_parameters._u_center_to_u0(pyLIMA_parameters,
                                                      x_center=new_x_center,
                                                      y_center=new_y_center)

        pyLIMA_parameters['t0'] = t_0
        pyLIMA_parameters['u0'] = u_0

    def check_data_in_event(self):
        """
        Find if astrometry and/or photometry data are present
        """
        for telescope in self.event.telescopes:

            if telescope.lightcurve is not None:
                self.photometry = True

            if telescope.astrometry is not None:
                self.astrometry = True

    def define_pyLIMA_standard_parameters(self):
        """
        Define the pyLIMA_standard dictionnary, i.e Paczynski parameters + second
        order parameters +fluxes parameters.
        Also define the standard parameters boundaries
        """

        model_dictionnary = self.paczynski_model_parameters()

        model_dictionnary_updated = self.astrometric_model_parameters(model_dictionnary)

        self.second_order_model_parameters(model_dictionnary_updated)

        self.telescopes_fluxes_model_parameters(model_dictionnary_updated)

        self.pyLIMA_standards_dictionnary = OrderedDict(
            sorted(model_dictionnary_updated.items(), key=lambda x: x[1]))

        self.standard_parameters_boundaries = \
            pyLIMA.priors.parameters_boundaries.parameters_boundaries(
                self.event, self.pyLIMA_standards_dictionnary)

    def astrometric_model_parameters(self, model_dictionnary):
        """
        Define the standard astrometric model parameters, i.e. add theta_E, pi_s,
        mu_source_N, mu_source_E, and
        ref_N, ref_E for each telescope containing astrometric data to the model
        dictionnary
        WARNING: users need to provide a parallax model if they treat astrometric data!

        Parameters
        ----------
        model_dictionnary : dict, a model dictionnary

        Returns
        -------
        model_dictionnary : dict, the updated model dictionnary

        """

        parameter = 0
        for telescope in self.event.telescopes:

            if (telescope.astrometry is not None) & (parameter == 0):

                if self.parallax_model[0] == 'None':
                    raise ValueError(
                        'There are astrometric data in this model, please define a '
                        'parallax model')
                    # print('Defining a default parallax model since we have
                    # astrometric data....')
                    # self.parallax_model = ['Full', np.mean(
                    # telescope.lightcurve['time'].value)]

                model_dictionnary['theta_E'] = len(model_dictionnary)
                model_dictionnary['pi_source'] = len(model_dictionnary)
                model_dictionnary['mu_source_N'] = len(model_dictionnary)
                model_dictionnary['mu_source_E'] = len(model_dictionnary)

                parameter += 1
                self.Jacobian_flag = 'Numerical'

            if (telescope.astrometry is not None) & (parameter == 1):
                model_dictionnary['position_source_N_' + telescope.name] = len(
                    model_dictionnary)
                model_dictionnary['position_source_E_' + telescope.name] = len(
                    model_dictionnary)
                # model_dictionnary['position_blend_N_' + telescope.name] = len(
                # model_dictionnary)
                # model_dictionnary['position_blend_E_' + telescope.name] = len(
                # model_dictionnary)

        return model_dictionnary

    def second_order_model_parameters(self, model_dictionnary):
        """
        Update the model dictionnary with the corresponding second order parameters

        Parameters
        ----------
        model_dictionnary : dict, a model dictionnary

        Returns
        -------
        model_dictionnary : dict the updated model dictionnary

        """
        import copy
        jack = copy.copy(self.Jacobian_flag)

        if self.parallax_model[0] != 'None':
            jack = 'Numerical'
            model_dictionnary['piEN'] = len(model_dictionnary)
            model_dictionnary['piEE'] = len(model_dictionnary)

            self.event.compute_parallax_all_telescopes(self.parallax_model)

        if self.double_source_model[0] == 'Static':
            jack = 'Numerical'
            model_dictionnary['delta_t0'] = len(model_dictionnary)
            model_dictionnary['delta_u0'] = len(model_dictionnary)

        if self.double_source_model[0] == 'Circular':
            jack = 'Numerical'
            model_dictionnary['xi_para'] = len(model_dictionnary)
            model_dictionnary['xi_perp'] = len(model_dictionnary)
            model_dictionnary['xi_angular_velocity'] = len(model_dictionnary)
            model_dictionnary['xi_phase'] = len(model_dictionnary)
            model_dictionnary['xi_inclination'] = len(model_dictionnary)
            model_dictionnary['xi_mass_ratio'] = len(model_dictionnary)

        if self.double_source_model[0] != 'None':
            jack = 'Numerical'

            if 'rho' in model_dictionnary.keys():
                model_dictionnary['rho_2'] = len(model_dictionnary)

            filters = [telescope.filter for telescope in self.event.telescopes]

            unique_filters = np.unique(filters)

            for filter in unique_filters:
                model_dictionnary['q_flux_' + filter] = len(model_dictionnary)

        if self.orbital_motion_model[0] == '2D':
            jack = 'Numerical'
            model_dictionnary['v_para'] = len(model_dictionnary)
            model_dictionnary['v_perp'] = len(model_dictionnary)

        if self.orbital_motion_model[0] == 'Circular':
            jack = 'Numerical'
            model_dictionnary['v_para'] = len(model_dictionnary)
            model_dictionnary['v_perp'] = len(model_dictionnary)
            model_dictionnary['v_radial'] = len(model_dictionnary)

        if self.orbital_motion_model[0] == 'Keplerian':
            jack = 'Numerical'
            model_dictionnary['v_para'] = len(model_dictionnary)
            model_dictionnary['v_perp'] = len(model_dictionnary)
            model_dictionnary['v_radial'] = len(model_dictionnary)
            model_dictionnary['r_s'] = len(model_dictionnary)
            model_dictionnary['a_s'] = len(model_dictionnary)

        self.Jacobian_flag = jack

        return model_dictionnary

    def telescopes_fluxes_model_parameters(self, model_dictionnary):
        """
        Update the model dictionnary with the corresponding telescope fluxes parameters

        Parameters
        ----------
        model_dictionnary : dict, a model dictionnary

        Returns
        -------
        model_dictionnary : dict the updated model dictionnary

         """
        for telescope in self.event.telescopes:

            if telescope.lightcurve is not None:

                model_dictionnary['fsource_' + telescope.name] = len(model_dictionnary)

                if self.blend_flux_parameter == 'fblend':
                    model_dictionnary['fblend_' + telescope.name] = len(
                        model_dictionnary)

                if self.blend_flux_parameter == 'gblend':
                    model_dictionnary['gblend_' + telescope.name] = len(
                        model_dictionnary)

                if self.blend_flux_parameter == 'ftotal':
                    model_dictionnary['ftotal_' + telescope.name] = len(
                        model_dictionnary)

                if self.blend_flux_parameter == 'noblend':
                    pass

        return model_dictionnary

    def define_model_parameters(self):
        """
        Define the model parameters dictionnary. It is different to the
        pyLIMA_standards_dictionnary
        if there is fancy parameters request.
        """

        self.model_dictionnary = self.pyLIMA_standards_dictionnary.copy()

        if self.origin[0] != 'center_of_mass':
            self.model_dictionnary['t_center'] = self.model_dictionnary.pop('t0')
            self.model_dictionnary['u_center'] = self.model_dictionnary.pop('u0')

        if self.fancy_parameters is not None:

            self.Jacobian_flag = 'Numerical'

            for key_parameter in self.fancy_parameters.fancy_parameters.keys():

                try:

                    self.model_dictionnary[self.fancy_parameters.fancy_parameters[
                        key_parameter]] = self.model_dictionnary.pop(
                        key_parameter)

                except KeyError:
                    print('I skip the fancy parameter ' + key_parameter + ', as it is '
                                                                          'not part '
                                                                          'of model '
                          + self.model_type())
                    pass

        self.model_dictionnary = OrderedDict(
            sorted(self.model_dictionnary.items(), key=lambda x: x[1]))

    def print_model_parameters(self):
        """
        Print the model parameters currently defined
        """
        self.define_model_parameters()

        print(self.model_dictionnary)

    def compute_the_microlensing_model(self, telescope, pyLIMA_parameters):
        """
        Find the microlensing model for given telescope astrometry and photometry

        Parameters
        ----------
        telescope : a telescope object
        pyLIMA_parameters : a pyLIMA_parameters object

        Returns
        -------
        microlensing_model : dict, the corresponding microlensing model for
        photometry and astromtry if avalaible
        """
        photometric_model = None
        astrometric_model = None

        if telescope.lightcurve is not None:
            magnification = self.model_magnification(telescope, pyLIMA_parameters)

            # f_source, f_blend = self.derive_telescope_flux(telescope,
            # pyLIMA_parameters, magnification)
            self.derive_telescope_flux(telescope, pyLIMA_parameters, magnification)

            f_source = pyLIMA_parameters['fsource_' + telescope.name]
            f_blend = pyLIMA_parameters['fblend_' + telescope.name]

            photometric_model = f_source * magnification + f_blend

        if telescope.astrometry is not None:
            astrometric_model = self.model_astrometry(telescope, pyLIMA_parameters)

        microlensing_model = {'photometry': photometric_model,
                              'astrometry': astrometric_model}

        return microlensing_model

    def derive_telescope_flux(self, telescope, pyLIMA_parameters, magnification):
        """
        Set fsource and fblend in pyLIMA_parameters. If not present, estimate vita
        linear regression with the given magnification

        Parameters
        ----------
        telescope : a telescope object
        pyLIMA_parameters : a pyLIMA_parameters object
        magnification : array, containing the magnificationa at time t
        """
        try:
            # Fluxes parameters are in the pyLIMA_parameters
            f_source = 2 * pyLIMA_parameters['fsource_' + telescope.name] / 2

            if self.blend_flux_parameter == 'fblend':
                f_blend = pyLIMA_parameters['fblend_' + telescope.name]

            if self.blend_flux_parameter == 'gblend':
                g_blend = pyLIMA_parameters['gblend_' + telescope.name]
                f_blend = f_source * g_blend

            if self.blend_flux_parameter == 'ftotal':
                f_total = pyLIMA_parameters['ftotal_' + telescope.name]
                f_blend = f_total-f_source

            if self.blend_flux_parameter == 'noblend':
                f_blend = 0

        except TypeError:

            # Fluxes parameters are estimated through np.polyfit
            lightcurve = telescope.lightcurve
            flux = lightcurve['flux'].value
            err_flux = lightcurve['err_flux'].value

            try:

                if self.blend_flux_parameter == 'noblend':

                    f_source = np.median(flux / magnification)
                    f_blend = 0.0

                else:

                    f_source, f_blend = np.polyfit(magnification, flux, 1,
                                                   w=1 / err_flux)

                    # if np.isinf(flux) | np.isinf(err_flux):
                    #    breakpoint()
                    # polynomial = np.polynomial.polynomial.Polynomial.fit(
                    #    magnification, flux, deg=1,  w=1 / err_flux)

                    # f_blend,f_source = polynomial.convert().coef
                    # breakpoint()
                    # from sklearn import linear_model, datasets
                    # ransac = linear_model.RANSACRegressor()
                    # ransac.fit(magnification.reshape(-1, 1), flux)
                    # f_source = ransac.estimator_.coef_[0]
                    # f_blend = ransac.estimator_.intercept_

            except ValueError:

                f_source = 0.0
                f_blend = 0.0

        pyLIMA_parameters['fsource_' + telescope.name] = f_source
        pyLIMA_parameters['fblend_' + telescope.name] = f_blend
        pyLIMA_parameters['gblend_' + telescope.name] = f_blend / f_source
        pyLIMA_parameters['ftotal_' + telescope.name] = f_source+f_blend

    def find_telescopes_fluxes(self, parameters):
        """
        Find fsource and fblend for all telescope for a given fancy_parameter

        Parameters
        ----------
        parameters :  list, a list of parameters

        Returns
        -------
        thefluxes : dict, a dictionnary containing the fluxes of all telescopes
        """
        pyLIMA_parameters = self.compute_pyLIMA_parameters(parameters)

        keys = []
        fluxes = []
        for telescope in self.event.telescopes:

            if telescope.lightcurve is not None:

                self.compute_the_microlensing_model(telescope, pyLIMA_parameters)

                f_source = pyLIMA_parameters['fsource_' + telescope.name]
                keys.append('fsource_' + telescope.name)
                fluxes.append(f_source)

                if self.blend_flux_parameter == 'fblend':
                    f_blend = pyLIMA_parameters['fblend_' + telescope.name]

                    keys.append('fblend_' + telescope.name)
                    fluxes.append(f_blend)

                if self.blend_flux_parameter == 'gblend':
                    f_blend = pyLIMA_parameters['fblend_' + telescope.name]

                    keys.append('gblend_' + telescope.name)
                    fluxes.append(f_blend / f_source)

                if self.blend_flux_parameter == 'ftotal':
                    f_blend = pyLIMA_parameters['fblend_' + telescope.name]

                    keys.append('ftotal_' + telescope.name)
                    fluxes.append(f_blend + f_source)

        thefluxes = collections.OrderedDict()

        for ind, key in enumerate(keys):
            thefluxes[key] = fluxes[ind]

        return thefluxes

    def compute_pyLIMA_parameters(self, model_parameters, fancy_parameters=True):
        """
         Realize the transformation between the fancy parameters to fit to the
         standard pyLIMA parameters needed to compute a model.

         Parameters
         ----------
         fancy_parameter :  list, a list of fancy parameters

         Returns
         -------
         pyLIMA_parameters : dict, a dictionnary the pyLIMA parameters
         """

        parameter_dictionnary = self.model_dictionnary.copy()
        pyLIMA_parameters = collections.OrderedDict()

        for ind, key_parameter in enumerate(parameter_dictionnary.keys()):

            try:

                pyLIMA_parameters[key_parameter] = model_parameters[ind]

            except IndexError:

                pyLIMA_parameters[key_parameter] = None

        if self.fancy_parameters is not None:
            self.fancy_to_pyLIMA_parameters(pyLIMA_parameters)

        if self.origin[0] != 'center_of_mass':
            self.change_origin(pyLIMA_parameters)

        if 'v_radial' in self.model_dictionnary.keys():

            v_para = pyLIMA_parameters['v_para']
            v_perp = pyLIMA_parameters['v_perp']
            v_radial = pyLIMA_parameters['v_radial']
            separation = pyLIMA_parameters['separation']

            if self.orbital_motion_model[0] == 'Circular':

                try:

                    r_s = -v_para / v_radial

                except ValueError:

                    v_radial = np.sign(v_radial) * 10 ** -20

                    r_s = -v_para / v_radial

                a_s = 1

            else:

                r_s = pyLIMA_parameters['r_s']
                a_s = pyLIMA_parameters['a_s']

            longitude_ascending_node, inclination, omega_peri, a_true, \
                orbital_velocity, eccentricity, true_anomaly, t_periastron, x, y, z = \
                orbital_motion_3D.orbital_parameters_from_position_and_velocities(
                    separation, r_s, a_s, v_para, v_perp, v_radial,
                    self.orbital_motion_model[1])

            Rmatrix = np.c_[x[:2], y[:2]]

            pyLIMA_parameters['Rmatrix'] = Rmatrix
            pyLIMA_parameters['a_true'] = a_true
            pyLIMA_parameters['eccentricity'] = eccentricity
            pyLIMA_parameters['orbital_velocity'] = orbital_velocity
            pyLIMA_parameters['t_periastron'] = t_periastron

        return pyLIMA_parameters

    def fancy_to_pyLIMA_parameters(self, fancy_parameters):

        for standard_key, fancy_key in self.fancy_parameters.fancy_parameters.items():

            try:

                value = getattr(self.fancy_parameters, standard_key)(fancy_parameters)
                fancy_parameters[standard_key] = value

            except Exception:

                pass

    def pyLIMA_to_fancy_parameters(self, pyLIMA_parameters):

        for standard_key, fancy_key in self.fancy_parameters.fancy_parameters.items():

            try:

                value = getattr(self.fancy_parameters, fancy_key)(pyLIMA_parameters)
                pyLIMA_parameters[fancy_key] = value

            except Exception :

                pass

    def sources_trajectory(self, telescope, pyLIMA_parameters, data_type=None):
        """
        Compute the trajectories of the two sources, if needed

        Parameters
        ----------
        telescope :  a telescope object
        pyLIMA_parameters : a pyLIMA_parameters objecr

        Returns
        -------
        source1_trajectory_x : the x coordinates of source 1
        source1_trajectory_y : the y coordinates of source 1
        source2_trajectory_x : the x coordinates of source 2
        source2_trajectory_y : the y coordinates of source 2
        """

        if data_type == 'photometry':

            lightcurve = telescope.lightcurve
            time = lightcurve['time'].value

            if 'piEN' in pyLIMA_parameters.keys():
                parallax_delta_positions = telescope.deltas_positions['photometry']

        if data_type == 'astrometry':

            astrometry = telescope.astrometry
            time = astrometry['time'].value

            if 'piEN' in pyLIMA_parameters.keys():
                parallax_delta_positions = telescope.deltas_positions['astrometry']

        tau = (time - pyLIMA_parameters['t0']) / pyLIMA_parameters['tE']
        beta = np.array([pyLIMA_parameters['u0']] * len(tau))

        if 'alpha' in pyLIMA_parameters.keys():

            alpha = pyLIMA_parameters['alpha']

        else:

            alpha = 0

        if self.parallax_model[0] != 'None':

            parallax_delta_tau, parallax_delta_beta = (
                self.parallax_trajectory_shifts(parallax_delta_positions,
                                                pyLIMA_parameters))

        else:

            parallax_delta_tau, parallax_delta_beta = 0, 0

        if self.orbital_motion_model[0] != 'None':

            dseparation, dalpha = orbital_motion.orbital_motion_shifts(
                self.orbital_motion_model,
                telescope.lightcurve['time'].value,
                pyLIMA_parameters)

            alpha -= dalpha  # Binary axes is fixed

        else:

            dseparation = np.array([0] * len(tau))
            dalpha = np.array([0] * len(tau))

        tau += parallax_delta_tau
        beta += parallax_delta_beta

        # double_source?
        if self.double_source_model[0] != 'None':  # then we have two sources

            (source1_delta_tau, source1_delta_beta, source2_delta_tau,
             source2_delta_beta) = self.xallarap_trajectory_shifts(
                time, pyLIMA_parameters, body='primary')

            tau2 = tau + source2_delta_tau
            beta2 = beta + source2_delta_beta

            lens_trajectory_x2 = tau2 * np.cos(alpha) - beta2 * np.sin(alpha)
            lens_trajectory_y2 = tau2 * np.sin(alpha) + beta2 * np.cos(alpha)

            source2_trajectory_x = -lens_trajectory_x2
            source2_trajectory_y = -lens_trajectory_y2


        else:

            source1_delta_tau, source1_delta_beta = 0, 0
            source2_trajectory_x, source2_trajectory_y = None, None

        tau1 = tau + source1_delta_tau
        beta1 = beta + source1_delta_beta

        lens_trajectory_x1 = tau1 * np.cos(alpha) - beta1 * np.sin(alpha)
        lens_trajectory_y1 = tau1 * np.sin(alpha) + beta1 * np.cos(alpha)

        source1_trajectory_x = -lens_trajectory_x1
        source1_trajectory_y = -lens_trajectory_y1

        return (source1_trajectory_x, source1_trajectory_y,
                source2_trajectory_x, source2_trajectory_y,
                dseparation, dalpha)

    def parallax_trajectory_shifts(self, parallax_delta_positions, pyLIMA_parameters):

        piE = np.array([pyLIMA_parameters['piEN'], pyLIMA_parameters['piEE']])

        parallax_delta_tau, parallax_delta_beta = (
            pyLIMA.parallax.parallax.compute_parallax_curvature(piE,
                                                                parallax_delta_positions))

        return parallax_delta_tau, parallax_delta_beta

    def xallarap_trajectory_shifts(self, time, pyLIMA_parameters, body='primary'):

        delta_position_1_1, delta_position_2_1, delta_position_1_2, delta_position_2_2 = (
            pyLIMA.xallarap.xallarap.xallarap_shifts(
            self.double_source_model, time, pyLIMA_parameters,
            body=body))

        if self.double_source_model[0] == 'Circular':

            xiE = np.array([pyLIMA_parameters['xi_para'], pyLIMA_parameters['xi_perp']])

            delta_position = np.array([delta_position_1_1 ,
                                       delta_position_2_1 ])

            source1_delta_tau, source1_delta_beta = (
                pyLIMA.xallarap.xallarap.compute_xallarap_curvature(xiE,
                                                                    delta_position))

            delta_position2 =  np.array([delta_position_1_2 ,
                                       delta_position_2_2 ])

            source2_delta_tau, source2_delta_beta = (
                pyLIMA.xallarap.xallarap.compute_xallarap_curvature(xiE,
                                                                    delta_position2))


        else:

            source1_delta_tau = delta_position_1_1
            source1_delta_beta = delta_position_2_1

            source2_delta_tau = delta_position_1_2
            source2_delta_beta = delta_position_2_2

        return (source1_delta_tau, source1_delta_beta, source2_delta_tau,
                source2_delta_beta)