import abc
import collections
from collections import OrderedDict

import numpy as np
import pyLIMA.parallax.parallax
import pyLIMA.priors.parameters_boundaries
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
    xallarap_model : list[str], the xallarap model (not implemented yet)
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

    def __init__(self, event, parallax=['None', 0.0], xallarap=['None'],
                 orbital_motion=['None', 0.0], blend_flux_parameter='fblend',
                 origin=['center_of_mass', [0.0, 0.0]], fancy_parameters={}):
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
        self.fancy_to_pyLIMA_dictionnary = fancy_parameters.copy()
        self.pyLIMA_to_fancy_dictionnary = dict(
            (v, k) for k, v in self.fancy_to_pyLIMA_dictionnary.items())

        self.pyLIMA_to_fancy = {}
        self.fancy_to_pyLIMA = {}

        self.Jacobian_flag = 'OK'
        self.standard_parameters_boundaries = []

        self.origin = origin

        self.check_data_in_event()
        self.define_pyLIMA_standard_parameters()
        self.define_fancy_parameters()

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
        fsource = getattr(pyLIMA_parameters, 'fsource_' + telescope.name)
        magnification_jacobian *= fsource

        if self.blend_flux_parameter == 'gblend':
            dfluxdfs = (amplification + getattr(pyLIMA_parameters,
                                                'gblend_' + telescope.name))
            dfluxdg = [getattr(pyLIMA_parameters, 'fsource_' + telescope.name)] * len(
                amplification)

            jacobi = np.c_[magnification_jacobian, dfluxdfs, dfluxdg].T

        if self.blend_flux_parameter == 'fblend':
            dfluxdfs = (amplification)
            dfluxdg = [1] * len(amplification)

            jacobi = np.c_[magnification_jacobian, dfluxdfs, dfluxdg].T

        if self.blend_flux_parameter == 'noblend':
            dfluxdfs = (amplification)

            jacobi = np.c_[magnification_jacobian, dfluxdfs].T

        print(jacobi)
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

        setattr(pyLIMA_parameters, 't0', t_0)
        setattr(pyLIMA_parameters, 'u0', u_0)

    def check_data_in_event(self):
        """
        Find if astrometry and/or photometry data are present
        """
        for telescope in self.event.telescopes:

            if telescope.lightcurve_magnitude is not None:
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

    def define_fancy_parameters(self):
        """
        Define the fancy parameters, if the origin is different than center of mass
        and if users defined fancy parameters.
        Also define the standard parameters boundaries
        """

        if self.origin[0] != 'center_of_mass':
            self.fancy_to_pyLIMA_dictionnary['t_center'] = 't0'
            self.fancy_to_pyLIMA_dictionnary['u_center'] = 'u0'

            self.pyLIMA_to_fancy_dictionnary = dict(
                (v, k) for k, v in self.fancy_to_pyLIMA_dictionnary.items())

        if len(self.fancy_to_pyLIMA_dictionnary) != 0:

            import pickle

            keys = self.fancy_to_pyLIMA_dictionnary.copy().keys()

            for key in keys:

                parameter = self.fancy_to_pyLIMA_dictionnary[key]

                if parameter in self.pyLIMA_standards_dictionnary.keys():

                    self.fancy_to_pyLIMA_dictionnary[key] = parameter

                    try:
                        self.pyLIMA_to_fancy[key] = pickle.loads(
                            pickle.dumps(getattr(pyLIMA_fancy_parameters, key)))
                        self.fancy_to_pyLIMA[parameter] = pickle.loads(
                            pickle.dumps(getattr(pyLIMA_fancy_parameters, parameter)))

                    except AttributeError:

                        self.pyLIMA_to_fancy[key] = None
                        self.fancy_to_pyLIMA[parameter] = None

                else:

                    self.fancy_to_pyLIMA_dictionnary.pop(key)
                    print(
                        'I skip the fancy parameter ' + parameter + ', as it is not '
                                                                    'part of model '
                        + self.model_type())

        self.define_model_parameters()

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
                    # telescope.lightcurve_flux['time'].value)]

                model_dictionnary['theta_E'] = len(model_dictionnary)
                model_dictionnary['pi_source'] = len(model_dictionnary)
                model_dictionnary['mu_source_N'] = len(model_dictionnary)
                model_dictionnary['mu_source_E'] = len(model_dictionnary)

                parameter += 1
                self.Jacobian_flag = 'No Way'

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
        jack = np.copy(self.Jacobian_flag)

        if self.parallax_model[0] != 'None':
            jack = 'Numerical'
            model_dictionnary['piEN'] = len(model_dictionnary)
            model_dictionnary['piEE'] = len(model_dictionnary)

            self.event.compute_parallax_all_telescopes(self.parallax_model)

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

        if self.Jacobian_flag != 'No Way':
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

            if telescope.lightcurve_flux is not None:

                model_dictionnary['fsource_' + telescope.name] = len(model_dictionnary)

                if self.blend_flux_parameter == 'fblend':
                    model_dictionnary['fblend_' + telescope.name] = len(
                        model_dictionnary)

                if self.blend_flux_parameter == 'gblend':
                    model_dictionnary['gblend_' + telescope.name] = len(
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

        if len(self.pyLIMA_to_fancy) != 0:

            self.Jacobian_flag = 'No Way'

            for key_parameter in self.fancy_to_pyLIMA_dictionnary.keys():

                try:
                    self.model_dictionnary[key_parameter] = self.model_dictionnary.pop(
                        self.fancy_to_pyLIMA_dictionnary[key_parameter])
                except ValueError:

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

        if telescope.lightcurve_flux is not None:
            magnification = self.model_magnification(telescope, pyLIMA_parameters)

            # f_source, f_blend = self.derive_telescope_flux(telescope,
            # pyLIMA_parameters, magnification)
            self.derive_telescope_flux(telescope, pyLIMA_parameters, magnification)

            f_source = getattr(pyLIMA_parameters, 'fsource_' + telescope.name)
            f_blend = getattr(pyLIMA_parameters, 'fblend_' + telescope.name)

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
            f_source = 2 * getattr(pyLIMA_parameters, 'fsource_' + telescope.name) / 2

            if self.blend_flux_parameter == 'fblend':
                f_blend = getattr(pyLIMA_parameters, 'fblend_' + telescope.name)

            if self.blend_flux_parameter == 'gblend':
                g_blend = getattr(pyLIMA_parameters, 'gblend_' + telescope.name)
                f_blend = f_source * g_blend

            if self.blend_flux_parameter == 'noblend':
                f_blend = 0

        except TypeError:

            # Fluxes parameters are estimated through np.polyfit
            lightcurve = telescope.lightcurve_flux
            flux = lightcurve['flux'].value
            err_flux = lightcurve['err_flux'].value

            try:

                if self.blend_flux_parameter == 'noblend':

                    f_source = np.median(flux / magnification)
                    f_blend = 0.0

                else:

                    # breakpoint()
                    f_source, f_blend = np.polyfit(magnification, flux, 1,
                                                   w=1 / err_flux)

                    # from sklearn import linear_model, datasets
                    # ransac = linear_model.RANSACRegressor()
                    # ransac.fit(magnification.reshape(-1, 1), flux)
                    # f_source = ransac.estimator_.coef_[0]
                    # f_blend = ransac.estimator_.intercept_

            except ValueError:

                f_source = 0.0
                f_blend = 0.0

        setattr(pyLIMA_parameters, 'fsource_' + telescope.name, f_source)
        setattr(pyLIMA_parameters, 'fblend_' + telescope.name, f_blend)

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

            if telescope.lightcurve_flux is not None:

                self.compute_the_microlensing_model(telescope, pyLIMA_parameters)

                f_source = getattr(pyLIMA_parameters, 'fsource_' + telescope.name)
                keys.append('fsource_' + telescope.name)
                fluxes.append(f_source)

                if self.blend_flux_parameter == 'fblend':
                    f_blend = getattr(pyLIMA_parameters, 'fblend_' + telescope.name)

                    keys.append('fblend_' + telescope.name)
                    fluxes.append(f_blend)

                if self.blend_flux_parameter == 'gblend':
                    f_blend = getattr(pyLIMA_parameters, 'fblend_' + telescope.name)

                    keys.append('gblend_' + telescope.name)
                    fluxes.append(f_blend / f_source)

        thefluxes = collections.namedtuple('parameters', keys)

        for ind, key in enumerate(keys):
            setattr(thefluxes, key, fluxes[ind])

        return thefluxes

    def compute_pyLIMA_parameters(self, fancy_parameters):
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

        model_parameters = collections.namedtuple('parameters',
                                                  self.model_dictionnary.keys())

        for key_parameter in self.model_dictionnary.keys():

            try:

                setattr(model_parameters, key_parameter,
                        fancy_parameters[self.model_dictionnary[key_parameter]])

            except IndexError:

                setattr(model_parameters, key_parameter, None)

        pyLIMA_parameters = self.fancy_parameters_to_pyLIMA_standard_parameters(
            model_parameters)

        if self.origin[0] != 'center_of_mass':
            self.change_origin(pyLIMA_parameters)

        if 'v_radial' in self.model_dictionnary.keys():

            v_para = pyLIMA_parameters.v_para
            v_perp = pyLIMA_parameters.v_perp
            v_radial = pyLIMA_parameters.v_radial
            separation = pyLIMA_parameters.separation

            if self.orbital_motion_model[0] == 'Circular':

                try:

                    r_s = -v_para / v_radial

                except ValueError:

                    v_radial = np.sign(v_radial) * 10 ** -20

                    r_s = -v_para / v_radial

                a_s = 1

            else:

                r_s = pyLIMA_parameters.r_s
                a_s = pyLIMA_parameters.a_s

            longitude_ascending_node, inclination, omega_peri, a_true, \
                orbital_velocity, eccentricity, true_anomaly, t_periastron, x, y, z = \
                orbital_motion_3D.orbital_parameters_from_position_and_velocities(
                    separation, r_s, a_s, v_para, v_perp, v_radial,
                    self.orbital_motion_model[1])

            Rmatrix = np.c_[x[:2], y[:2]]

            setattr(pyLIMA_parameters, 'Rmatrix', Rmatrix)
            setattr(pyLIMA_parameters, 'a_true', a_true)
            setattr(pyLIMA_parameters, 'eccentricity', eccentricity)
            setattr(pyLIMA_parameters, 'orbital_velocity', orbital_velocity)
            setattr(pyLIMA_parameters, 't_periastron', t_periastron)

        return pyLIMA_parameters

    def fancy_parameters_to_pyLIMA_standard_parameters(self, fancy_parameters):
        """
        Transform the fancy parameters to the pyLIMA standards. The output got all
        the necessary standard attributes, example t0, u0, tE...

        Parameters
        ----------
        fancy_parameters :  a pyLIMA_parameters object

        Returns
        -------
        fancy_parameters : dict, an updated dictionnary the pyLIMA parameters
        """

        if len(self.fancy_to_pyLIMA) != 0:

            for key_parameter in self.fancy_to_pyLIMA.keys():

                try:

                    setattr(fancy_parameters, key_parameter,
                            self.fancy_to_pyLIMA[key_parameter](fancy_parameters))

                except TypeError:

                    pass

        return fancy_parameters

    def pyLIMA_standard_parameters_to_fancy_parameters(self, pyLIMA_parameters):
        """
        Transform the pyLIMA standards parameters to the fancy parameters. The output
        got all
        the necessary fancy attributes.

        Parameters
        ----------
        pyLIMA_parameter :  a pyLIMA parameter object

        :Returns
        -------
        pyLIMA_parameters : dict, the updated pyLIMA parameter containing the fancy
        parameters
        """

        if len(self.pyLIMA_to_fancy) != 0:

            for key_parameter in self.pyLIMA_to_fancy.keys():
                setattr(pyLIMA_parameters, key_parameter,
                        self.pyLIMA_to_fancy[key_parameter](pyLIMA_parameters))

        return pyLIMA_parameters

    def source_trajectory(self, telescope, pyLIMA_parameters, data_type=None):
        """
        Compute the microlensing source trajectory associated to a telescope for the
        given parameters for the photometry
        or astrometry data

        Parameters
        ----------
        telescope : a telescope object
        pyLIMA_parameters : a pyLIMA_parameters object
        data_type : str, photometry or astrometry

        Returns
        ----------
        source_trajectory_x : array, the source x position
        source_trajectory_y : array, the source y position
        dseparation : array, the modification of binary separation if orbital motion
        is present
        dalpha : array, the modification of the lens trajectory angle due to the
        orbital motion of the lens
        """
        # Linear basic trajectory

        if data_type == 'photometry':

            lightcurve = telescope.lightcurve_flux
            time = lightcurve['time'].value

            if 'piEN' in pyLIMA_parameters._fields:
                delta_positions = telescope.deltas_positions['photometry']

        if data_type == 'astrometry':

            astrometry = telescope.astrometry
            time = astrometry['time'].value

            if 'piEN' in pyLIMA_parameters._fields:
                delta_positions = telescope.deltas_positions['astrometry']

        tau = (time - pyLIMA_parameters.t0) / pyLIMA_parameters.tE
        beta = np.array([pyLIMA_parameters.u0] * len(tau))

        # These following second order induce curvatures in the source trajectory
        # Parallax?

        if 'piEN' in pyLIMA_parameters._fields:

            try:

                piE = np.array([pyLIMA_parameters.piEN, pyLIMA_parameters.piEE])
                parallax_delta_tau, parallax_delta_beta = \
                    pyLIMA.parallax.parallax.compute_parallax_curvature(
                        piE, delta_positions)

                tau += parallax_delta_tau
                beta += parallax_delta_beta

            except ValueError:

                pass

        # Xallarap?
        if 'XiEN' in pyLIMA_parameters._fields:
            # XiE = np.array([pyLIMA_parameters.XiEN, pyLIMA_parameters.XiEE])
            # ra = pyLIMA_parameters.ra_xallarap
            # dec = pyLIMA_parameters.dec_xallarap
            # period = pyLIMA_parameters.period_xallarap

            # if 'eccentricity_xallarap' in pyLIMA_parameters._fields:
            #    eccentricity = pyLIMA_parameters.eccentricity_xallarap
            #    t_periastron = pyLIMA_parameters.t_periastron_xallarap

            #    orbital_elements = [telescope.lightcurve_flux[:, 0], ra, dec,
            #    period, eccentricity, t_periastron]
            #    xallarap_delta_tau, xallarap_delta_beta =
            #    microlxallarap.compute_xallarap_curvature(XiE,
            #      orbital_elements,
            #      mode='elliptic')
            # else:

            #    orbital_elements = [telescope.lightcurve_flux[:, 0], ra, dec, period]
            #    xallarap_delta_tau, xallarap_delta_beta =
            #    microlxallarap.compute_xallarap_curvature(XiE,
            #    orbital_elements)

            # tau += xallarap_delta_tau
            # beta += xallarap_delta_beta

            ### To be implemented
            pass
        if 'alpha' in pyLIMA_parameters._fields:

            alpha = pyLIMA_parameters.alpha

        else:

            alpha = 0

        # Orbital motion?

        if self.orbital_motion_model[0] != 'None':

            dseparation, dalpha = orbital_motion.orbital_motion_shifts(
                self.orbital_motion_model,
                telescope.lightcurve_flux['time'].value,
                pyLIMA_parameters)
            alpha += dalpha

        else:

            dseparation = np.array([0] * len(tau))
            dalpha = np.array([0] * len(tau))

        lens_trajectory_x = tau * np.cos(alpha) - beta * np.sin(alpha)
        lens_trajectory_y = tau * np.sin(alpha) + beta * np.cos(alpha)

        source_trajectory_x = -lens_trajectory_x
        source_trajectory_y = -lens_trajectory_y

        return source_trajectory_x, source_trajectory_y, dseparation, dalpha
