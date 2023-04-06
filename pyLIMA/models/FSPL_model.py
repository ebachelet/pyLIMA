import numpy as np
import pkg_resources
from scipy import interpolate, misc

from pyLIMA.models.ML_model import MLmodel
from pyLIMA.magnification import magnification_FSPL, magnification_Jacobian
from pyLIMA.astrometry import astrometric_shifts, astrometric_positions

resource_path = '/'.join(('data', 'Yoo_B0B1.dat'))
template = pkg_resources.resource_filename('pyLIMA', resource_path)

try:

    yoo_table = np.loadtxt(template)

except:

    print('ERROR : No Yoo_B0B1.dat file found, please check!')

b0b1 = yoo_table
zz = b0b1[:, 0]
b0 = b0b1[:, 1]
b1 = b0b1[:, 2]

interpol_b0 = interpolate.interp1d(zz, b0, kind='linear')
interpol_b1 = interpolate.interp1d(zz, b1, kind='linear')

dB0 = misc.derivative(lambda x: interpol_b0(x), zz[1:-1], dx=10 ** -4, order=3)
dB1 = misc.derivative(lambda x: interpol_b1(x), zz[1:-1], dx=10 ** -4, order=3)
dB0 = np.append(2.0, dB0)
dB0 = np.concatenate([dB0, [dB0[-1]]])
dB1 = np.append((2.0 - 3 * np.pi / 4), dB1)
dB1 = np.concatenate([dB1, [dB1[-1]]])
interpol_db0 = interpolate.interp1d(zz, dB0, kind='linear')
interpol_db1 = interpolate.interp1d(zz, dB1, kind='linear')
yoo_table = [zz, interpol_b0, interpol_b1, interpol_db0, interpol_db1]


class FSPLmodel(MLmodel):

    def __init__(self, event, parallax=['None', 0.0], xallarap=['None'],
                 orbital_motion=['None', 0.0], origin=['center_of_mass', [0,0]], blend_flux_parameter='fblend',fancy_parameters={}):

        super().__init__(event, parallax=parallax, xallarap=xallarap,
                         orbital_motion=orbital_motion, origin=origin, blend_flux_parameter=blend_flux_parameter,
                         fancy_parameters=fancy_parameters)

        self.yoo_table = yoo_table

    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: FSPL
        :rtype: string
        """

        return 'FSPL'

    def paczynski_model_parameters(self):
        """ Define the FSPL standard parameters, [t0,u0,tE,rho]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'tE': 2, 'rho': 3}
        self.Jacobian_flag='Analytical'

        return model_dictionary

    def model_astrometry(self, telescope, pyLIMA_parameters):


        """ The astrometric shifts associated to a PSPL model. More details in microlmagnification module.

               :param object telescope: a telescope object. More details in telescope module.
               :param object pyLIMA_parameters: a namedtuple which contain the parameters


               :return: astro_shifts
               :rtype: array_like
        """


        #########THIS IS PSPL ASTROMETRY, HERE FOR TESTING....################
        if telescope.astrometry is not None:

            source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters, data_type='astrometry')

            # Blended centroid shifts....
            #magnification = self.model_magnification(telescope, pyLIMA_parameters)
            #try:
            #    g_blend = f_blending/f_source
            #    shifts = astrometric_shifts.PSPL_shifts_with_blend(source_trajectory_x, source_trajectory_y, pyLIMA_parameters.theta_E,g_blend)
            #    angle = np.arctan2(source_trajectory_y,source_trajectory_x)
            #    shifts = np.array([shifts*np.cos(angle), shifts*np.sin(angle)])

            #except:

            shifts = astrometric_shifts.PSPL_shifts_no_blend(source_trajectory_x, source_trajectory_y,
                                                                 pyLIMA_parameters.theta_E)

            delta_ra, delta_dec = astrometric_positions.xy_shifts_to_NE_shifts(shifts,pyLIMA_parameters.piEN,
                                                                                pyLIMA_parameters.piEE)

            position_ra, position_dec = astrometric_positions.source_astrometric_position(telescope, pyLIMA_parameters,
                                                                                          shifts=(delta_ra, delta_dec),
                                                                                          time_ref=self.parallax_model[
                                                                                              1])

            astro_shifts = np.array([position_ra, position_dec])

        else:

            astro_shifts = None

        return astro_shifts
    def model_magnification(self, telescope, pyLIMA_parameters, return_impact_parameter=False):
        """ The magnification associated to a FSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification, impact_parameter
        :rtype: array_like,array_like
        """
        if telescope.lightcurve_flux is not None:
            source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters,
                                                                                 data_type='photometry')
            rho = pyLIMA_parameters.rho
            gamma = telescope.ld_gamma

            magnification = magnification_FSPL.magnification_FSPL_Yoo(source_trajectory_x, source_trajectory_y,
                                                                              rho,
                                                                              gamma, self.yoo_table,
                                                                              return_impact_parameter)
        else:

            magnification = None

        return magnification

    def model_magnification_Jacobian(self, telescope, pyLIMA_parameters):
        """ The derivative of a FSPL model

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: jacobi
        :rtype: array_like
        """

        if self.Jacobian_flag == 'Analytical':

            magnification_jacobian= magnification_Jacobian.magnification_FSPL_Jacobian(self, telescope,pyLIMA_parameters)

        else:

            magnification_jacobian = magnification_Jacobian.magnification_numerical_Jacobian(self, telescope,
                                                                                             pyLIMA_parameters)
        amplification = self.model_magnification(telescope, pyLIMA_parameters, return_impact_parameter=True)

        return magnification_jacobian, amplification

    def astrometry_Jacobian(self, telescope, pyLIMA_parameters):
        """ The derivative of a PSPL model lightcurve

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: jacobi
        :rtype: array_like
        """

        pass