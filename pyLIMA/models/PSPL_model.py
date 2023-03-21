import numpy as np

from pyLIMA.models.ML_model import MLmodel
from pyLIMA.magnification import magnification_PSPL
from pyLIMA.astrometry import astrometric_shifts, astrometric_positions
from pyLIMA.magnification import magnification_Jacobian

class PSPLmodel(MLmodel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: PSPL
        :rtype: string
        """
        return 'PSPL'

    def paczynski_model_parameters(self):
        """ Define the PSPL standard parameters, [t0,u0,tE]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'tE': 2}
        self.Jacobian_flag='Analytical'

        return model_dictionary

    def model_astrometry(self, telescope, pyLIMA_parameters):

        """ The astrometric shifts associated to a PSPL model. More details in microlmagnification module.

               :param object telescope: a telescope object. More details in telescope module.
               :param object pyLIMA_parameters: a namedtuple which contain the parameters


               :return: astro_shifts
               :rtype: array_like
        """

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
        """ The magnification associated to a PSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :param boolean return_impact_parameter: if the impact parameter is needed or not

        :return: magnification
        :rtype: array_like
        """

        if telescope.lightcurve_flux is not None:

            source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters,
                                                                                 data_type='photometry')

            magnification = magnification_PSPL.magnification_PSPL(source_trajectory_x, source_trajectory_y,
                                                                              return_impact_parameter)
        else:

            magnification = None

        return magnification

    def model_magnification_Jacobian(self, telescope, pyLIMA_parameters):
        """ The derivative of a PSPL model lightcurve

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: jacobi
        :rtype: array_like
        """

        if self.Jacobian_flag == 'Analytical':

            magnification_jacobian, amplification = magnification_Jacobian.magnification_PSPL_Jacobian(self,telescope,pyLIMA_parameters)

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

