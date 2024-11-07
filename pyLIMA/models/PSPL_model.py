import numpy as np
from pyLIMA.astrometry import astrometric_shifts, astrometric_positions
from pyLIMA.magnification import magnification_Jacobian
from pyLIMA.magnification import magnification_PSPL
from pyLIMA.models.ML_model import MLmodel


class PSPLmodel(MLmodel):

    def model_type(self):

        return 'PSPL'

    def paczynski_model_parameters(self):
        """
        [t0,u0,tE]
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'tE': 2}
        self.Jacobian_flag = 'Analytical'

        return model_dictionary

    def model_astrometry(self, telescope, pyLIMA_parameters):
        """
        The astrometric shifts associated to a PSPL model.
        See https://ui.adsabs.harvard.edu/abs/2000ApJ...534..213D/abstract
            https://ui.adsabs.harvard.edu/abs/2017IJMPD..2641015N/abstract
            https://ui.adsabs.harvard.edu/abs/2022ApJ...933...83S/abstract

        Parameters
        ----------
        telescope : a telescope object
        pyLIMA_parameters : a pyLIMA_parameters object

        Returns
        -------
        astro_shifts : array, [shifts_N,shifts_E] are the projected astrometric
        microlensing shifts projected in the North, East
        """
        if telescope.astrometry is not None:

            (source1_trajectory_x, source1_trajectory_y,
             source2_trajectory_x, source2_trajectory_y,
             dseparation, dalpha) = self.sources_trajectory(
                telescope, pyLIMA_parameters,
                data_type='astrometry')

            # Blended centroid shifts....
            # magnification = self.model_magnification(telescope, pyLIMA_parameters)
            # try:
            #    g_blend = f_blending/f_source
            #    shifts = astrometric_shifts.PSPL_shifts_with_blend(
            #    source_trajectory_x, source_trajectory_y, pyLIMA_parameters.theta_E,
            #    g_blend)
            #    angle = np.arctan2(source_trajectory_y,source_trajectory_x)
            #    shifts = np.array([shifts*np.cos(angle), shifts*np.sin(angle)])

            # except:

            shifts = astrometric_shifts.PSPL_shifts_no_blend(source1_trajectory_x,
                                                             source1_trajectory_y,
                                                             pyLIMA_parameters[
                                                                 'theta_E'])

            delta_ra, delta_dec = astrometric_positions.xy_shifts_to_NE_shifts(shifts,
                                                                               pyLIMA_parameters['piEN'],
                                                                               pyLIMA_parameters['piEE'])

            position_ra, position_dec = \
                astrometric_positions.source_astrometric_positions(
                    telescope, pyLIMA_parameters,
                    shifts=(delta_ra, delta_dec),
                    time_ref=self.parallax_model[
                        1])

            astro_shifts = np.array([position_ra, position_dec])

        else:

            astro_shifts = None

        return astro_shifts

    def model_magnification(self, telescope, pyLIMA_parameters,
                            return_impact_parameter=False):

        if telescope.lightcurve is not None:

            (source1_trajectory_x, source1_trajectory_y,
             source2_trajectory_x, source2_trajectory_y,
             dseparation, dalpha) = self.sources_trajectory(
                telescope, pyLIMA_parameters,
                data_type='photometry')

            source1_magnification = magnification_PSPL.magnification_PSPL(
                source1_trajectory_x,source1_trajectory_y,return_impact_parameter)

            if source2_trajectory_x is not None:

                source2_magnification = magnification_PSPL.magnification_PSPL(
                    source2_trajectory_x,
                    source2_trajectory_y,
                    return_impact_parameter)

                blend_magnification_factor = pyLIMA_parameters['q_flux_' +
                                                               telescope.filter]
                effective_magnification = (
                        source1_magnification +
                        source2_magnification *
                        blend_magnification_factor)

                magnification = effective_magnification
                #breakpoint()
            else:

                magnification = source1_magnification

        else:

            magnification = None

        return magnification

    def model_magnification_Jacobian(self, telescope, pyLIMA_parameters):
        """
        [d(At)/dt0,dA(t)/du0,dA(t)/dtE]
        """

        if self.Jacobian_flag == 'Analytical':

            magnification_jacobian, amplification = \
                magnification_Jacobian.magnification_PSPL_Jacobian(
                    self, telescope, pyLIMA_parameters)

        else:

            magnification_jacobian = \
                magnification_Jacobian.magnification_numerical_Jacobian(
                    self, telescope,
                    pyLIMA_parameters)
            amplification = self.model_magnification(telescope, pyLIMA_parameters,
                                                     return_impact_parameter=False)

        return magnification_jacobian, amplification

    def astrometry_Jacobian(self, telescope, pyLIMA_parameters):

        pass
