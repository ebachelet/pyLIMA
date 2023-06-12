from pyLIMA.models.ML_model import MLmodel
from pyLIMA.magnification import magnification_FSPL, magnification_Jacobian
class FSPLmodel(MLmodel):

    def __init__(self, event, parallax=['None', 0.0], xallarap=['None'],
                 orbital_motion=['None', 0.0], origin=['center_of_mass', [0,0]], blend_flux_parameter='fblend',fancy_parameters={}):

        super().__init__(event, parallax=parallax, xallarap=xallarap,
                         orbital_motion=orbital_motion, origin=origin, blend_flux_parameter=blend_flux_parameter,
                         fancy_parameters=fancy_parameters)

    @property
    def model_type(self):

        return 'FSPL'

    def paczynski_model_parameters(self):
        """
        [to,u0,tE,rho]
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'tE': 2, 'rho': 3}
        self.Jacobian_flag = 'Analytical'

        return model_dictionary

    def model_astrometry(self, telescope, pyLIMA_parameters):
        ##########THIS IS PSPL ASTROMETRY, HERE FOR TESTING....################
        #if telescope.astrometry is not None:

        #    source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters, data_type='astrometry')

        #    # Blended centroid shifts....
        #    #magnification = self.model_magnification(telescope, pyLIMA_parameters)
        #    #try:
        #    #    g_blend = f_blending/f_source
        #    #    shifts = astrometric_shifts.PSPL_shifts_with_blend(source_trajectory_x, source_trajectory_y, pyLIMA_parameters.theta_E,g_blend)
        #    #    angle = np.arctan2(source_trajectory_y,source_trajectory_x)
        #    #    shifts = np.array([shifts*np.cos(angle), shifts*np.sin(angle)])

        #    #except:

        #    shifts = astrometric_shifts.PSPL_shifts_no_blend(source_trajectory_x, source_trajectory_y,
        #                                                         pyLIMA_parameters.theta_E)

        #    delta_ra, delta_dec = astrometric_positions.xy_shifts_to_NE_shifts(shifts,pyLIMA_parameters.piEN,
        #                                                                        pyLIMA_parameters.piEE)

        #    position_ra, position_dec = astrometric_positions.source_astrometric_position(telescope, pyLIMA_parameters,
        #                                                                                  shifts=(delta_ra, delta_dec),
        #                                                                                  time_ref=self.parallax_model[
        #                                                                                      1])

        #    astro_shifts = np.array([position_ra, position_dec])

        #else:

        #    astro_shifts = None

        #return astro_shifts

        pass

    def model_magnification(self, telescope, pyLIMA_parameters, return_impact_parameter=False):
        """
        The FSPL magnification, see  http://adsabs.harvard.edu/abs/2004ApJ...603..139Y
        """
        if telescope.lightcurve_flux is not None:
            source_trajectory_x, source_trajectory_y, _, _ = self.source_trajectory(telescope, pyLIMA_parameters,
                                                                                 data_type='photometry')
            rho = pyLIMA_parameters.rho
            gamma = telescope.ld_gamma

            magnification = magnification_FSPL.magnification_FSPL_Yoo(source_trajectory_x, source_trajectory_y,
                                                                              rho,
                                                                              gamma,
                                                                              return_impact_parameter)
        else:

            magnification = None

        return magnification

    def model_magnification_Jacobian(self, telescope, pyLIMA_parameters):
        """
        [dA(t)/dt0,dA(t)/du0,dA(t)/dtE,dA(t)/drho]
        """
        if self.Jacobian_flag == 'Analytical':

            magnification_jacobian= magnification_Jacobian.magnification_FSPL_Jacobian(self, telescope,pyLIMA_parameters)

        else:

            magnification_jacobian = magnification_Jacobian.magnification_numerical_Jacobian(self, telescope,
                                                                                             pyLIMA_parameters)

        amplification = self.model_magnification(telescope, pyLIMA_parameters, return_impact_parameter=False)

        return magnification_jacobian, amplification

    def astrometry_Jacobian(self, telescope, pyLIMA_parameters):

        pass