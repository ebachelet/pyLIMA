from pyLIMA.magnification import magnification_FSPL, magnification_Jacobian
from pyLIMA.models.ML_model import MLmodel


class FSPLmodel(MLmodel):

    def __init__(self, event, parallax=['None', 0.0], double_source=['None',0],
                 orbital_motion=['None', 0.0], origin=['center_of_mass', [0, 0]],
                 blend_flux_parameter='ftotal', fancy_parameters=None):

        super().__init__(event, parallax=parallax, double_source=double_source,
                         orbital_motion=orbital_motion, origin=origin,
                         blend_flux_parameter=blend_flux_parameter,
                         fancy_parameters=fancy_parameters)

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

        pass

    def model_magnification(self, telescope, pyLIMA_parameters,
                            return_impact_parameter=False):
        """
        The FSPL magnification, see  http://adsabs.harvard.edu/abs/2004ApJ...603..139Y
        """
        if telescope.lightcurve is not None:

            rho = pyLIMA_parameters['rho']
            gamma = telescope.ld_gamma

            (source1_trajectory_x, source1_trajectory_y,
             source2_trajectory_x, source2_trajectory_y,
             dseparation, dalpha) = self.sources_trajectory(
                telescope, pyLIMA_parameters,
                data_type='photometry')


            source1_magnification = magnification_FSPL.magnification_FSPL_Yoo(
                source1_trajectory_x, source1_trajectory_y,rho,
                gamma,return_impact_parameter)

            if source2_trajectory_x is not None:

                rho2 = pyLIMA_parameters['rho_2']


                #Need to change to gamma2

                source2_magnification = magnification_FSPL.magnification_FSPL_Yoo(
                    source2_trajectory_x, source2_trajectory_y, rho2, gamma,
                    return_impact_parameter)

                blend_magnification_factor = pyLIMA_parameters['q_flux_' +
                                                               telescope.filter]
                effective_magnification = (
                        source1_magnification +
                        source2_magnification *
                        blend_magnification_factor)

                magnification = effective_magnification

            else:

                magnification = source1_magnification

        else:

            magnification = None

        return magnification

    def model_magnification_Jacobian(self, telescope, pyLIMA_parameters):
        """
        [dA(t)/dt0,dA(t)/du0,dA(t)/dtE,dA(t)/drho]
        """

        if self.Jacobian_flag == 'Analytical':

            magnification_jacobian = magnification_Jacobian.magnification_FSPL_Jacobian(
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
