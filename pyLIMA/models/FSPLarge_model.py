from pyLIMA.magnification import magnification_VBB
from pyLIMA.models.FSPL_model import FSPLmodel


class FSPLargemodel(FSPLmodel):

    def model_type(self):

        return 'FSPLarge'

    def paczynski_model_parameters(self):
        """
        [to,u0,tE,rho]
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'tE': 2, 'rho': 3}
        self.Jacobian_flag = 'Numerical'

        return model_dictionary
    def model_magnification(self, telescope, pyLIMA_parameters,
                            return_impact_parameter=False):
        """
        The finite source magnification of large source (i.e. no Yoo approximation),
        using VBB instead. Slower obviously...
        See https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.2188B/abstract
            https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5157B/abstract
        """


        rho = pyLIMA_parameters['rho']
        linear_limb_darkening = telescope.ld_a1
        sqrt_limb_darkening = telescope.ld_a2

        (source1_trajectory_x, source1_trajectory_y,
         source2_trajectory_x, source2_trajectory_y,
         dseparation, dalpha) = self.sources_trajectory(
            telescope, pyLIMA_parameters,
            data_type='photometry')

        if (sqrt_limb_darkening is not None) & (sqrt_limb_darkening > 0):

            source1_magnification = magnification_VBB.magnification_FSPL(source1_trajectory_x,
                                                        source1_trajectory_y,
                                                        rho, linear_limb_darkening,
                                                        sqrt_limb_darkening)
        else:

            source1_magnification = magnification_VBB.magnification_FSPL(source1_trajectory_x,
                                                        source1_trajectory_y,
                                                        rho, linear_limb_darkening)

        if source2_trajectory_x is not None:

            rho_2 = pyLIMA_parameters['rho_2']

            # Need to change to gamma2

            if (sqrt_limb_darkening is not None) & (sqrt_limb_darkening > 0):

                source2_magnification = magnification_VBB.magnification_FSPL(
                    source2_trajectory_x,
                    source2_trajectory_y,
                    rho_2, linear_limb_darkening,
                    sqrt_limb_darkening)
            else:

                source1_magnification = magnification_VBB.magnification_FSPL(
                    source1_trajectory_x,
                    source1_trajectory_y,
                    rho_2, linear_limb_darkening)

            blend_magnification_factor = pyLIMA_parameters['q_flux_' + telescope.filter]
            effective_magnification = (
                    source1_magnification +
                    source2_magnification *
                    blend_magnification_factor)

            magnification = effective_magnification

        else:

            magnification = source1_magnification

        return magnification
