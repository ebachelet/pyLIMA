from pyLIMA.magnification import magnification_VBB
from pyLIMA.models.USBL_model import USBLmodel


class FSBLmodel(USBLmodel):
    def model_type(self):

        return 'FSBL'

    def model_magnification(self, telescope, pyLIMA_parameters,
                            return_impact_parameter=None):
        """
        The magnification associated to a FSBL model, that includes limb-darkening.
        See
        https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.2188B/abstract
        https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5157B/abstract
        """

        if telescope.lightcurve is not None:

            linear_limb_darkening = telescope.ld_a1

            (source1_trajectory_x, source1_trajectory_y,
             source2_trajectory_x, source2_trajectory_y,
             dseparation, dalpha) = self.sources_trajectory(
                telescope, pyLIMA_parameters,
                data_type='photometry')

            separation = dseparation + pyLIMA_parameters['separation']

            source1_magnification = magnification_VBB.magnification_FSBL(separation,
                                                     pyLIMA_parameters['mass_ratio'],
                                                     source1_trajectory_x,
                                                     source1_trajectory_y,
                                                     pyLIMA_parameters['rho'],
                                                     linear_limb_darkening)

            if source2_trajectory_x is not None:
                # need to update limb_darkening

                source2_magnification = magnification_VBB.magnification_FSBL(separation,
                                                     pyLIMA_parameters['mass_ratio'],
                                                     source1_trajectory_x,
                                                     source1_trajectory_y,
                                                     pyLIMA_parameters['rho_2'],
                                                     linear_limb_darkening)

                blend_magnification_factor = pyLIMA_parameters['q_flux_' +
                                                               telescope.filter]
                effective_magnification = (
                        source1_magnification +
                        source2_magnification *
                        blend_magnification_factor)

                magnification_FSBL = effective_magnification

            else:

                magnification_FSBL = source1_magnification


        else:

            magnification_FSBL = None

        if return_impact_parameter:

            return magnification_FSBL, None

        else:

            return magnification_FSBL
