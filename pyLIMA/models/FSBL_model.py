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

        if telescope.lightcurve_flux is not None:

            source_trajectoire = self.source_trajectory(telescope, pyLIMA_parameters,
                                                        data_type='photometry')

            linear_limb_darkening = telescope.ld_a1

            separation = source_trajectoire[2] + pyLIMA_parameters.separation

            magnification_FSBL = \
                magnification_VBB.magnification_FSBL(separation,
                                                     pyLIMA_parameters.mass_ratio,
                                                     source_trajectoire[0],
                                                     source_trajectoire[1],
                                                     pyLIMA_parameters.rho,
                                                     linear_limb_darkening)

        else:

            magnification_FSBL = None

        if return_impact_parameter:

            return magnification_FSBL, None

        else:

            return magnification_FSBL
