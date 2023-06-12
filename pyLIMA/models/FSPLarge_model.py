from pyLIMA.magnification import magnification_VBB
from pyLIMA.models.FSPL_model import FSPLmodel


class FSPLargemodel(FSPLmodel):

    def model_magnification(self, telescope, pyLIMA_parameters,
                            return_impact_parameter=False):
        """
        The finite source magnification of large source (i.e. no Yoo approximation),
        using VBB instead. Slower obviously...
        See https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.2188B/abstract
            https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5157B/abstract
        """

        source_trajectory_x, source_trajectory_y, _, _ = self.source_trajectory(
            telescope, pyLIMA_parameters,
            data_type='photometry')

        rho = pyLIMA_parameters.rho
        linear_limb_darkening = telescope.ld_a1

        try:

            sqrt_limb_darkening = telescope.ld_a2

            return magnification_VBB.magnification_FSPL(source_trajectory_x,
                                                        source_trajectory_y,
                                                        rho, linear_limb_darkening,
                                                        sqrt_limb_darkening)
        except ValueError:

            return magnification_VBB.magnification_FSPL(source_trajectory_x,
                                                        source_trajectory_y,
                                                        rho, linear_limb_darkening)
