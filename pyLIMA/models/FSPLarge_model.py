from pyLIMA.models.FSPL_model import FSPLmodel
from pyLIMA.magnification import magnification_VBB


class FSPLargemodel(FSPLmodel):
    @property
    def model_magnification(self, telescope, pyLIMA_parameters, return_impact_parameter=False):
        """ The magnification associated to a FSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: magnification, impact_parameter
        :rtype: array_like,array_like
        """

        source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope,
                                                                                                    pyLIMA_parameters,
                                                                                                    data_type='astrometry')
                                                                             )
        rho = pyLIMA_parameters.rho

        try:

            linear_limb_darkening = telescope.gamma * 6 / (4 + 2 * telescope.gamma + telescope.sigma)
            sqrt_limb_darkening = telescope.sigma * 5 / (4 + 2 * telescope.gamma + telescope.sigma)

            return magnification_VBB.magnification_FSPL(source_trajectory_x, source_trajectory_y,
                                                                             rho, linear_limb_darkening,
                                                                             sqrt_limb_darkening)
        except:

            linear_limb_darkening = telescope.gamma * 3 / (2 + telescope.gamma)

            return magnification_VBB.magnification_FSPL(source_trajectory_x, source_trajectory_y,
                                                                             rho, linear_limb_darkening)
