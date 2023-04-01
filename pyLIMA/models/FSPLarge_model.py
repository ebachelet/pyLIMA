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

        source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters,
                                                                             data_type='astrometry')

        rho = pyLIMA_parameters.rho
        linear_limb_darkening = telescope.ld_a1

        try:

            sqrt_limb_darkening = telescope.ld_a2

            return magnification_VBB.magnification_FSPL(source_trajectory_x, source_trajectory_y,
                                                                             rho, linear_limb_darkening,
                                                                             sqrt_limb_darkening)
        except:


            return magnification_VBB.magnification_FSPL(source_trajectory_x, source_trajectory_y,
                                                                             rho, linear_limb_darkening)
