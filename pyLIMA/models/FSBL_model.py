from pyLIMA.models.USBL_model import USBLmodel
from pyLIMA.magnification import magnification_VBB

class FSBLmodel(USBLmodel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: FSBL
        :rtype: string
        """
        return 'FSBL'

    def model_magnification(self, telescope, pyLIMA_parameters, return_impact_parameter=None):
        """ The magnification associated to a USBL model.
            From Bozza  2010 : http://adsabs.harvard.edu/abs/2010MNRAS.408.2188B

            :param object telescope: a telescope object. More details in telescope module.
            :param object pyLIMA_parameters: a namedtuple which contain the parameters
            :return: magnification,
            :rtype: array_like,
        """

        if telescope.lightcurve_flux is not None:

            source_trajectoire = self.source_trajectory(telescope, pyLIMA_parameters, data_type='photometry')

            linear_limb_darkening = telescope.gamma * 3 / (2 + telescope.gamma)

            separation = source_trajectoire[2] + 10 ** pyLIMA_parameters.logs

            magnification_FSBL = \
                magnification_VBB.magnification_FSBL(separation, 10 ** pyLIMA_parameters.logq,
                                                   source_trajectoire[0], source_trajectoire[1],
                                                   pyLIMA_parameters.rho, linear_limb_darkening)


        else:

            magnification_FSBL = None

        return magnification_FSBL