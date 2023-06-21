from pyLIMA.magnification import magnification_VBB
from pyLIMA.models.USBL_model import USBLmodel


class PSBLmodel(USBLmodel):
    def model_type(self):

        return 'PSBL'

    def paczynski_model_parameters(self):
        """
        [t0,u0,tE,s,q,alpha]
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'tE': 2, 'separation': 3, 'mass_ratio': 4,
                            'alpha': 5}

        self.Jacobian_flag = 'No way'

        return model_dictionary

    def model_astrometry(self, telescope, pyLIMA_parameters):

        pass

    def model_magnification(self, telescope, pyLIMA_parameters,
                            return_impact_parameter=None):
        """
        The magnification associated to a PSBL model. No finite source effect ==>
        very fast
        """
        if telescope.lightcurve_flux is not None:

            source_trajectoire = self.source_trajectory(telescope, pyLIMA_parameters,
                                                        data_type='photometry')

            separation = source_trajectoire[2] + pyLIMA_parameters.separation

            magnification_PSBL = \
                magnification_VBB.magnification_PSBL(separation,
                                                     pyLIMA_parameters.mass_ratio,
                                                     source_trajectoire[0],
                                                     source_trajectoire[1])

        else:

            magnification_PSBL = None

        return magnification_PSBL
