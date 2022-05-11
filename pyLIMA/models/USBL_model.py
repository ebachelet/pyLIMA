import numpy as np

from pyLIMA.models.ML_model import MLmodel
from pyLIMA.astrometry import astrometric_shifts
from pyLIMA.magnification import magnification_VBB

class USBLmodel(MLmodel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: USBL
        :rtype: string
        """
        return 'USBL'

    def paczynski_model_parameters(self):
        """ Define the USBL standard parameters, [to,uo,tE,rho, s,q,alpha]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2, 'rho': 3, 'logs': 4, 'logq': 5, 'alpha': 6}

        self.Jacobian_flag = 'No way'

        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters, return_impact_parameter=None):
        """ The magnification associated to a USBL model.
            From Bozza  2010 : http://adsabs.harvard.edu/abs/2010MNRAS.408.2188B

            :param object telescope: a telescope object. More details in telescope module.
            :param object pyLIMA_parameters: a namedtuple which contain the parameters
            :return: magnification,
            :rtype: array_like,
        """

        if telescope.astrometry is not None:

            source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters, data_type='astrometry')

            import pyLIMA.magnification.magnification_VBB
            pyLIMA.magnification.magnification_VBB.VBB.astrometry = True


            shifts = astrometric_shifts.PSPL_shifts(source_trajectory_x, source_trajectory_y, pyLIMA_parameters.theta_E)

            angle = np.arctan2(pyLIMA_parameters.piEE, pyLIMA_parameters.piEN)

            Deltay = shifts[0] * np.cos(angle) - np.sin(angle) * shifts[1]
            Deltax = shifts[0] * np.sin(angle) + np.cos(angle) * shifts[1]

            shifts = [Deltax,Deltay]

        else:
            shifts = None

        if telescope.lightcurve_flux is not None:

            source_trajectoire = self.source_trajectory(telescope, pyLIMA_parameters, data_type='photometry')

            separation = source_trajectoire[2] + 10 ** pyLIMA_parameters.logs

            magnification_USBL = \
               magnification_VBB.magnification_USBL(separation, 10 ** pyLIMA_parameters.logq,
                                                                          source_trajectoire[0], source_trajectoire[1],
                                                                          pyLIMA_parameters.rho)

        else:

            magnification_USBL = None

        magnification_model = {'magnification':magnification_USBL,'astrometry':shifts}

        return magnification_model

