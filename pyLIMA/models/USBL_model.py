from pyLIMA.models.ML_model import MLmodel
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
        """ Define the USBL standard parameters, [to,uo,tE,rho, logs,logq,alpha]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'tE': 2, 'rho': 3, 'separation': 4, 'mass_ratio': 5, 'alpha': 6}

        self.Jacobian_flag = 'No way'

        return model_dictionary

    def model_astrometry(self, telescope, pyLIMA_parameters):

        pass

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

            separation = source_trajectoire[2] + pyLIMA_parameters.separation
            print(separation,pyLIMA_parameters.mass_ratio)
            magnification_USBL = \
               magnification_VBB.magnification_USBL(separation, pyLIMA_parameters.mass_ratio,
                                                                          source_trajectoire[0], source_trajectoire[1],
                                                                          pyLIMA_parameters.rho)

        else:

            magnification_USBL = None


        return magnification_USBL

