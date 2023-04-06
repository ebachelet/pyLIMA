import numpy as np

from pyLIMA.models.ML_model import MLmodel
from pyLIMA.magnification import magnification_VBB

class USBLmodel(MLmodel):

    def __init__(self, event, parallax=['None', 0.0], xallarap=['None'],
                 orbital_motion=['None', 0.0], blend_flux_parameter='fblend',
                 origin = ['center_of_mass', [0,0]], fancy_parameters={}):
        """The fit class has to be intialized with an event object."""

        super().__init__(event, parallax=parallax, xallarap=xallarap, orbital_motion=orbital_motion,
                         blend_flux_parameter=blend_flux_parameter, origin=origin, fancy_parameters=fancy_parameters)


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

        self.Jacobian_flag = 'Numerical'

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

            #self.u0_t0_from_uc_tc(pyLIMA_parameters)

            source_trajectoire = self.source_trajectory(telescope, pyLIMA_parameters, data_type='photometry')

            separation = source_trajectoire[2] + pyLIMA_parameters.separation
            magnification_USBL = \
               magnification_VBB.magnification_USBL(separation, pyLIMA_parameters.mass_ratio,
                                                                          source_trajectoire[0], source_trajectoire[1],
                                                                          pyLIMA_parameters.rho)
        else:

            magnification_USBL = None

        if return_impact_parameter:

            return magnification_USBL,None
        else:
            return magnification_USBL

#    def find_origin(self, pyLIMA_parameters):

#        if self.origin == 'center_of_mass':

#            self.x_center = 0

#        if self.origin == 'primary':

#            center_of_mass = -(pyLIMA_parameters.separation * pyLIMA_parameters.mass_ratio) / (
#                        1 + pyLIMA_parameters.mass_ratio)

#            self.x_center = center_of_mass

#        if self.origin == 'companion':

#            center_of_mass = -(pyLIMA_parameters.separation * pyLIMA_parameters.mass_ratio) / (
#                    1 + pyLIMA_parameters.mass_ratio)

#            self.x_center = center_of_mass + pyLIMA_parameters.separation

#    def u0_t0_from_uc_tc(self,pyLIMA_parameters):

#        self.find_origin(pyLIMA_parameters)

#        new_origin_x = self.x_center
#        new_origin_y = self.y_center

#        t0 = pyLIMA_parameters.t0 - pyLIMA_parameters.tE * (new_origin_x * np.cos(pyLIMA_parameters.alpha) +
#                                                            new_origin_y * np.sin(pyLIMA_parameters.alpha))

#        u0 = pyLIMA_parameters.u0 - (new_origin_x * np.sin(pyLIMA_parameters.alpha) -
#                                     new_origin_y * np.cos(pyLIMA_parameters.alpha))

#        setattr(pyLIMA_parameters, 't0', t0)
#        setattr(pyLIMA_parameters, 'u0', u0)

#    def uc_tc_from_u0_t0(self, pyLIMA_parameters):#

#        self.find_origin(pyLIMA_parameters)

#        new_origin_x = self.x_center
#        new_origin_y = self.y_center

#        tc = pyLIMA_parameters.t0 + pyLIMA_parameters.tE * (new_origin_x * np.cos(pyLIMA_parameters.alpha) +
#                                              new_origin_y * np.sin(pyLIMA_parameters.alpha))

#        uc = pyLIMA_parameters.u0 + (new_origin_x * np.sin(pyLIMA_parameters.alpha) -
#                                     new_origin_y * np.cos(pyLIMA_parameters.alpha))

#        setattr(pyLIMA_parameters, 't0', tc)
#        setattr(pyLIMA_parameters, 'u0', uc)
