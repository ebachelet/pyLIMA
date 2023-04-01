import numpy as np

from pyLIMA.models.ML_model import MLmodel
import copy

from pyLIMA.magnification import magnification_FSPL

class DSPLmodel(MLmodel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: DSPL
        :rtype: string
        """
        return 'DSPL'

    def paczynski_model_parameters(self):
        """ Define the PSPL standard parameters, [t0,u0,tE]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'t0': 0, 'u0': 1,'delta_t0': 0, 'delta_u0': 1, 'tE': 4, 'rho_1':5,'rho_2':6}

        filters = [telescope.filter for telescope in self.event.telescopes]

        unique_filters = np.unique(filters)

        for filter in unique_filters:
            model_dictionary['q_flux_' + filter] = len(model_dictionary)

        self.Jacobian_flag = 'No way'

        return model_dictionary


    def model_astrometry(self, telescope, pyLIMA_parameters):
        """ The astrometric shifts associated to a PSPL model. More details in microlmagnification module.

           :param object telescope: a telescope object. More details in telescope module.
           :param object pyLIMA_parameters: a namedtuple which contain the parameters


           :return: astro_shifts
           :rtype: array_like
        """
        pass

    def model_magnification(self, telescope, pyLIMA_parameters, return_impact_parameter=False):
        """ The magnification associated to a PSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :param boolean return_impact_parameter: if the impact parameter is needed or not

        :return: magnification
        :rtype: array_like
        """

        if telescope.lightcurve_flux is not None:
            source1_trajectory_x, source1_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters,
                                                                                   data_type='photometry')

            pyLIMA_parameters_2 = copy.deepcopy(pyLIMA_parameters)
            pyLIMA_parameters_2['t0'] += pyLIMA_parameters_2['delta_t0']
            pyLIMA_parameters_2['u0'] += pyLIMA_parameters_2['delta_u0']

            source2_trajectory_x, source2_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters_2,
                                                                                   data_type='photometry')


            source1_magnification = magnification_FSPL.magnification_FSPL_Yoo(source1_trajectory_x, source1_trajectory_y,
                                                                              pyLIMA_parameters.rho_1,telescope.ld_gamma1,self.yoo_table,
                                                                              return_impact_parameter)

            source2_magnification = magnification_FSPL.magnification_FSPL_Yoo(source2_trajectory_x,
                                                                              source2_trajectory_y,
                                                                              pyLIMA_parameters.rho_2, telescope.ld_gamma2,
                                                                              self.yoo_table,
                                                                              return_impact_parameter)

            blend_magnification_factor = getattr(pyLIMA_parameters, 'q_flux_' + telescope.filter)

            effective_magnification = (source1_magnification + source2_magnification * blend_magnification_factor) / (1 + blend_magnification_factor)

            magnification = effective_magnification

        else:

            magnification = None

        return magnification

