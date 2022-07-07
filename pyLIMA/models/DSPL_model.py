import numpy as np

from pyLIMA.models.ML_model import MLmodel
import copy

from pyLIMA.magnification import magnification_PSPL

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
        model_dictionary = {'t0': 0, 'u0': 1, 'delta_t0': 2, 'delta_u0': 3, 'tE': 4}

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

            source1_trajectory_x, source1_trajectory_y, source2_trajectory_x, source2_trajectory_y = \
                self.sources_trajectories(telescope, pyLIMA_parameters)

            source1_magnification = magnification_PSPL.magnification_PSPL(source1_trajectory_x, source1_trajectory_y)

            source2_magnification = magnification_PSPL.magnification_PSPL(source2_trajectory_x, source2_trajectory_y)

            blend_magnification_factor = getattr(pyLIMA_parameters, 'q_flux_' + telescope.filter)

            effective_magnification = (source1_magnification + source2_magnification * blend_magnification_factor) / (1 + blend_magnification_factor)

            magnification = effective_magnification

        else:

            magnification = None

        return magnification

    def sources_trajectories(self, telescope, pyLIMA_parameters):

        source1_trajectory_x, source1_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters,
                                                                               data_type='photometry')

        pyLIMA_parameters_2 = copy.deepcopy(pyLIMA_parameters)
        setattr(pyLIMA_parameters_2, 't0',  pyLIMA_parameters.t0+pyLIMA_parameters.delta_t0)
        setattr(pyLIMA_parameters_2, 'u0', pyLIMA_parameters.u0 + pyLIMA_parameters.delta_u0)

        source2_trajectory_x, source2_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters_2,
                                                                               data_type='photometry')

        return source1_trajectory_x, source1_trajectory_y, source2_trajectory_x, source2_trajectory_y
