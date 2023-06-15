import numpy as np
from pyLIMA.magnification import magnification_PSPL
from pyLIMA.models.ML_model import MLmodel


class DSPLmodel(MLmodel):

    def model_type(self):

        return 'DSPL'

    def paczynski_model_parameters(self):
        """
        [t0,u0,delta_t0,delta_u0,q_flux_i]
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'delta_t0': 2, 'delta_u0': 3, 'tE': 4}

        filters = [telescope.filter for telescope in self.event.telescopes]

        unique_filters = np.unique(filters)

        for filter in unique_filters:
            model_dictionary['q_flux_' + filter] = len(model_dictionary)

        self.Jacobian_flag = 'No way'

        return model_dictionary

    def model_astrometry(self, telescope, pyLIMA_parameters):

        pass

    def model_magnification(self, telescope, pyLIMA_parameters,
                            return_impact_parameter=False):
        """
        The weighted (by q_flux) sum of the two source magnifications
        """
        if telescope.lightcurve_flux is not None:

            source1_trajectory_x, source1_trajectory_y, source2_trajectory_x, \
                source2_trajectory_y = \
                self.sources_trajectory(telescope, pyLIMA_parameters)

            source1_magnification = magnification_PSPL.magnification_PSPL(
                source1_trajectory_x, source1_trajectory_y)

            source2_magnification = magnification_PSPL.magnification_PSPL(
                source2_trajectory_x, source2_trajectory_y)

            blend_magnification_factor = getattr(pyLIMA_parameters,
                                                 'q_flux_' + telescope.filter)

            effective_magnification = (
                                              source1_magnification +
                                              source2_magnification *
                                              blend_magnification_factor) / (
                                              1 + blend_magnification_factor)

            magnification = effective_magnification

        else:

            magnification = None

        return magnification

    def sources_trajectory(self, telescope, pyLIMA_parameters):
        """
        Compute the trajectories of the two sources

        Parameters
        ----------
        telescope :  a telescope object
        pyLIMA_parameters : a pyLIMA_parameters objecr

        Returns
        -------
        source1_trajectory_x : the x coordinates of source 1
        source1_trajectory_y : the y coordinates of source 1
        source2_trajectory_x : the x coordinates of source 2
        source2_trajectory_y : the y coordinates of source 2
        """
        source1_trajectory_x, source1_trajectory_y, _, _ = self.source_trajectory(
            telescope, pyLIMA_parameters,
            data_type='photometry')

        parameters = [getattr(pyLIMA_parameters, i) for i in pyLIMA_parameters._fields]

        pyLIMA_parameters_2 = self.compute_pyLIMA_parameters(parameters)

        pyLIMA_parameters_2.t0 = pyLIMA_parameters_2.t0 + pyLIMA_parameters_2.delta_t0
        pyLIMA_parameters_2.u0 = pyLIMA_parameters_2.u0 + pyLIMA_parameters_2.delta_u0

        source2_trajectory_x, source2_trajectory_y, _, _ = self.source_trajectory(
            telescope, pyLIMA_parameters_2,
            data_type='photometry')

        return source1_trajectory_x, source1_trajectory_y, source2_trajectory_x, \
            source2_trajectory_y
