import numpy as np
from pyLIMA.magnification import magnification_FSPL
from pyLIMA.models.DSPL_model import DSPLmodel


class DFSPLmodel(DSPLmodel):
    def model_type(self):

        return 'DFSPL'

    def paczynski_model_parameters(self):
        """
        [t0,u0,delta_t0,delta_u0,tE,rho_1,rho_2,q_flux_i]
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'delta_t0': 2, 'delta_u0': 3, 'tE': 4,
                            'rho_1': 5, 'rho_2': 6}

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

            source1_magnification = magnification_FSPL.magnification_FSPL_Yoo(
                source1_trajectory_x, source1_trajectory_y,
                pyLIMA_parameters.rho_1, telescope.ld_gamma1,
                return_impact_parameter)

            source2_magnification = magnification_FSPL.magnification_FSPL_Yoo(
                source2_trajectory_x,
                source2_trajectory_y,
                pyLIMA_parameters.rho_2, telescope.ld_gamma2,
                return_impact_parameter)

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
