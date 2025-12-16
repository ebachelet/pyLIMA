import jax.numpy as jnp
from pyLIMA.magnification import magnification_JAX as magnification_JAX
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
        if telescope.lightcurve is not None:

            (source1_trajectory_x, source1_trajectory_y,
             source2_trajectory_x, source2_trajectory_y,
             dseparation, dalpha) = self.sources_trajectory(
                telescope, pyLIMA_parameters,
                data_type='photometry')

            # Use microlux-based PSBL via JAX wrapper (no VBB fallback)
            source1_magnification = magnification_JAX.magnification_PSBL_microlux(
                float(pyLIMA_parameters['t0']),
                float(pyLIMA_parameters['u0']),
                float(pyLIMA_parameters['tE']),
                float(pyLIMA_parameters['mass_ratio']),
                float(pyLIMA_parameters['separation']),
                float(pyLIMA_parameters.get('alpha', 0.0)),
                telescope.lightcurve['time'].value,
                return_info=False
            )

            if source2_trajectory_x is not None:
                # Use microlux for second source as well
                source2_magnification = magnification_JAX.magnification_PSBL_microlux(
                    float(pyLIMA_parameters['t0']),
                    float(pyLIMA_parameters['u0']),
                    float(pyLIMA_parameters['tE']),
                    float(pyLIMA_parameters['mass_ratio']),
                    float(pyLIMA_parameters['separation']),
                    float(pyLIMA_parameters.get('alpha', 0.0)),
                    telescope.lightcurve['time'].value,
                    return_info=False
                )

                blend_magnification_factor = pyLIMA_parameters['q_flux_' +
                                                               telescope.filter]
                effective_magnification = (
                        source1_magnification +
                        source2_magnification *
                        blend_magnification_factor)

                magnification_PSBL = effective_magnification

            else:

                magnification_PSBL = source1_magnification

        else:

            magnification_PSBL = None

        return magnification_PSBL
