import jax.numpy as jnp
from pyLIMA.magnification import magnification_JAX as magnification_JAX
from pyLIMA.models.USBL_model import USBLmodel


class FSBLmodel(USBLmodel):
    def model_type(self):

        return 'FSBL'

    def model_magnification(self, telescope, pyLIMA_parameters,
                            return_impact_parameter=None):
        """
        The magnification associated to a FSBL model, that includes limb-darkening.
        See
        https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.2188B/abstract
        https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5157B/abstract
        """

        if telescope.lightcurve is not None:

            linear_limb_darkening = getattr(telescope, 'ld_a1', None)

            (source1_trajectory_x, source1_trajectory_y,
             source2_trajectory_x, source2_trajectory_y,
             dseparation, dalpha) = self.sources_trajectory(
                telescope, pyLIMA_parameters,
                data_type='photometry')

            # Use microlux-based FSBL via JAX wrapper (no VBB fallback)
            source1_magnification = magnification_JAX.magnification_FSBL_microlux(
                float(pyLIMA_parameters['t0']),
                float(pyLIMA_parameters['u0']),
                float(pyLIMA_parameters['tE']),
                float(pyLIMA_parameters['rho']),
                float(pyLIMA_parameters['mass_ratio']),
                float(pyLIMA_parameters['separation']),
                float(pyLIMA_parameters.get('alpha', 0.0)),
                telescope.lightcurve['time'].value,
                limb_darkening_coeff=linear_limb_darkening if linear_limb_darkening is not None else None
            )

            if source2_trajectory_x is not None:
                # Use microlux for second source as well
                source2_magnification = magnification_JAX.magnification_FSBL_microlux(
                    float(pyLIMA_parameters['t0']),
                    float(pyLIMA_parameters['u0']),
                    float(pyLIMA_parameters['tE']),
                    float(pyLIMA_parameters.get('rho_2', pyLIMA_parameters['rho'])),
                    float(pyLIMA_parameters['mass_ratio']),
                    float(pyLIMA_parameters['separation']),
                    float(pyLIMA_parameters.get('alpha', 0.0)),
                    telescope.lightcurve['time'].value,
                    limb_darkening_coeff=linear_limb_darkening if linear_limb_darkening is not None else None
                )

                blend_magnification_factor = pyLIMA_parameters['q_flux_' +
                                                               telescope.filter]
                effective_magnification = (
                        source1_magnification +
                        source2_magnification *
                        blend_magnification_factor)

                magnification_FSBL = effective_magnification

            else:

                magnification_FSBL = source1_magnification


        else:

            magnification_FSBL = None

        if return_impact_parameter:

            return magnification_FSBL, None

        else:

            return magnification_FSBL
