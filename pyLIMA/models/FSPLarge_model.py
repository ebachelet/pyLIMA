import jax.numpy as jnp
from pyLIMA.magnification import magnification_VBB
from pyLIMA.magnification import magnification_JAX as magnification_JAX
from pyLIMA.models.FSPL_model import FSPLmodel


class FSPLargemodel(FSPLmodel):

    def model_type(self):

        return 'FSPLarge'

    def paczynski_model_parameters(self):
        """
        [to,u0,tE,rho]
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'tE': 2, 'rho': 3}
        self.Jacobian_flag = 'Numerical'

        return model_dictionary
    def model_magnification(self, telescope, pyLIMA_parameters,
                            return_impact_parameter=False):
        """
        The finite source magnification of large source (i.e. no Yoo approximation),
        using VBB instead. Slower obviously...
        See https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.2188B/abstract
            https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5157B/abstract
        """


        rho = float(pyLIMA_parameters['rho'])
        linear_limb_darkening = getattr(telescope, 'ld_a1', None)
        sqrt_limb_darkening = getattr(telescope, 'ld_a2', None)

        (source1_trajectory_x, source1_trajectory_y,
         source2_trajectory_x, source2_trajectory_y,
         dseparation, dalpha) = self.sources_trajectory(
            telescope, pyLIMA_parameters,
            data_type='photometry')

        # Prefer microlux via JAX wrapper; fall back to VBB implementation
        try:
            source1_magnification = magnification_JAX.magnification_FSPLarge_microlux(
                float(pyLIMA_parameters['t0']),
                float(pyLIMA_parameters['u0']),
                float(pyLIMA_parameters['tE']),
                rho,
                telescope.lightcurve['time'].value,
                limb_darkening_coeff=linear_limb_darkening if linear_limb_darkening is not None else None,
                return_impact_parameter=return_impact_parameter
            )
        except Exception:
            # fallback to legacy VBB implementation
            if (sqrt_limb_darkening is not None) and (sqrt_limb_darkening > 0):
                source1_magnification = magnification_VBB.magnification_FSPL(
                    source1_trajectory_x, source1_trajectory_y, rho, linear_limb_darkening, sqrt_limb_darkening
                )
            else:
                source1_magnification = magnification_VBB.magnification_FSPL(
                    source1_trajectory_x, source1_trajectory_y, rho, linear_limb_darkening
                )

        if source2_trajectory_x is not None:

            rho_2 = float(pyLIMA_parameters.get('rho_2', rho))

            try:
                source2_magnification = magnification_JAX.magnification_FSPLarge_microlux(
                    float(pyLIMA_parameters['t0']),
                    float(pyLIMA_parameters['u0']),
                    float(pyLIMA_parameters['tE']),
                    rho_2,
                    telescope.lightcurve['time'].value,
                    limb_darkening_coeff=linear_limb_darkening if linear_limb_darkening is not None else None,
                    return_impact_parameter=return_impact_parameter
                )
            except Exception:
                # fallback to legacy VBB implementation
                if (sqrt_limb_darkening is not None) and (sqrt_limb_darkening > 0):
                    source2_magnification = magnification_VBB.magnification_FSPL(
                        source2_trajectory_x, source2_trajectory_y, rho_2, linear_limb_darkening, sqrt_limb_darkening
                    )
                else:
                    source2_magnification = magnification_VBB.magnification_FSPL(
                        source2_trajectory_x, source2_trajectory_y, rho_2, linear_limb_darkening
                    )

            blend_magnification_factor = pyLIMA_parameters['q_flux_' + telescope.filter]
            effective_magnification = (
                    source1_magnification +
                    source2_magnification *
                    blend_magnification_factor)

            magnification = effective_magnification

        else:

            magnification = source1_magnification

        return magnification
