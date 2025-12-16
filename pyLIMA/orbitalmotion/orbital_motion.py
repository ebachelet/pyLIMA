import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional, List

from pyLIMA.orbitalmotion import orbital_motion_2D, orbital_motion_3D

# Optional imports for JAX structures
try:
    from pyLIMA.jax_telescope import JaxTelescope
    from pyLIMA.jax_event import JaxEvent, EventParameters
except ImportError:
    JaxTelescope = None
    JaxEvent = None
    EventParameters = None


def orbital_motion_shifts(orbital_motion_model: Union[List, Tuple],
                         time: Union[float, jnp.ndarray],
                         pyLIMA_parameters: Union[Dict[str, Any], 'EventParameters']) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the orbital motion shifts depending on the model.
    JAX-compatible version supporting both 2D and 3D (Keplerian) models.
    See https://ui.adsabs.harvard.edu/abs/2011ApJ...738...87S/abstract

    Parameters
    ----------
    orbital_motion_model : list or tuple
        [model_type, t0_om] where model_type is '2D', 'Circular', or 'Keplerian'
    time : float or jnp.ndarray
        Time(s) to evaluate
    pyLIMA_parameters : dict or EventParameters
        Model parameters

    Returns
    -------
    dseparation : jnp.ndarray
        Binary separation variation
    dalpha : jnp.ndarray
        Lens trajectory angle variation due to motion
    """
    model_type = orbital_motion_model[0]
    t0_om = orbital_motion_model[1]

    if model_type == '2D':
        # 2D linear orbital motion model
        if isinstance(pyLIMA_parameters, dict):
            v_para = pyLIMA_parameters.get('v_para', 0.0)
            v_perp = pyLIMA_parameters.get('v_perp', 0.0)
            separation = pyLIMA_parameters.get('separation', 1.0)
        else:
            # EventParameters case
            v_para = pyLIMA_parameters.additional_params.get('v_para', 0.0)
            v_perp = pyLIMA_parameters.additional_params.get('v_perp', 0.0)
            separation = pyLIMA_parameters.additional_params.get('separation', 1.0)

        ds_dt = v_para * separation
        dseparation = orbital_motion_2D.orbital_motion_2D_separation_shift(time, t0_om, ds_dt)
        dalpha_dt = v_perp
        dalpha = orbital_motion_2D.orbital_motion_2D_trajectory_shift(time, t0_om, dalpha_dt)

    else:
        # 3D Keplerian orbital motion model
        dseparation, dalpha = orbital_motion_3D.orbital_motion_keplerian(
            time, pyLIMA_parameters, orbital_motion_model
        )

    return dseparation, dalpha


def orbital_motion_for_telescope(telescope: Optional['JaxTelescope'],
                                 orbital_motion_model: Union[List, Tuple],
                                 pyLIMA_parameters: Union[Dict[str, Any], 'EventParameters']) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute orbital motion shifts for a specific telescope.
    
    JAX-compatible wrapper that extracts time data from telescope and computes shifts.

    Parameters
    ----------
    telescope : JaxTelescope or Telescope, optional
        The telescope object
    orbital_motion_model : list or tuple
        [model_type, t0_om]
    pyLIMA_parameters : dict or EventParameters
        Model parameters

    Returns
    -------
    dseparation : jnp.ndarray
        Separation variation
    dalpha : jnp.ndarray
        Angle variation
    """
    # Extract time from telescope
    if isinstance(telescope, JaxTelescope):
        if telescope.lightcurve is not None:
            time = telescope.lightcurve.time
        elif telescope.astrometry is not None:
            time = telescope.astrometry.time
        else:
            raise ValueError(f"Telescope {telescope.name} has no data")
    else:
        # Legacy telescope case
        if telescope.lightcurve is not None:
            time = jnp.asarray(telescope.lightcurve['time'].value)
        elif telescope.astrometry is not None:
            time = jnp.asarray(telescope.astrometry['time'].value)
        else:
            raise ValueError(f"Telescope {telescope.name} has no data")

    return orbital_motion_shifts(orbital_motion_model, time, pyLIMA_parameters)


def event_orbital_motions(event: 'JaxEvent',
                         orbital_motion_model: Union[List, Tuple],
                         pyLIMA_parameters: Union[Dict[str, Any], 'EventParameters']) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute orbital motion for all telescopes in a JaxEvent.

    Parameters
    ----------
    event : JaxEvent
        The JAX event object
    orbital_motion_model : list or tuple
        [model_type, t0_om]
    pyLIMA_parameters : dict or EventParameters
        Model parameters

    Returns
    -------
    orbital_motions : dict
        Dictionary mapping telescope names to (dseparation, dalpha) tuples
    """
    orbital_motions = {}

    for tel in event.telescopes:
        if tel.lightcurve is not None or tel.astrometry is not None:
            dsep, dpha = orbital_motion_for_telescope(tel, orbital_motion_model, pyLIMA_parameters)
            orbital_motions[tel.name] = (dsep, dpha)

    return orbital_motions


def apply_orbital_motion_to_magnification(magnification: jnp.ndarray,
                                         dseparation: jnp.ndarray,
                                         dalpha: jnp.ndarray,
                                         source_trajectory_x: jnp.ndarray,
                                         source_trajectory_y: jnp.ndarray,
                                         separation_0: float = 1.0) -> jnp.ndarray:
    """
    Apply orbital motion corrections to magnification calculations.
    
    Accounts for the changing binary separation and orientation due to orbital motion.

    Parameters
    ----------
    magnification : jnp.ndarray
        Initial magnification array
    dseparation : jnp.ndarray
        Separation variation
    dalpha : jnp.ndarray
        Angle variation
    source_trajectory_x : jnp.ndarray
        Source x positions
    source_trajectory_y : jnp.ndarray
        Source y positions
    separation_0 : float, optional
        Initial binary separation (default: 1.0)

    Returns
    -------
    magnification_corrected : jnp.ndarray
        Magnification with orbital motion applied
    """
    # This is a placeholder for orbital motion correction
    # The actual correction depends on the specific lens model
    # For now, return the magnification as-is (orbital motion affects the
    # lens position in the source trajectory computation)
    return magnification


def combined_orbital_model(time: Union[float, jnp.ndarray],
                          pyLIMA_parameters: Union[Dict[str, Any], 'EventParameters'],
                          orbital_motion_model: Optional[Union[List, Tuple]] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unified interface to compute orbital motion for any model.
    
    Automatically detects and applies the appropriate orbital motion model.

    Parameters
    ----------
    time : float or jnp.ndarray
        Time(s) to evaluate
    pyLIMA_parameters : dict or EventParameters
        Model parameters
    orbital_motion_model : list or tuple, optional
        [model_type, t0_om]. If None, tries to detect from pyLIMA_parameters

    Returns
    -------
    dseparation : jnp.ndarray
        Separation variation
    dalpha : jnp.ndarray
        Angle variation
    """
    if orbital_motion_model is None:
        # Try to detect from parameters
        if isinstance(pyLIMA_parameters, dict):
            model_type = pyLIMA_parameters.get('orbital_motion_type', '2D')
            t0_om = pyLIMA_parameters.get('t0_om', pyLIMA_parameters.get('t0', 0.0))
        else:
            model_type = pyLIMA_parameters.additional_params.get('orbital_motion_type', '2D')
            t0_om = pyLIMA_parameters.additional_params.get('t0_om', pyLIMA_parameters.t0)
        
        orbital_motion_model = (model_type, t0_om)

    return orbital_motion_shifts(orbital_motion_model, time, pyLIMA_parameters)
