import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional

# Optional imports for JAX structures
try:
    from pyLIMA.jax_telescope import JaxTelescope
    from pyLIMA.jax_event import JaxEvent, EventParameters
except ImportError:
    JaxTelescope = None
    JaxEvent = None
    EventParameters = None


@jax.jit
def orbital_motion_2D_trajectory_shift(time: Union[float, jnp.ndarray],
                                       t0_om: float,
                                       dalpha_dt: float) -> jnp.ndarray:
    """
    Compute the trajectory curvature induced by the linear orbital motion of the lens.
    
    JAX-compatible and JIT-compiled version.

    Parameters
    ----------
    time : float or jnp.ndarray
        Time(s) to evaluate, in days (JD)
    t0_om : float
        Time of reference of the orbital motion
    dalpha_dt : float
        Linear rate of lens rotation, in radians/year

    Returns
    -------
    dalpha : jnp.ndarray
        Variation of the lens trajectory angle due to lens motion
    """
    time = jnp.asarray(time)
    t0_om = jnp.asarray(t0_om)
    dalpha_dt = jnp.asarray(dalpha_dt)

    dalpha = dalpha_dt * (time - t0_om) / 365.25

    return dalpha


@jax.jit
def orbital_motion_2D_separation_shift(time: Union[float, jnp.ndarray],
                                       t0_om: float,
                                       ds_dt: float) -> jnp.ndarray:
    """
    Compute the separation change induced by the linear orbital motion of the lens.
    
    JAX-compatible and JIT-compiled version.

    Parameters
    ----------
    time : float or jnp.ndarray
        Time(s) to evaluate, in days (JD)
    t0_om : float
        Time of reference of the orbital motion
    ds_dt : float
        Linear rate of lens separation change, in Einstein radii/year

    Returns
    -------
    dseparation : jnp.ndarray
        Variation of the lens separation due to lens motion
    """
    time = jnp.asarray(time)
    t0_om = jnp.asarray(t0_om)
    ds_dt = jnp.asarray(ds_dt)

    dseparation = ds_dt * (time - t0_om) / 365.25

    return dseparation


# JAX-specific utility functions for orbital motion

@jax.jit
def orbital_motion_trajectory_and_separation(time: Union[float, jnp.ndarray],
                                             t0_om: float,
                                             dalpha_dt: float,
                                             ds_dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute both trajectory and separation shifts simultaneously (efficient).
    
    Parameters
    ----------
    time : float or jnp.ndarray
        Time(s) to evaluate
    t0_om : float
        Time of reference of the orbital motion
    dalpha_dt : float
        Linear rate of lens rotation (radians/year)
    ds_dt : float
        Linear rate of lens separation change (Einstein radii/year)
    
    Returns
    -------
    dalpha : jnp.ndarray
        Trajectory angle variation
    dseparation : jnp.ndarray
        Separation variation
    """
    time = jnp.asarray(time)
    t0_om = jnp.asarray(t0_om)
    dt = (time - t0_om) / 365.25
    
    dalpha = dalpha_dt * dt
    dseparation = ds_dt * dt
    
    return dalpha, dseparation


def orbital_motion_for_telescope(telescope: Optional['JaxTelescope'],
                                 time: Union[float, jnp.ndarray],
                                 pyLIMA_parameters: Union[Dict[str, Any], 'EventParameters']) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute orbital motion shifts for a specific telescope.
    
    JAX-compatible wrapper that extracts parameters and computes both shifts.
    
    Parameters
    ----------
    telescope : JaxTelescope or Telescope, optional
        The telescope object (used mainly for data time reference)
    time : float or jnp.ndarray
        Time(s) to evaluate
    pyLIMA_parameters : dict or EventParameters
        Model parameters containing t0_om, dalpha_dt, ds_dt
    
    Returns
    -------
    dalpha : jnp.ndarray
        Trajectory angle variation
    dseparation : jnp.ndarray
        Separation variation
    """
    # Extract orbital motion parameters
    if isinstance(pyLIMA_parameters, dict):
        t0_om = pyLIMA_parameters.get('t0_om', pyLIMA_parameters.get('t0', 0.0))
        dalpha_dt = pyLIMA_parameters.get('dalpha_dt', 0.0)
        ds_dt = pyLIMA_parameters.get('ds_dt', 0.0)
    else:
        # EventParameters case
        t0_om = pyLIMA_parameters.additional_params.get('t0_om', pyLIMA_parameters.t0)
        dalpha_dt = pyLIMA_parameters.additional_params.get('dalpha_dt', 0.0)
        ds_dt = pyLIMA_parameters.additional_params.get('ds_dt', 0.0)
    
    return orbital_motion_trajectory_and_separation(time, t0_om, dalpha_dt, ds_dt)


def event_orbital_motions(event: 'JaxEvent',
                         pyLIMA_parameters: Union[Dict[str, Any], 'EventParameters']) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute orbital motion for all telescopes in a JaxEvent.
    
    Parameters
    ----------
    event : JaxEvent
        The JAX event object containing telescopes
    pyLIMA_parameters : dict or EventParameters
        Model parameters
    
    Returns
    -------
    orbital_motions : dict
        Dictionary mapping telescope names to (dalpha, dseparation) tuples
    """
    orbital_motions = {}
    
    for tel in event.telescopes:
        if tel.lightcurve is not None:
            times = tel.lightcurve.time
            dalpha, dseparation = orbital_motion_for_telescope(tel, times, pyLIMA_parameters)
            orbital_motions[tel.name] = (dalpha, dseparation)
        elif tel.astrometry is not None:
            times = tel.astrometry.time
            dalpha, dseparation = orbital_motion_for_telescope(tel, times, pyLIMA_parameters)
            orbital_motions[tel.name] = (dalpha, dseparation)
    
    return orbital_motions


@jax.jit
def batch_orbital_motion(time_batch: jnp.ndarray,
                         t0_om: float,
                         dalpha_dt: float,
                         ds_dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute orbital motion for a batch of times using vmap.
    
    Useful for efficient vectorized computation across multiple time points.
    
    Parameters
    ----------
    time_batch : jnp.ndarray
        Shape (n_batch,) array of times
    t0_om : float
        Time of reference
    dalpha_dt : float
        Rate of trajectory change
    ds_dt : float
        Rate of separation change
    
    Returns
    -------
    dalpha_batch : jnp.ndarray
        Shape (n_batch,) trajectory variations
    dseparation_batch : jnp.ndarray
        Shape (n_batch,) separation variations
    """
    # vmap over time dimension
    vmapped_motion = jax.vmap(lambda t: orbital_motion_trajectory_and_separation(t, t0_om, dalpha_dt, ds_dt))
    dalpha_batch, dseparation_batch = vmapped_motion(time_batch)
    
    return dalpha_batch, dseparation_batch


def apply_orbital_motion_to_trajectory(source_trajectory_x: jnp.ndarray,
                                       source_trajectory_y: jnp.ndarray,
                                       dalpha: jnp.ndarray,
                                       ds: jnp.ndarray,
                                       s_initial: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply orbital motion (rotation and separation change) to source trajectory.
    
    Parameters
    ----------
    source_trajectory_x : jnp.ndarray
        Source x positions
    source_trajectory_y : jnp.ndarray
        Source y positions
    dalpha : jnp.ndarray
        Angle variation due to orbital motion
    ds : jnp.ndarray
        Separation variation
    s_initial : float, optional
        Initial binary separation (default: 1.0)
    
    Returns
    -------
    trajectory_x_corrected : jnp.ndarray
        Corrected source x positions
    trajectory_y_corrected : jnp.ndarray
        Corrected source y positions
    """
    source_trajectory_x = jnp.asarray(source_trajectory_x)
    source_trajectory_y = jnp.asarray(source_trajectory_y)
    dalpha = jnp.asarray(dalpha)
    ds = jnp.asarray(ds)
    
    # Convert to polar coordinates
    r = jnp.sqrt(source_trajectory_x ** 2 + source_trajectory_y ** 2)
    alpha = jnp.arctan2(source_trajectory_y, source_trajectory_x)
    
    # Apply orbital motion: rotation and separation change
    alpha_corrected = alpha + dalpha
    s_corrected = s_initial + ds
    
    # Convert back to Cartesian
    trajectory_x_corrected = r * jnp.cos(alpha_corrected)
    trajectory_y_corrected = r * jnp.sin(alpha_corrected)
    
    return trajectory_x_corrected, trajectory_y_corrected


def combined_orbital_and_lens_motion(source_trajectory_x: jnp.ndarray,
                                     source_trajectory_y: jnp.ndarray,
                                     dalpha: jnp.ndarray,
                                     ds: jnp.ndarray,
                                     s_initial: float = 1.0,
                                     apply_to_lens: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply combined orbital motion to lens or source trajectory.
    
    Parameters
    ----------
    source_trajectory_x : jnp.ndarray
        X positions (source or lens depending on apply_to_lens)
    source_trajectory_y : jnp.ndarray
        Y positions
    dalpha : jnp.ndarray
        Angle variation
    ds : jnp.ndarray
        Separation variation
    s_initial : float, optional
        Initial reference separation
    apply_to_lens : bool, optional
        If True, apply to lens; if False, apply to source (default: True)
    
    Returns
    -------
    trajectory_x : jnp.ndarray
        Corrected x positions
    trajectory_y : jnp.ndarray
        Corrected y positions
    """
    if apply_to_lens:
        # For binary lens, rotation is around the center of mass
        return apply_orbital_motion_to_trajectory(source_trajectory_x, source_trajectory_y,
                                                 dalpha, ds, s_initial)
    else:
        # For source, typically only small corrections
        return apply_orbital_motion_to_trajectory(source_trajectory_x, source_trajectory_y,
                                                 dalpha, ds, s_initial)
