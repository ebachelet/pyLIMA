import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any

# Optional imports for JAX structures
try:
    from pyLIMA.jax_telescope import JaxTelescope
    from pyLIMA.jax_event import JaxEvent, EventParameters
except ImportError:
    JaxTelescope = None
    JaxEvent = None
    EventParameters = None


@jax.jit
def PSPL_shifts_no_blend(source_x: Union[float, jnp.ndarray], 
                         source_y: Union[float, jnp.ndarray], 
                         theta_E: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    The PSPL astrometric shifts without blend, see https://arxiv.org/pdf/1705.01767.pdf
    
    JAX-compatible and JIT-compiled version.

    Parameters
    ----------
    source_x : float or jnp.ndarray
        The positions of the source in x
    source_y : float or jnp.ndarray
        The positions of the source in y
    theta_E : float
        The angular Einstein ring radius in mas

    Returns
    -------
    shifts_x : jnp.ndarray
        The astrometric shift in x
    shifts_y : jnp.ndarray
        The astrometric shift in y
    """
    source_x = jnp.asarray(source_x)
    source_y = jnp.asarray(source_y)
    theta_E = jnp.asarray(theta_E)
    
    denominator = source_x ** 2 + source_y ** 2 + 2
    
    shifts_x = source_x / denominator * theta_E
    shifts_y = source_y / denominator * theta_E

    return shifts_x, shifts_y


@jax.jit
def PSPL_shifts_with_blend(source_x: Union[float, jnp.ndarray], 
                           source_y: Union[float, jnp.ndarray], 
                           theta_E: float,
                           g_blend: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    The PSPL astrometric shifts with blend, see https://arxiv.org/pdf/1705.01767.pdf
    
    JAX-compatible and JIT-compiled version.

    Parameters
    ----------
    source_x : float or jnp.ndarray
        The positions of the source in x
    source_y : float or jnp.ndarray
        The positions of the source in y
    theta_E : float
        The angular Einstein ring radius in mas
    g_blend : float
        The blend ratio, i.e. f_blend/f_source

    Returns
    -------
    shifts_x : jnp.ndarray
        The astrometric shift in x
    shifts_y : jnp.ndarray
        The astrometric shift in y
    """
    source_x = jnp.asarray(source_x)
    source_y = jnp.asarray(source_y)
    theta_E = jnp.asarray(theta_E)
    g_blend = jnp.asarray(g_blend)
    
    u_square = source_x ** 2 + source_y ** 2
    u = jnp.sqrt(u_square)
    factor = jnp.sqrt(4 + u_square)

    delta_s = (u - g_blend * u_square * factor) / (2 + u_square + g_blend * u * factor)
    shifts_magnitude = (delta_s + g_blend / (1 + g_blend) * u) * theta_E

    alpha = jnp.arctan2(source_y, source_x)
    shifts_x = shifts_magnitude * jnp.cos(alpha)
    shifts_y = shifts_magnitude * jnp.sin(alpha)

    return shifts_x, shifts_y


# JAX-specific utility functions for JaxEvent and JaxTelescope

def compute_shifts_for_telescope(telescope: Optional['JaxTelescope'],
                                 source_x: Union[float, jnp.ndarray],
                                 source_y: Union[float, jnp.ndarray],
                                 pyLIMA_parameters: Union[Dict[str, Any], 'EventParameters'],
                                 blend_mode: str = 'no_blend') -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute astrometric shifts for a telescope.
    
    JAX-compatible wrapper that handles both legacy and JaxTelescope objects.
    
    Parameters
    ----------
    telescope : JaxTelescope or Telescope, optional
        The telescope object. If None, uses parameters from pyLIMA_parameters
    source_x : float or jnp.ndarray
        Source x position(s)
    source_y : float or jnp.ndarray
        Source y position(s)
    pyLIMA_parameters : dict or EventParameters
        Model parameters containing theta_E and optionally g_blend
    blend_mode : str, optional
        'no_blend' or 'with_blend' (default: 'no_blend')
    
    Returns
    -------
    shifts_x : jnp.ndarray
        Astrometric shifts in x
    shifts_y : jnp.ndarray
        Astrometric shifts in y
    """
    # Extract theta_E
    if isinstance(pyLIMA_parameters, dict):
        theta_E = pyLIMA_parameters.get('theta_E', 1.0)
        g_blend = pyLIMA_parameters.get('g_blend', 0.0)
    else:
        theta_E = pyLIMA_parameters.additional_params.get('theta_E', 1.0)
        g_blend = pyLIMA_parameters.additional_params.get('g_blend', 0.0)
    
    if blend_mode == 'with_blend':
        return PSPL_shifts_with_blend(source_x, source_y, theta_E, g_blend)
    else:
        return PSPL_shifts_no_blend(source_x, source_y, theta_E)


def event_astrometric_shifts(event: 'JaxEvent',
                             source_trajectory: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]],
                             pyLIMA_parameters: Union[Dict[str, Any], 'EventParameters'],
                             blend_mode: str = 'no_blend') -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute astrometric shifts for all telescopes in a JaxEvent.
    
    Parameters
    ----------
    event : JaxEvent
        The JAX event object containing telescopes
    source_trajectory : dict
        Dictionary mapping telescope names to (source_x, source_y) tuples
    pyLIMA_parameters : dict or EventParameters
        Model parameters
    blend_mode : str, optional
        'no_blend' or 'with_blend' (default: 'no_blend')
    
    Returns
    -------
    shifts : dict
        Dictionary mapping telescope names to (shifts_x, shifts_y) tuples
    """
    shifts = {}
    for tel in event.telescopes:
        if tel.name in source_trajectory:
            source_x, source_y = source_trajectory[tel.name]
            shifts[tel.name] = compute_shifts_for_telescope(
                tel, source_x, source_y, pyLIMA_parameters, blend_mode
            )
    return shifts


@jax.jit
def PSPL_shifts_no_blend_jitted(source_x: jnp.ndarray,
                                source_y: jnp.ndarray,
                                theta_E: float) -> jnp.ndarray:
    """
    JIT-compiled PSPL shifts without blend (returns stacked array).
    
    Parameters
    ----------
    source_x : jnp.ndarray
        Source x positions
    source_y : jnp.ndarray
        Source y positions
    theta_E : float
        Einstein radius in mas
    
    Returns
    -------
    shifts : jnp.ndarray
        Shape (2, n_data) array of [shifts_x, shifts_y]
    """
    shifts_x, shifts_y = PSPL_shifts_no_blend(source_x, source_y, theta_E)
    return jnp.stack([shifts_x, shifts_y])


@jax.jit
def PSPL_shifts_with_blend_jitted(source_x: jnp.ndarray,
                                  source_y: jnp.ndarray,
                                  theta_E: float,
                                  g_blend: float) -> jnp.ndarray:
    """
    JIT-compiled PSPL shifts with blend (returns stacked array).
    
    Parameters
    ----------
    source_x : jnp.ndarray
        Source x positions
    source_y : jnp.ndarray
        Source y positions
    theta_E : float
        Einstein radius in mas
    g_blend : float
        Blend ratio f_blend/f_source
    
    Returns
    -------
    shifts : jnp.ndarray
        Shape (2, n_data) array of [shifts_x, shifts_y]
    """
    shifts_x, shifts_y = PSPL_shifts_with_blend(source_x, source_y, theta_E, g_blend)
    return jnp.stack([shifts_x, shifts_y])


def batch_shifts(source_x_batch: jnp.ndarray,
                 source_y_batch: jnp.ndarray,
                 theta_E: float,
                 g_blend: float = 0.0,
                 blend_mode: str = 'no_blend') -> jnp.ndarray:
    """
    Compute shifts for a batch of source positions using vmap.
    
    Useful for computing shifts across multiple events or time points efficiently.
    
    Parameters
    ----------
    source_x_batch : jnp.ndarray
        Shape (n_batch,) array of x positions
    source_y_batch : jnp.ndarray
        Shape (n_batch,) array of y positions
    theta_E : float
        Einstein radius in mas
    g_blend : float, optional
        Blend ratio (default: 0.0)
    blend_mode : str, optional
        'no_blend' or 'with_blend' (default: 'no_blend')
    
    Returns
    -------
    shifts : jnp.ndarray
        Shape (n_batch, 2) array of [shifts_x, shifts_y] for each source position
    """
    if blend_mode == 'with_blend':
        shifts_fn = lambda x, y: PSPL_shifts_with_blend(x, y, theta_E, g_blend)
    else:
        shifts_fn = lambda x, y: PSPL_shifts_no_blend(x, y, theta_E)
    
    # vmap over both source_x and source_y
    vmapped_shifts = jax.vmap(shifts_fn)
    shifts_x, shifts_y = vmapped_shifts(source_x_batch, source_y_batch)
    
    return jnp.stack([shifts_x, shifts_y], axis=1)
