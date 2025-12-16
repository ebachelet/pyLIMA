import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any

# Optional imports for JAX structures
try:
    from pyLIMA.jax_telescope import JaxTelescope, AstrometryData, LightcurveData
    from pyLIMA.jax_event import JaxEvent, EventParameters
except ImportError:
    JaxTelescope = None
    JaxEvent = None
    AstrometryData = None
    LightcurveData = None


def xy_shifts_to_NE_shifts(xy_shifts, piEN, piEE):
    """
    Transform the x,y positions into the North, East reference,
    i.e. rotate by phi = arctan2(piEE,piEN)
    
    JAX-compatible version: works with both JAX and NumPy arrays.

    Parameters
    ----------
    xy_shifts : array, the x and y positions
    piEN : float, the North component of the parallax vector
    piEE : float, a list of [string,float] indicating the xallarap model

    Returns
    -------
    Detlta_ra : array, the astrometric shift in North
    Detlta_dec : array, the astrometric shift in East
    """
    # Convert to JAX arrays if needed
    centroid_x = jnp.asarray(xy_shifts[0])
    centroid_y = jnp.asarray(xy_shifts[1])
    piEN = jnp.asarray(piEN)
    piEE = jnp.asarray(piEE)

    angle = jnp.arctan2(piEE, piEN)

    Delta_dec = centroid_x * jnp.cos(angle) - jnp.sin(angle) * centroid_y
    Delta_ra = centroid_x * jnp.sin(angle) + jnp.cos(angle) * centroid_y

    return Delta_ra, Delta_dec


def astrometric_positions_of_the_source(telescope, pyLIMA_parameters, time_ref=None):
    """
    The source astrometric positions without astrometric shifts
    
    JAX-compatible version: works with JaxTelescope, legacy Telescope, and dicts.

    Parameters
    ----------
    telescope : JaxTelescope or Telescope
        The telescope object containing astrometric data
    pyLIMA_parameters : dict or EventParameters
        The dictionary containing the model parameters
    time_ref : float, optional
        Time of reference chosen, default is t0

    Returns
    -------
    position_ra : jnp.ndarray
        The astrometric position in North, degree or pixels
    position_dec : jnp.ndarray
        The astrometric position in East, degree or pixels
    """
    # Extract time from JaxTelescope or legacy Telescope
    if isinstance(telescope, JaxTelescope):
        if telescope.astrometry is None:
            raise ValueError(f"Telescope {telescope.name} has no astrometry data")
        time = telescope.astrometry.time
        astrometry_unit = 'deg'  # JAX stores in floats, assume degrees
        pixel_scale = telescope.pixel_scale
        telescope_name = telescope.name
    else:
        time = jnp.asarray(telescope.astrometry['time'].value)
        astrometry_unit = telescope.astrometry['ra'].unit
        pixel_scale = telescope.pixel_scale
        telescope_name = telescope.name

    # Extract parameters
    if isinstance(pyLIMA_parameters, dict):
        if time_ref is None:
            time_ref = pyLIMA_parameters['t0']
        ref_N = jnp.asarray(pyLIMA_parameters.get(f'position_source_N_{telescope_name}', 0.0))
        ref_E = jnp.asarray(pyLIMA_parameters.get(f'position_source_E_{telescope_name}', 0.0))
        mu_N = jnp.asarray(pyLIMA_parameters.get('mu_source_N', 0.0))
        mu_E = jnp.asarray(pyLIMA_parameters.get('mu_source_E', 0.0))
        parallax_source = jnp.asarray(pyLIMA_parameters.get('pi_source', 0.0))
    else:
        # EventParameters case
        if time_ref is None:
            time_ref = pyLIMA_parameters.t0
        ref_N = jnp.asarray(pyLIMA_parameters.additional_params.get(f'position_source_N_{telescope_name}', 0.0))
        ref_E = jnp.asarray(pyLIMA_parameters.additional_params.get(f'position_source_E_{telescope_name}', 0.0))
        mu_N = jnp.asarray(pyLIMA_parameters.additional_params.get('mu_source_N', 0.0))
        mu_E = jnp.asarray(pyLIMA_parameters.additional_params.get('mu_source_E', 0.0))
        parallax_source = jnp.asarray(pyLIMA_parameters.additional_params.get('pi_source', 0.0))

    # Extract Earth positions
    if isinstance(telescope, JaxTelescope):
        if telescope.ephemeris_astrometry is None:
            earth_vector = jnp.zeros(2)
        else:
            earth_vector = telescope.ephemeris_astrometry.positions[:2]
    else:
        earth_vector = jnp.asarray(telescope.Earth_positions_projected['astrometry'])
    
    Earth_projected = earth_vector * parallax_source  # mas

    if astrometry_unit == 'deg':
        position_N = mu_N / 365.25 * (time - time_ref) / 3600 / 1000 + ref_N  # deg
        position_E = mu_E / 365.25 * (time - time_ref) / 3600 / 1000 + ref_E

        position_dec = position_N - Earth_projected[0] / 3600 / 1000
        position_ra = position_E - Earth_projected[1] / 3600 / 1000
    else:
        position_N = mu_N / 365.25 * (time - time_ref) + ref_N  # pix
        position_E = mu_E / 365.25 * (time - time_ref) + ref_E

        position_dec = position_N - Earth_projected[0] / pixel_scale
        position_ra = position_E - Earth_projected[1] / pixel_scale

    return position_ra, position_dec


def source_astrometric_positions(telescope, pyLIMA_parameters, shifts=None,
                                 time_ref=None):
    """
    The source astrometric positions with or without astrometric shifts
    
    JAX-compatible version: works with JaxTelescope, legacy Telescope, and dicts.

    Parameters
    ----------
    telescope : JaxTelescope or Telescope
        The telescope object containing astrometric data
    pyLIMA_parameters : dict or EventParameters
        The dictionary containing the model parameters
    shifts : array, optional
        North and East astrometric shifts in mas or pixel
    time_ref : float, optional
        Time of reference chosen, default is t0

    Returns
    -------
    position_ra : jnp.ndarray
        The astrometric position in North, degree or pixels
    position_dec : jnp.ndarray
        The astrometric position in East, degree or pixels
    """
    if shifts is None:
        shifts = jnp.zeros(2)
    else:
        shifts = jnp.asarray(shifts)

    position_E, position_N = astrometric_positions_of_the_source(telescope,
                                                                 pyLIMA_parameters,
                                                                 time_ref)

    # Determine units
    if isinstance(telescope, JaxTelescope):
        astrometry_unit = 'deg'  # JAX stores in floats, assume degrees
        pixel_scale = telescope.pixel_scale
    else:
        astrometry_unit = telescope.astrometry['ra'].unit
        pixel_scale = telescope.pixel_scale

    if astrometry_unit == 'deg':
        position_ra = shifts[0] / 3600. / 1000 + position_E
        position_dec = shifts[1] / 3600. / 1000 + position_N
    else:
        position_ra = shifts[0] / pixel_scale + position_E
        position_dec = shifts[1] / pixel_scale + position_N

    return position_ra, position_dec


def lens_astrometric_positions(model, telescope, pyLIMA_parameters, time_ref=None):
    """
    The lens astrometric positions
    
    JAX-compatible version: works with JaxTelescope, legacy Telescope, and dicts.

    Parameters
    ----------
    model : object
        The microlensing model object to compute the source trajectory relative to lens
    telescope : JaxTelescope or Telescope
        The telescope object containing astrometric data
    pyLIMA_parameters : dict or EventParameters
        The dictionary containing the model parameters
    time_ref : float, optional
        Time of reference chosen, default is t0

    Returns
    -------
    lens_EN : Tuple[jnp.ndarray, jnp.ndarray]
        The astrometric positions (ra, dec) in North, degree or pixels
    """
    source_EN = source_astrometric_positions(telescope, pyLIMA_parameters, shifts=None,
                                             time_ref=time_ref)

    (source1_trajectory_x, source1_trajectory_y,
    source2_trajectory_x, source2_trajectory_y,
    dseparation, dalpha) = model.sources_trajectory(
        telescope, pyLIMA_parameters,
        data_type='astrometry')

    source_relative_to_lens_EN = xy_shifts_to_NE_shifts(
        (source1_trajectory_x, source1_trajectory_y),
        pyLIMA_parameters.get('piEN', 0.0) if isinstance(pyLIMA_parameters, dict) else pyLIMA_parameters.additional_params.get('piEN', 0.0),
        pyLIMA_parameters.get('piEE', 0.0) if isinstance(pyLIMA_parameters, dict) else pyLIMA_parameters.additional_params.get('piEE', 0.0))

    if isinstance(pyLIMA_parameters, dict):
        theta_E = jnp.asarray(pyLIMA_parameters.get('theta_E', 1.0))
    else:
        theta_E = jnp.asarray(pyLIMA_parameters.additional_params.get('theta_E', 1.0))

    source_relative_to_lens_EN = (jnp.array(source_relative_to_lens_EN) * theta_E)  # mas

    # Determine units and pixel scale
    if isinstance(telescope, JaxTelescope):
        astrometry_unit = 'deg'  # JAX stores in floats, assume degrees
        pixel_scale = telescope.pixel_scale
    else:
        astrometry_unit = telescope.astrometry['ra'].unit
        pixel_scale = telescope.pixel_scale

    if astrometry_unit == 'deg':
        lens_EN = source_EN - source_relative_to_lens_EN / 3600. / 1000.
    else:
        lens_EN = source_EN - source_relative_to_lens_EN / pixel_scale

    return lens_EN


# JAX-specific utility functions for JaxEvent and JaxTelescope

def event_astrometric_positions(event: 'JaxEvent', telescope_name: str, 
                                pyLIMA_parameters: Dict[str, Any],
                                time_ref: Optional[float] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get astrometric positions for a specific telescope in a JaxEvent.
    
    Parameters
    ----------
    event : JaxEvent
        The JAX event object containing telescopes
    telescope_name : str
        Name of the telescope to extract
    pyLIMA_parameters : dict or EventParameters
        Model parameters
    time_ref : float, optional
        Time of reference, default is t0
    
    Returns
    -------
    position_ra : jnp.ndarray
        Right ascension positions
    position_dec : jnp.ndarray
        Declination positions
    """
    # Find the telescope
    telescope = None
    for tel in event.telescopes:
        if tel.name == telescope_name:
            telescope = tel
            break
    
    if telescope is None:
        raise ValueError(f"Telescope {telescope_name} not found in event {event.name}")
    
    if telescope.astrometry is None:
        raise ValueError(f"Telescope {telescope_name} has no astrometry data")
    
    return source_astrometric_positions(telescope, pyLIMA_parameters, shifts=None, time_ref=time_ref)


@jax.jit
def xy_shifts_to_NE_shifts_jitted(xy_shifts: jnp.ndarray, piEN: float, piEE: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled version of xy_shifts_to_NE_shifts for maximum performance.
    
    Parameters
    ----------
    xy_shifts : jnp.ndarray
        Shape (2,) array of [x, y] shifts
    piEN : float
        North parallax component
    piEE : float
        East parallax component
    
    Returns
    -------
    Delta_ra : jnp.ndarray
        Astrometric shift in RA
    Delta_dec : jnp.ndarray
        Astrometric shift in Dec
    """
    centroid_x = xy_shifts[0]
    centroid_y = xy_shifts[1]
    
    angle = jnp.arctan2(piEE, piEN)
    
    Delta_dec = centroid_x * jnp.cos(angle) - jnp.sin(angle) * centroid_y
    Delta_ra = centroid_x * jnp.sin(angle) + jnp.cos(angle) * centroid_y
    
    return Delta_ra, Delta_dec


def get_all_telescopes_astrometry(event: 'JaxEvent', 
                                  pyLIMA_parameters: Dict[str, Any],
                                  time_ref: Optional[float] = None) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Extract astrometric positions from all telescopes in a JaxEvent.
    
    Parameters
    ----------
    event : JaxEvent
        The JAX event object
    pyLIMA_parameters : dict or EventParameters
        Model parameters
    time_ref : float, optional
        Time of reference
    
    Returns
    -------
    positions : dict
        Dictionary mapping telescope names to (position_ra, position_dec) tuples
    """
    positions = {}
    for telescope in event.telescopes:
        if telescope.astrometry is not None:
            positions[telescope.name] = source_astrometric_positions(
                telescope, pyLIMA_parameters, shifts=None, time_ref=time_ref
            )
    return positions
