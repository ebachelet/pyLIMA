import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from pyLIMA.jax_telescope import JaxTelescope, LightcurveData, AstrometryData, EphemerisData


@dataclass(frozen=True)
class JaxEvent:
    """JAX-compatible Event class for microlensing analysis.
    
    This replaces the mutable Event class with an immutable structure
    suitable for functional programming and JAX autodiff.
    
    Attributes
    ----------
    kind : str
        Type of event (e.g., 'Microlensing')
    name : str
        Event name
    ra : float
        Right ascension in degrees (0-360)
    dec : float
        Declination in degrees (-90 to 90)
    North : jnp.ndarray
        North vector projected in the plane of sky (shape: (3,))
    East : jnp.ndarray
        East vector projected in the plane of sky (shape: (3,))
    telescopes : tuple
        Tuple of JaxTelescope objects (immutable)
    survey : Optional[str]
        Survey associated with the event for alignment
    """
    
    kind: str
    name: str
    ra: float
    dec: float
    North: jnp.ndarray
    East: jnp.ndarray
    telescopes: Tuple[JaxTelescope, ...]
    survey: Optional[str] = None
    
    @staticmethod
    def register_pytree():
        """Register JaxEvent as JAX pytree for autodiff."""
        def flatten(event):
            values = (
                event.kind, event.name, event.ra, event.dec,
                event.North, event.East, event.telescopes, event.survey
            )
            keys = (
                'kind', 'name', 'ra', 'dec',
                'North', 'East', 'telescopes', 'survey'
            )
            return values, keys
        
        def unflatten(keys, values):
            return JaxEvent(*values)
        
        jax.tree_util.register_pytree_node(
            JaxEvent,
            flatten,
            unflatten
        )
    
    def __post_init__(self):
        """Validate event parameters after initialization."""
        if not isinstance(self.name, str):
            raise ValueError(f'Event name must be string, got {type(self.name)}')
        
        if not (0 <= self.ra <= 360):
            raise ValueError(f'RA must be 0-360°, got {self.ra}')
        
        if not (-90 <= self.dec <= 90):
            raise ValueError(f'Dec must be -90 to 90°, got {self.dec}')
        
        if len(self.telescopes) == 0:
            raise ValueError('Event must have at least one telescope')
    
    def telescopes_names(self) -> List[str]:
        """Get list of telescope names.
        
        Returns
        -------
        list : telescope names
        """
        return [tel.name for tel in self.telescopes]
    
    def check_event(self) -> 'JaxEvent':
        """Validate event and remove empty telescopes.
        
        Returns a new JaxEvent with only valid telescopes.
        
        Returns
        -------
        JaxEvent : validated event
        """
        # Filter out empty telescopes
        valid_telescopes = []
        for tel in self.telescopes:
            if tel.lightcurve is not None or tel.astrometry is not None:
                valid_telescopes.append(tel)
            else:
                print(f'WARNING: Telescope {tel.name} is empty (no lightcurve or astrometry), removing it')
        
        if len(valid_telescopes) == 0:
            raise ValueError('Event has no telescopes with data after validation')
        
        # Return new event with valid telescopes
        return JaxEvent(
            kind=self.kind,
            name=self.name,
            ra=self.ra,
            dec=self.dec,
            North=self.North,
            East=self.East,
            telescopes=tuple(valid_telescopes),
            survey=self.survey
        )
    
    def find_survey(self, choice: Optional[str] = None) -> 'JaxEvent':
        """Find survey telescope and reorder to be first.
        
        Parameters
        ----------
        choice : str, optional
            Survey name. If None, uses first telescope.
        
        Returns
        -------
        JaxEvent : new event with reordered telescopes
        """
        survey_name = choice or self.telescopes[0].name
        names = self.telescopes_names()
        
        # Find survey telescope
        matching_indices = [i for i, name in enumerate(names) if survey_name in name]
        
        if not matching_indices:
            print(f'WARNING: No telescope containing "{survey_name}" found')
            return self
        
        # Reorder so survey is first
        survey_idx = matching_indices[0]
        reordered = [self.telescopes[survey_idx]]
        reordered.extend([self.telescopes[i] for i in range(len(self.telescopes)) if i != survey_idx])
        
        return JaxEvent(
            kind=self.kind,
            name=self.name,
            ra=self.ra,
            dec=self.dec,
            North=self.North,
            East=self.East,
            telescopes=tuple(reordered),
            survey=survey_name
        )
    
    def total_number_of_data_points(self, choice: str = 'flux') -> int:
        """Compute total number of photometric or astrometric data points.
        
        Parameters
        ----------
        choice : str, 'flux', 'magnitude', or 'astrometry'
        
        Returns
        -------
        int : total number of data points across all telescopes
        """
        n_data = 0
        for tel in self.telescopes:
            n_data += tel.n_data(choice)
        return n_data
    
    def get_photometric_data(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Extract all photometric data from all telescopes.
        
        Returns
        -------
        times : jnp.ndarray, shape (n_total,)
            All observation times
        fluxes : jnp.ndarray, shape (n_total,)
            All flux observations
        errors : jnp.ndarray, shape (n_total,)
            All flux errors
        """
        times_list = []
        fluxes_list = []
        errors_list = []
        
        for tel in self.telescopes:
            if tel.lightcurve is not None:
                times_list.append(tel.lightcurve.time)
                fluxes_list.append(tel.lightcurve.flux)
                errors_list.append(tel.lightcurve.err_flux)
        
        return (
            jnp.concatenate(times_list),
            jnp.concatenate(fluxes_list),
            jnp.concatenate(errors_list)
        )
    
    def get_astrometric_data(self) -> Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Extract all astrometric data from all telescopes.
        
        Returns
        -------
        times : jnp.ndarray, shape (n_total,)
            All observation times
        ras : jnp.ndarray, shape (n_total,)
            All RA observations
        ras_err : jnp.ndarray, shape (n_total,)
            All RA errors
        decs : jnp.ndarray, shape (n_total,)
            All Dec observations
        decs_err : jnp.ndarray, shape (n_total,)
            All Dec errors
        """
        times_list = []
        ras_list = []
        ras_err_list = []
        decs_list = []
        decs_err_list = []
        
        for tel in self.telescopes:
            if tel.astrometry is not None:
                times_list.append(tel.astrometry.time)
                ras_list.append(tel.astrometry.ra)
                ras_err_list.append(tel.astrometry.err_ra)
                decs_list.append(tel.astrometry.dec)
                decs_err_list.append(tel.astrometry.err_dec)
        
        if not times_list:
            return None
        
        return (
            jnp.concatenate(times_list),
            jnp.concatenate(ras_list),
            jnp.concatenate(ras_err_list),
            jnp.concatenate(decs_list),
            jnp.concatenate(decs_err_list)
        )


# Register pytree
JaxEvent.register_pytree()


def convert_event_to_jax(old_event) -> JaxEvent:
    """Convert legacy pyLIMA Event to JAX-compatible JaxEvent.
    
    Parameters
    ----------
    old_event : Event
        Legacy pyLIMA Event object
    
    Returns
    -------
    JaxEvent : JAX-compatible event structure
    """
    from pyLIMA.jax_telescope import convert_telescope_to_jax
    
    # Convert all telescopes
    jax_telescopes = tuple(convert_telescope_to_jax(tel) for tel in old_event.telescopes)
    
    # Convert North/East vectors
    North = jnp.asarray(old_event.North, dtype=jnp.float32)
    East = jnp.asarray(old_event.East, dtype=jnp.float32)
    
    return JaxEvent(
        kind=old_event.kind,
        name=old_event.name,
        ra=float(old_event.ra),
        dec=float(old_event.dec),
        North=North,
        East=East,
        telescopes=jax_telescopes,
        survey=old_event.survey
    )


@dataclass(frozen=True)
class EventParameters:
    """Event model parameters for JAX inference.
    
    Attributes
    ----------
    t0 : float
        Time of maximum magnification (JD)
    u0 : float
        Minimum impact parameter (dimensionless)
    tE : float
        Einstein crossing time (days)
    rho : float
        Source radius normalized to Einstein radius
    s : float
        Binary separation (Einstein radii) - for binary lenses
    q : float
        Mass ratio - for binary lenses
    alpha : float
        Angle from source to lens trajectory (radians)
    parallax_north : float
        Parallax vector North component
    parallax_east : float
        Parallax vector East component
    additional_params : dict
        Additional model parameters (e.g., limb darkening)
    """
    t0: float
    u0: float
    tE: float
    rho: float = 0.0
    s: float = 1.0
    q: float = 0.5
    alpha: float = 0.0
    parallax_north: float = 0.0
    parallax_east: float = 0.0
    additional_params: dict = None
    
    def to_dict(self) -> dict:
        """Convert parameters to dictionary.
        
        Returns
        -------
        dict : parameter dictionary
        """
        params = {
            't0': self.t0,
            'u0': self.u0,
            'tE': self.tE,
            'rho': self.rho,
            's': self.s,
            'q': self.q,
            'alpha': self.alpha,
            'parallax_north': self.parallax_north,
            'parallax_east': self.parallax_east,
        }
        if self.additional_params:
            params.update(self.additional_params)
        return params


def north_east_vectors(ra_degrees: float, dec_degrees: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute North and East unit vectors for a sky coordinate.
    
    Parameters
    ----------
    ra_degrees : float
        Right ascension in degrees
    dec_degrees : float
        Declination in degrees
    
    Returns
    -------
    North : jnp.ndarray, shape (3,)
        North unit vector
    East : jnp.ndarray, shape (3,)
        East unit vector
    """
    ra_rad = jnp.deg2rad(ra_degrees)
    dec_rad = jnp.deg2rad(dec_degrees)
    
    # Target direction vector
    cos_dec = jnp.cos(dec_rad)
    sin_dec = jnp.sin(dec_rad)
    cos_ra = jnp.cos(ra_rad)
    sin_ra = jnp.sin(ra_rad)
    
    target = jnp.array([
        cos_dec * cos_ra,
        cos_dec * sin_ra,
        sin_dec
    ])
    
    # East vector (perpendicular to meridian)
    east = jnp.array([
        -sin_ra,
        cos_ra,
        0.0
    ])
    
    # North vector (perpendicular to both target and east)
    north = jnp.cross(target, east)
    
    return north, east


def create_jax_event(
    name: str,
    ra: float,
    dec: float,
    jax_telescopes: List[JaxTelescope],
    kind: str = 'Microlensing',
    survey: Optional[str] = None
) -> JaxEvent:
    """Create a JaxEvent with proper coordinate vectors.
    
    Parameters
    ----------
    name : str
        Event name
    ra : float
        Right ascension (0-360°)
    dec : float
        Declination (-90 to 90°)
    jax_telescopes : list of JaxTelescope
        List of telescopes observing the event
    kind : str, optional
        Event type (default: 'Microlensing')
    survey : str, optional
        Survey name
    
    Returns
    -------
    JaxEvent : JAX-compatible event
    """
    north, east = north_east_vectors(ra, dec)
    
    return JaxEvent(
        kind=kind,
        name=name,
        ra=ra,
        dec=dec,
        North=north,
        East=east,
        telescopes=tuple(jax_telescopes),
        survey=survey
    )