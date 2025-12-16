import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

# Register pytrees for JAX autodiff
@dataclass(frozen=True)
class LightcurveData:
    """JAX-compatible lightcurve data structure.
    
    Attributes:
        time: Array of observation times (JD)
        mag: Array of magnitudes
        err_mag: Array of magnitude errors
        flux: Array of fluxes (W/m^2)
        err_flux: Array of flux errors
        inv_err_flux: Array of inverse flux errors (for chi2)
    """
    time: jnp.ndarray
    mag: jnp.ndarray
    err_mag: jnp.ndarray
    flux: jnp.ndarray
    err_flux: jnp.ndarray
    inv_err_flux: jnp.ndarray
    
    @staticmethod
    def register_pytree():
        """Register LightcurveData as JAX pytree for autodiff."""
        def flatten(lc):
            values = (lc.time, lc.mag, lc.err_mag, lc.flux, lc.err_flux, lc.inv_err_flux)
            keys = ('time', 'mag', 'err_mag', 'flux', 'err_flux', 'inv_err_flux')
            return values, keys
        
        def unflatten(keys, values):
            return LightcurveData(*values)
        
        jax.tree_util.register_pytree_node(
            LightcurveData,
            flatten,
            unflatten
        )
    
    def n_data(self) -> int:
        """Number of photometric data points."""
        return len(self.time)


@dataclass(frozen=True)
class AstrometryData:
    """JAX-compatible astrometry data structure.
    
    Attributes:
        time: Array of observation times (JD)
        ra: Array of right ascensions (degrees or pixels)
        err_ra: Array of RA errors
        dec: Array of declinations (degrees or pixels)
        err_dec: Array of Dec errors
    """
    time: jnp.ndarray
    ra: jnp.ndarray
    err_ra: jnp.ndarray
    dec: jnp.ndarray
    err_dec: jnp.ndarray
    
    @staticmethod
    def register_pytree():
        """Register AstrometryData as JAX pytree."""
        def flatten(ast):
            values = (ast.time, ast.ra, ast.err_ra, ast.dec, ast.err_dec)
            keys = ('time', 'ra', 'err_ra', 'dec', 'err_dec')
            return values, keys
        
        def unflatten(keys, values):
            return AstrometryData(*values)
        
        jax.tree_util.register_pytree_node(
            AstrometryData,
            flatten,
            unflatten
        )
    
    def n_data(self) -> int:
        """Number of astrometric data points."""
        return len(self.time)


@dataclass(frozen=True)
class EphemerisData:
    """JAX-compatible ephemeris data (Earth/spacecraft positions).
    
    Attributes:
        positions: Array of XYZ positions (AU)
        speeds: Array of XYZ velocities (AU/day)
        sidereal_times: Array of sidereal angles (for Earth-based)
        telescope_positions: Array of telescope relative positions
    """
    positions: jnp.ndarray
    speeds: jnp.ndarray
    sidereal_times: Optional[jnp.ndarray] = None
    telescope_positions: Optional[jnp.ndarray] = None
    
    @staticmethod
    def register_pytree():
        """Register EphemerisData as JAX pytree."""
        def flatten(eph):
            values = (eph.positions, eph.speeds, eph.sidereal_times, eph.telescope_positions)
            keys = ('positions', 'speeds', 'sidereal_times', 'telescope_positions')
            return values, keys
        
        def unflatten(keys, values):
            return EphemerisData(*values)
        
        jax.tree_util.register_pytree_node(
            EphemerisData,
            flatten,
            unflatten
        )


@dataclass(frozen=True)
class JaxTelescope:
    """JAX-compatible Telescope class for microlensing light curve modeling.
    
    This replaces the mutable Telescope class with an immutable structure
    suitable for functional programming and JAX autodiff.
    
    Attributes:
        name: Telescope name (str)
        filter: Filter used (str)
        pixel_scale: Pixel scale (mas/pix)
        location: 'Earth' or 'Space'
        altitude: Altitude in meters
        longitude: Longitude in degrees
        latitude: Latitude in degrees
        lightcurve: LightcurveData object
        astrometry: AstrometryData object or None
        ephemeris_photometry: EphemerisData for photometric observations
        ephemeris_astrometry: EphemerisData for astrometric observations or None
        spacecraft_name: Satellite name for JPL Horizons
        ld_gamma: Microlensing gamma limb darkening coefficient
        ld_sigma: Microlensing sigma limb darkening coefficient
        ld_a1: Classical a1 limb darkening coefficient
        ld_a2: Classical a2 limb darkening coefficient
    """
    name: str
    filter: str
    pixel_scale: float
    location: str
    altitude: float
    longitude: float
    latitude: float
    lightcurve: LightcurveData
    ephemeris_photometry: EphemerisData
    astrometry: Optional[AstrometryData] = None
    ephemeris_astrometry: Optional[EphemerisData] = None
    spacecraft_name: Optional[str] = None
    ld_gamma: float = 0.0
    ld_sigma: float = 0.0
    ld_a1: float = 0.0
    ld_a2: float = 0.0
    
    @staticmethod
    def register_pytree():
        """Register JaxTelescope as JAX pytree for autodiff."""
        def flatten(tel):
            values = (
                tel.name, tel.filter, tel.pixel_scale, tel.location,
                tel.altitude, tel.longitude, tel.latitude,
                tel.lightcurve, tel.ephemeris_photometry,
                tel.astrometry, tel.ephemeris_astrometry,
                tel.spacecraft_name,
                tel.ld_gamma, tel.ld_sigma, tel.ld_a1, tel.ld_a2
            )
            keys = (
                'name', 'filter', 'pixel_scale', 'location',
                'altitude', 'longitude', 'latitude',
                'lightcurve', 'ephemeris_photometry',
                'astrometry', 'ephemeris_astrometry',
                'spacecraft_name',
                'ld_gamma', 'ld_sigma', 'ld_a1', 'ld_a2'
            )
            return values, keys
        
        def unflatten(keys, values):
            return JaxTelescope(*values)
        
        jax.tree_util.register_pytree_node(
            JaxTelescope,
            flatten,
            unflatten
        )
    
    def n_data(self, choice: str = 'flux') -> int:
        """Returns the number of photometric or astrometric data.
        
        Parameters
        ----------
        choice : str, 'flux', 'magnitude', or 'astrometry'
        
        Returns
        -------
        int : number of data points
        """
        if choice in ['flux', 'magnitude']:
            return self.lightcurve.n_data()
        elif choice == 'astrometry':
            return self.astrometry.n_data() if self.astrometry else 0
        return 0
    
    def define_limb_darkening_coefficients(self) -> 'JaxTelescope':
        """Convert between LD coefficient conventions (returns new JaxTelescope).
        
        Transforms ld_gamma/ld_sigma <-> ld_a1/ld_a2
        See https://iopscience.iop.org/article/10.1086/378196/pdf
        
        Returns
        -------
        JaxTelescope : new telescope with updated LD coefficients
        """
        ld_gamma = self.ld_gamma
        ld_sigma = self.ld_sigma
        ld_a1 = self.ld_a1
        ld_a2 = self.ld_a2
        
        # Convert from classical to microlensing LD
        if self.ld_gamma == 0.0 and (self.ld_a1 != 0.0 or self.ld_a2 != 0.0):
            denom = 15.0 - 5.0 * self.ld_a1 - 3.0 * self.ld_a2
            ld_gamma = 10.0 * self.ld_a1 / denom
            ld_sigma = 12.0 * self.ld_a2 / denom
        
        # Convert from microlensing to classical LD
        elif self.ld_a1 == 0.0 and (self.ld_gamma != 0.0 or self.ld_sigma != 0.0):
            denom = 4.0 + 2.0 * self.ld_gamma + self.ld_sigma
            ld_a1 = 6.0 * self.ld_gamma / denom
            ld_a2 = 5.0 * self.ld_sigma / denom
        
        return JaxTelescope(
            name=self.name,
            filter=self.filter,
            pixel_scale=self.pixel_scale,
            location=self.location,
            altitude=self.altitude,
            longitude=self.longitude,
            latitude=self.latitude,
            lightcurve=self.lightcurve,
            ephemeris_photometry=self.ephemeris_photometry,
            astrometry=self.astrometry,
            ephemeris_astrometry=self.ephemeris_astrometry,
            spacecraft_name=self.spacecraft_name,
            ld_gamma=ld_gamma,
            ld_sigma=ld_sigma,
            ld_a1=ld_a1,
            ld_a2=ld_a2
        )


# Register all pytrees
LightcurveData.register_pytree()
AstrometryData.register_pytree()
EphemerisData.register_pytree()
JaxTelescope.register_pytree()


def convert_telescope_to_jax(old_telescope) -> JaxTelescope:
    """Convert legacy pyLIMA Telescope to JAX-compatible JaxTelescope.
    
    Parameters
    ----------
    old_telescope : Telescope
        Legacy pyLIMA Telescope object
    
    Returns
    -------
    JaxTelescope : JAX-compatible telescope structure
    """
    # Extract lightcurve data
    if old_telescope.lightcurve is not None:
        lc = LightcurveData(
            time=jnp.asarray(old_telescope.lightcurve['time'].value, dtype=jnp.float32),
            mag=jnp.asarray(old_telescope.lightcurve['mag'].value, dtype=jnp.float32),
            err_mag=jnp.asarray(old_telescope.lightcurve['err_mag'].value, dtype=jnp.float32),
            flux=jnp.asarray(old_telescope.lightcurve['flux'].value, dtype=jnp.float32),
            err_flux=jnp.asarray(old_telescope.lightcurve['err_flux'].value, dtype=jnp.float32),
            inv_err_flux=jnp.asarray(old_telescope.lightcurve['inv_err_flux'].value, dtype=jnp.float32)
        )
    else:
        lc = None
    
    # Extract astrometry data
    if old_telescope.astrometry is not None:
        ast = AstrometryData(
            time=jnp.asarray(old_telescope.astrometry['time'].value, dtype=jnp.float32),
            ra=jnp.asarray(old_telescope.astrometry['ra'].value, dtype=jnp.float32),
            err_ra=jnp.asarray(old_telescope.astrometry['err_ra'].value, dtype=jnp.float32),
            dec=jnp.asarray(old_telescope.astrometry['dec'].value, dtype=jnp.float32),
            err_dec=jnp.asarray(old_telescope.astrometry['err_dec'].value, dtype=jnp.float32)
        )
    else:
        ast = None
    
    # Extract ephemeris data
    eph_photo = EphemerisData(
        positions=jnp.asarray(old_telescope.Earth_positions.get('photometry', []), dtype=jnp.float32),
        speeds=jnp.asarray(old_telescope.Earth_speeds.get('photometry', []), dtype=jnp.float32),
        sidereal_times=jnp.asarray(old_telescope.sidereal_times.get('photometry', []), dtype=jnp.float32),
        telescope_positions=jnp.asarray(old_telescope.telescope_positions.get('photometry', []), dtype=jnp.float32)
    )
    
    eph_astro = None
    if old_telescope.astrometry is not None:
        eph_astro = EphemerisData(
            positions=jnp.asarray(old_telescope.Earth_positions.get('astrometry', []), dtype=jnp.float32),
            speeds=jnp.asarray(old_telescope.Earth_speeds.get('astrometry', []), dtype=jnp.float32),
            sidereal_times=jnp.asarray(old_telescope.sidereal_times.get('astrometry', []), dtype=jnp.float32),
            telescope_positions=jnp.asarray(old_telescope.telescope_positions.get('astrometry', []), dtype=jnp.float32)
        )
    
    return JaxTelescope(
        name=old_telescope.name,
        filter=old_telescope.filter,
        pixel_scale=old_telescope.pixel_scale,
        location=old_telescope.location,
        altitude=old_telescope.altitude,
        longitude=old_telescope.longitude,
        latitude=old_telescope.latitude,
        lightcurve=lc,
        ephemeris_photometry=eph_photo,
        astrometry=ast,
        ephemeris_astrometry=eph_astro,
        spacecraft_name=old_telescope.spacecraft_name,
        ld_gamma=old_telescope.ld_gamma,
        ld_sigma=old_telescope.ld_sigma,
        ld_a1=old_telescope.ld_a1,
        ld_a2=old_telescope.ld_a2
    )