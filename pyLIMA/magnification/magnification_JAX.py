"""JAX-compatible magnification utilities.

Provides an analytic PSPL implementation (fully JAX) and an adapter
to the local `microlux` package for binary/complex lenses.

Functions are written to accept jnp arrays and to be jittable/vmap-able.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

import microlux

def _to_jnp(x: Any) -> jnp.ndarray:
    return jnp.asarray(x)


@jax.jit
def magnification_PSPL(tau: jnp.ndarray, beta: jnp.ndarray) -> Any:
    """Analytic Paczynski Point-Source Point-Lens magnification.

    Parameters
    ----------
    tau : array_like
        (t - t0) / tE
    beta : array_like
        impact parameter (u0) or array of same length as `tau`.

    Returns
    -------
    A : jnp.ndarray
        Magnification array
    """
    tau = _to_jnp(tau)
    beta = _to_jnp(beta)
    u = jnp.sqrt(tau ** 2 + beta ** 2)
    u2 = u ** 2
    A = (u2 + 2.0) / (u * jnp.sqrt(u2 + 4.0))
    return A

def impact_parameter_PSPL(tau: jnp.ndarray, beta: jnp.ndarray) -> Any:
    """Analytic Paczynski Point-Source Point-Lens magnification.

    Parameters
    ----------
    tau : array_like
        (t - t0) / tE
    beta : array_like
        impact parameter (u0) or array of same length as `tau`.

    Returns
    -------
    u : jnp.ndarray
        imparct parameter array
    """
    tau = _to_jnp(tau)
    beta = _to_jnp(beta)
    u = jnp.sqrt(tau ** 2 + beta ** 2)
    u2 = u ** 2
    A = (u2 + 2.0) / (u * jnp.sqrt(u2 + 4.0))
    return u


def magnification_FSPL_microlux(
    t0: float,
    u0: float,
    tE: float,
    rho: float,
    times: jnp.ndarray,
    limb_darkening_coeff: Optional[float] = None,
    tol: float = 1e-2,
    retol: float = 0.001,
    return_impact_parameter: bool = False,
) -> Any:
    """Finite Source Point Lens magnification using microlux.

    Uses microlux.point_light_curve for single-lens finite source effects.
    Supports limb darkening.

    Parameters
    ----------
    t0 : float
        Time of peak (t0)
    u0 : float
        Impact parameter (u0)
    tE : float
        Einstein crossing time
    rho : float
        Normalized source radius
    times : jnp.ndarray
        Array of times at which to evaluate magnification
    limb_darkening_coeff : float, optional
        Linear limb darkening coefficient. If None, no limb darkening.
    tol : float
        Absolute tolerance for contour integration
    retol : float
        Relative tolerance for contour integration
    return_impact_parameter : bool
        If True return tuple (A, u) else return A only.

    Returns
    -------
    A : jnp.ndarray
        Magnification array
    u : jnp.ndarray (optional)
        Impact parameter u(t) when `return_impact_parameter` is True.
    """
    times = _to_jnp(times)
    tau = (times - t0) / tE
    trajectory_l = tau + 1j * u0  # simplified trajectory in low-mass frame

    # For single lens (q=0, s=0), microlux.point_light_curve works directly
    # We use it as the foundation; for true finite source we delegate to extended_light_curve
    mag, cond = microlux.point_light_curve(trajectory_l, s=0.0, q=0.0, rho=rho, tol=tol)

    if limb_darkening_coeff is not None:
        # Use extended_light_curve with limb darkening
        ld_instance = microlux.limb_darkening.LinearLimbDarkening(limb_darkening_coeff)
        mag = microlux.extended_light_curve(
            trajectory_l,
            s=0.0,
            q=0.0,
            rho=rho,
            tol=tol,
            retol=retol,
            limb_darkening=ld_instance,
        )

    return mag


def magnification_PSBL_microlux(
    t0: float,
    u0: float,
    tE: float,
    q: float,
    s: float,
    alpha_deg: float,
    times: jnp.ndarray,
    tol: float = 1e-2,
    retol: float = 0.001,
    return_info: bool = False,
) -> Any:
    """Point Source Binary Lens magnification using microlux.

    Parameters
    ----------
    t0 : float
        Time of peak
    u0 : float
        Impact parameter
    tE : float
        Einstein crossing time
    q : float
        Mass ratio (companion / primary)
    s : float
        Normalized separation between binary components
    alpha_deg : float
        Angle of source trajectory relative to binary axis (in degrees)
    times : jnp.ndarray
        Array of times at which to evaluate magnification
    tol : float
        Absolute tolerance for contour integration
    retol : float
        Relative tolerance for contour integration
    return_info : bool
        If True, return tuple (mag, info) else return mag only.

    Returns
    -------
    mag : jnp.ndarray
        Magnification array
    info : tuple (optional)
        Debug information when return_info is True.
    """
    times = _to_jnp(times)
    rho = 0.0  # point source
    mag = microlux.binary_mag(
        t0, u0, tE, rho, q, s, alpha_deg, times, tol=tol, retol=retol, return_info=return_info
    )
    return mag


def magnification_FSBL_microlux(
    t0: float,
    u0: float,
    tE: float,
    rho: float,
    q: float,
    s: float,
    alpha_deg: float,
    times: jnp.ndarray,
    limb_darkening_coeff: Optional[float] = None,
    tol: float = 1e-2,
    retol: float = 0.001,
    return_info: bool = False,
) -> Any:
    """Finite Source Binary Lens magnification using microlux.

    Parameters
    ----------
    t0 : float
        Time of peak
    u0 : float
        Impact parameter
    tE : float
        Einstein crossing time
    rho : float
        Normalized source radius
    q : float
        Mass ratio (companion / primary)
    s : float
        Normalized separation between binary components
    alpha_deg : float
        Angle of source trajectory relative to binary axis (in degrees)
    times : jnp.ndarray
        Array of times at which to evaluate magnification
    limb_darkening_coeff : float, optional
        Linear limb darkening coefficient
    tol : float
        Absolute tolerance for contour integration
    retol : float
        Relative tolerance for contour integration
    return_info : bool
        If True, return tuple (mag, info) else return mag only.

    Returns
    -------
    mag : jnp.ndarray
        Magnification array
    info : tuple (optional)
        Debug information when return_info is True.
    """
    times = _to_jnp(times)
    mag = microlux.binary_mag(
        t0,
        u0,
        tE,
        rho,
        q,
        s,
        alpha_deg,
        times,
        limb_darkening_coeff=limb_darkening_coeff,
        tol=tol,
        retol=retol,
        return_info=return_info,
    )
    return mag


def magnification_USBL_microlux(
    t0: float,
    u0: float,
    tE: float,
    rho: float,
    q: float,
    s: float,
    alpha_deg: float,
    times: jnp.ndarray,
    tol: float = 1e-2,
    retol: float = 0.001,
) -> jnp.ndarray:
    """Uniform Source Binary Lens magnification using microlux.

    Alias for FSBL with rho > 0 (finite source).

    Parameters
    ----------
    t0 : float
        Time of peak
    u0 : float
        Impact parameter
    tE : float
        Einstein crossing time
    rho : float
        Normalized source radius
    q : float
        Mass ratio (companion / primary)
    s : float
        Normalized separation between binary components
    alpha_deg : float
        Angle of source trajectory relative to binary axis (in degrees)
    times : jnp.ndarray
        Array of times at which to evaluate magnification
    tol : float
        Absolute tolerance for contour integration
    retol : float
        Relative tolerance for contour integration

    Returns
    -------
    mag : jnp.ndarray
        Magnification array
    """
    times = _to_jnp(times)
    mag = microlux.binary_mag(
        t0, u0, tE, rho, q, s, alpha_deg, times, tol=tol, retol=retol
    )
    return mag


def magnification_FSPLarge_microlux(
    t0: float,
    u0: float,
    tE: float,
    rho: float,
    times: jnp.ndarray,
    limb_darkening_coeff: Optional[float] = None,
    tol: float = 1e-2,
    retol: float = 0.001,
    return_impact_parameter: bool = False,
) -> Any:
    """Finite Source Point Lens (large source) magnification using microlux.

    Alias for FSPL with potentially larger source radii.
    Uses microlux contour integration for accurate finite source effects.

    Parameters
    ----------
    t0 : float
        Time of peak
    u0 : float
        Impact parameter
    tE : float
        Einstein crossing time
    rho : float
        Normalized source radius (large)
    times : jnp.ndarray
        Array of times at which to evaluate magnification
    limb_darkening_coeff : float, optional
        Linear limb darkening coefficient
    tol : float
        Absolute tolerance for contour integration
    retol : float
        Relative tolerance for contour integration
    return_impact_parameter : bool
        If True return tuple (A, u) else return A only.

    Returns
    -------
    A : jnp.ndarray
        Magnification array
    u : jnp.ndarray (optional)
        Impact parameter u(t) when `return_impact_parameter` is True.
    """
    return magnification_FSPL_microlux(
        t0,
        u0,
        tE,
        rho,
        times,
        limb_darkening_coeff=limb_darkening_coeff,
        tol=tol,
        retol=retol,
        return_impact_parameter=return_impact_parameter,
    )


def _get_times_from_telescope(telescope: Any) -> jnp.ndarray:
    """Attempt to extract times from a telescope-like object (legacy or JAX).

    Tries common access patterns used across the codebase.
    """
    # Legacy: telescope.lightcurve['time'].value
    try:
        t = telescope.lightcurve["time"].value
        return _to_jnp(t)
    except Exception:
        pass

    # Jax LightcurveData: telescope.lightcurve.time
    try:
        t = telescope.lightcurve.time
        return _to_jnp(t)
    except Exception:
        pass

    # direct attribute
    try:
        t = telescope.time
        return _to_jnp(t)
    except Exception:
        raise ValueError("Could not extract time array from telescope object")





__all__ = [
    "magnification_PSPL",
    "magnification_FSPL_microlux",
    "magnification_FSPLarge_microlux",
    "magnification_PSBL_microlux",
    "magnification_FSBL_microlux",
    "magnification_USBL_microlux",
]
