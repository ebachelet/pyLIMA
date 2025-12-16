"""Xallarap utilities converted to JAX.

This module provides functions to compute small position shifts induced by
orbital motion of the source (xallarap). Functions are JAX-friendly and
compatible with JAX pytrees (use `jax.numpy` and `jax.jit`).
"""
from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp


@jax.jit
def circular_xallarap(
    time: jnp.ndarray, t0_xi: float, xi_angular_velocity: float, xi_phase: float, xi_inclination: float
) -> jnp.ndarray:
    """Compute circular xallarap displacement (two components) at `time`.

    Returns array([dx, dy]) where dx/dy are the positional shifts in natural
    units (same as original implementation).
    """
    time = jnp.asarray(time)
    omega = xi_angular_velocity * (time - t0_xi) + xi_phase

    separation_1 = jnp.cos(omega)
    separation_2 = jnp.sin(xi_inclination) * jnp.sin(omega)

    return jnp.stack([separation_1, separation_2])


@jax.jit
def xallarap_shifts(xallarap_model: Any, time: jnp.ndarray, pyLIMA_parameters: dict, body: str = "primary") -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the two-component separations for primary and companion.

    Returns (s1_1, s2_1, s1_2, s2_2) where indices 1/2 are primary/companion.
    Inputs may be scalars or arrays; outputs are JAX arrays broadcast to `time`.
    """
    time = jnp.asarray(time)

    model_type = xallarap_model[0] if isinstance(xallarap_model, (list, tuple)) else xallarap_model

    # default zeros
    s11 = jnp.zeros_like(time, dtype=jnp.float64)
    s21 = jnp.zeros_like(time, dtype=jnp.float64)
    s12 = jnp.zeros_like(time, dtype=jnp.float64)
    s22 = jnp.zeros_like(time, dtype=jnp.float64)

    if model_type == "Static":
        # static offset: only companion has offset in this implementation
        delta_t0 = pyLIMA_parameters.get("delta_t0", 0.0)
        delta_u0 = pyLIMA_parameters.get("delta_u0", 0.0)
        s11 = jnp.zeros_like(time)
        s21 = jnp.zeros_like(time)
        s12 = -delta_t0 / pyLIMA_parameters.get("tE", 1.0)
        s22 = jnp.asarray(delta_u0)
        return s11, s21, s12, s22

    if model_type == "Circular":
        xi_angular_velocity = pyLIMA_parameters["xi_angular_velocity"]
        xi_phase = pyLIMA_parameters["xi_phase"]
        xi_inclination = pyLIMA_parameters["xi_inclination"]

        mass_ratio = pyLIMA_parameters.get("xi_mass_ratio", 1.0)
        mass_1 = mass_ratio / (1.0 + mass_ratio)

        origin_1 = jnp.cos(xi_phase) * mass_1
        origin_2 = jnp.sin(xi_phase) * jnp.sin(xi_inclination) * mass_1

        delta_pos = circular_xallarap(time, xallarap_model[1], xi_angular_velocity, xi_phase, xi_inclination)

        dx = delta_pos[0] * mass_1
        dy = delta_pos[1] * mass_1

        s11 = dx - origin_1
        s21 = dy - origin_2

        inv_mass = 1.0 / mass_ratio
        s12 = -delta_pos[0] * inv_mass - origin_1
        s22 = -delta_pos[1] * inv_mass - origin_2

        return s11, s21, s12, s22

    # fallback: zeros
    return s11, s21, s12, s22


@jax.jit
def compute_xallarap_curvature(xiE: jnp.ndarray, delta_positions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute curvature shifts induced by xallarap vector xiE and telescope offsets.

    xiE: [xiN, xiE]
    delta_positions: [d_N, d_E]

    Returns (delta_tau, delta_beta)
    """
    xiE = jnp.asarray(xiE)
    delta_positions = jnp.asarray(delta_positions)

    delta_tau = xiE[0] * delta_positions[0] + xiE[1] * delta_positions[1]
    delta_beta = xiE[0] * delta_positions[1] - xiE[1] * delta_positions[0]

    return delta_tau, delta_beta

