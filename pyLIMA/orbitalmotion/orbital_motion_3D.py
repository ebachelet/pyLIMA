import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional
from jax.interpreters import ad

try:
    from pyLIMA.jax_telescope import JaxTelescope
    from pyLIMA.jax_event import JaxEvent, EventParameters
except Exception:
    JaxTelescope = None
    JaxEvent = None
    EventParameters = None


# Helper function to handle symbolic zeros in JAX autodiff
def make_zero(tan):
    """
    Helper function to handle symbolic zeros (i.e. parameters that are not being differentiated).
    
    Parameters
    ----------
    tan : ad.Zero or array-like
        Tangent vector that may be a symbolic zero
    
    Returns
    -------
    result : array-like
        Zero array with proper shape if symbolic, else returns tangent as-is
    """
    if type(tan) is ad.Zero:
        return ad.zeros_like_aval(tan.aval)
    else:
        return tan


# See https://jax.exoplanet.codes/en/latest/tutorials/core-from-scratch/
# JAX solver for Kepler's equation (starter + refiner + vectorized solver)
def kepler_starter(mean_anom, ecc):
    ome = 1 - ecc
    M2 = jnp.square(mean_anom)
    alpha = 3 * jnp.pi / (jnp.pi - 6 / jnp.pi)
    alpha += 1.6 / (jnp.pi - 6 / jnp.pi) * (jnp.pi - mean_anom) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * mean_anom
    q = 2 * alphad * ome - M2
    q2 = jnp.square(q)
    w = (jnp.abs(r) + jnp.sqrt(q2 * q + r * r)) ** (1. / 3.)
    # cbrt might be negative-aware; keep sign
    w = jnp.sign(w) * jnp.cbrt(jnp.abs(w))
    return (2 * r * w / (jnp.square(w) + w * q + q2) + mean_anom) / d


def kepler_refiner(mean_anom, ecc, ecc_anom):
    ome = 1 - ecc
    sE = ecc_anom - jnp.sin(ecc_anom)
    cE = 1 - jnp.cos(ecc_anom)

    f_0 = ecc * sE + ecc_anom * ome - mean_anom
    f_1 = ecc * cE + ome
    f_2 = ecc * (ecc_anom - sE)
    f_3 = 1 - f_1
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24)

    return ecc_anom + dE


@jax.jit
@jnp.vectorize
def kepler_solver_impl(mean_anom, ecc):
    mean_anom = mean_anom % (2 * jnp.pi)

    # We restrict to the range [0, pi)
    high = mean_anom > jnp.pi
    mean_anom_small = jnp.where(high, 2 * jnp.pi - mean_anom, mean_anom)

    # Solve
    ecc_anom = kepler_starter(mean_anom_small, ecc)
    ecc_anom = kepler_refiner(mean_anom_small, ecc, ecc_anom)

    # Re-wrap back into the full range
    ecc_anom = jnp.where(high, 2 * jnp.pi - ecc_anom, ecc_anom)

    return ecc_anom


@jax.custom_jvp
def kepler_solver(mean_anom, ecc):
    """
    JAX-compatible Kepler solver with custom JVP for improved autodiff.
    
    Uses the implicit function theorem to compute derivatives through the solver,
    enabling gradient-based inference with HMC/NUTS samplers.
    
    Parameters
    ----------
    mean_anom : float or jnp.ndarray
        Mean anomaly (radians)
    ecc : float or jnp.ndarray
        Eccentricity (0 <= ecc < 1)
    
    Returns
    -------
    ecc_anom : float or jnp.ndarray
        Eccentric anomaly (radians)
    """
    return kepler_solver_impl(mean_anom, ecc)


@kepler_solver.defjvp
def kepler_solver_jvp(primals, tangents):
    """
    Custom JVP (Jacobian-vector product) for Kepler solver using implicit function theorem.
    
    This enables efficient differentiation through the Kepler equation solver,
    which is critical for HMC/NUTS sampling in pyLIMA.
    
    The derivative dE/dM (where E is eccentric anomaly, M is mean anomaly) is computed via:
    dE/dM = 1 / (1 - e*cos(E))
    
    And dE/de is computed as:
    dE/de = sin(E) / (1 - e*cos(E))
    """
    mean_anom, ecc = primals
    d_mean_anom, d_ecc = tangents

    # Run the solver to compute eccentric anomaly
    ecc_anom = kepler_solver(mean_anom, ecc)

    # Implicit function theorem: F(E, M, e) = E - e*sin(E) - M = 0
    # dF/dE = 1 - e*cos(E)
    # dE/dM = -dF/dM / dF/dE = 1 / (1 - e*cos(E))
    # dE/de = -dF/de / dF/dE = sin(E) / (1 - e*cos(E))
    
    dE_dM = 1.0 / (1.0 - ecc * jnp.cos(ecc_anom))
    dE_de = jnp.sin(ecc_anom) * dE_dM
    
    # Apply chain rule: handle symbolic zeros properly
    d_ecc_anom = dE_dM * make_zero(d_mean_anom) + dE_de * make_zero(d_ecc)

    return ecc_anom, d_ecc_anom


def eccentric_anomaly_function(time: Union[jnp.ndarray, np.ndarray], ellipticity: float,
                               t_periastron: float, speed: float) -> jnp.ndarray:
    """
    Solve Kepler's equation for an array of times using the JAX solver.
    
    This uses the custom JVP version for autodiff support in inference.

    Parameters
    ----------
    time : array-like
        Observation times (JD)
    ellipticity : float
        Eccentricity of the orbit (0 <= e < 1)
    t_periastron : float
        Time of periastron passage (JD)
    speed : float
        Orbital angular velocity (mean motion, cycles per day; orbital_velocity / 365.25)

    Returns
    -------
    eccentric_anomalies : jnp.ndarray
        Eccentric anomalies at each time (radians)
    """
    time = jnp.asarray(time)
    ecc = jnp.asarray(ellipticity)
    # mean anomaly: M = n * (t - t_periastron)
    mean_anom = speed * (time - t_periastron)
    # Use custom JVP version for autodiff
    return kepler_solver(mean_anom, ecc)


def orbital_motion_keplerian(time, pyLIMA_parameters: Dict[str, Any], om_model):
    """
    Circular and Keplerian orbital motion of the lens.
    Converted to use JAX arrays where possible; returns JAX arrays.
    """
    Rmatrix = jnp.asarray(pyLIMA_parameters['Rmatrix'])
    orbital_velocity = pyLIMA_parameters['orbital_velocity']
    a_true = pyLIMA_parameters['a_true']
    t_periastron = pyLIMA_parameters['t_periastron']

    time = jnp.asarray(time)

    if om_model[0] == 'Circular':  # Circular
        theta = orbital_velocity * (time - t_periastron) / 365.25
        r_prime = a_true * jnp.vstack([jnp.cos(theta), jnp.sin(theta)])
        r_microlens = jnp.matmul(Rmatrix, r_prime)
    else:  # Keplerian
        eccentricity = pyLIMA_parameters['eccentricity']
        ecc_anom = eccentric_anomaly_function(time, eccentricity, t_periastron,
                                              orbital_velocity / 365.25)
        r_prime = a_true * jnp.vstack([jnp.cos(ecc_anom) - eccentricity,
                                       jnp.sqrt(1 - eccentricity ** 2) * jnp.sin(ecc_anom)])
        r_microlens = jnp.matmul(Rmatrix, r_prime)

    sep = jnp.sqrt(r_microlens[0] ** 2 + r_microlens[1] ** 2)
    angle = jnp.arctan2(r_microlens[1], r_microlens[0])

    separation0 = jnp.asarray(pyLIMA_parameters.get('separation', 1.0))
    angle_0 = 0.0
    return sep - separation0, angle - angle_0


def state_orbital_elements(separation_0, r_s, a_s, v_para, v_perp, v_radial):
    """
    Compute the stated orbital parameters from the microlensing parameters.
    Uses numpy where symbolic matrix operations are easier; returns numpy arrays
    for compatibility with downstream code that expects numpy.
    """
    # Keep using numpy here for robust linear algebra and compatibility
    separation_z = r_s * separation_0
    r_0 = np.array([separation_0, 0, separation_z])

    r_norm = np.sum(r_0 ** 2) ** 0.5
    a_true = a_s * r_norm

    v_0 = r_0[0] * np.array([v_para, v_perp, v_radial])
    h_0 = np.cross(r_0, v_0)

    GMass = np.sum(v_0 ** 2) / (-1 / a_true + 2 / r_norm)

    e_0 = np.cross(v_0, h_0) / GMass - r_0 / r_norm

    orbital_velocity = (GMass / a_true ** 3) ** 0.5

    return e_0, h_0, r_0, v_0, r_norm, separation_z, a_true, GMass, orbital_velocity


def orbital_parameters_from_position_and_velocities(separation_0, r_s, a_s, v_para,
                                                    v_perp, v_radial, t0_om):
    """
    Compute the (unscaled) orbital parameters from the microlensing parameters.
    Some parts rely on numpy/scipy; return values are numpy/pure Python.
    """
    e_0, h_0, r_0, v_0, r_norm, separation_z, a_true, GMass, orbital_velocity = \
        state_orbital_elements(separation_0, r_s, a_s, v_para, v_perp, v_radial)

    eccentricity = np.sqrt(np.sum(e_0 ** 2))
    h_norm = np.sqrt(np.sum(h_0 ** 2))
    z_0 = h_0 / h_norm

    inclination = np.arccos(z_0[2])

    N = np.cross([0, 0, 1], h_0)
    longitude_ascending_node = np.arctan2(N[1], N[0])

    if np.abs(np.dot(r_0, v_0)) < 1e-10:  # Circular
        eccentricity = 0.0
        cosw = separation_0 / a_true * np.cos(longitude_ascending_node)
        sinw = separation_z / np.sin(inclination) / a_true
        omega_peri = np.arctan2(sinw, cosw)

        true_anomaly = omega_peri
        t_periastron = t0_om

        try:
            from scipy.spatial.transform import Rotation
            Rmatrix = Rotation.from_euler("ZXZ", [-omega_peri, -inclination,
                                                  -longitude_ascending_node])
            x_0, y_0, z_0 = Rmatrix.as_matrix()
        except Exception:
            # Fallback approximate axes
            x_0 = np.array([1.0, 0.0, 0.0])
            y_0 = np.array([0.0, 1.0, 0.0])
            z_0 = np.array([0.0, 0.0, 1.0])
    else:  # Keplerian
        x_0 = e_0 / np.linalg.norm(e_0)
        z_0 = h_0 / h_norm
        y_0 = np.cross(z_0, x_0)
        r_0_norm = r_0 / r_norm

        omega_peri = np.arctan2(x_0[2], y_0[2])

        cos_true_anomaly = np.dot(r_0_norm, x_0)
        sin_true_anomaly = np.dot(r_0_norm, y_0)

        true_anomaly = np.arctan2(sin_true_anomaly, cos_true_anomaly)

        eccentric_anomaly = np.arctan2(np.sqrt(1 - eccentricity ** 2) * np.sin(true_anomaly),
                                      (np.cos(true_anomaly) + eccentricity))

        t_periastron = t0_om - (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)) / orbital_velocity * 365.25

    return longitude_ascending_node, inclination, omega_peri, a_true, \
        orbital_velocity, eccentricity, true_anomaly, \
        t_periastron, x_0, y_0, z_0
import numpy as np
import jax
import jax.numpy as jnp

def orbital_motion_keplerian(time, pyLIMA_parameters, om_model):
    """"
    Circular and Keplerian orbital motion of the lens. Use the same parametrization
    from Bozza 2021.
    See https://ui.adsabs.harvard.edu/abs/2021MNRAS.505..126B/abstract
        https://ui.adsabs.harvard.edu/abs/2011ApJ...738...87S/abstract

    Parameters
    ----------
    time : array, containing the time to treat
    pyLIMA_parameters : a pyLIMA_parameters object
    om_model :  list, [str,float] the orbital motion model

    Returns
    -------
    dseparation : array, containts the variation of the lens separation due to the
    motion of the lens
    dalpha : array, containts the variation of the lens trajectory angle due to the
    motion of the lens
    """
    Rmatrix = pyLIMA_parameters['Rmatrix']
    orbital_velocity = pyLIMA_parameters['orbital_velocity']
    a_true = pyLIMA_parameters['a_true']
    t_periastron = pyLIMA_parameters['t_periastron']

    if om_model[0] == 'Circular':  # Circular

        theta = orbital_velocity * (time - t_periastron) / 365.25

        r_prime = a_true * np.array([np.cos(theta), np.sin(theta)])

        r_microlens = np.dot(Rmatrix, r_prime)

    else:  # Keplerian

        eccentricity = pyLIMA_parameters['eccentricity']

        eccentric_anomaly = eccentric_anomaly_function(time, eccentricity, t_periastron,
                                                       orbital_velocity / 365.25)

        r_prime = a_true * np.array([np.cos(eccentric_anomaly) - eccentricity,
                                     (1 - eccentricity ** 2) ** 0.5 * np.sin(
                                         eccentric_anomaly)])

        r_microlens = np.dot(Rmatrix, r_prime)

    sep = (r_microlens[0] ** 2 + r_microlens[1] ** 2) ** 0.5
    angle = np.arctan2(r_microlens[1], r_microlens[0])

    separation0 = pyLIMA_parameters['separation']
    angle_0 = 0
    ### Just to check,
    # to_om = om_model[1]
    # eccentricity = pyLIMA_parameters.eccentricity

    # eccentric_anomaly_0 = eccentric_anomaly_function([to_om], eccentricity,
    # t_periastron, orbital_velocity/365.25)
    # r_prime_0 = np.array([np.cos(eccentric_anomaly_0) - eccentricity,
    # (1 - eccentricity ** 2) ** 0.5 *
    #                      np.sin(eccentric_anomaly_0)]) * a_true# / rE
    # r_microlens_0 = np.dot(Rmatrix, r_prime_0)
    # separation0 = (r_microlens_0[0] ** 2 + r_microlens_0[1] ** 2) ** 0.5
    # angle_0 = np.arctan2(r_microlens_0[1], r_microlens_0[0])
    # breakpoint()
    return sep - separation0, (angle - angle_0)


def orbital_parameters_from_position_and_velocities(separation_0, r_s, a_s, v_para,
                                                    v_perp, v_radial, t0_om):
    """
    Compute the (unscaled!, i.e. GMass is not known) orbital parameters from the
    microlensing parameters
    See https://ui.adsabs.harvard.edu/abs/2021MNRAS.505..126B/abstract
        https://ui.adsabs.harvard.edu/abs/2011ApJ...738...87S/abstract
        https://orbital-mechanics.space/classical-orbital-elements/orbital-elements
        -and-the-state-vector.html

    Parameters
    ----------
    separation_0 : float, the binary separation at t0_om
    r_s : float, the ratio of the radial separation s_z over the projected separation
    s_0
    a_s : float, the ratio of the microlesing separation over the true (unscaled)
    semi-major axis
    v_para : float, the rotation speed along the s_0 axis, i.e. 1/s ds/dt
    v_perp : float, the rotation speed perpendicular to s_0 axis, i.e. dalpha/dt
    v_radial : float, the rotation speed in the z axis, i.e. 1/s ds_z/dt
    t0_om:  float, the time of reference of the orbital _motion model

    Returns
    -------
    longitude_ascending_node : float
    inclination : float
    omega_peri : float
    orbital_velocity : float
    eccentricity : float
    true_anomaly : float
    t_periastron :  float
    x_0 : float, the first column of the Rotation matrix
    y_0 : float, the second column of the Rotation matrix
    z_0 : float, the first column of the Rotation matrix
    """
    e_0, h_0, r_0, v_0, r_norm, separation_z, a_true, GMass, orbital_velocity = \
        state_orbital_elements(
            separation_0, r_s, a_s, v_para, v_perp, v_radial)

    # From Skowron2011, Bozza2020
    # and https://orbital-mechanics.space/classical-orbital-elements/orbital-elements
    # -and-the-state-vector.html

    eccentricity = np.sum(e_0 ** 2) ** 0.5
    h_norm = np.sum(h_0 ** 2) ** 0.5
    z_0 = h_0 / h_norm

    inclination = np.arccos(z_0[2])

    N = np.cross([0, 0, 1], h_0)
    longitude_ascending_node = np.arctan2(N[1], N[0])

    if np.abs(np.dot(r_0, v_0)) < 10 ** -10:  # Circular

        eccentricity = 0
        cosw = separation_0 / a_true * np.cos(longitude_ascending_node)
        sinw = separation_z / np.sin(inclination) / a_true
        omega_peri = np.arctan2(sinw, cosw)

        true_anomaly = omega_peri
        t_periastron = t0_om

        from scipy.spatial.transform import Rotation

        Rmatrix = Rotation.from_euler("ZXZ", [-omega_peri, -inclination,
                                              -longitude_ascending_node])
        x_0, y_0, z_0 = Rmatrix.as_matrix()

    else:  # Keplerian

        x_0 = e_0 / eccentricity
        z_0 = h_0 / h_norm
        y_0 = np.cross(z_0, x_0)
        r_0_norm = r_0 / r_norm

        omega_peri = np.arctan2(x_0[2], y_0[2])

        cos_true_anomaly = np.dot(r_0_norm, x_0)
        sin_true_anomaly = np.dot(r_0_norm, y_0)

        true_anomaly = np.arctan2(sin_true_anomaly, cos_true_anomaly)

        eccentric_anomaly = np.arctan2(
            (1 - eccentricity ** 2) ** 0.5 * np.sin(true_anomaly),
            (np.cos(true_anomaly) + eccentricity))

        t_periastron = t0_om - \
                       (eccentric_anomaly - eccentricity * np.sin(
                           eccentric_anomaly)) / orbital_velocity * 365.25

    return longitude_ascending_node, inclination, omega_peri, a_true, \
        orbital_velocity, eccentricity, true_anomaly, \
        t_periastron, x_0, y_0, z_0


def state_orbital_elements(separation_0, r_s, a_s, v_para, v_perp, v_radial):
    """
    Compute the stated orbital parameters from the microlensing parameters
    See https://ui.adsabs.harvard.edu/abs/2021MNRAS.505..126B/abstract
        https://ui.adsabs.harvard.edu/abs/2011ApJ...738...87S/abstract
        https://orbital-mechanics.space/classical-orbital-elements/orbital-elements
        -and-the-state-vector.html

    Parameters
    ----------
    separation_0 : float, the binary separation at t0_om
    r_s : float, the ratio of the radial separation s_z over the projected separation
    s_0
    a_s : float, the ratio of the microlesing separation over the true (unscaled)
    semi-major axis
    v_para : float, the rotation speed along the s_0 axis, i.e. 1/s ds/dt
    v_perp : float, the rotation speed perpendicular to s_0 axis, i.e. dalpha/dt
    v_radial : float, the rotation speed in the z axis, i.e. 1/s ds_z/dt

    Returns
    -------
    e_0 : array, the eccentricity vector
    h_0 : array, the specific angular momentum vector
    r_0 : array, the microlensing separation vector at time t0_om
    v_0 : array, the microlensing speed vector at time t0_om
    r_norm : float, the norm of the separation vector
    separation_z : float, the radial separation
    a_true : float,the semi-major-axis
    GMass : float, the unscaled mass
    orbital_velocity : float, the orbital velocity of the lens
    """
    separation_z = r_s * separation_0
    r_0 = np.array([separation_0, 0, separation_z])

    r_norm = np.sum(r_0 ** 2) ** 0.5
    a_true = a_s * r_norm

    v_0 = r_0[0] * np.array([v_para, v_perp, v_radial])

    h_0 = np.cross(r_0, v_0)

    GMass = np.sum(v_0 ** 2) / (-1 / a_true + 2 / r_norm)

    e_0 = np.cross(v_0, h_0) / GMass - r_0 / r_norm

    orbital_velocity = (GMass / a_true ** 3) ** 0.5

    return e_0, h_0, r_0, v_0, r_norm, separation_z, a_true, GMass, orbital_velocity


def eccentric_anomaly_function(time, ellipticity, t_periastron, speed):
    """
    Solve the Kepler equation, see https://github.com/dfm/kepler.py

    Parameters
    ----------
    time : array, the time to treat
    ellipticity : float, the eccentricity of the orbit
    t_periastron : float, the time of periastron of the orbit
    speed : float, the orbital velocity

    Returns
    -------
    eccentricities : array, the associated eccentric anomalies at time t
    """
    import kepler

    eccentricities = []

    for t in time:
        phase = speed * (t - t_periastron)
        phase = phase % (2 * np.pi)

        ecc = kepler.solve(phase, ellipticity)

        eccentricities.append(ecc)

    return eccentricities


