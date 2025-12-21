"""
JAX Nested Sampling (jaxns) for pyLIMA microlensing inference.

Provides nested sampling via jaxns for parameter estimation with:
- Evidence computation (model comparison)
- Efficient multimodal posterior exploration
- Full posterior samples with importance weights

Example:
--------
    from pyLIMA.inference.jaxns_inference import PSPLInferenceNS
    
    # Create inference object
    inf = PSPLInferenceNS(event, model)
    
    # Set priors
    inf.set_priors({'t0': (2459000, 2460000),
                    'u0': (0.01, 2.0),
                    'tE': (1, 200)})
    
    # Run nested sampling
    inf.run_nested_sampling(num_live_points=500, seed=0)
    
    # Get results
    samples, weights = inf.get_weighted_samples()
    evidence = inf.get_evidence()
    inf.print_summary()
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp

try:
    from jaxns import NestedSampler
    from jaxns.framework.model import Model
    from jaxns.framework.prior import Prior
    JAXNS_AVAILABLE = True
except ImportError:
    JAXNS_AVAILABLE = False

tfpd = tfp.distributions


class PSPLInferenceNS:
    def __init__(self, event, model):
        """
        Initialize nested sampling inference object.

        Parameters
        ----------
        event : Event
            pyLIMA event object with telescopes and data
        model : MLmodel
            pyLIMA model (e.g., PSPLmodel, FSPLmodel)
        """
        if not JAXNS_AVAILABLE:
            raise ImportError("jaxns not installed. Please install from /home/mrabus/github/jaxns or pip install jaxns")

        self.event = event
        self.model = model
        self.priors = {}
        self.sampler = None
        self.results = None
        self.samples = None
        self._prior_names = []

    def set_priors(self, priors: Dict[str, Tuple[float, float]]) -> None:
        """
        Set uniform priors for parameters.

        Parameters
        ----------
        priors : dict
            Mapping of parameter_name -> (lower_bound, upper_bound)
        """
        self.priors = priors
        self._prior_names = list(priors.keys())
        print(f"Priors set for: {self._prior_names}")

    def _get_data_for_telescope(self, telescope) -> Tuple[np.ndarray, np.ndarray]:
        """Extract flux and error flux from telescope (handles both formats)."""
        try:
            flux = np.asarray(telescope.lightcurve.flux)
            err_flux = np.asarray(telescope.lightcurve.err_flux)
        except (TypeError, AttributeError):
            flux = np.asarray(telescope.lightcurve['flux'].value)
            err_flux = np.asarray(telescope.lightcurve['err_flux'].value)

        return flux, err_flux

    def _log_likelihood(self, params: Dict[str, float]) -> float:
        """
        Compute log-likelihood for nested sampling.

        Parameters
        ----------
        params : dict
            Parameter dict with keys matching prior names

        Returns
        -------
        float
            log-likelihood value
        """
        # Build parameter vector following model ordering
        param_vector = []
        for model_param in self.model.model_dictionnary.keys():
            param_vector.append(params.get(model_param, None))

        # Compute pyLIMA parameters
        pyLIMA_params = self.model.compute_pyLIMA_parameters(param_vector)

        # Compute likelihood from all telescopes
        chi2_total = 0.0

        for telescope in self.event.telescopes:
            if telescope.lightcurve is None:
                continue

            # Get magnification
            mag = self.model.model_magnification(telescope, pyLIMA_params)
            mag = jnp.asarray(mag)

            # Get data
            flux_data, err_flux = self._get_data_for_telescope(telescope)
            flux_data = jnp.asarray(flux_data)
            err_flux = jnp.asarray(err_flux)

            # Compute fsource and fblend by weighted linear regression
            X = jnp.vstack([mag, jnp.ones_like(mag)]).T
            w = 1.0 / (err_flux ** 2)
            XT_W = (X.T * w)
            A = XT_W @ X
            b = XT_W @ flux_data

            def solve_safe(A, b):
                return jnp.where(jnp.linalg.cond(A) < 1e12,
                                  jnp.linalg.solve(A, b),
                                  jnp.array([0.0, 0.0]))

            theta = solve_safe(A, b)
            fs = theta[0]
            fb = theta[1]
            flux_model = fs * mag + fb

            # Accumulate chi2
            residuals = (flux_data - flux_model) / err_flux
            chi2_total = chi2_total + jnp.sum(residuals ** 2)

        # Log-likelihood (Gaussian)
        return -0.5 * chi2_total

    def run_nested_sampling(
        self,
        num_live_points: int = 500,
        seed: int = 0,
        max_samples: int = 10000,
        num_parallel_workers: int = 1,
        progress_bar: bool = True
    ) -> None:
        """
        Run nested sampling.

        Parameters
        ----------
        num_live_points : int
            Number of live points for nested sampling
        seed : int
            Random seed for reproducibility
        max_samples : int
            Maximum number of samples to generate
        num_parallel_workers : int
            Number of parallel workers for likelihood evaluation
        progress_bar : bool
            Whether to show progress bar
        """
        print(f"Starting Nested Sampling: {num_live_points} live points, max_samples={max_samples}")
        # Build prior model for jaxns
        def prior_model():
            params = {}
            for pname in self._prior_names:
                low, high = self.priors[pname]
                params[pname] = yield Prior(tfpd.Uniform(low=low, high=high), name=pname)
            return params

        # Likelihood consumes the params dict returned by prior_model
        def log_likelihood(params):
            return self._log_likelihood(params)

        # Construct jaxns Model
        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

        # Create and run sampler
        key = jax.random.PRNGKey(seed)
        sampler = NestedSampler(
            model=model,
            max_samples=max_samples,
            num_live_points=num_live_points,
            verbose=progress_bar
        )

        key, subkey = jax.random.split(key)
        termination_reason, state = sampler(subkey)
        self.results = sampler.to_results(termination_reason, state, trim=True)

        print("Nested Sampling completed successfully!")

    def get_weighted_samples(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Get weighted samples from nested sampling.

        Returns
        -------
        samples : dict
            Mapping param_name -> array of parameter samples
        log_weights : np.ndarray
            Log-weights for importance sampling
        """
        if self.results is None:
            raise RuntimeError("Nested sampling not yet run. Call run_nested_sampling() first.")

        # Extract samples dict from results (already in X-space)
        samples_dict = {k: np.asarray(v) for k, v in self.results.samples.items()}

        # Use log_dp_mean as log-weights (evidence contribution per sample)
        log_w = np.asarray(self.results.log_dp_mean)
        log_w = log_w - np.logaddexp.reduce(log_w)
        weights = np.exp(log_w)

        return samples_dict, weights

    def get_evidence(self) -> Tuple[float, float]:
        """
        Get log-evidence and uncertainty.

        Returns
        -------
        log_evidence : float
            Log of marginal likelihood
        log_evidence_err : float
            Uncertainty (1-sigma)
        """
        if self.results is None:
            raise RuntimeError("Nested sampling not yet run. Call run_nested_sampling() first.")

        log_evidence = float(np.asarray(self.results.log_Z_mean))
        log_evidence_err = float(np.asarray(self.results.log_Z_uncert))

        return log_evidence, log_evidence_err

    def get_posterior_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute posterior statistics from weighted samples.

        Returns
        -------
        dict
            Mapping param_name -> {'mean': ..., 'std': ..., 'median': ..., 'q16': ..., 'q84': ...}
        """
        if self.results is None:
            raise RuntimeError("Nested sampling not yet run. Call run_nested_sampling() first.")

        samples, weights = self.get_weighted_samples()

        stats = {}
        for pname in self._prior_names:
            samp = samples[pname]
            # Weighted percentiles
            sorted_idx = np.argsort(samp)
            sorted_samp = samp[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cum_weights = np.cumsum(sorted_weights)

            stats[pname] = {
                'mean': float(np.average(samp, weights=weights)),
                'std': float(np.sqrt(np.average((samp - np.average(samp, weights=weights))**2, weights=weights))),
                'median': float(sorted_samp[np.searchsorted(cum_weights, 0.5)]),
                'q16': float(sorted_samp[np.searchsorted(cum_weights, 0.16)]),
                'q84': float(sorted_samp[np.searchsorted(cum_weights, 0.84)]),
            }

        return stats

    def best_fit_dict(self) -> Dict[str, float]:
        """Get best-fit (weighted mean) parameters as dict."""
        stats = self.get_posterior_stats()
        return {name: stats[name]['mean'] for name in stats.keys()}

    def print_summary(self) -> None:
        """Print nested sampling summary."""
        if self.results is None:
            raise RuntimeError("Nested sampling not yet run. Call run_nested_sampling() first.")

        log_evidence, log_evidence_err = self.get_evidence()
        print(f"\n=== Nested Sampling Summary ===")
        print(f"Log-Evidence: {log_evidence:.2f} ± {log_evidence_err:.2f}")
        try:
            n_samps = int(np.asarray(self.results.total_num_samples))
        except Exception:
            n_samps = len(next(iter(self.results.samples.values())))
        print(f"Number of samples: {n_samps}")

        stats = self.get_posterior_stats()
        print(f"\n=== Posterior Statistics ===")
        for pname, s in stats.items():
            print(f"{pname:12s}: {s['median']:.4f} + {s['q84'] - s['median']:.4f} - {s['median'] - s['q16']:.4f}")

    def plot_posteriors(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot 1D posterior distributions from nested sampling.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.results is None:
            raise RuntimeError("Nested sampling not yet run. Call run_nested_sampling() first.")

        samples, weights = self.get_weighted_samples()
        n_params = len(self._prior_names)

        fig, axes = plt.subplots(n_params, 1, figsize=figsize)
        if n_params == 1:
            axes = [axes]

        for i, pname in enumerate(self._prior_names):
            ax = axes[i]
            samp = samples[pname]

            ax.hist(samp, bins=50, weights=weights, density=True, alpha=0.7, color='blue')
            ax.axvline(np.average(samp, weights=weights), color='red', linestyle='--', label='Mean')
            ax.axvline(np.median(samp), color='green', linestyle='--', label='Median')
            ax.set_xlabel(pname)
            ax.set_ylabel('Probability Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_best_fit_model(self, telescope) -> np.ndarray:
        """Compute model prediction at best-fit parameters."""
        best_fit = self.best_fit_dict()

        param_vector = []
        for model_param in self.model.model_dictionnary.keys():
            param_vector.append(best_fit.get(model_param, None))

        pyLIMA_params = self.model.compute_pyLIMA_parameters(param_vector)
        return self.model.model_magnification(telescope, pyLIMA_params)

    def best_fit_magnification_at_times(self, telescope, times: np.ndarray) -> np.ndarray:
        """
        Compute magnification at best-fit parameters for arbitrary time stamps.

        Parameters
        ----------
        telescope : Telescope
            Telescope context (parallax, configuration) used by the model
        times : np.ndarray
            Array of time stamps for which to compute magnification

        Returns
        -------
        np.ndarray
            Magnification array at the provided times
        """
        best_fit = self.best_fit_dict()

        # Build parameter vector
        param_vector = []
        for model_param in self.model.model_dictionnary.keys():
            param_vector.append(best_fit.get(model_param, None))

        pyLIMA_params = self.model.compute_pyLIMA_parameters(param_vector)

        # Temporarily override telescope times
        restore_attr = None
        restore_dict = None
        try:
            restore_attr = getattr(telescope.lightcurve, 'time', None)
            telescope.lightcurve.time = np.asarray(times)
            mag = self.model.model_magnification(telescope, pyLIMA_params)
        except Exception:
            try:
                restore_dict = telescope.lightcurve['time'].value
                telescope.lightcurve['time'].value = np.asarray(times)
                mag = self.model.model_magnification(telescope, pyLIMA_params)
            finally:
                if restore_dict is not None:
                    telescope.lightcurve['time'].value = restore_dict
        else:
            if restore_attr is not None:
                telescope.lightcurve.time = restore_attr

        return np.asarray(mag)

    def best_fit_flux_at_times(self, telescope, times: np.ndarray) -> np.ndarray:
        """
        Compute flux model at best-fit parameters for arbitrary times.

        Uses weighted least squares on the telescope's observed flux to
        estimate fsource and fblend, then evaluates flux at `times`.

        Parameters
        ----------
        telescope : Telescope
            Telescope context used by the model
        times : np.ndarray
            Array of time stamps to evaluate flux

        Returns
        -------
        np.ndarray
            Flux model at the provided times
        """
        # Best-fit magnification at provided times
        mag_times = self.best_fit_magnification_at_times(telescope, times)

        # Fit fs, fb on the telescope's actual data
        flux_data, err_flux = self._get_data_for_telescope(telescope)
        mag_data = self.get_best_fit_model(telescope)

        X = np.vstack([np.asarray(mag_data), np.ones_like(mag_data)]).T
        w = 1.0 / (np.asarray(err_flux) ** 2)
        XT_W = X.T * w
        A = XT_W @ X
        b = XT_W @ np.asarray(flux_data)
        try:
            theta = np.linalg.solve(A, b) if np.linalg.cond(A) < 1e12 else np.array([0.0, 0.0])
        except np.linalg.LinAlgError:
            theta = np.array([0.0, 0.0])
        fs, fb = float(theta[0]), float(theta[1])

        return fs * np.asarray(mag_times) + fb

    def predictive_samples(self, telescope, num_samples: int = 100) -> np.ndarray:
        """Generate posterior predictive samples."""
        if self.results is None:
            raise RuntimeError("Nested sampling not yet run. Call run_nested_sampling() first.")

        samples, weights = self.get_weighted_samples()

        # Draw sample indices weighted by importance weights
        n_total = len(list(samples.values())[0])
        sample_indices = np.random.choice(n_total, size=num_samples, p=weights)

        # Observed data for fitting flux parameters
        flux_data, err_flux = self._get_data_for_telescope(telescope)
        flux_data = np.asarray(flux_data)
        err_flux = np.asarray(err_flux)

        predictions = []

        for idx in sample_indices:
            param_dict = {}
            for pname in self._prior_names:
                param_dict[pname] = float(samples[pname][idx])

            param_vector = []
            for model_param in self.model.model_dictionnary.keys():
                param_vector.append(param_dict.get(model_param, None))

            pyLIMA_params = self.model.compute_pyLIMA_parameters(param_vector)
            mag = np.asarray(self.model.model_magnification(telescope, pyLIMA_params))

            # Weighted least squares to infer fsource/fblend
            X = np.vstack([mag, np.ones_like(mag)]).T
            w = 1.0 / (err_flux ** 2)
            XT_W = X.T * w
            A = XT_W @ X
            b = XT_W @ flux_data

            try:
                if np.linalg.cond(A) < 1e12:
                    theta = np.linalg.solve(A, b)
                else:
                    theta = np.array([0.0, 0.0])
            except np.linalg.LinAlgError:
                theta = np.array([0.0, 0.0])

            fs = float(theta[0])
            fb = float(theta[1])
            flux = fs * mag + fb

            predictions.append(np.asarray(flux))

        return np.array(predictions)


# Convenience function
def fit_pspl_nested_sampling(
    event,
    model=None,
    priors: Optional[Dict] = None,
    num_live_points: int = 500,
    seed: int = 0
) -> PSPLInferenceNS:
    """
    Quick interface to run PSPL nested sampling with default settings.

    Parameters
    ----------
    event : Event
        pyLIMA event
    model : MLmodel, optional
        If None, creates PSPLmodel(event)
    priors : dict, optional
        Prior bounds. If None, uses reasonable defaults from data.
    num_live_points : int
        Number of live points for nested sampling
    seed : int
        Random seed

    Returns
    -------
    PSPLInferenceNS
        Fitted inference object with samples
    """
    from pyLIMA.models import PSPL_model
    from pyLIMA.priors import guess

    if model is None:
        model = PSPL_model.PSPLmodel(event)

    guess_dict, fs = guess.initial_guess_PSPL(event)

    if priors is None:
        priors = {
            't0': (guess_dict['t0'] - 50, guess_dict['t0'] + 50),
            'u0': (max(0.01, guess_dict['u0'] - 1), guess_dict['u0'] + 1),
            'tE': (max(1, guess_dict['tE'] - 100), guess_dict['tE'] + 100),
        }

    inf = PSPLInferenceNS(event, model)
    inf.set_priors(priors)
    inf.run_nested_sampling(num_live_points=num_live_points, seed=seed)

    return inf
