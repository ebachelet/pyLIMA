"""
NumPyro-based Bayesian inference for pyLIMA microlensing models.

Provides HMC/NUTS sampling for parameter estimation with full posterior inference,
uncertainties, and correlations.

Example:
--------
    from pyLIMA.inference.numpyro_inference import PSPLInference
    
    # Create inference object
    inf = PSPLInference(event, model)
    
    # Set priors
    inf.set_priors({'t0': (2459000, 2460000),
                    'u0': (0.01, 2.0),
                    'tE': (1, 200)})
    
    # Run sampling
    inf.run_mcmc(num_warmup=1000, num_samples=2000, seed=0)
    
    # Get results
    samples = inf.get_samples()
    inf.print_summary()
    inf.plot_posteriors()
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import arviz as az
import matplotlib.pyplot as plt


class PSPLInference:
    """Bayesian inference for PSPL microlensing models using NumPyro/NUTS."""
    
    def __init__(self, event, model):
        """
        Initialize inference object.
        
        Parameters
        ----------
        event : Event
            pyLIMA event object with telescopes and data
        model : MLmodel
            pyLIMA model (e.g., PSPLmodel, FSPLmodel)
        """
        self.event = event
        self.model = model
        self.priors = {}
        self.mcmc = None
        self.samples = None
        self._prior_names = []
        
    def set_priors(self, priors: Dict[str, Tuple[float, float]]) -> None:
        """
        Set uniform priors for parameters.
        
        Parameters
        ----------
        priors : dict
            Mapping of parameter_name -> (lower_bound, upper_bound)
            e.g. {'t0': (2459000, 2460000), 'u0': (0.01, 2.0), 'tE': (1, 200)}
        """
        self.priors = priors
        self._prior_names = list(priors.keys())
        print(f"Priors set for: {self._prior_names}")
        
    def _get_data_for_telescope(self, telescope) -> Tuple[np.ndarray, np.ndarray]:
        """Extract flux and error flux from telescope (handles both formats)."""
        try:
            # JaxTelescope format
            flux = np.asarray(telescope.lightcurve.flux)
            err_flux = np.asarray(telescope.lightcurve.err_flux)
        except (TypeError, AttributeError):
            # Legacy dict format
            flux = np.asarray(telescope.lightcurve['flux'].value)
            err_flux = np.asarray(telescope.lightcurve['err_flux'].value)
        
        return flux, err_flux
    
    def _likelihood_model(self):
        """NumPyro model defining priors and likelihood."""
        
        # Sample from priors
        params_dict = {}
        for param_name, (low, high) in self.priors.items():
            params_dict[param_name] = numpyro.sample(
                param_name,
                dist.Uniform(low, high)
            )
        
        # Build parameter vector following model ordering
        param_vector = []
        for model_param in self.model.model_dictionnary.keys():
            param_vector.append(params_dict.get(model_param, None))
        
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
            # Solve for [fsource, fblend] in flux = fsource * mag + fblend
            # Design matrix X = [mag, 1]
            X = jnp.vstack([mag, jnp.ones_like(mag)]).T
            w = 1.0 / (err_flux ** 2)
            # Normal equations: (X^T W X) theta = X^T W y
            XT_W = (X.T * w)
            A = XT_W @ X
            b = XT_W @ flux_data
            # Safe solve: if A is singular, fallback to zeros
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
        
        # Log-likelihood as Gaussian
        # numpyro.sample('obs', dist.Normal(0, 1), obs=jnp.sqrt(chi2_total))
        numpyro.factor('log_likelihood', -0.5 * chi2_total)
    
    def run_mcmc(self, num_warmup: int = 1000, num_samples: int = 2000,
                 num_chains: int = 1, seed: int = 0,
                 dense_mass: bool = False,
                 init_params: Optional[Dict[str, float]] = None) -> None:
        """
        Run HMC/NUTS sampling.
        
        Parameters
        ----------
        num_warmup : int
            Number of warmup (burn-in) iterations
        num_samples : int
            Number of post-warmup samples per chain
        num_chains : int
            Number of independent chains
        seed : int
            Random seed for reproducibility
        dense_mass : bool
            If True, use dense mass matrix (full covariance); else diagonal
        """
        print(f"Starting MCMC: {num_warmup} warmup, {num_samples} samples, {num_chains} chains")

        # If no init_params provided, construct sensible ones from data/prior midpoints
        if init_params is None:
            init_params = {}
            try:
                from pyLIMA.priors import guess as guess_mod
                guess_dict, _ = guess_mod.initial_guess_PSPL(self.event)
            except Exception:
                guess_dict = {}

            for pname, (low, high) in self.priors.items():
                if pname in guess_dict:
                    init_val = float(guess_dict[pname])
                else:
                    init_val = float(0.5 * (low + high))
                init_params[pname] = init_val

        # If the user provided init_params explicitly, use an init strategy
        # that sets parameter values to those provided (init_to_value). This
        # helps NumPyro find valid initial params for complex models.
        init_strategy = None
        if init_params is not None:
            try:
                init_strategy = init_to_value(init_params)
            except Exception:
                init_strategy = None

        # Create kernel (after we possibly defined init_strategy)
        if init_strategy is not None:
            kernel = NUTS(self._likelihood_model, dense_mass=dense_mass, init_strategy=init_strategy)
        else:
            kernel = NUTS(self._likelihood_model, dense_mass=dense_mass)

        # Instantiate MCMC object
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains
        )

        # Run MCMC with provided or constructed initial parameters
        # If we passed an init_strategy, `init_params` to .run is not necessary,
        # but passing it is harmless; keep for compatibility.
        self.mcmc.run(jax.random.PRNGKey(seed), init_params=init_params)
        
        print("MCMC completed successfully!")
        
    def get_samples(self) -> Dict[str, np.ndarray]:
        """Get posterior samples as dictionary."""
        if self.mcmc is None:
            raise RuntimeError("MCMC not yet run. Call run_mcmc() first.")
        
        self.samples = self.mcmc.get_samples()
        return self.samples
    
    def print_summary(self) -> None:
        """Print posterior summary statistics."""
        if self.mcmc is None:
            raise RuntimeError("MCMC not yet run. Call run_mcmc() first.")
        
        self.mcmc.print_summary()
    
    def get_posterior_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute posterior statistics (mean, std, percentiles).
        
        Returns
        -------
        dict
            Mapping param_name -> {'mean': ..., 'std': ..., 'q16': ..., 'q84': ...}
        """
        if self.samples is None:
            self.get_samples()
        
        stats = {}
        for param_name in self._prior_names:
            if param_name not in self.samples:
                continue
            
            samples_1d = self.samples[param_name].flatten()
            stats[param_name] = {
                'mean': float(np.mean(samples_1d)),
                'std': float(np.std(samples_1d)),
                'median': float(np.median(samples_1d)),
                'q16': float(np.percentile(samples_1d, 16)),
                'q84': float(np.percentile(samples_1d, 84)),
            }
        
        return stats
    
    def best_fit_dict(self) -> Dict[str, float]:
        """Get best-fit (median posterior) parameters as dict."""
        stats = self.get_posterior_stats()
        return {name: stats[name]['median'] for name in stats.keys()}
    
    def plot_posteriors(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot 1D posterior distributions and 2D corner plots.
        
        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.samples is None:
            self.get_samples()
        
        # Convert to arviz InferenceData for easy plotting
        idata = az.from_numpyro(self.mcmc)
        
        # Plot posteriors
        fig = plt.figure(figsize=figsize)
        az.plot_posterior(idata, figsize=figsize)
        plt.tight_layout()
        
        return fig
    
    def plot_trace(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot trace plots (MCMC chains over iterations).
        
        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.samples is None:
            self.get_samples()
        
        idata = az.from_numpyro(self.mcmc)
        fig = az.plot_trace(idata, figsize=figsize)
        plt.tight_layout()
        
        return fig
    
    def get_best_fit_model(self, telescope) -> np.ndarray:
        """
        Compute model prediction at best-fit parameters.
        
        Parameters
        ----------
        telescope : Telescope
            Telescope for which to compute model
        
        Returns
        -------
        np.ndarray
            Model magnification at best-fit
        """
        best_fit = self.best_fit_dict()
        
        # Build parameter vector
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
            # Attribute-style
            restore_attr = getattr(telescope.lightcurve, 'time', None)
            telescope.lightcurve.time = np.asarray(times)
            mag = self.model.model_magnification(telescope, pyLIMA_params)
        except Exception:
            # Dict-style
            try:
                restore_dict = telescope.lightcurve['time'].value
                telescope.lightcurve['time'].value = np.asarray(times)
                mag = self.model.model_magnification(telescope, pyLIMA_params)
            finally:
                if restore_dict is not None:
                    telescope.lightcurve['time'].value = restore_dict
        else:
            # Restore attribute
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
        """
        Generate posterior predictive samples (model predictions sampled from posterior).
        
        Parameters
        ----------
        telescope : Telescope
            Telescope for which to compute predictions
        num_samples : int
            Number of posterior samples to draw
        
        Returns
        -------
        np.ndarray
            Shape (num_samples, num_datapoints) of flux predictions
        """
        if self.samples is None:
            self.get_samples()
        
        # Draw random sample indices from posterior
        n_total = len(list(self.samples.values())[0].flatten())
        sample_indices = np.random.choice(n_total, size=num_samples, replace=False)
        
        # Get observed data for flux parameter fitting
        flux_data, err_flux = self._get_data_for_telescope(telescope)
        flux_data = np.asarray(flux_data)
        err_flux = np.asarray(err_flux)
        
        predictions = []
        
        for idx in sample_indices:
            # Extract parameter values at this sample
            param_dict = {}
            for param_name in self._prior_names:
                flat_samples = self.samples[param_name].flatten()
                param_dict[param_name] = float(flat_samples[idx])
            
            # Build parameter vector
            param_vector = []
            for model_param in self.model.model_dictionnary.keys():
                param_vector.append(param_dict.get(model_param, None))
            
            pyLIMA_params = self.model.compute_pyLIMA_parameters(param_vector)
            
            # Get magnification
            mag = np.asarray(self.model.model_magnification(telescope, pyLIMA_params))
            
            # Compute fsource and fblend by weighted least squares (same as in likelihood)
            # Solve for [fsource, fblend] in flux = fsource * mag + fblend
            X = np.vstack([mag, np.ones_like(mag)]).T
            w = 1.0 / (err_flux ** 2)
            XT_W = X.T * w  # Broadcast weights across rows
            A = XT_W @ X
            b = XT_W @ flux_data
            
            # Solve with error handling
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


# Convenience function for quick inference
def fit_pspl_mcmc(event, model=None, priors: Optional[Dict] = None,
                  num_warmup: int = 1000, num_samples: int = 2000,
                  seed: int = 0) -> PSPLInference:
    """
    Quick interface to run PSPL inference with default settings.
    
    Parameters
    ----------
    event : Event
        pyLIMA event
    model : MLmodel, optional
        If None, creates PSPLmodel(event)
    priors : dict, optional
        Prior bounds. If None, uses reasonable defaults from data.
    num_warmup : int
        Number of warmup iterations
    num_samples : int
        Number of posterior samples
    seed : int
        Random seed
    
    Returns
    -------
    PSPLInference
        Fitted inference object with samples
    """
    from pyLIMA.models import PSPL_model
    from pyLIMA.priors import guess
    
    if model is None:
        model = PSPL_model.PSPLmodel(event)
    
    # Get initial guess for priors
    guess_dict, fs = guess.initial_guess_PSPL(event)
    
    if priors is None:
        # Use reasonable bounds around initial guess
        priors = {
            't0': (guess_dict['t0'] - 50, guess_dict['t0'] + 50),
            'u0': (max(0.01, guess_dict['u0'] - 1), guess_dict['u0'] + 1),
            'tE': (max(1, guess_dict['tE'] - 100), guess_dict['tE'] + 100),
        }
    
    # Run inference
    inf = PSPLInference(event, model)
    inf.set_priors(priors)
    inf.run_mcmc(num_warmup=num_warmup, num_samples=num_samples, seed=seed)
    
    return inf
