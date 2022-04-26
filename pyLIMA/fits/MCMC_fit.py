import scipy
import time as python_time
import numpy as np
import sys
import emcee

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions


class MCMCfit(MLfit):

    def __init__(self, model, rescaling_photometry=False, MCMC_walkers=2, MCMC_links = 5000, telescopes_fluxes_method='MCMC'):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, rescaling_photometry)

        self.MCMC_walkers = MCMC_walkers #times number of dimension!
        self.MCMC_links = MCMC_links
        self.MCMC_chains = []
        self.telescopes_fluxes_method = telescopes_fluxes_method
        self.fit_time = 0 #s

    def fit_type(self):
        return "Monte Carlo Markov Chain (Affine Invariant)"

    def objective_function(self, fit_process_parameters):

        model_parameters = fit_process_parameters[:-len(self.model.event.telescopes)]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.rescaling_photometry:

            residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                   pyLIMA_parameters,
                                                                                                   norm=True,
                                                                                                   rescaling_photometry_parameters=fit_process_parameters[
                                                                                                                                   -len(
                                                                                                                                       self.model.event.telescopes):])
        else:

            residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                   pyLIMA_parameters,
                                                                                                   norm=True,
                                                                                                   rescaling_photometry_parameters=None)

        photometric_likelihood = -0.5*(np.sum(residus ** 2 + 2 * np.log(errflux)) + len(errflux) * 1.8378770664093453)
        # astrometric_residuals = pyLIMA.fits.residuals.all_telescope_astrometric_chi2(self.model.event, pyLIMA_parameters)

        return photometric_likelihood


    def fit(self, computational_pool=False):

        start = python_time.time()

        self.guess = self.initial_guess()

        if self.telescopes_fluxes_method != 'MCMC':
            limit_parameters = len(self.model.parameters_boundaries)
            best_solution = self.guess[:limit_parameters]
        else:
            limit_parameters = len(self.guess)
            best_solution = self.guess

        if self.rescaling_photometry:

            for telescope in self.model.event.telescopes:

                best_solution.append(0.0)

        number_of_parameters = len(best_solution)
        nwalkers = self.MCMC_walkers * number_of_parameters
        nlinks = self.MCMC_links

        # Initialize the population of MCMC
        population = best_solution+np.random.randn(nwalkers,number_of_parameters)

        if computational_pool:

            pool = computational_pool

        try:
            # create a new MPI pool
            from schwimmbad import MPIPool
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
        except:

            pass

        sampler = emcee.EnsembleSampler(nwalkers, number_of_parameters, self.objective_function,
                                        a=2.0, pool=pool)

        sampler.run_mcmc(population, nlinks, progress=True)

        mcmc_shape = list(sampler.get_chain().shape)
        mcmc_shape[-1] += 1
        MCMC_chains = np.zeros(mcmc_shape)
        MCMC_chains[:,:,:-1] = sampler.get_chain()
        MCMC_chains[:,:,-1] = sampler.get_log_prob()

        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        self.MCMC_chains = MCMC_chains

