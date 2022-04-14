import scipy
import time as python_time
import numpy as np
import sys
import emcee

from pyLIMA.fits.fit import MLfit
import pyLIMA.fits.residuals


class MCMCfit(MLfit):

    def __init__(self, event, model, MCMC_walkers=2, MCMC_links = 5000, telescopes_fluxes_method='MCMC'):
        """The fit class has to be intialized with an event object."""

        self.event = event
        self.model = model
        self.MCMC_walkers = MCMC_walkers #times number of dimension!
        self.MCMC_links = MCMC_links
        self.MCMC_chains = []
        self.telescopes_fluxes_method = telescopes_fluxes_method
        self.fit_time = 0 #s

    def fit_type(self):
        return "Monte Carlo Markov Chain (Affine Invariant)"

    def objective_function(self, fit_process_parameters):

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)

        photometric_chi2 = pyLIMA.fits.residuals.all_telescope_photometric_chi2(self.event, self.model,
                                                                                          pyLIMA_parameters)

        #astrometric_residuals = pyLIMA.fits.residuals.all_telescope_astrometric_chi2(self.event, pyLIMA_parameters)

        return -0.5*photometric_chi2

    def fit(self,pool=None):

        start = python_time.time()

        self.guess = self.initial_guess()



        if self.telescopes_fluxes_method != 'MCMC':
            limit_parameters = len(self.model.parameters_boundaries)
            best_solution = self.guess[:limit_parameters]
        else:
            limit_parameters = len(self.guess)
            best_solution = self.guess

        number_of_parameters = len(best_solution)
        nwalkers = self.MCMC_walkers * number_of_parameters
        nlinks = self.MCMC_links

        # Initialize the population of MCMC
        population = best_solution+np.random.randn(nwalkers,number_of_parameters)


        try:
            # create a new MPI pool
            from schwimmbad import MPIPool
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
        except:

            pass

        sampler = emcee.EnsembleSampler(nwalkers,number_of_parameters , self.objective_function,
                                        a=2.0, pool=pool)

        sampler.run_mcmc(population, nlinks, progress=True)

        mcmc_shape = list(sampler.get_chain().shape)
        mcmc_shape[-1] += 1
        MCMC_chains = np.zeros(mcmc_shape)
        MCMC_chains[:,:,:-1] = sampler.get_chain()
        MCMC_chains[:,:,-1] = sampler.get_log_prob()

        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        self.MCMC_chains = MCMC_chains

