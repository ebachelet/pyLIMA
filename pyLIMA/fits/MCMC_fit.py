import time as python_time
import numpy as np
import sys
import emcee

from pyLIMA.fits.ML_fit import MLfit
from multiprocessing import Manager

class MCMCfit(MLfit):

    def __init__(self, model, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', MCMC_walkers=2, MCMC_links = 5000):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry, telescopes_fluxes_method=telescopes_fluxes_method)


        self.MCMC_walkers = MCMC_walkers #times number of dimension!
        self.MCMC_links = MCMC_links

    def fit_type(self):
        return "Monte Carlo Markov Chain (Affine Invariant)"

    def objective_function(self, fit_process_parameters):

        objective = self.standard_objective_function(fit_process_parameters)

        return -objective

    def fit(self, initial_population=[], computational_pool=False):

        start_time = python_time.time()

        if initial_population == []:

            best_solution = self.initial_guess()

            number_of_parameters = len(best_solution)
            nwalkers = self.MCMC_walkers * number_of_parameters

            # Initialize the population of MCMC
            eps = 10**-1
            floors = np.floor(np.round(best_solution))
            initial = best_solution-floors
            mask = initial == 0
            initial[mask] = eps

            deltas = initial*np.random.uniform(-eps,eps,(nwalkers,number_of_parameters))
            population = best_solution+deltas

        else:

            population = initial_population[:,:-1]
            number_of_parameters = population.shape[1]

            nwalkers = self.MCMC_walkers * number_of_parameters

        nlinks = self.MCMC_links

        if computational_pool:

            pool = computational_pool

        else:
            pool = None

        try:
            # create a new MPI pool
            from schwimmbad import MPIPool
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
        except:

            pass

        if pool:

            with pool:

                sampler = emcee.EnsembleSampler(nwalkers, number_of_parameters, self.objective_function,
                                                a=2.0, pool=pool)

                sampler.run_mcmc(population, nlinks, progress=True)
        else:

            sampler = emcee.EnsembleSampler(nwalkers, number_of_parameters, self.objective_function,
                                            a=2.0, pool=pool)

            sampler.run_mcmc(population, nlinks, progress=True)

        computation_time = python_time.time() - start_time
        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')

        self.trials = np.array(self.trials)
        self.trials[:,-1] *= -1

        MCMC_chains = self.reconstruct_chains(sampler.get_chain(),sampler.get_log_prob())

        best_model_index = np.where(MCMC_chains[:, :, -1] == MCMC_chains[:, :, -1].max())
        fit_results = MCMC_chains[np.unique(best_model_index[0])[0], np.unique(best_model_index[1])[0], :-1]
        fit_log_likelihood = MCMC_chains[np.unique(best_model_index[0])[0], np.unique(best_model_index[1])[0], -1]

        print('best_model:', fit_results, 'ln(likelihood)', fit_log_likelihood)

        self.fit_results = {'best_model': fit_results, 'ln(likelihood)': fit_log_likelihood,
                            'MCMC_chains': MCMC_chains, 'fit_time': computation_time}


    def reconstruct_chains(self, mcmc_samples, mcmc_prob):

        if self.trials.shape[1] > mcmc_samples.shape[2]+1:

            rangei, rangej, rangek = mcmc_samples.shape

            unique_mcmc = np.unique(mcmc_prob,return_index=True,return_inverse=True,return_counts=True)
            unique_trials = np.unique(self.trials[:,-1],return_index=True,return_inverse=True,return_counts=True)

            trials_index = []

            for value in unique_mcmc[0]:

                indices = np.where(unique_trials[0] == value)[0][0]
                trials_index.append(indices)

            MCMC_chains = self.trials[unique_trials[1]][trials_index][unique_mcmc[2]].reshape(rangei, rangej, self.trials.shape[1])


        else:

            MCMC_chains = mcmc_samples

        return MCMC_chains

    def samples_to_plot(self):


        chains = self.fit_results['MCMC_chains']
        samples = chains.reshape(-1,chains.shape[2])
        samples_to_plot = samples[int(len(samples)/2):]

        return samples_to_plot