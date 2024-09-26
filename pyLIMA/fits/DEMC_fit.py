import sys
import time as python_time

import emcee
import numpy as np
from pyLIMA.fits.ML_fit import MLfit
from pyLIMA.priors import parameters_priors


class DEMCfit(MLfit):
    """
    Under Construction
    """
    def __init__(self, model, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', loss_function='likelihood',
                 DEMC_walkers=2, DEMC_links=5000):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry,
                         telescopes_fluxes_method=telescopes_fluxes_method,
                         loss_function=loss_function)

        self.DEMC_walkers = DEMC_walkers  # times number of dimension!
        self.DEMC_links = DEMC_links
        self.DEMC_chains = []
        self.priors = parameters_priors.default_parameters_priors(self.fit_parameters)

    def fit_type(self):
        return "Monte Carlo Markov Chain (Affine Invariant)"

    def objective_function(self, fit_process_parameters):

        likelihood = -self.model_likelihood(fit_process_parameters)

        # Priors
        priors = self.get_priors(fit_process_parameters)

        likelihood += -priors

        return likelihood

    def fit(self, initial_population=[], computational_pool=False):

        start_time = python_time.time()
        # Safety, recompute in case user changes boundaries after init
        self.priors = parameters_priors.default_parameters_priors(self.fit_parameters)

        number_of_parameters = len(self.fit_parameters)
        nwalkers = self.DEMC_walkers * number_of_parameters

        if initial_population == []:

            import scipy.stats as ss
            sampler = ss.qmc.LatinHypercube(d=len(self.fit_parameters))
            # self.betas = np.logspace(-3, 0, number_of_walkers)
            # self.betas = np.linspace(0,1, number_of_walkers)

            for i in range(nwalkers):

                individual = sampler.random(n=1)[0]

                for ind, j in enumerate(self.fit_parameters.keys()):
                    individual[ind] = individual[ind] * (self.fit_parameters[j][1][1] -
                                                         self.fit_parameters[j][1][
                                                             0]) + \
                                      self.fit_parameters[j][1][0]

                individual = np.array(individual)
                individual = np.r_[individual]

                objective = self.objective_function(individual)
                individual = np.r_[individual, objective]

                initial_population.append(individual)

        population = np.array(initial_population)[:, :-1]

        nlinks = self.DEMC_links

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

        except ModuleNotFoundError:

            pass

        if pool:

            with pool:

                sampler = emcee.EnsembleSampler(nwalkers, number_of_parameters,
                                                self.objective_function,
                                                a=2.0,
                                                moves=[(emcee.moves.DEMove(), 0.8), (
                                                    emcee.moves.DESnookerMove(), 0.2)],
                                                pool=pool)

                sampler.run_mcmc(population, nlinks, progress=True)
        else:

            sampler = emcee.EnsembleSampler(nwalkers, number_of_parameters,
                                            self.objective_function,
                                            a=2.0, pool=pool)

            sampler.run_mcmc(population, nlinks, progress=True)

        computation_time = python_time.time() - start_time

        mcmc_shape = list(sampler.get_chain().shape)
        mcmc_shape[-1] += 1
        DEMC_chains = np.zeros(mcmc_shape)
        DEMC_chains[:, :, :-1] = sampler.get_chain()
        DEMC_chains[:, :, -1] = sampler.get_log_prob()

        print(sys._getframe().f_code.co_name, ' : ' + self.fit_type() + ' fit SUCCESS')

        best_model_index = np.where(
            DEMC_chains[:, :, -1] == DEMC_chains[:, :, -1].max())
        fit_results = DEMC_chains[np.unique(best_model_index[0])[0],
                      np.unique(best_model_index[1])[0], :-1]
        fit_log_likelihood = DEMC_chains[
            np.unique(best_model_index[0])[0], np.unique(best_model_index[1])[0], -1]

        print('best_model:', fit_results, self.loss_function, fit_log_likelihood)

        self.fit_results = {'best_model': fit_results,
                            self.loss_function: fit_log_likelihood,
                            'DEMC_chains': DEMC_chains, 'fit_time': computation_time}

    def fit_outputs(self):
        from pyLIMA.outputs import pyLIMA_plots
        pyLIMA_plots.plot_lightcurves(self.model, self.fit_results['best_model'])
        pyLIMA_plots.plot_geometry(self.model, self.fit_results['best_model'])

        parameters = [key for key in self.model.model_dictionnary.keys() if
                      ('source' not in key) and ('blend' not in key)]

        chains = self.fit_results['MCMC_chains']
        samples = chains.reshape(-1, chains.shape[2])
        samples_to_plot = samples[int(len(samples) / 2):, :len(parameters)]
        pyLIMA_plots.plot_distribution(samples_to_plot, parameters_names=parameters)
