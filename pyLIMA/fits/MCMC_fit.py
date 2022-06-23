import time as python_time
import numpy as np
import sys
import emcee

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions

class MCMCfit(MLfit):

    def __init__(self, model, fancy_parameters=False, rescale_photometry=False, rescale_astrometry=False, telescopes_fluxes_method='fit', MCMC_walkers=2, MCMC_links = 5000):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, fancy_parameters=fancy_parameters, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry, telescopes_fluxes_method='polyfit')

        self.MCMC_walkers = MCMC_walkers #times number of dimension!
        self.MCMC_links = MCMC_links
        self.MCMC_chains = []
        self.telescopes_fluxes_method = telescopes_fluxes_method
        self.fit_time = 0 #s

    def fit_type(self):
        return "Monte Carlo Markov Chain (Affine Invariant)"

    def objective_function(self, fit_process_parameters):

        likelihood = 0

        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.model.photometry:

            if self.rescale_photometry:

                rescaling_photometry_parameters = 10**(fit_process_parameters[self.rescale_photometry_parameters_index])

                residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                       pyLIMA_parameters,
                                                                                                       norm=True,
                                                                                                       rescaling_photometry_parameters=rescaling_photometry_parameters)
            else:

                residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                       pyLIMA_parameters,
                                                                                                       norm=True,
                                                                                                       rescaling_photometry_parameters=None)

            photometric_likelihood = 0.5 * (np.sum(residus ** 2 + np.log(2 * np.pi * errflux ** 2)))

            likelihood += photometric_likelihood

        if self.model.astrometry:
            residus, errors = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                  pyLIMA_parameters,
                                                                                                  norm=True,
                                                                                                  rescaling_astrometry_parameters=None)

            astrometric_likelihood = 0.5 * (np.sum(residus ** 2 + np.log(2 * np.pi * errors ** 2)))

            likelihood += astrometric_likelihood

        return -likelihood


    def fit(self, computational_pool=False):

        start = python_time.time()

        best_solution = self.initial_guess()

        number_of_parameters = len(best_solution)
        nwalkers = self.MCMC_walkers * number_of_parameters
        nlinks = self.MCMC_links

        # Initialize the population of MCMC
        population = best_solution*np.random.uniform(0.999999, 1.000001, (nwalkers,number_of_parameters))+np.random.uniform(-1, 1,(nwalkers,number_of_parameters))*10**-4


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

        mcmc_shape = list(sampler.get_chain().shape)
        mcmc_shape[-1] += 1
        MCMC_chains = np.zeros(mcmc_shape)
        MCMC_chains[:,:,:-1] = sampler.get_chain()
        MCMC_chains[:,:,-1] = sampler.get_log_prob()

        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        self.MCMC_chains = MCMC_chains

