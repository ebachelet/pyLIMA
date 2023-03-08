import time as python_time
import numpy as np
import sys
import emcee

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions
from pyLIMA.outputs import pyLIMA_plots

class MCMCfit(MLfit):

    def __init__(self, model, fancy_parameters=False, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', MCMC_walkers=2, MCMC_links = 5000):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, fancy_parameters=fancy_parameters, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry, telescopes_fluxes_method=telescopes_fluxes_method)

        self.MCMC_walkers = MCMC_walkers #times number of dimension!
        self.MCMC_links = MCMC_links
        self.MCMC_chains = []

    def fit_type(self):
        return "Monte Carlo Markov Chain (Affine Invariant)"

    def objective_function(self, fit_process_parameters):

        likelihood = 0

        for ind, parameter in enumerate(self.fit_parameters.keys()):

            if (fit_process_parameters[ind]<self.fit_parameters[parameter][1][0]) | (fit_process_parameters[ind]>self.fit_parameters[parameter][1][1]):

                return -np.inf

        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.model.photometry:

            if self.rescale_photometry:

                rescaling_photometry_parameters = 10 ** (
                fit_process_parameters[self.rescale_photometry_parameters_index])

                photometric_likelihood = pyLIMA.fits.objective_functions.all_telescope_photometric_likelihood(self.model,
                                                                                                              pyLIMA_parameters,
                                                                                                              rescaling_photometry_parameters= rescaling_photometry_parameters)
            else:

                photometric_likelihood = pyLIMA.fits.objective_functions.all_telescope_photometric_likelihood(self.model,
                                                                                                              pyLIMA_parameters)

            likelihood += photometric_likelihood

        if self.model.astrometry:

            if self.rescale_astrometry:

                rescaling_astrometry_parameters = 10 ** (
                fit_process_parameters[self.rescale_astrometry_parameters_index])

                residuals = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                pyLIMA_parameters,
                                                                                                norm=True,
                                                                                                rescaling_astrometry_parameters=rescaling_astrometry_parameters)

            else:

                residuals = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                pyLIMA_parameters,
                                                                                                norm=True,
                                                                                                rescaling_astrometry_parameters=None)

            residus = np.r_[residuals[:, 0], residuals[:, 2]]  # res_ra,res_dec
            errors = np.r_[residuals[:, 1], residuals[:, 3]]  # err_res_ra,err_res_dec

            astrometric_likelihood = 0.5 * np.sum(residus ** 2 + 2 * np.log(errors) + np.log(2 * np.pi))

            likelihood += astrometric_likelihood

        # Priors
        if np.isnan(likelihood):
            return -np.inf

        return -likelihood


    def fit(self, initial_population=[], computational_pool=False):

        start_time = python_time.time()

        if initial_population == []:

            best_solution = self.initial_guess()

            number_of_parameters = len(best_solution)
            nwalkers = self.MCMC_walkers * number_of_parameters
            nlinks = self.MCMC_links

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

        mcmc_shape = list(sampler.get_chain().shape)
        mcmc_shape[-1] += 1
        MCMC_chains = np.zeros(mcmc_shape)
        MCMC_chains[:, :, :-1] = sampler.get_chain()
        MCMC_chains[:, :, -1] = sampler.get_log_prob()

        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')

        best_model_index = np.where(MCMC_chains[:, :, -1] == MCMC_chains[:, :, -1].max())
        fit_results = MCMC_chains[np.unique(best_model_index[0])[0], np.unique(best_model_index[1])[0], :-1]
        fit_log_likelihood = MCMC_chains[np.unique(best_model_index[0])[0], np.unique(best_model_index[1])[0], -1]

        print('best_model:', fit_results, 'ln(likelihood)', fit_log_likelihood)

        self.fit_results = {'best_model': fit_results, 'ln(likelihood)': fit_log_likelihood,
                            'MCMC_chains': MCMC_chains, 'fit_time': computation_time}


    def fit_outputs(self):

        pyLIMA_plots.plot_lightcurves(self.model, self.fit_results['best_model'])
        pyLIMA_plots.plot_geometry(self.model, self.fit_results['best_model'])

        parameters = [key for key in self.model.model_dictionnary.keys() if ('source' not in key) and ('blend' not in key)]

        chains = self.fit_results['MCMC_chains']
        samples = chains.reshape(-1,chains.shape[2])
        samples_to_plot = samples[int(len(samples)/2):,:len(parameters)]
        pyLIMA_plots.plot_distribution(samples_to_plot,parameters_names = parameters )