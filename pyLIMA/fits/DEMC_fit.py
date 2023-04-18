import time as python_time
import numpy as np
import sys
import emcee

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions
from pyLIMA.outputs import pyLIMA_plots

class DEMCfit(MLfit):

    def __init__(self, model, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', DEMC_walkers=2, DEMC_links = 5000):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry, telescopes_fluxes_method=telescopes_fluxes_method)

        self.DEMC_walkers = DEMC_walkers #times number of dimension!
        self.DEMC_links = DEMC_links
        self.DEMC_chains = []

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


        priors = self.get_priors()

        for ind, prior_pdf in enumerate(priors):

            if prior_pdf is not None:

                probability = prior_pdf.pdf(fit_process_parameters[ind])

                if probability > 0:

                    likelihood += -np.log(probability)

                else:

                    likelihood = np.inf

        return -likelihood


    def fit(self, initial_population=[], computational_pool=False):

        start_time = python_time.time()

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
                    individual[ind] = individual[ind] * (self.fit_parameters[j][1][1] - self.fit_parameters[j][1][0]) + \
                                      self.fit_parameters[j][1][0]

                individual = np.array(individual)
                individual = np.r_[individual]

                objective = self.objective_function(individual)
                individual = np.r_[individual, objective]

                initial_population.append(individual)

        population = np.array(initial_population)[:,:-1]



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
        except:

            pass

        if pool:

            with pool:

                sampler = emcee.EnsembleSampler(nwalkers, number_of_parameters, self.objective_function,
                                                a=2.0, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
                                                pool=pool)

                sampler.run_mcmc(population, nlinks, progress=True)
        else:

            sampler = emcee.EnsembleSampler(nwalkers, number_of_parameters, self.objective_function,
                                            a=2.0, pool=pool)

            sampler.run_mcmc(population, nlinks, progress=True)

        computation_time = python_time.time() - start_time

        mcmc_shape = list(sampler.get_chain().shape)
        mcmc_shape[-1] += 1
        DEMC_chains = np.zeros(mcmc_shape)
        DEMC_chains[:, :, :-1] = sampler.get_chain()
        DEMC_chains[:, :, -1] = sampler.get_log_prob()

        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')

        best_model_index = np.where(DEMC_chains[:, :, -1] == DEMC_chains[:, :, -1].max())
        fit_results = DEMC_chains[np.unique(best_model_index[0])[0], np.unique(best_model_index[1])[0], :-1]
        fit_log_likelihood = DEMC_chains[np.unique(best_model_index[0])[0], np.unique(best_model_index[1])[0], -1]

        print('best_model:', fit_results, 'ln(likelihood)', fit_log_likelihood)

        self.fit_results = {'best_model': fit_results, 'ln(likelihood)': fit_log_likelihood,
                            'DEMC_chains': DEMC_chains, 'fit_time': computation_time}


    def fit_outputs(self):

        pyLIMA_plots.plot_lightcurves(self.model, self.fit_results['best_model'])
        pyLIMA_plots.plot_geometry(self.model, self.fit_results['best_model'])

        parameters = [key for key in self.model.model_dictionnary.keys() if ('source' not in key) and ('blend' not in key)]

        chains = self.fit_results['MCMC_chains']
        samples = chains.reshape(-1,chains.shape[2])
        samples_to_plot = samples[int(len(samples)/2):,:len(parameters)]
        pyLIMA_plots.plot_distribution(samples_to_plot,parameters_names = parameters )