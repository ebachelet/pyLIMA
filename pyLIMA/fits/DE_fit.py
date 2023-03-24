import scipy
import time as python_time
import numpy as np
import sys
from multiprocessing import Manager



from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions
from pyLIMA.outputs import pyLIMA_plots


class DEfit(MLfit):

    def __init__(self, model, fancy_parameters=False, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', DE_population_size=10, max_iteration=10000,
                 display_progress=False):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, fancy_parameters=fancy_parameters, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry, telescopes_fluxes_method=telescopes_fluxes_method)

        self.population = Manager().list() # to be recognize by all process during parallelization
        self.DE_population_size = DE_population_size #Times number of dimensions!
        self.fit_boundaries = []
        self.max_iteration = max_iteration
        self.fit_time = 0 #s
        self.display_progress = display_progress

    def fit_type(self):
        return "Differential Evolution"

    def objective_function(self, fit_process_parameters):

        likelihood = 0
        #print(fit_process_parameters)
        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.model.photometry:

            if self.rescale_photometry:

                rescaling_photometry_parameters = 10 ** (
                    fit_process_parameters[self.rescale_photometry_parameters_index])

                photometric_likelihood = pyLIMA.fits.objective_functions.all_telescope_photometric_likelihood(
                    self.model,
                    pyLIMA_parameters,
                    rescaling_photometry_parameters=rescaling_photometry_parameters)
            else:

                photometric_likelihood = pyLIMA.fits.objective_functions.all_telescope_photometric_likelihood(
                    self.model,
                    pyLIMA_parameters)

            likelihood += photometric_likelihood

        if self.model.astrometry:

            if self.rescale_astrometry:


                rescaling_astrometry_parameters = 10**(fit_process_parameters[self.rescale_astrometry_parameters_index])

                residuals = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                      pyLIMA_parameters,
                                                                                                      norm=True,
                                                                                                      rescaling_astrometry_parameters= rescaling_astrometry_parameters)

            else:

                residuals = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                   pyLIMA_parameters,
                                                                                                   norm=True,
                                                                                                   rescaling_astrometry_parameters=None)

            residus = np.r_[residuals[:,0],residuals[:,2]] #res_ra,res_dec
            errors = np.r_[residuals[:,1],residuals[:,3]] #err_res_ra,err_res_dec


            astrometric_likelihood = 0.5 * (np.sum(residus ** 2 + np.log(2 * np.pi * errors ** 2)))

            likelihood += astrometric_likelihood

        # Priors
        priors = self.get_priors()

        for ind, prior_pdf in enumerate(priors):

            if prior_pdf is not None:

                probability = prior_pdf.pdf(fit_process_parameters[ind])

                if probability > 0:

                    likelihood += -np.log(probability)

                else:

                    likelihood = np.inf

        self.population.append(fit_process_parameters.tolist() + [likelihood])

        return likelihood

    def fit(self, initial_population=[], computational_pool=None):

        start_time = python_time.time()

        if computational_pool:

            worker = computational_pool.map

        else:

            worker = 1


        if initial_population == []:

            init = 'latinhypercube'

        else:

            init = initial_population

        bounds = [self.fit_parameters[key][1] for key in self.fit_parameters.keys()]

        differential_evolution_estimation = scipy.optimize.differential_evolution(self.objective_function,
                                                                                  bounds=bounds,
                                                                                  mutation=(0.5, 1.5), popsize=int(self.DE_population_size),
                                                                                  maxiter=self.max_iteration, tol=0.00,
                                                                                  atol=1.0, strategy='rand1bin',
                                                                                  recombination=0.5, polish=False, init=init,
                                                                                  disp=self.display_progress, workers=worker)

        print('DE converge to objective function : f(x) = ', str(differential_evolution_estimation['fun']))
        print('DE converge to parameters : = ', differential_evolution_estimation['x'].astype(str))

        fit_results = differential_evolution_estimation['x']
        fit_log_likelihood = differential_evolution_estimation['fun']

        computation_time = python_time.time() - start_time
        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')

        DE_population = np.array(self.population)

        print('best_model:', fit_results, '-ln(likelihood)', fit_log_likelihood)

        self.fit_results = {'best_model': fit_results, '-(ln_likelihood)' : fit_log_likelihood, 'fit_time': computation_time,
                            'DE_population': DE_population, 'fit_time' : computation_time}

    def fit_outputs(self):

        pyLIMA_plots.plot_lightcurves(self.model, self.fit_results['best_model'])
        pyLIMA_plots.plot_geometry(self.model, self.fit_results['best_model'])

        parameters = [key for key in self.model.model_dictionnary.keys() if ('source' not in key) and ('blend' not in key)]


        samples_to_plot = self.fit_results['DE_population'][:,:len(parameters)]
        pyLIMA_plots.plot_distribution(samples_to_plot,parameters_names = parameters )