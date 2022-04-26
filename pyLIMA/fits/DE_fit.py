import scipy
import time as python_time
import numpy as np
import sys
from multiprocessing import Process, Manager

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions
from pyLIMA.priors import parameters_boundaries

class DEfit(MLfit):

    def __init__(self, model, rescaling_photometry=False, DE_population_size=10, max_iteration=10000, telescopes_fluxes_method='DE'):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, rescaling_photometry)

        self.DE_population = Manager().list() # to be recognize by all process during parallelization
        self.DE_population_size = DE_population_size
        self.fit_boundaries = []
        self.max_iteration = max_iteration
        self.telescopes_fluxes_method = telescopes_fluxes_method
        self.fit_time = 0 #s

    def fit_type(self):
        return "Differential Evolution"

    def objective_function(self, fit_process_parameters):

        likelihood = 0
        model_parameters = fit_process_parameters[:-len(self.model.event.telescopes)]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.model.photometry:

            if self.rescaling_photometry:

                residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                       pyLIMA_parameters,
                                                                                                       norm=True,
                                                                                                       rescaling_photometry_parameters=fit_process_parameters[
                                                                                                                                       -len(self.model.event.telescopes):])
            else:

                residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                       pyLIMA_parameters,
                                                                                                       norm=True,
                                                                                                       rescaling_photometry_parameters=None)

            photometric_likelihood = 0.5*(np.sum(residus ** 2 + 2 * np.log(errflux)) + len(errflux)*1.8378770664093453)

            likelihood += photometric_likelihood

        if self.model.astrometry:

            residus, errors= pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                   pyLIMA_parameters,
                                                                                                   norm=True,
                                                                                                   rescaling_astrometry_parameters=None)

            astrometric_likelihood = 0.5*(np.sum(residus ** 2 + 2 * np.log(errors)) + len(errors)*1.8378770664093453)

            likelihood += astrometric_likelihood
        self.DE_population.append(fit_process_parameters.tolist() + [likelihood])
        return likelihood

    def fit(self, computational_pool=None):

        starting_time = python_time.time()

        if computational_pool:

            worker = computational_pool.map

        else:

            worker = 1

        bounds = self.model.parameters_boundaries

        if self.telescopes_fluxes_method == 'DE':

            bounds += parameters_boundaries.telescopes_fluxes_boundaries(self.model)

        if self.rescaling_photometry:

            bounds += parameters_boundaries.rescaling_boundaries(self.model)

        self.fit_boundaries = bounds

        n_data = 0
        for telescope in self.model.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        differential_evolution_estimation = scipy.optimize.differential_evolution(self.objective_function,
                                                                                  bounds=bounds,
                                                                                  mutation=(0.1,1.9), popsize=int(self.DE_population_size),
                                                                                  maxiter=self.max_iteration, tol=0.0001,
                                                                                  atol=0.000, strategy='rand1bin',
                                                                                  recombination=0.25, polish=True, init='latinhypercube',
                                                                                  disp=True, workers=worker)


        # paczynski_parameters are all parameters to compute the model, excepted the telescopes fluxes.
        paczynski_parameters = differential_evolution_estimation['x'].tolist()

        print('DE converge to objective function : f(x) = ', str(differential_evolution_estimation['fun']))
        print('DE converge to parameters : = ', differential_evolution_estimation['x'].astype(str))

        fit_results = np.hstack((differential_evolution_estimation['x'],differential_evolution_estimation['fun']))

        computation_time = python_time.time() - starting_time
        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        self.DE_population = np.array(self.DE_population)
        self.fit_results = fit_results
        self.fit_time = computation_time