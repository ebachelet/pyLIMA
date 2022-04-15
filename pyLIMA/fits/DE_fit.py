import scipy
import time as python_time
import numpy as np
import sys

from pyLIMA.fits.fit import MLfit
import pyLIMA.fits.objective_functions


class DEfit(MLfit):

    def __init__(self, model, DE_population_size = 10, max_iteration = 10000):
        """The fit class has to be intialized with an event object."""

        self.model = model
        self.DE_population = []
        self.fit_results = []
        self.DE_population_size = DE_population_size
        self.max_iteration = max_iteration
        self.fit_time = 0 #s

    def fit_type(self):
        return "Differential Evolution"

    def objective_function(self, fit_process_parameters):

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)

        photometric_chi2 = pyLIMA.fits.objective_functions.all_telescope_photometric_chi2(self.model,
                                                                                          pyLIMA_parameters)

        #astrometric_residuals = pyLIMA.fits.residuals.all_telescope_astrometric_chi2(self.model.event, pyLIMA_parameters)
        self.DE_population.append(fit_process_parameters.tolist() + [photometric_chi2])
        return photometric_chi2

    def fit(self, computational_pool=None):

        starting_time = python_time.time()

        if computational_pool:

            worker = computational_pool.map

        else:

            worker = 1

        n_data = 0
        for telescope in self.model.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        differential_evolution_estimation = scipy.optimize.differential_evolution(self.objective_function,
                                                                                  bounds=self.model.parameters_boundaries,
                                                                                  mutation=(0.5,1.5), popsize=int(self.DE_population_size),
                                                                                  maxiter=self.max_iteration, tol=0.00,
                                                                                  atol=0.001*n_data, strategy='best1bin',
                                                                                  recombination=0.7, polish=True, init='latinhypercube',
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