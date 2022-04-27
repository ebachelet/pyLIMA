import scipy
import time as python_time
import numpy as np
import sys
from multiprocessing import Manager
from collections import OrderedDict


from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions

class DEfit(MLfit):

    def __init__(self, model, rescale_photometry=False, DE_population_size=10, max_iteration=10000, telescopes_fluxes_method='DE'):
        """The fit class has to be intialized with an event object."""

        self.telescopes_fluxes_method = telescopes_fluxes_method

        super().__init__(model, rescale_photometry)

        self.DE_population = Manager().list() # to be recognize by all process during parallelization
        self.DE_population_size = DE_population_size #Times number of dimensions!
        self.fit_boundaries = []
        self.max_iteration = max_iteration
        self.fit_time = 0 #s

    def fit_type(self):
        return "Differential Evolution"

    def define_fit_parameters(self):

        fit_parameters_dictionnary = self.model.paczynski_model_parameters()

        fit_parameters_dictionnary_updated = self.model.astrometric_model_parameters(fit_parameters_dictionnary)

        fit_parameters_dictionnary_updated = self.model.second_order_model_parameters(
            fit_parameters_dictionnary_updated)

        if self.telescopes_fluxes_method == 'DE':
            fit_parameters_dictionnary_updated = self.model.telescopes_fluxes_model_parameters(
            fit_parameters_dictionnary_updated)

        if self.rescale_photometry:

            for telescope in self.model.event.telescopes:

                if telescope.lightcurve_flux is not None:
                    fit_parameters_dictionnary_updated['k_photometry_' + telescope.name] = \
                        len(fit_parameters_dictionnary_updated)

        self.fit_parameters = OrderedDict(
            sorted(fit_parameters_dictionnary_updated.items(), key=lambda x: x[1]))

        self.model_parameters_index = [self.model.model_dictionnary[i] for i in self.model.model_dictionnary.keys() if i in self.fit_parameters.keys()]
        self.rescale_photometry_parameters_index = [self.fit_parameters[i] for i in self.fit_parameters.keys() if 'k_photometry' in i]

    def objective_function(self, fit_process_parameters):

        likelihood = 0


        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.model.photometry:

            if self.rescale_photometry:

                rescaling_photometry_parameters = fit_process_parameters[self.rescale_photometry_parameters_index]

                residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                       pyLIMA_parameters,
                                                                                                       norm=True,
                                                                                                       rescaling_photometry_parameters=rescaling_photometry_parameters)
            else:

                residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                       pyLIMA_parameters,
                                                                                                       norm=True,
                                                                                                       rescaling_photometry_parameters=None)

            photometric_likelihood = 0.5*(np.sum(residus ** 2 + np.log(2*np.pi*errflux**2)))

            likelihood += photometric_likelihood

        if self.model.astrometry:

            residus, errors= pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                   pyLIMA_parameters,
                                                                                                   norm=True,
                                                                                                   rescaling_astrometry_parameters=None)

            astrometric_likelihood = 0.5*(np.sum(residus ** 2 + np.log(2*np.pi*errors**2)))

            likelihood += astrometric_likelihood

        self.DE_population.append(fit_process_parameters.tolist() + [likelihood])

        return likelihood

    def fit(self, computational_pool=None):

        starting_time = python_time.time()

        if computational_pool:

            worker = computational_pool.map

        else:

            worker = 1

        bounds = self.fit_parameters_boundaries

        n_data = 0
        for telescope in self.model.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        differential_evolution_estimation = scipy.optimize.differential_evolution(self.objective_function,
                                                                                  bounds=bounds,
                                                                                  mutation=(0.1, 1.9), popsize=int(self.DE_population_size),
                                                                                  maxiter=self.max_iteration, tol=0.0,
                                                                                  atol=1, strategy='rand2bin',
                                                                                  recombination=0.5, polish=True, init='latinhypercube',
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