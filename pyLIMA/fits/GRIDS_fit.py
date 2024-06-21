import sys
import time as python_time

import numpy as np
from pyLIMA.fits.ML_fit import MLfit
from pyLIMA.fits import DE_fit
from tqdm import tqdm


class GRIDfit(MLfit):
    """
    Performs fits on fix grids of parameters, fitting left parameters with DE.
    Standard way to find preliminary binary models. Efficient but slow...

    Attributes
    ----------
    DE_population_size : int, the scale of the population, i.e. the number
    of DE individuals = DE_population_size*len(fit_parameters)
    max_iteration : int, the total number of iteration
    fix_parameters : dict, the parameters that are set on the grid
    grid_resolution : int, the resolution of the grid for each grid parameters
    """
    def __init__(self, model, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', DE_population_size=5,
                 max_iteration=2000,
                 fix_parameters=[], grid_resolution=10):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry,
                         telescopes_fluxes_method=telescopes_fluxes_method)

        self.DE_population_size = DE_population_size
        self.max_iteration = max_iteration
        self.fix_parameters = fix_parameters
        self.grid_resolution = grid_resolution
        self.intervals = []

    def fit_type(self):
        return "Grids"

    def construct_the_hyper_grid(self):
        """Define the grid. ON CONSTRUCTION.
        """

        parameters_on_the_grid = []

        for parameter_name in self.fix_parameters:
            parameter_range = self.fit_parameters[parameter_name][1]
            self.intervals.append((parameter_range[1]-parameter_range[0])/(self.grid_resolution))

            parameters_on_the_grid.append( np.arange(parameter_range[0], parameter_range[1],
                                                     self.intervals[-1]))
        parameters_on_the_grid = np.array(parameters_on_the_grid)
        params = map(np.asarray, parameters_on_the_grid)
        grid = np.broadcast_arrays(
            *[x[(slice(None),) + (None,) * i] for i, x in enumerate(params)])

        reformate_grid = np.vstack(grid).reshape(len(parameters_on_the_grid), -1).T

        return reformate_grid

    def objective_function(self, fit_process_parameters):

        objective = self.standard_objective_function(fit_process_parameters)

        return objective

    def fit_on_grid_pixel(self, *fixed_parameters):

        fixed_parameters, computational_pool = fixed_parameters[0]

        fixed_parameters = np.ravel(fixed_parameters)
        defit = DE_fit.DEfit(self.model,DE_population_size= self.DE_population_size,display_progress=False,
                             strategy='best1bin', loss_function='chi2',max_iteration=self.max_iteration)

        for key in self.fit_parameters:

            defit.fit_parameters[key][1] = self.fit_parameters[key][1]

        for ind, key in enumerate(self.fix_parameters):

            defit.fit_parameters[key][1] = [fixed_parameters[ind]+self.intervals[ind]/2, fixed_parameters[ind]+self.intervals[ind]/2]

        defit.fit(computational_pool=computational_pool)
        fitted_parameters = defit.fit_results['best_model']
        best_model = np.append( fitted_parameters,self.objective_function(fitted_parameters))
        return best_model

    def fit(self, computational_pool=None):


        hyper_grid = self.construct_the_hyper_grid()
        start_time = python_time.time()

        self.bounds = [self.fit_parameters[key][1] for key in self.fit_parameters.keys()]

        #if computational_pool is not None:

            #with computational_pool as pool, tqdm(total=len(hyper_grid)) as pbar:

            #    res = [pool.apply_async(self.fit_on_grid_pixel, args=(hyper_grid[i],),
            #                            callback=lambda _: pbar.update(1)) for i in
            #           range(len(hyper_grid))]

            #    population = np.array([r.get() for r in res])

        #else:
        #    population = []

        #    for j in tqdm(range(len(hyper_grid))):
        #        new_step = self.fit_on_grid_pixel([hyper_grid[j]])
        #        population.append(new_step)

        ###This is not clean, can not have a parallelization of the grid because of the Manage List of the DE fit

        population = []

        for j in tqdm(range(len(hyper_grid))):
            new_step = self.fit_on_grid_pixel([hyper_grid[j], computational_pool])
            population.append(new_step)

        GRIDS_population = np.array(population)

        computation_time = python_time.time() - start_time
        print(sys._getframe().f_code.co_name,
              ' : ' + self.fit_type() + ' fit SUCCESS')

        best_model_index = GRIDS_population[:, -1].argmin()

        print('best_model:', GRIDS_population[best_model_index, :-1],
              self.loss_function, GRIDS_population[best_model_index, -1])

        self.fit_results = {'best_model': GRIDS_population[best_model_index, :-1],
                            self.loss_function: GRIDS_population[
                                best_model_index, -1],
                            'fit_time': computation_time,
                            'GRIDS_population': GRIDS_population}

