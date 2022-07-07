import numpy as np
import time as python_time
from tqdm import tqdm
import scipy.optimize as so

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions


class GRIDfit(MLfit):

    def __init__(self, model, fancy_parameters=False, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', DEMC_population_size=10, max_iteration=10000,
                 fix_parameters = [], grid_resolution = 50):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, fancy_parameters=fancy_parameters, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry, telescopes_fluxes_method=telescopes_fluxes_method)

        self.population = [] # to be recognize by all process during parallelization
        self.DEMC_population_size = DEMC_population_size #Times number of dimensions!
        self.max_iteration = max_iteration
        self.fix_parameters = fix_parameters
        self.grid_resolution = grid_resolution

    def fit_type(self):
        return "Grids"

    def reconstruct_fit_process_parameters(self, moving_parameters, fix_parameters):
        """
        """

        fit_process_parameters = []

        ind_fix = 0
        ind_move = 0

        for key in list(self.fit_parameters.keys()):

            if key not in self.fix_parameters:

                fit_process_parameters.append(moving_parameters[ind_move])
                ind_move += 1

            else:

                fit_process_parameters.append(fix_parameters[ind_fix])
                ind_fix += 1

        return np.array(fit_process_parameters)

    def construct_the_hyper_grid(self, parameters):
        """Define the grid. ON CONSTRUCTION.
        """
        params = map(np.asarray, parameters)
        grid = np.broadcast_arrays(*[x[(slice(None),) + (None,) * i] for i, x in enumerate(params)])

        reformate_grid = np.vstack(grid).reshape(len(parameters), -1).T

        return reformate_grid

    def objective_function(self, moving_parameters, fixed_parameters):

        fit_process_parameters = self.reconstruct_fit_process_parameters(moving_parameters, fixed_parameters)
        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        likelihood = 0

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

            photometric_likelihood = 0.5*(np.sum(residus ** 2 + np.log(2*np.pi*errflux**2)))

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

            astrometric_likelihood = 0.5*(np.sum(residus ** 2 + np.log(2*np.pi*errors**2)))

            likelihood += astrometric_likelihood

        return likelihood

    def fit_on_grid_pixel(self, *fixed_parameters):

        fixed_parameters = np.ravel(fixed_parameters)
        differential_evolution_estimation = so.differential_evolution(self.objective_function,
                                                                                  bounds=self.bounds,
                                                                                  mutation=(0.5, 1.5),
                                                                                  popsize=2,
                                                                                  args=([fixed_parameters]),
                                                                                  maxiter=1000, tol=0.00,
                                                                                  atol=1.0, strategy='rand1bin',
                                                                                  recombination=0.5, polish=True,
                                                                                  init='latinhypercube',
                                                                                  disp=False)



        print(fixed_parameters, differential_evolution_estimation['fun'])
        fitted_parameters = differential_evolution_estimation['x']
        best_model = self.reconstruct_fit_process_parameters(fitted_parameters, fixed_parameters)
        best_model = np.append(best_model, differential_evolution_estimation['fun'])
        return best_model


    def fit(self, computational_pool=None):

        parameters_on_the_grid = []

        for parameter_name in self.fix_parameters:

            parameter_range = self.model.parameters_boundaries[self.model.model_dictionnary[parameter_name]]

            parameters_on_the_grid.append(np.linspace(parameter_range[0], parameter_range[1],
                            self.grid_resolution))

        hyper_grid = self.construct_the_hyper_grid(parameters_on_the_grid)

        start_time = python_time.time()

        self.bounds = [self.fit_parameters[key][1] for key in self.fit_parameters.keys() if key not in self.fix_parameters]

        if computational_pool is not None:

            new_step = np.array(computational_pool.starmap(self.fit_on_grid_pixel, hyper_grid))

        else:

            for j, ind in enumerate(hyper_grid):

                new_step = self.fit_on_grid_pixel([hyper_grid[j]])

                import pdb;
                pdb.set_trace()

        #grid_results = list(
        #    computational_map(emcee.ensemble._function_wrapper(self.optimization_on_grid_pixel, args=[], kwargs={}),
        # #                     hyper_grid))

        #return np.array(grid_results)


