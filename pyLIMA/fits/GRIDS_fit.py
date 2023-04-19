import numpy as np
import time as python_time
from tqdm import tqdm
import scipy.optimize as so
import sys

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions
from pyLIMA.outputs import pyLIMA_plots

class GRIDfit(MLfit):

    def __init__(self, model, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', DE_population_size=2, max_iteration=2000,
                 fix_parameters = [], grid_resolution = 10):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry, telescopes_fluxes_method=telescopes_fluxes_method)

        self.DE_population_size = DE_population_size
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

    def construct_the_hyper_grid(self):
        """Define the grid. ON CONSTRUCTION.
        """

        parameters_on_the_grid = []

        for parameter_name in self.fix_parameters:

            parameter_range = self.fit_parameters[parameter_name][1]

            parameters_on_the_grid.append(np.linspace(parameter_range[0], parameter_range[1],
                                                      self.grid_resolution))

        parameters_on_the_grid = np.array(parameters_on_the_grid)
        params = map(np.asarray, parameters_on_the_grid)
        grid = np.broadcast_arrays(*[x[(slice(None),) + (None,) * i] for i, x in enumerate(params)])

        reformate_grid = np.vstack(grid).reshape(len(parameters_on_the_grid), -1).T

        return reformate_grid

    def objective_function(self, moving_parameters, fixed_parameters):

        fit_process_parameters = self.reconstruct_fit_process_parameters(moving_parameters, fixed_parameters)
        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        likelihood = 0

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


            astrometric_likelihood = 0.5*np.sum(residus ** 2 + 2*np.log(errors)+np.log(2*np.pi))

            likelihood += astrometric_likelihood

        # Priors
        priors = self.get_priors()

        for ind, prior_pdf in enumerate(priors):

            if prior_pdf is not None:

                pdf = prior_pdf(fit_process_parameters[ind])

                if pdf > 0:

                    likelihood += -np.log(pdf)

                else:

                    likelihood = np.inf
        return likelihood

    def fit_on_grid_pixel(self, *fixed_parameters):

        fixed_parameters = np.ravel(fixed_parameters)
        differential_evolution_estimation = so.differential_evolution(self.objective_function,
                                                                                  bounds=self.bounds,
                                                                                  mutation=(0.5, 1.5),
                                                                                  popsize=self.DE_population_size,
                                                                                  args=([fixed_parameters]),
                                                                                  maxiter=self.max_iteration, tol=0.00,
                                                                                  atol=1.0, strategy='rand1bin',
                                                                                  recombination=0.5, polish=True,
                                                                                  init='latinhypercube',
                                                                                  disp=False)

        fitted_parameters = differential_evolution_estimation['x']
        best_model = self.reconstruct_fit_process_parameters(fitted_parameters, fixed_parameters)
        best_model = np.append(best_model, differential_evolution_estimation['fun'])
        return best_model


    def fit(self, computational_pool=None):

        hyper_grid = self.construct_the_hyper_grid()

        start_time = python_time.time()

        self.bounds = [self.fit_parameters[key][1] for key in self.fit_parameters.keys() if key not in self.fix_parameters]



        if computational_pool is not None:

            with computational_pool as pool, tqdm(total=len(hyper_grid)) as pbar:

                res = [pool.apply_async(self.fit_on_grid_pixel, args=(hyper_grid[i],),
                                    callback=lambda _: pbar.update(1)) for i in range(len(hyper_grid))]

                population = np.array([r.get() for r in res])

        else:
            population = []

            for j in tqdm(range(self.max_iteration)):

                new_step = self.fit_on_grid_pixel([hyper_grid[j]])
                population.append(new_step)

            GRIDS_population = np.array(population)

            computation_time = python_time.time() - start_time
            print(sys._getframe().f_code.co_name, ' : ' + self.fit_type() + ' fit SUCCESS')


            best_model_index = GRIDS_population[:,-1].argmin()

            print('best_model:', GRIDS_population[best_model_index,:-1], '-ln(likelihood)', GRIDS_population[best_model_index,-1])

            self.fit_results = {'best_model': GRIDS_population[best_model_index,:-1], '-(ln_likelihood)': GRIDS_population[best_model_index,-1],
                                'fit_time': computation_time,
                                'GRIDS_population': GRIDS_population}

    def fit_outputs(self):

        pyLIMA_plots.plot_lightcurves(self.model, self.fit_results['best_model'])
        pyLIMA_plots.plot_geometry(self.model, self.fit_results['best_model'])


