import sys
import time as python_time

import numpy as np
import scipy
from pyLIMA.fits.ML_fit import MLfit
from tqdm import tqdm


class DEfit(MLfit):
    """
    Differential Evolution from Storn & Price. Please cite the paper :
    https://link.springer.com/article/10.1023/A:1008202821328

    Attributes
    -----------
    DE_population_size : int, the scale of the population, i.e. the number
    of DE individuals = DE_population_size*len(fit_parameters)
    max_iteration : int, the total number of iteration
    display_progress : bool, turns on to display progress
    strategy : str, 'best1bin' or 'rand1bin' (default)
    """
    def __init__(self, model, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', loss_function='likelihood',
                 DE_population_size=10, max_iteration=10000,
                 display_progress=False, strategy='rand1bin'):

        super().__init__(model, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry,
                         telescopes_fluxes_method=telescopes_fluxes_method,
                         loss_function=loss_function)

        self.DE_population_size = DE_population_size  # Times number of dimensions!
        self.max_iteration = max_iteration
        self.fit_time = 0  # s
        self.display_progress = display_progress
        self.strategy = strategy

    def fit_type(self):

        return "Differential Evolution"

    def objective_function(self, fit_process_parameters):

        objective = self.standard_objective_function(fit_process_parameters)

        return objective

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

        differential_evolution_estimation = scipy.optimize.differential_evolution(
            self.objective_function,
            bounds=bounds,
            mutation=(0.5, 1.5), popsize=int(self.DE_population_size),
            maxiter=self.max_iteration, tol=0.00,
            atol=1.0, strategy=self.strategy,
            recombination=0.5, polish=False, init=init,
            disp=self.display_progress, workers=worker)

        self.trials = np.array(self.trials)

        print('DE converge to objective function : f(x) = ',
              str(differential_evolution_estimation['fun']))
        print('DE converge to parameters : = ',
              differential_evolution_estimation['x'].astype(str))

        fit_results = differential_evolution_estimation['x']
        fit_log_likelihood = differential_evolution_estimation['fun']

        computation_time = python_time.time() - start_time
        print(sys._getframe().f_code.co_name, ' : ' + self.fit_type() + ' fit SUCCESS')

        DE_population = self.trials

        print('best_model:', fit_results, '-ln(likelihood)', fit_log_likelihood)

        self.fit_results = {'best_model': fit_results,
                            '-(ln_likelihood)': fit_log_likelihood,
                            'fit_time': computation_time,
                            'DE_population': DE_population}

    def samples_to_plot(self):

        samples = self.fit_results['DE_population']

        return samples


class DEfitnew(MLfit):
    """
    Under Construction
    """
    def __init__(self, model, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', DE_population_size=10,
                 max_iteration=10000,
                 display_progress=False, strategy='rand1bin'):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, rescale_photometry=rescale_photometry,
                         rescale_astrometry=rescale_astrometry,
                         telescopes_fluxes_method=telescopes_fluxes_method)

        self.DE_population_size = DE_population_size  # Times number of dimensions!
        self.max_iteration = max_iteration
        self.fit_time = 0  # s
        self.display_progress = display_progress
        self.strategy = strategy

    def fit_type(self):
        return "Differential Evolution"

    def objective_function(self, fit_process_parameters):

        likelihood, pyLIMA_parameters = self.model_likelihood(fit_process_parameters)
        likelihood *= -1

        # Priors
        priors = self.get_priors(fit_process_parameters)

        likelihood += -priors

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

        solver = scipy.optimize._differentialevolution.DifferentialEvolutionSolver(
            self.objective_function, bounds=bounds,
            mutation=(0.5, 1.5),
            popsize=int(self.DE_population_size),
            maxiter=1, tol=0.00, atol=1.0,
            strategy=self.strategy,
            recombination=0.5, polish=False,
            init=init, disp=self.display_progress,
            workers=worker)

        if initial_population == []:

            solver.init_population_lhs()

        else:

            solver.init_population_array(init)

        pop = []
        pop_energies = []

        for loop in tqdm(range(self.max_iteration)):
            solver.__next__()

            pop.append(np.copy(solver._scale_parameters(solver.population)))
            pop_energies.append(np.copy(solver.population_energies))

            print('Best:', pop[-1][0], pop_energies[-1][0])
            # converged = solver.converged()

            # if converged:

            #    break

        pop = np.array(pop)
        pop_energies = np.array(pop_energies)

        print('DE converge to objective function : f(x) = ', str(pop_energies[-1, 0]))
        print('DE converge to parameters : = ', pop[-1, 0].astype(str))

        fit_results = pop[-1, 0]
        fit_log_likelihood = pop_energies[-1, 0]

        computation_time = python_time.time() - start_time
        print(sys._getframe().f_code.co_name, ' : ' + self.fit_type() + ' fit SUCCESS')

        DE_population = np.zeros((pop.shape[0], pop.shape[1], pop.shape[2] + 1))

        DE_population[:, :, :-1] = pop
        DE_population[:, :, -1] = pop_energies

        print('best_model:', fit_results, '-ln(likelihood)', fit_log_likelihood)

        self.fit_results = {'best_model': fit_results,
                            '-(ln_likelihood)': fit_log_likelihood,
                            'fit_time': computation_time,
                            'DE_population': DE_population}

    def samples_to_plot(self):

        chains = self.fit_results['DE_population']
        samples = chains.reshape(-1, chains.shape[2])

        return samples
