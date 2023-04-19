import time as python_time
import numpy as np
import sys
from collections import OrderedDict
from pymoo.core.problem import ElementwiseProblem

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions
from pyLIMA.priors import parameters_boundaries

class MLProblem(ElementwiseProblem):

    def __init__(self, bounds, objective_photometry=None, objective_astrometry=None, **kwargs):

        n_var = len(bounds)
        n_obj = 0

        if objective_photometry is not None:

            n_obj += 1
            self.objective_photometry = objective_photometry

        else:

            self.objective_photometry = None

        if objective_astrometry is not None:

            n_obj += 1
            self.objective_astrometry = objective_astrometry

        else:

            self.objective_astrometry = None


        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=np.array([i[0] for i in bounds]),
                         xu=np.array([i[1] for i in bounds]),
                         **kwargs)




    def _evaluate(self, x, out, *args, **kwargs):

        objectives = []

        if self.objective_photometry is not None:

            objectives.append(self.objective_photometry(x))

        if self.objective_astrometry is not None:

            objectives.append(self.objective_astrometry(x))

        out["F"] = objectives


class NGSA2fit(MLfit):

    def fit_type(self):
        return "Non-dominated Sorting Genetic Algorithm"



    def likelihood_photometry(self, fit_process_parameters):

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


            likelihood = photometric_likelihood

        return likelihood

    def likelihood_astrometry(self, fit_process_parameters):

        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

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

            astrometric_likelihood = 0.5 * (np.sum(residus ** 2 + np.log(2 * np.pi * errors ** 2)))

        return astrometric_likelihood


    def fit(self,computational_pool=None):

        starting_time = python_time.time()

        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.factory import get_sampling, get_crossover, get_mutation

        number_of_parameters = len(self.fit_parameters.keys())

        algorithm = NSGA2(
            pop_size=40,
            n_offsprings=10,)
            #sampling=get_sampling("real_random"),
            #crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            #mutation=get_mutation("real_pm", eta=20),
            #eliminate_duplicates=True)

        from pymoo.factory import get_termination

        #termination = get_termination('default',f_tol = 10**-8,  n_max_gen=5000,)

        if self.model.astrometry:

            astrometry = self.likelihood_astrometry

        else:

            astrometry = None

        if self.model.photometry:

            photometry = self.likelihood_photometry

        else:

            photometry = None

        bounds = [self.fit_parameters[key][1] for key in self.fit_parameters.keys()]


        from pymoo.optimize import minimize
        from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
        from pymoo.algorithms.soo.nonconvex.isres import ISRES
        from pymoo.core.problem import StarmapParallelization

        if computational_pool:
            runner = StarmapParallelization(computational_pool.starmap)

        problem = MLProblem(bounds, photometry, astrometry,elementwise_runner=runner)

        breakpoint()

        algorithm = ISRES(n_offsprings=len(self.fit_parameters)*10, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)

        res = minimize(problem, algorithm, ("n_gen", 1000),verbose=True, seed=1)

        breakpoint()
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=True)

        X = res.X
        F = res.F
        n_evals = np.array([e.evaluator.n_eval for e in res.history])
        opt = np.array([e.opt[0].F for e in res.history])

        #mask =np.argmin((F[:,0]/F[:,0].max())**2+(F[:,1]/F[:,1].max())**2)

        import pdb;
        pdb.set_trace()

        print('DE converge to objective function : f(x) = ', str(differential_evolution_estimation['fun']))
        print('DE converge to parameters : = ', differential_evolution_estimation['x'].astype(str))

        fit_results = np.hstack((differential_evolution_estimation['x'],differential_evolution_estimation['fun']))

        computation_time = python_time.time() - starting_time
        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        self.DE_population = np.array(self.DE_population)
        self.fit_results = fit_results
        self.fit_time = computation_time