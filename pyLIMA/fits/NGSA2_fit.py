# import time as python_time
import numpy as np

from pyLIMA.fits.ML_fit import MLfit

from pymoo.core.problem import ElementwiseProblem


class MLProblem(ElementwiseProblem):

    def __init__(self, bounds, objective_photometry=None, objective_astrometry=None,
                 **kwargs):

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
    """
    Under Construction
    """

    def __init__(self, model, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='polyfit', loss_function='likelihood',
                 ):
        if int(np.__version__[0]) >= 2:
            raise NotImplementedError(
                "This fit is not yet supported for numpy>=2. Downgrade to numpy if you must. "
                "See GitHub issue (https://github.com/anyoptimization/pymoo/issues/606)")
        else:

            super().__init__(model, rescale_photometry=rescale_photometry,
                             rescale_astrometry=rescale_astrometry,
                             telescopes_fluxes_method=telescopes_fluxes_method,
                             loss_function=loss_function)
    def fit_type(self):
        return "Non-dominated Sorting Genetic Algorithm"

    def fit(self, computational_pool=None):

        # starting_time = python_time.time()

        from pymoo.algorithms.moo.nsga2 import NSGA2

        algorithm = NSGA2(pop_size=100)

        if self.model.astrometry is not None:

            astrometry = self.likelihood_astrometry
        else:
            astrometry = None

        if self.model.photometry is not None:

            photometry = self.likelihood_photometry
        else:
            photometry = None

        bounds = [self.fit_parameters[key][1] for key in self.fit_parameters.keys()]

        problem = MLProblem(bounds, photometry, astrometry)

        from pymoo.optimize import minimize

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 200),
                       seed=1,
                       save_history=True,
                       verbose=True)

        print(res)
        # X = res.X
        # F = res.F
        # n_evals = np.array([e.evaluator.n_eval for e in res.history])
        # opt = np.array([e.opt[0].F for e in res.history])

        # mask =np.argmin((F[:,0]/F[:,0].max())**2+(F[:,1]/F[:,1].max())**2)
        # computation_time = python_time.time() - start_time

        ###NOT COMPLETE
        breakpoint()
