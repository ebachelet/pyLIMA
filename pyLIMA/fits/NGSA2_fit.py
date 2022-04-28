import scipy
import time as python_time
import numpy as np
import sys
from collections import OrderedDict
from pymoo.core.problem import ElementwiseProblem

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions
from pyLIMA.priors import parameters_boundaries

class MLProblem(ElementwiseProblem):

    def __init__(self, bounds, objective_photometry=None, objective_astrometry=None):

        n_var = len(bounds)
        n_obj = 0

        if objective_photometry is not None:

            n_obj += 1
            self.objective_photometry = objective_photometry

        else:

            self.objective_photometry = None

        if objective_astrometry is not None:
            #pass
            n_obj += 1
            self.objective_astrometry = objective_astrometry

        else:
            #pass
            self.objective_astrometry = None

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=np.array([i[0] for i in bounds]),
                         xu=np.array([i[1] for i in bounds]))




    def _evaluate(self, x, out, *args, **kwargs):

        objectives = []

        if self.objective_photometry is not None:

            objectives.append(self.objective_photometry(x))

        if self.objective_astrometry is not None:
            #pass
            objectives.append(self.objective_astrometry(x))

        out["F"] = objectives


class NGSA2fit(MLfit):

    def __init__(self, model, rescale_photometry=False, telescopes_fluxes_method='NGSA2'):
        """The fit class has to be intialized with an event object."""


        self.telescopes_fluxes_method = telescopes_fluxes_method

        super().__init__(model, rescale_photometry)


    def fit_type(self):
        return "Non-dominated Sorting Genetic Algorithm"

    def define_fit_parameters(self):

        fit_parameters_dictionnary = self.model.paczynski_model_parameters()

        fit_parameters_dictionnary_updated = self.model.astrometric_model_parameters(fit_parameters_dictionnary)

        fit_parameters_dictionnary_updated = self.model.second_order_model_parameters(
            fit_parameters_dictionnary_updated)

        if self.telescopes_fluxes_method == 'NGSA2':

            fit_parameters_dictionnary_updated = self.model.telescopes_fluxes_model_parameters(
                fit_parameters_dictionnary_updated)

        if self.rescale_photometry:

            for telescope in self.model.event.telescopes:

                if telescope.lightcurve_flux is not None:

                    fit_parameters_dictionnary_updated['logk_photometry_' + telescope.name] = \
                        len(fit_parameters_dictionnary_updated)

        self.fit_parameters = OrderedDict(
            sorted(fit_parameters_dictionnary_updated.items(), key=lambda x: x[1]))

        self.model_parameters_index = [self.model.model_dictionnary[i] for i in self.model.model_dictionnary.keys() if
                                       i in self.fit_parameters.keys()]
        self.rescale_photometry_parameters_index = [self.fit_parameters[i] for i in self.fit_parameters.keys() if
                                                    'logk_photometry' in i]

        fit_parameters_boundaries = parameters_boundaries.parameters_boundaries(self.model.event, self.fit_parameters)

        for ind, key in enumerate(self.fit_parameters.keys()):
            self.fit_parameters[key] = [ind, fit_parameters_boundaries[ind]]

    def likelihood_photometry(self, fit_process_parameters):

        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.model.photometry:

            if self.rescale_photometry:

                rescaling_photometry_parameters = np.exp(fit_process_parameters[self.rescale_photometry_parameters_index])

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

            likelihood = photometric_likelihood

        return likelihood

    def likelihood_astrometry(self, fit_process_parameters):

        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.model.astrometry:

            residus, errors = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                  pyLIMA_parameters,
                                                                                                  norm=True,
                                                                                                  rescaling_astrometry_parameters=None)

            astrometric_likelihood = 0.5 * (np.sum(residus ** 2 + np.log(2 * np.pi * errors ** 2)))

            likelihood = astrometric_likelihood

        return likelihood


    def fit(self):

        starting_time = python_time.time()

        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.factory import get_sampling, get_crossover, get_mutation

        algorithm = NSGA2(
            pop_size=40,
            n_offsprings=10,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.5, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )

        from pymoo.factory import get_termination

        termination = get_termination('default',n_max_gen=10000,)

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