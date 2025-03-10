import time as python_time

import numpy as np
from pyLIMA.fits.LM_fit import LMfit

from iminuit import Minuit


class LeastSquares:
    """
    Generic least-squares cost function with error for minuit.
    """

    errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly

    def __init__(self, objective_function):

        self.objective_function = objective_function
    def __call__(self, *par):  # we must accept a variable number of model parameters

        residuals = self.objective_function(par[0])

        objective = np.sum(residuals**2)
        return objective

class MINUITfit(LMfit):

    def fit_type(self):

        return "Minuit"

    def fit(self):

        starting_time = python_time.time()

        # use the analytical Jacobian (faster) if no second order are present,
        # else let the
        # algorithm find it.
        self.guess = self.initial_guess()

        if self.guess is None:
            return

        least_squares = LeastSquares(self.objective_function)
        minuit = Minuit(least_squares, self.guess)
        minuit.limits = [self.fit_parameters[key][1] for key in self.fit_parameters]

        minuit.migrad()


        fit_results =  np.array([minuit.params[f].value for f in range(len(
                self.fit_parameters))])


        fit_chi2 = minuit.fval

        covariance_matrix = minuit.covariance

        computation_time = python_time.time() - starting_time

        print(self.fit_type() + ' fit SUCCESS')
        #print('best_model:', fit_results, self.loss_function, fit_chi2)

        self.fit_results = {'best_model': fit_results, self.loss_function: fit_chi2,
                            'fit_time': computation_time,
                            'covariance_matrix': covariance_matrix,
                            'fit_object':minuit}

        self.print_fit_results()
