import sys
import time as python_time

import numpy as np
import scipy
from pyLIMA.fits.LM_fit import LMfit


class MINIMIZEfit(LMfit):
    """
    Under Construction
    """
    def __init__(self, model, telescopes_fluxes_method='fit', loss_function='chi2'):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, telescopes_fluxes_method=telescopes_fluxes_method,
                         loss_function=loss_function)

        self.guess = []

    def fit_type(self):

        return "Minimize fit"

    def objective_function(self, fit_process_parameters):

        likelihood = self.standard_objective_function(fit_process_parameters)

        return likelihood

    def fit(self):

        starting_time = python_time.time()
        self.population = []
        # use the analytical Jacobian (faster) if no second order are present,
        # else let the
        # algorithm find it.
        self.guess = self.initial_guess()

        if self.guess is None:
            return

        bounds_min = [self.fit_parameters[key][1][0] for key in
                      self.fit_parameters.keys()]
        bounds_max = [self.fit_parameters[key][1][1] for key in
                      self.fit_parameters.keys()]
        bounds = [[bounds_min[i], bounds_max[i]] for i in range(len(bounds_max))]
        n_data = 0

        for telescope in self.model.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        # if self.model.Jacobian_flag != 'No Way':

        #    jacobian_function = self.residuals_Jacobian

        # else:

        # jacobian_function = '2-point'

        minimize_fit = scipy.optimize.minimize(self.objective_function, self.guess,
                                               bounds=bounds, options={'maxiter': 50000,
                                                                       'disp': True}, )

        self.population = np.array(self.population)

        fit_results = minimize_fit['x'].tolist()
        fit_chi2 = minimize_fit['fun']  # likelihood

        try:

            ###Currently ugly#####

            mask = self.population[:, -1] < self.population[:,
                                            -1].min() + 36  # Valerio magic 10%

            covariance_matrix = np.cov(self.population[mask, :-1].T)


        except IndexError:

            covariance_matrix = np.zeros((len(self.fit_parameters),
                                          len(self.fit_parameters)))

        computation_time = python_time.time() - starting_time

        print(sys._getframe().f_code.co_name, ' : ' + self.fit_type() + ' fit SUCCESS')
        print('best_model:', fit_results, self.loss_function, fit_chi2)

        self.fit_results = {'best_model': fit_results, self.loss_function: fit_chi2,
                            'fit_time': computation_time,
                            'covariance_matrix': covariance_matrix}
