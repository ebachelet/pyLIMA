import sys
import time as python_time

import numpy as np
import scipy
from pyLIMA.fits.LM_fit import LMfit


class MINIMIZEfit(LMfit):

    def fit_type(self):

        return "Minimize fit"

    def objective_function(self, fit_process_parameters):

        likelihood = -self.model_likelihood(fit_process_parameters)

        # Priors
        priors = self.get_priors(fit_process_parameters)

        likelihood += -priors

        return likelihood

    def fit(self):

        starting_time = python_time.time()
        self.population = []
        # use the analytical Jacobian (faster) if no second order are present,
        # else let the
        # algorithm find it.
        self.guess = self.initial_guess()

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
                                               method='Nelder-Mead',
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


        except:

            covariance_matrix = np.zeros((len(self.fit_parameters),
                                          len(self.fit_parameters)))

        computation_time = python_time.time() - starting_time

        print(sys._getframe().f_code.co_name, ' : ' + self.fit_type() + ' fit SUCCESS')
        print('best_model:', fit_results, ' likelihood:', fit_chi2)

        self.fit_results = {'best_model': fit_results, 'likelihood': fit_chi2,
                            'fit_time': computation_time,
                            'covariance_matrix': covariance_matrix}
