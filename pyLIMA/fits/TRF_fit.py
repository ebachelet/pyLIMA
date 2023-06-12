import sys
import time as python_time

import numpy as np
import scipy
from pyLIMA.fits.LM_fit import LMfit


class TRFfit(LMfit):

    def fit_type(self):

        return "Trust Region Reflective"

    def fit(self):

        starting_time = python_time.time()

        # use the analytical Jacobian (faster) if no second order are present,
        # else let the
        # algorithm find it.
        self.guess = self.initial_guess()
        bounds_min = [self.fit_parameters[key][1][0] for key in
                      self.fit_parameters.keys()]
        bounds_max = [self.fit_parameters[key][1][1] for key in
                      self.fit_parameters.keys()]

        n_data = 0

        for telescope in self.model.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        if self.model.Jacobian_flag != 'No Way':

            jacobian_function = self.residuals_Jacobian

        else:

            jacobian_function = '2-point'

        if self.loss_function == 'soft_l1':

            loss = 'soft_l1'
            jacobian_function = '2-point'

        else:

            loss = 'linear'

        trf_fit = scipy.optimize.least_squares(self.objective_function, self.guess,
                                               method='trf',
                                               bounds=(bounds_min, bounds_max),
                                               max_nfev=50000, jac=jacobian_function,
                                               loss=loss, xtol=10 ** -10,
                                               ftol=10 ** -10, gtol=10 ** -10)

        fit_results = trf_fit['x'].tolist()
        fit_chi2 = trf_fit['cost'] * 2  # chi2

        try:
            # Try to extract the covariance matrix from the levenberg-marquard_fit
            # output
            covariance_matrix = np.linalg.pinv(np.dot(trf_fit['jac'].T, trf_fit['jac']))

        except ValueError:

            covariance_matrix = np.zeros((len(self.fit_parameters),
                                          len(self.fit_parameters)))

        covariance_matrix *= fit_chi2 / (n_data - len(self.model.model_dictionnary))
        computation_time = python_time.time() - starting_time

        print(sys._getframe().f_code.co_name, ' : ' + self.fit_type() + ' fit SUCCESS')
        print('best_model:', fit_results, ' chi2:', fit_chi2)

        self.fit_results = {'best_model': fit_results, 'chi2': fit_chi2,
                            'fit_time': computation_time,
                            'covariance_matrix': covariance_matrix}
