import scipy
import time as python_time
import numpy as np
import sys

from pyLIMA.fits.LM_fit import LMfit
import pyLIMA.fits.residuals


class TRFfit(LMfit):

    def fit_type(self):
        return "Trust Region Reflective"

    def fit(self):

        starting_time = python_time.time()

        # use the analytical Jacobian (faster) if no second order are present, else let the
        # algorithm find it.
        self.guess = self.initial_guess()
        bounds_min = [i[0] for i in self.model.parameters_boundaries] + [0, -np.inf] * len(
            self.model.event.telescopes)
        bounds_max = [i[1] for i in self.model.parameters_boundaries] + [np.inf, np.inf] * len(
            self.model.event.telescopes)


        if self.rescaling_photometry:

            for telescope in self.model.event.telescopes:

                self.guess.append(0.1)
                bounds_min.append(0)
                bounds_max.append(10)
        n_data = 0
        for telescope in self.model.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        n_parameters = len(self.model.model_dictionnary)

        #if self.model.Jacobian_flag == 'OK':

        # No Jacobian now

        lm_fit = scipy.optimize.least_squares(self.objective_function, self.guess, method='trf',
                                              bounds=(bounds_min, bounds_max),  max_nfev=50000, xtol=10 ** -10,
                                              ftol=10**-10, gtol=10 ** -10)

        
        fit_result = lm_fit['x'].tolist()
        fit_result.append(lm_fit['cost'])  # likelihood

        try:
            # Try to extract the covariance matrix from the levenberg-marquard_fit output
            covariance_matrix = np.linalg.pinv(np.dot(lm_fit['jac'].T, lm_fit['jac']))

        except:

            covariance_matrix = np.zeros((len(self.model.model_dictionnary),
                                          len(self.model.model_dictionnary)))

        #covariance_matrix *= fit_result[-1] / (n_data - len(self.model.model_dictionnary))
        computation_time = python_time.time() - starting_time

        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        print(fit_result)
        self.fit_results = fit_result
        self.fit_covariance_matrix = covariance_matrix
        self.fit_time = computation_time
