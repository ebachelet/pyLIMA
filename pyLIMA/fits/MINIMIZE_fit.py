import scipy
import time as python_time
import numpy as np
import sys

from pyLIMA.fits.LM_fit import LMfit
import pyLIMA.fits.objective_functions


class MINIMIZEfit(LMfit):

    def fit_type(self):

        return "Minimize fit"

    def objective_function(self, fit_process_parameters):

        likelihood = 0
        # print(fit_process_parameters)
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

            likelihood += photometric_likelihood

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

            likelihood += astrometric_likelihood

        # Priors
        priors = self.get_priors()

        for ind, prior_pdf in enumerate(priors):

            if prior_pdf is not None:

                probability = prior_pdf.pdf(fit_process_parameters[ind])

                if probability > 0:

                    likelihood += -np.log(probability)

                else:

                    likelihood = np.inf
        self.population.append(fit_process_parameters.tolist() + [likelihood])

        return likelihood

    def fit(self):

        starting_time = python_time.time()
        self.population = []
        # use the analytical Jacobian (faster) if no second order are present, else let the
        # algorithm find it.
        self.guess = self.initial_guess()

        bounds_min = [self.fit_parameters[key][1][0] for key in self.fit_parameters.keys()]
        bounds_max = [self.fit_parameters[key][1][1] for key in self.fit_parameters.keys()]
        bounds = [[bounds_min[i],bounds_max[i]] for i in range(len(bounds_max))]
        n_data = 0

        for telescope in self.model.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        #if self.model.Jacobian_flag != 'No Way':

        #    jacobian_function = self.residuals_Jacobian

        #else:

        #jacobian_function = '2-point'

        minimize_fit = scipy.optimize.minimize(self.objective_function, self.guess,method='Nelder-Mead',
                                               bounds=bounds, options={'maxiter': 50000, 'disp': True},)

        self.population = np.array(self.population)

        fit_results = minimize_fit['x'].tolist()
        fit_chi2 = minimize_fit['fun']  # likelihood

        try:

            ###Currently ugly#####

            mask = self.population[:, -1] < self.population[:, -1].min() +36 #Valerio magic 10%

            covariance_matrix = np.cov(self.population[mask,:-1].T)


        except:

            covariance_matrix = np.zeros((len(self.fit_parameters),
                                          len(self.fit_parameters)))

        computation_time = python_time.time() - starting_time

        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        print('best_model:', fit_results, ' likelihood:', fit_chi2)

        self.fit_results = {'best_model': fit_results, 'likelihood': fit_chi2, 'fit_time': computation_time,
                            'covariance_matrix': covariance_matrix}

