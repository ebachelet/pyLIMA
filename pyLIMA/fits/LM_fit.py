import scipy
import time as python_time
import numpy as np
import sys

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions
from pyLIMA.outputs import pyLIMA_plots

class LMfit(MLfit):

    def __init__(self, model, fancy_parameters=False,telescopes_fluxes_method='fit'):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, fancy_parameters=fancy_parameters, telescopes_fluxes_method=telescopes_fluxes_method)

        self.guess = []

    def fit_type(self):
        return "Levenberg-Marquardt"

    def objective_function(self, fit_process_parameters):

        likelihood = []

        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.model.photometry:

            residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                       pyLIMA_parameters,
                                                                                                       norm=True,
                                                                                                       rescaling_photometry_parameters=None)

            likelihood = np.append(likelihood, residus)


        if self.model.astrometry:

            residuals = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                  pyLIMA_parameters,
                                                                                                  norm=True,
                                                                                                  rescaling_astrometry_parameters=None)
            residus = np.r_[residuals[:, 0], residuals[:, 2]]  # res_ra,res_dec
            likelihood = np.append(likelihood, residus)

        return likelihood



    def fit(self):

        start_time = python_time.time()

        # use the analytical Jacobian (faster) if no second order are present, else let the
        # algorithm find it.
        self.guess = self.initial_guess()


        n_data = 0
        for telescope in self.model.event.telescopes:
            n_data = n_data + telescope.n_data('flux')



        if self.model.Jacobian_flag != 'No Way':

            jacobian_function = self.residuals_Jacobian

        else:

            jacobian_function = '2-point'

        lm_fit = scipy.optimize.least_squares(self.objective_function, self.guess, method='lm',  max_nfev=50000,
                                              jac=jacobian_function, xtol=10**-10, ftol=10**-10, gtol=10 ** -10)

        fit_results = lm_fit['x'].tolist()
        fit_chi2 = lm_fit['cost']*2 #chi2

        try:
            # Try to extract the covariance matrix from the levenberg-marquard_fit output
            covariance_matrix = np.linalg.pinv(np.dot(lm_fit['jac'].T,lm_fit['jac']))

        except:

            covariance_matrix = np.zeros((len(self.model.model_dictionnary),
                                          len(self.model.model_dictionnary)))


        covariance_matrix *= fit_chi2/(n_data-len(self.model.model_dictionnary))
        computation_time = python_time.time() - start_time

        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        print('best_model:', fit_results, ' chi2:', fit_chi2)

        self.fit_results = {'best_model': fit_results, 'chi2' : fit_chi2, 'fit_time': computation_time,
                            'covariance_matrix': covariance_matrix}

    def fit_outputs(self):

        pyLIMA_plots.plot_lightcurves(self.model, self.fit_results['best_model'])
        pyLIMA_plots.plot_geometry(self.model, self.fit_results['best_model'])

        parameters = [key for key in self.model.model_dictionnary.keys() if ('source' not in key) and ('blend' not in key)]

        samples = np.random.multivariate_normal(self.fit_results['best_model'], self.fit_results['covariance_matrix'],10000)
        samples_to_plot = samples[:,:len(parameters)]
        pyLIMA_plots.plot_distribution(samples_to_plot,parameters_names = parameters )

