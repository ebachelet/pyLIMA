import scipy
import time as python_time
import numpy as np
import sys

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions


class TRFfit(MLfit):

    def __init__(self, model, rescale_photometry=False):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, rescale_photometry)

        self.guess = []
        self.fit_results = []
        self.fit_covariance_matrix = []
        self.fit_time = 0 #s


    def fit_type(self):
        return "Trust Region Reflective"

    def objective_function(self, fit_process_parameters):

        likelihood = []

        model_parameters = fit_process_parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.model.photometry:

            if self.rescale_photometry:

                rescaling_photometry_parameters = fit_process_parameters[self.rescale_photometry_parameters_index]
                residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                       pyLIMA_parameters,
                                                                                                       norm=True,
                                                                                                       rescaling_photometry_parameters=rescaling_photometry_parameters)


            else:

                residus, errflux = pyLIMA.fits.objective_functions.all_telescope_photometric_residuals(self.model,
                                                                                                   pyLIMA_parameters,
                                                                                                   norm=True,
                                                                                                   rescaling_photometry_parameters=None)

            likelihood = np.append(likelihood, residus ** 2+np.log(2*np.pi)+2*np.log(errflux))

        if self.model.astrometry:
            residus, errors = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                  pyLIMA_parameters,
                                                                                                  norm=True,
                                                                                                  rescaling_astrometry_parameters=None)

            likelihood = np.append(likelihood, residus ** 2+np.log(2*np.pi)+2*np.log(errors))

        return likelihood

    def fit(self):

        starting_time = python_time.time()

        # use the analytical Jacobian (faster) if no second order are present, else let the
        # algorithm find it.
        self.guess = self.initial_guess()

        bounds_min = [i[0] for i in self.fit_parameters_boundaries]
        bounds_max = [i[1] for i in self.fit_parameters_boundaries]
        n_data = 0
        for telescope in self.model.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        n_parameters = len(self.model.model_dictionnary)


        # No Jacobian now
        lm_fit = scipy.optimize.least_squares(self.objective_function, self.guess, method='trf',loss=self.loss,
                                              bounds=(bounds_min, bounds_max),  max_nfev=50000, xtol=None,
                                              ftol=None, gtol=10 ** -1)

        import pdb;
        pdb.set_trace()
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

    def loss(self, z):

        fz = np.sqrt(z)
        dfz = 1/(2*fz)
        d2fz = -1/(4*fz**3)

        return np.array([fz, dfz, d2fz])


