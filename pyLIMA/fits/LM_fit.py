import scipy
import time as python_time
import numpy as np
import sys

from pyLIMA.fits.ML_fit import MLfit
import pyLIMA.fits.objective_functions

class LMfit(MLfit):

    def __init__(self, model):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, rescalephotometry=False)

        self.guess = []
        self.fit_results = []
        self.fit_covariance_matrix = []
        self.fit_time = 0 #s

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

            likelihood = np.append(likelihood, residus**2)


        if self.model.astrometry:

            residus, errors = pyLIMA.fits.objective_functions.all_telescope_astrometric_residuals(self.model,
                                                                                                  pyLIMA_parameters,
                                                                                                  norm=True,
                                                                                                  rescaling_astrometry_parameters=None)

            likelihood = np.append(likelihood, residus**2)

        return (2*likelihood)**0.5

    def fit(self):

        starting_time = python_time.time()

        # use the analytical Jacobian (faster) if no second order are present, else let the
        # algorithm find it.
        self.guess = self.initial_guess()


        n_data = 0
        for telescope in self.model.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        n_parameters = len(self.model.model_dictionnary)

        #if self.model.Jacobian_flag == 'OK':

        # No Jacobian now
        lm_fit = scipy.optimize.least_squares(self.objective_function, self.guess, method='lm',  max_nfev=50000,
                                              gtol=10 ** -10)



        fit_result = lm_fit['x'].tolist()
        fit_result.append(lm_fit['cost']) #likelihood


        try:
            # Try to extract the covariance matrix from the levenberg-marquard_fit output
            covariance_matrix = np.linalg.pinv(np.dot(lm_fit['jac'].T,lm_fit['jac']))

        except:

            covariance_matrix = np.zeros((len(self.model.model_dictionnary),
                                          len(self.model.model_dictionnary)))


        covariance_matrix *= fit_result[-1]/(n_data-len(self.model.model_dictionnary))
        computation_time = python_time.time() - starting_time

        print(sys._getframe().f_code.co_name, ' : '+self.fit_type()+' fit SUCCESS')
        print(fit_result)
        self.fit_results = fit_result
        self.fit_covariance_matrix = covariance_matrix
        self.fit_time = computation_time

    def jacobian(self, fit_process_parameters):
        """Return the analytical Jacobian matrix, if requested by method LM.
        Available only for PSPL and FSPL without second_order.

        :param list fit_process_parameters: the model parameters ingested by the correpsonding
                                            fitting routine.
        :return: a numpy array which represents the jacobian matrix
        :rtype: array_like
        """

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)

        count = 0
        # import pdb;
        # pdb.set_trace()
        for telescope in self.model.event.telescopes:

            if count == 0:

                _jacobi = self.model.model_Jacobian(telescope, pyLIMA_parameters)

            else:

                _jacobi = np.c_[_jacobi, self.model.model_Jacobian(telescope, pyLIMA_parameters)]

            count += 1

        # The objective function is : (data-model)/errors

        _jacobi = -_jacobi
        jacobi = _jacobi[:-2]
        # Split the fs and g derivatives in several columns correpsonding to
        # each observatories
        start_index = 0
        dresdfs = _jacobi[-2]
        dresdg = _jacobi[-1]

        for telescope in self.model.event.telescopes:
            derivative_fs = np.zeros((len(dresdfs)))
            derivative_g = np.zeros((len(dresdg)))
            index = np.arange(start_index, start_index + len(telescope.lightcurve_flux['time'].value))
            derivative_fs[index] = dresdfs[index]
            derivative_g[index] = dresdg[index]
            jacobi = np.r_[jacobi, np.array([derivative_fs, derivative_g])]

            start_index = index[-1] + 1

        return jacobi.T