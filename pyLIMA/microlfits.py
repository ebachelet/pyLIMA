# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:39:32 2015

@author: ebachelet
"""
from __future__ import division
import time as python_time

import numpy as np
import scipy.optimize
import collections

import emcee

import microlmodels
import microloutputs
import microlguess
import microltoolbox


class MLFits(object):
    """
    ######## Fitter module ########

    This module fits the event with the selected attributes.

    **WARNING**: All fits (and so results) are made using data in flux.

    Attributes :

        event : the event object on which you perform the fit on. More details on the event module.

        model : The microlensing model you want to fit. Has to be an object define in
                microlmodels module.
                More details on the microlmodels module.

        method : The fitting method you want to use for the fit.

        guess : The guess you can give to the fit or the guess return by the initial_guess function.

        fit_results : the fit parameters returned by method LM and DE.

        fit_covariance : the fit parameters covariance matrix returned by method LM and DE.

        fit_time : the time needed to fit.

        MCMC_chains : the MCMC chains returns by the MCMC method

        MCMC_probabilities : the objective function computed for each chains of the MCMC method

        fluxes_MCMC_method : a string describing how you want to estimate the model fluxes for the MCMC method.

        outputs : the standard pyLIMA outputs. More details in the microloutputs module.

    :param object event: the event object on which you perform the fit on. More details on the
    event module.


    """

    def __init__(self, event):
        """The fit class has to be intialized with an event object."""

        self.event = event
        self.model = microlmodels.MLModels(event)
        self.method = 'None'
        self.guess = 0.0
        self.outputs = []
        self.fit_results = []
        self.fit_covariance = []
        self.fit_time = []
        self.MCMC_chains = []
        self.MCMC_probabilities = []
        self.fluxes_MCMC_method = ''

    def mlfit(self, model, method, flux_estimation_MCMC):
        """This function realize the requested microlensing fit, and set the according results
        attributes.

        :param object model: the model object requested. More details on the microlmodels module.

        :param string method: The fitting method you want to use. Has to be a string  in :

                                 'LM' --> Levenberg-Marquardt algorithm. Based on the
                                 scipy.optimize.leastsq routine.
                                          **WARNING** : the parameter maxfev (number of maximum
                                          iterations) is set to 50000
                                          the parameter ftol (relative precision on the chi^2) is
                                          set to 0.00001
                                          your fit may not converge because of these limits.
                                          The starting points of this method are found using the
                                          initial_guess method.
                                          Obviously, this can fail. In this case, switch to
                                          method 'DE'.

                                 'DE' --> Differential evolution algoritm. Based on the
                                 scipy.optimize.differential_evolution.
                                          Look Storn & Price (1997) : "Differential Evolution – A
                                          Simple and Efficient Heuristic for global Optimization
                                          over Continuous Spaces"
                                          Because this method is heuristic, it is not 100% sure a
                                          satisfying solution is found. Just relaunch :)
                                          The result is then use as a starting point for the 'LM'
                                          method.


                                 'MCMC' --> Monte-Carlo Markov Chain algorithm. Based on the
                                 emcee python package :
                                          " emcee: The MCMC Hammer" (Foreman-Mackey et al. 2013).
                                          The inital population is computed around the best
                                          solution return by
                                          the 'DE' method.

        Note that a sanity check is done post-fit to assess the fit quality with the check_fit
        function.
        """

        self.event.check_event()

        self.model = model
        self.method = method
        self.fluxes_MCMC_method = flux_estimation_MCMC

        self.model.define_model_parameters()

        if self.method == 'LM':
            self.guess = self.initial_guess()
            self.fit_results, self.fit_covariance, self.fit_time = self.lmarquardt()

        if self.method == 'DE':
            self.fit_results, self.fit_covariance, self.fit_time = self.differential_evolution()

        if self.method == 'MCMC':
            self.MCMC_chains, self.MCMC_probabilities = self.MCMC()

        fit_quality_flag = 'Good Fit'

        if self.method != 'MCMC':
            fit_quality_flag = self.check_fit()

        if fit_quality_flag == 'Bad Fit':

            if self.method == 'LM':

                print 'We have to change method, this fit was unsuccessfull. We decided to switch ' \
                      '' \
                      'method to "DE"'

                self.method = 'DE'
                self.mlfit(self.model, self.method, self.fluxes_MCMC_method)

            else:

                print 'Unfortunately, this is too hard for pyLIMA :('

    def check_fit(self):
        """Check if the fit results and covariance make sens.

         0.0 terms or a negative term in the diagonal covariance matrix indicate the fit is not
         reliable.

         A negative source flux is also counted as a bad fit.

         A negative rho or rho> 0.1 is also consider as a bad fit

         :return: a flag indicated good or bad fit ('Good Fit' or 'Bad Fit')
         :rtype: string
        """

        flag_quality = 'Good Fit'
        negative_covariance_diagonal = np.diag(self.fit_covariance) < 0

        if (0.0 in self.fit_covariance) | (True in negative_covariance_diagonal) | \
                (np.isnan(self.fit_covariance).any()) | (np.isinf(self.fit_covariance).any()):
            print 'Your fit probably wrong. Cause ==> bad covariance matrix'
            flag_quality = 'Bad Fit'
            return flag_quality

        for i in self.event.telescopes:

            if self.fit_results[self.model.model_dictionnary['fs_' + i.name]] < 0:
                print 'Your fit probably wrong. Cause ==> negative source flux for telescope ' + \
                      i.name
                flag_quality = 'Bad Fit'
                return flag_quality

        if 'rho' in self.model.model_dictionnary.keys():

            if (self.fit_results[self.model.model_dictionnary['rho']] > 0.1) | \
                    (self.fit_results[self.model.model_dictionnary['rho']] < 0.0):
                print 'Your fit probably wrong. Cause ==> bad rho '
                flag_quality = 'Bad Fit'
                return flag_quality

        return flag_quality

    def initial_guess(self):
        """Try to estimate the microlensing parameters. Only use for PSPL and FSPL
           models. More details on microlguess module.

           :return guess_parameters: a list containing parameters guess related to the model.
           :rtype: list
        """
        # Estimate  the Paczynski parameters

        if self.model.paczynski_model == 'PSPL':
            guess_paczynski_parameters, f_source = microlguess.initial_guess_PSPL(self.event)

        if self.model.paczynski_model == 'FSPL':
            guess_paczynski_parameters, f_source = microlguess.initial_guess_FSPL(self.event)

        # Estimate  the telescopes fluxes (flux_source + g_blending) parameters, with a PSPL model

        fake_model = microlmodels.MLModels(self.event, 'PSPL')
        telescopes_fluxes = self.find_fluxes(guess_paczynski_parameters, fake_model)

        # The survey fluxes are already known from microlguess
        telescopes_fluxes[0] = f_source
        telescopes_fluxes[1] = 0.0

        if self.model.parallax_model[0] != 'None':
            guess_paczynski_parameters = guess_paczynski_parameters + [0.0, 0.0]

        if self.model.xallarap_model[0] != 'None':
            guess_paczynski_parameters = guess_paczynski_parameters + [0, 0]

        if self.model.orbital_motion_model[0] != 'None':
            guess_paczynski_parameters = guess_paczynski_parameters + [0, 0]

        if self.model.source_spots_model != 'None':
            guess_paczynski_parameters = guess_paczynski_parameters + [0]

        guess_paczynski_parameters += telescopes_fluxes

        guess_parameters_pyLIMA_standards = collections.namedtuple('parameters',
                                                                   self.model.pyLIMA_standards_dictionnary.keys())

        for key_parameter in self.model.pyLIMA_standards_dictionnary.keys():

            try:
                setattr(guess_parameters_pyLIMA_standards, key_parameter,
                        guess_paczynski_parameters[self.model.pyLIMA_standards_dictionnary[key_parameter]])

            except:

                pass

        fancy_parameters_guess = self.model.pyLIMA_standard_parameters_to_fancy_parameters(
            guess_parameters_pyLIMA_standards)

        model_guess_parameters = []
        for key_parameter in self.model.model_dictionnary.keys():
            model_guess_parameters.append(getattr(fancy_parameters_guess, key_parameter))

        return model_guess_parameters

    def MCMC(self):
        """ The MCMC method. Construct starting points of the chains around
            the best solution found by the 'DE' method.
            The objective function is :func:`chichi_MCMC`. Optimization
            is made on Paczynski parameters, fs and g are found using a linear fit (np.polyfit).

            Based on the emcee python package :
            " emcee: The MCMC Hammer" (Foreman-Mackey et al. 2013).
            Have a look here : http://dan.iel.fm/emcee/current/

            :return: a tuple containing (MCMC_chains, MCMC_probabilities)
            :rtype: tuple

            **WARNING** :
                   nwalkers is set to 200
                   nlinks is set to 100
                   nwalkers*nlinks MCMC steps in total
        """

        nwalkers = 200
        nlinks = 100

        differential_evolution_estimation = self.differential_evolution()[0]

        self.guess = differential_evolution_estimation

        print 'pre MCMC done'
        # Best solution

        best_solution = self.guess

        if self.fluxes_MCMC_method == 'MCMC':
            limit_parameters = len(self.model.model_dictionnary.keys())

        else:
            limit_parameters = len(self.model.parameters_boundaries)

        nwalkers = 200

        # Initialize the population of MCMC
        population = []

        count_walkers = 0
        while count_walkers < nwalkers:

            # Construct an individual of the population around the best solution.
            individual = []
            for parameter_key in self.model.model_dictionnary.keys()[:limit_parameters]:

                parameter_trial = microlguess.MCMC_parameters_initialization(parameter_key,
                                                                             self.model.model_dictionnary,
                                                                             best_solution)

                if parameter_trial:

                    for parameter in parameter_trial:
                        individual.append(parameter)

            # fluxes = self.find_fluxes(individual,self.model)
            # individual += fluxes

            chichi = self.chichi_MCMC(individual)

            if chichi != -np.inf:
                # np.array(individual)
                population.append(np.array(individual))
                count_walkers += 1
        # number_of_parameters = number_of_paczynski_parameters + len(fluxes)
        # number_of_parameters = number_of_paczynski_parameters

        number_of_parameters = len(individual)
        sampler = emcee.EnsembleSampler(nwalkers, number_of_parameters, self.chichi_MCMC, a=2.0)

        # First estimation using population as a starting points.

        final_positions, final_probabilities, state = sampler.run_mcmc(population, nlinks)
        print 'MCMC preburn done'
        sampler.reset()

        # Final estimation using the previous output.

        sampler.run_mcmc(final_positions, nlinks)

        MCMC_chains = sampler.chain
        MCMC_probabilities = sampler.lnprobability

        return MCMC_chains, MCMC_probabilities

    def differential_evolution(self):
        """  The DE method. Differential evolution algorithm. The objective function is
        :func:`chichi_differential_evolution`.
         Based on the scipy.optimize.differential_evolution.
         Look Storn & Price (1997) :
         "Differential Evolution – A Simple and Efficient Heuristic for
         global Optimization over Continuous Spaces"

         :return: a tuple containing (fit_results, fit_covariance, computation_time)
         :rtype: tuple

         **WARNING** :
                   tol (relative standard deviation of the objective function) is set to 10^-6
                   popsize (the total number of individuals is :
                   popsize*number_of_paczynski_parameters)
                   is set to 20 mutation is set to (0.5, 1.5)
                   recombination is set to 0.6
                   These parameters can avoid the fit to properly converge (expected to be rare :)).
                   Just relaunch should be fine.
        """

        starting_time = python_time.time()
        differential_evolution_estimation = scipy.optimize.differential_evolution(
            self.chichi_differential_evolution,
            bounds=self.model.parameters_boundaries,
            mutation=0.6, popsize=20,
            tol=0.000001,
            recombination=0.6, polish='True',
            disp=False
        )

        # paczynski_parameters are all parameters to compute the model, excepted the telescopes fluxes.
        paczynski_parameters = differential_evolution_estimation['x'].tolist()

        print 'DE converge to objective function : f(x) = ', str(differential_evolution_estimation['fun'])
        # Construct the guess for the LM method. In principle, guess and outputs of the LM
        # method should be very close.

        self.guess = paczynski_parameters + self.find_fluxes(paczynski_parameters, self.model)

        fit_results, fit_covariance, fit_time = self.lmarquardt()

        computation_time = python_time.time() - starting_time

        return fit_results, fit_covariance, computation_time

    def lmarquardt(self):
        """The LM method. This is based on the Levenberg-Marquardt algorithm:

           "A Method for the Solution of Certain Problems in Least Squares"
           Levenberg, K. Quart. Appl. Math. 2, 1944, p. 164-168
           "An Algorithm for Least-Squares Estimation of Nonlinear Parameters"
           Marquardt, D. SIAM J. Appl. Math. 11, 1963, p. 431-441

           Based on scipy.optimize.leastsq python routine, which is based on MINPACK's lmdif and
           lmder
           algorithms (fortran based).

           The objective function is :func:`residuals_LM`.
           The starting point parameters are self.guess.
           the Jacobian is given by :func:`LM_Jacobian`.

           The fit is performed on all parameters : Paczynski parameters and telescopes fluxes.

           :return: a tuple containing (fit_results, fit_covariance, computation_time)
           :rtype: tuple

           **WARNING**:
                     ftol (relative error desired in the sum of square) is set to 10^-6
                     maxfev (maximum number of function call) is set to 50000
                     These limits can avoid the fit to properly converge (expected to be rare :))
        """
        starting_time = python_time.time()

        # use the analytical Jacobian (faster) if no second order are present, else let the
        # algorithm find it.
        ### NEED CHANGE ###
        # import pdb; pdb.set_trace()

        if (self.model.parallax_model[0] == 'None') & (len(self.model.fancy_to_pyLIMA_dictionnary) == 0):
            lmarquardt_fit = scipy.optimize.leastsq(self.residuals_LM, self.guess, maxfev=50000,
                                                    Dfun=self.LM_Jacobian, col_deriv=1, full_output=1, ftol=10 ** -6,
                                                    xtol=10 ** -10, gtol=10 ** -5)
        else:

            lmarquardt_fit = scipy.optimize.leastsq(self.residuals_LM, self.guess, maxfev=50000, full_output=1,
                                                    ftol=10 ** -6, xtol=10 ** -10,
                                                    gtol=10 ** -5)

        computation_time = python_time.time() - starting_time

        fit_result = lmarquardt_fit[0].tolist()
        fit_result.append(microltoolbox.chichi(self.residuals_LM, lmarquardt_fit[0]))

        n_data = 0.0

        for telescope in self.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        n_parameters = len(self.model.model_dictionnary)

        try:
            # Try to extract the covariance matrix from the lmarquard_fit output

            if np.all(lmarquardt_fit[1].diagonal() > 0) & (lmarquardt_fit[1] is not None):
                # Normalise the output by the reduced chichi
                covariance_matrix = lmarquardt_fit[1] * fit_result[-1] / (n_data - n_parameters)

            # Try to do it "manually"
            else:

                print ' Attempt to construct a rough covariance matrix'
                jacobian = self.LM_Jacobian(fit_result)

                covariance_matrix = np.linalg.inv(np.dot(jacobian, jacobian.T))
                # Normalise the output by the reduced chichi
                covariance_matrix = covariance_matrix * fit_result[-1] / (n_data - n_parameters)

                # Construct a dummy covariance matrix
                if np.any(lmarquardt_fit[1].diagonal() > 0):
                    print 'Bad covariance covariance matrix'
                    covariance_matrix = np.zeros((len(self.model.model_dictionnary),
                                                  len(self.model.model_dictionnary)))

        # Construct a dummy covariance matrix
        except:
            print 'Bad covariance covariance matrix'
            covariance_matrix = np.zeros((len(self.model.model_dictionnary),
                                          len(self.model.model_dictionnary)))

        # import pdb; pdb.set_trace()
        return fit_result, covariance_matrix, computation_time

    def LM_Jacobian(self, fit_process_parameters):
        """Return the analytical Jacobian matrix, if requested by method LM.
        Available only for PSPL and FSPL without second_order.

        :param list fit_process_parameters: the model parameters ingested by the correpsonding
        fitting routine.
        :return: a numpy array which represents the jacobian matrix
        :rtype: array_like
        """

        jacobi = self.model.model_Jacobian(fit_process_parameters)
        return jacobi

    def residuals_LM(self, fit_process_parameters):
        """The normalized residuals associated to the model and parameters.

           :param list fit_process_parameters: the model parameters ingested by the correpsonding
           fitting routine.

           :return: a numpy array which represents the residuals_i for each telescope,
           residuals_i=(data_i-model_i)/sigma_i
           :rtype: array_like
           The sum of square residuals gives chi^2.
        """

        # Construct an np.array with each telescope residuals
        residuals = np.array([])
        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)
        for telescope in self.event.telescopes:
            # Find the residuals of telescope observation regarding the parameters and model
            residus, priors = self.model_residuals(telescope, pyLIMA_parameters)
            # no prior here
            residuals = np.append(residuals, residus)

        return residuals

    def chichi_telescopes(self, fit_process_parameters):
        """Return a list of chi^2 (float) for individuals telescopes.

        :param list fit_process_parameters: the model parameters ingested by the correpsonding
        fitting routine.

        :returns: the chi^2 for each telescopes

        :rtype: list
        """

        residuals = self.residuals_LM(fit_process_parameters)
        chichi_list = []
        start_index = 0
        for telescope in self.event.telescopes:
            chichi_list.append(
                (residuals[start_index:start_index + len(telescope.lightcurve_flux)] ** 2).sum())

            start_index += len(telescope.lightcurve_flux)

        return chichi_list

    def chichi_differential_evolution(self, fit_process_parameters):
        """Return the chi^2 for the DE method. There is some priors here.

        :param list fit_process_parameters: the model parameters ingested by the correpsonding
        fitting routine.

        :returns: the chi^2

        :rtype: float
        """
        residuals = np.array([])
        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)
        for telescope in self.event.telescopes:

            # Find the residuals of telescope observation regarding the parameters and model

            residus, priors = self.model_residuals(telescope, pyLIMA_parameters)

            # Little prior here, need to be changed
            if priors == np.inf:
                return np.inf

            residuals = np.append(residuals, residus)

        chichi = (residuals ** 2).sum()
        return chichi

    def chichi_MCMC(self, fit_process_parameters):
        """Return the chi^2 for the MCMC method. There is some priors here.

        :param list fit_process_parameters: the model parameters ingested by the correpsonding
        fitting routine.

        :returns: the chi^2

        :rtype: float
        """
        # prior_limit = microlpriors.microlensing_parameters_limits_priors(fit_process_parameters, self.model.parameters_boundaries)

        # if prior_limit == np.inf:

        # return -np.inf

        chichi = 0
        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)
        for telescope in self.event.telescopes:

            # Find the residuals of telescope observation regarding the parameters and model
            residus, priors = self.model_residuals(telescope, pyLIMA_parameters)

            # Little prior here, need to be chaneged
            if priors == np.inf:

                return -np.inf

            else:

                chichi += (residus ** 2).sum()
                # Little prior here, need to be changed

                chichi += priors

        return -chichi

    def model_residuals(self, telescope, fit_process_parameters):
        """ Compute the residuals and the priors of a telescope lightcurve according to the model.

        :param object telescope: a telescope object. More details in telescopes module.
        :param list fit_process_parameters: the model parameters ingested by the correpsonding
        fitting routine.

        :return: the residuals in flux, the priors
        :rtype: array_like, float
        """
        lightcurve = telescope.lightcurve_flux

        flux = lightcurve[:, 1]
        errflux = lightcurve[:, 2]

        microlensing_model = self.model.compute_the_microlensing_model(telescope, fit_process_parameters)

        residuals = (flux - microlensing_model[0]) / errflux

        priors = microlensing_model[1:]

        return residuals, priors

    def find_fluxes(self, fit_process_parameters, model):
        """Find telescopes flux associated (fs,g) to the model. Used for initial_guess and LM
        method.

        :param fit_process_parameters: the model parameters ingested by the correpsonding fitting
        routine.
        :param model: the on which you want to compute the fs,g parameters.

        :return: a list of tuple with the (fs,g) telescopes flux parameters.
        :rtype: list
        """

        telescopes_fluxes = []
        pyLIMA_parameters = model.compute_pyLIMA_parameters(fit_process_parameters)
        for telescope in self.event.telescopes:
            lightcurve = telescope.lightcurve_flux

            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]

            amplification = model.model_magnification(telescope, pyLIMA_parameters)[0]

            f_source, f_blending = np.polyfit(amplification, flux, 1, w=1 / errflux)
            # Prior here
            if f_source < 0:

                telescopes_fluxes.append(np.min(flux))
                telescopes_fluxes.append(0.0)
            else:
                telescopes_fluxes.append(f_source)
                telescopes_fluxes.append(f_blending / f_source)
        return telescopes_fluxes

    def produce_outputs(self):
        """ Produce the standard outputs for a fit.
        More details in microloutputs module.
        """

        if self.method != 'MCMC':

            outputs = microloutputs.LM_outputs(self)

        else:

            outputs = microloutputs.MCMC_outputs(self)

        self.outputs = outputs
