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
import warnings
import emcee
import sys
import copy
from collections import OrderedDict
import multiprocessing as mp

from pyLIMA import microlmodels
from pyLIMA import microloutputs
from pyLIMA import microlguess
from pyLIMA import microltoolbox
from pyLIMA import microlpriors
from pyLIMA import microlcaustics

warnings.filterwarnings("ignore")

class FitException(Exception):
    pass

class MLFits(object):
    """
    ######## Fitter module ########

    This class contains the method to fit the event with the selected attributes.

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
        self.model = microlmodels.ModelPSPL(event)
        self.method = 'None'
        self.guess = []
        self.outputs = []
        self.fit_results = []
        self.fit_covariance = []
        self.fit_time = []
        self.DE_population = []
        self.binary_regime = None
        self.MCMC_chains = []
        self.MCMC_probabilities = []
        self.fluxes_MCMC_method = ''
        self.pool = None




    def mlfit(self, model, method, DE_population_size=10, flux_estimation_MCMC='MCMC', fix_parameters_dictionnary=None,
              grid_resolution=10, computational_pool=None, binary_regime=None):
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
                                          solution returned by
                                          the 'DE' method.


        :param int DE_population_size:  The population factor desired for the DE method. Default is 10.

        :param string flux_estimation_MCMC: The desired method to estimate the fluxes (f_source and g) of the
                                             telescopes. 'MCMC' will do this through an MCMC method (default) when
                                             everything else will do this thanks to a 1D polyfit through np.polyfit.

        Note that a sanity check is done post-fit to assess the fit quality with the check_fit
        function.
        """
        print('')
        print('Start fit on ' + self.event.name + ', with model ' + model.model_type + ' and method ' + method)
        self.event.check_event()

        self.model = model
        self.method = method
        self.fluxes_MCMC_method = flux_estimation_MCMC
        self.DE_population_size = DE_population_size

        self.model.define_model_parameters()

        if method != 'DE':

            self.check_parameters_boundaries()

        if computational_pool:

            pool = computational_pool
        else:
            pool = None

        if pool:
            manager = mp.Manager()
            self.DE_population = manager.list()

        self.binary_regime = binary_regime

        if self.method == 'LM':
            number_of_data = self.event.total_number_of_data_points()
            if number_of_data <= (len(self.model.model_dictionnary)):

                print("You do not have enough data points to use this method (LM), please switch to other methods." \
                      " Given the requested total model " + str(self.model.model_dictionnary.keys()) + \
                      " you need at least " + str(
                    len(self.model.model_dictionnary)) + ' data points to use the method LM!')
                return

            else:

                self.fit_results, self.fit_covariance, self.fit_time = self.lmarquardt()

        if self.method == 'TRF':
            self.fit_results, self.fit_covariance, self.fit_time = self.trust_region_reflective()
        if self.method == 'DE':
            self.fit_results, self.fit_covariance, self.fit_time = self.differential_evolution(pool)

        if self.method == 'MCMC':
            self.MCMC_chains = self.MCMC(pool)

        if self.method == 'GRIDS':
            self.fix_parameters_dictionnary = OrderedDict(
                sorted(fix_parameters_dictionnary.items(), key=lambda x: x[1]))
            self.grid_resolution = grid_resolution
            self.grid_parameters = self.grids()

        fit_quality_flag = 'Good Fit'

        #if self.method != 'MCMC':
        #    fit_quality_flag = self.check_fit()

        #if fit_quality_flag == 'Bad Fit':

        #    if self.method == 'LM':

        #        print('We have to change method, this fit was unsuccessfull. We decided to switch ' \
        #             '' \
         #             'method to "DE"')

                # self.method = 'DE'
                # self.mlfit(self.model, self.method, self.fluxes_MCMC_method)

         #   else:

         #      print('Unfortunately, this is too hard for pyLIMA :(')

    def check_parameters_boundaries(self):
        """Check if the the parameters guess are inside the parameters boundaries.

                 return: raise an Error
        """
        if self.model.parameters_guess != []:

            for index,parameter in enumerate(self.model.parameters_guess):

                 if (parameter<self.model.parameters_boundaries[index][0]) | (parameter>self.model.parameters_boundaries[index][1]):

                     print('ERROR :Guess parameters provided are outside the specified boundaries')
                     raise FitException('Parameters guess are outside the parameters boundaries')

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
        number_of_data = self.event.total_number_of_data_points()
        if number_of_data >= (len(self.model.model_dictionnary) + 2 * len(self.event.telescopes)):

            if (0.0 in self.fit_covariance):
                print('Your fit probably wrong. Cause ==> bad covariance matrix')
                flag_quality = 'Bad Fit'
                return flag_quality

        if (True in negative_covariance_diagonal) | \
                (np.isnan(self.fit_covariance).any()) | (np.isinf(self.fit_covariance).any()):
            print('Your fit probably wrong. Cause ==> bad covariance matrix')
            flag_quality = 'Bad Fit'
            return flag_quality

        for i in self.event.telescopes:

            if self.fit_results[self.model.model_dictionnary['fs_' + i.name]] < 0:
                print('Your fit probably wrong. Cause ==> negative source flux for telescope ' + \
                      i.name)
                flag_quality = 'Bad Fit'
                return flag_quality

        if 'rho' in list(self.model.model_dictionnary.keys()):

            if (self.fit_results[self.model.model_dictionnary['rho']] > 0.1) | \
                    (self.fit_results[self.model.model_dictionnary['rho']] < 0.0):
                print('Your fit probably wrong. Cause ==> bad rho ')
                flag_quality = 'Bad Fit'
                return flag_quality

        return flag_quality

    def initial_guess(self):
        """Try to estimate the microlensing parameters. Only use for PSPL and FSPL
           models. More details on microlguess module.

           :return guess_parameters: a list containing parameters guess related to the model.
           :rtype: list
        """

        if len(self.model.parameters_guess) == 0:

            # Estimate  the Paczynski parameters

            if self.model.model_type == 'PSPL':
                guess_paczynski_parameters, f_source = microlguess.initial_guess_PSPL(self.event)

            if self.model.model_type == 'FSPL':
                guess_paczynski_parameters, f_source = microlguess.initial_guess_FSPL(self.event)

            if self.model.model_type == 'DSPL':
                guess_paczynski_parameters, f_source = microlguess.initial_guess_DSPL(self.event)

            # Estimate  the telescopes fluxes (flux_source + g_blending) parameters

            
            telescopes_fluxes = self.find_fluxes(guess_paczynski_parameters, self.model)

            # The survey fluxes are already known from microlguess
            telescopes_fluxes[0] = f_source
            telescopes_fluxes[1] = 0.0

            if  'piEN' in self.model.model_dictionnary.keys():
                guess_paczynski_parameters = guess_paczynski_parameters + [0.0, 0.0]

            if  'XiEN' in self.model.model_dictionnary.keys():
                guess_paczynski_parameters = guess_paczynski_parameters + [0, 0]

            if 'dsdt' in self.model.model_dictionnary.keys():
                guess_paczynski_parameters = guess_paczynski_parameters + [0, 0]

            if 'spot_size' in self.model.model_dictionnary.keys():
                guess_paczynski_parameters = guess_paczynski_parameters + [0]



        else:

            guess_paczynski_parameters = list(self.model.parameters_guess)

            telescopes_fluxes = self.find_fluxes(guess_paczynski_parameters, self.model)

        guess_paczynski_parameters += telescopes_fluxes

        
        print(sys._getframe().f_code.co_name, ' : Initial parameters guess SUCCESS')
        return guess_paczynski_parameters

    def MCMC(self,pool):
        """ The MCMC method. Construct starting points of the chains around
            the best solution found by the 'DE' method.
            The objective function is :func:`chichi_MCMC`. Telescope flux (fs and g), can be optimized thanks to MCMC if
            flux_estimation_MCMC is 'MCMC', either they are derived through np.polyfit.

            Based on the emcee python package :
            " emcee: The MCMC Hammer" (Foreman-Mackey et al. 2013).
            Have a look here : http://dan.iel.fm/emcee/current/

            :return: a tuple containing (MCMC_chains, MCMC_probabilities)
            :rtype: tuple

            **WARNING** :
                   nwalkers is set to 16 times the len of pazynski_parameters
                   nlinks is set to 1000
                   5*nwalkers*nlinks MCMC steps in total
        """



        # start = python_time.time()

        if len(self.model.parameters_guess) == 0:

            differential_evolution_estimation = self.differential_evolution(pool)[0]
            self.DE_population_size = 10
            self.guess = differential_evolution_estimation

        else:

            self.guess = list(self.model.parameters_guess)
            self.guess += self.find_fluxes(self.guess, self.model)

        # Best solution

        limit_parameters = len(self.model.parameters_boundaries)
        if self.fluxes_MCMC_method != 'MCMC':
            best_solution = self.guess[:limit_parameters]
        else:
            best_solution = self.guess
        nwalkers = 16 * len(best_solution)
        nlinks = 1000

        # Initialize the population of MCMC
        population = []

        count_walkers = 0

        while count_walkers < nwalkers:

            # Construct an individual of the population around the best solution.
            individual = []
            for parameter_key in list(self.model.model_dictionnary.keys())[:limit_parameters]:

                parameter_trial = microlguess.MCMC_parameters_initialization(parameter_key,
                                                                             self.model.model_dictionnary,
                                                                             best_solution)

                if parameter_trial:

                    for parameter in parameter_trial:
                        individual.append(parameter)

            if self.fluxes_MCMC_method == 'MCMC':
                fluxes = self.find_fluxes(individual, self.model)
                individual += fluxes

            chichi = self.chichi_MCMC(individual)

            if chichi != -np.inf:
                # np.array(individual)
                # print count_walkers

                population.append(np.array(individual))
                count_walkers += 1

        print('pre MCMC done')

        number_of_parameters = len(individual)

        try:
            # create a new MPI pool
            from schwimmbad import MPIPool
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
        except:

            pass

        sampler = emcee.EnsembleSampler(nwalkers, number_of_parameters, self.chichi_MCMC,
                                        a=2.0, pool= pool)
        # First estimation using population as a starting points.

        final_positions, final_probabilities, state = sampler.run_mcmc(population, nlinks, progress=True)

        print('MCMC preburn done')


        sampler.reset()

        sampler.run_mcmc(final_positions, nlinks, progress=True)
        MCMC_chains = np.c_[sampler.get_chain().reshape(nlinks*nwalkers,number_of_parameters),sampler.get_log_prob().reshape(nlinks*nwalkers)]

        # Final estimation using the previous output.
        #for positions, probabilities, states in sampler.sample(final_positions, iterations=  nlinks,
        #                                                       storechain=True):
        #    chains = np.c_[positions, probabilities]
        #    if MCMC_chains is not None:

        #        MCMC_chains = np.r_[MCMC_chains, chains]
        #    else:

        #        MCMC_chains = chains

        print(sys._getframe().f_code.co_name, ' : MCMC fit SUCCESS')
        return MCMC_chains

    def chichi_MCMC(self, fit_process_parameters):
        """Return the chi^2 for the MCMC method. There is some priors here.

        :param list fit_process_parameters: the model parameters ingested by the correpsonding
                                            fitting routine.

        :returns: here, the return is -chi^2/2 (likelihood)

        :rtype: float
        """

        chichi = 0

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)

        prior = microlpriors.priors_on_models(pyLIMA_parameters, self.model, binary_regime = self.binary_regime)

        if prior != np.inf:

            chichi -= prior

        else:

            return -np.inf

        for telescope in self.event.telescopes:
            # Find the residuals of telescope observation regarding the parameters and model
            residus = self.model_residuals(telescope, pyLIMA_parameters)

            chichi += (residus ** 2).sum()

        return -chichi / 2

    def differential_evolution(self,pool):
        """  The DE method. Differential evolution algorithm. The objective function is
        :func:`chichi_differential_evolution`. The flux parameters are estimated through np.polyfit.
         Based on the scipy.optimize.differential_evolution.
         Look Storn & Price (1997) :
         "Differential Evolution – A Simple and Efficient Heuristic for
         global Optimization over Continuous Spaces"

         :return: a tuple containing (fit_results, fit_covariance, computation_time)
         :rtype: tuple

         **WARNING** :
                   tol (relative standard deviation of the objective function) is set to 10^-4

                   popsize (the total number of individuals is :
                   popsize*number_of_paczynski_parameters)
                   is set to DE_population_size

                   mutation is set to (0.1, 1.5)

                   recombination is set to 0.7

                   These parameters can avoid the fit to properly converge (expected to be rare :)).
                   Just relaunch should be fine.
        """
        starting_time = python_time.time()
        if pool:
             worker = pool.map
        else:
             worker = 1
        differential_evolution_estimation = scipy.optimize.differential_evolution(
            self.chichi_differential_evolution,
            bounds=self.model.parameters_boundaries,
            mutation=(0.5,1.5), popsize=int(self.DE_population_size), maxiter=100000, tol=0.0,
            atol=1, strategy='rand1bin',
            recombination=0.7, polish=True, init='latinhypercube',
            disp=True,workers = worker,
        )


        # paczynski_parameters are all parameters to compute the model, excepted the telescopes fluxes.
        paczynski_parameters = differential_evolution_estimation['x'].tolist()

        print('DE converge to objective function : f(x) = ', str(differential_evolution_estimation['fun']))
        print('DE converge to parameters : = ', differential_evolution_estimation['x'].astype(str))

        self.DE_population = np.array(self.DE_population)

        # Construct the guess for the LM method. In principle, guess and outputs of the LM
        # method should be very close.

        number_of_data = self.event.total_number_of_data_points()
        if number_of_data <= (len(self.model.model_dictionnary)):

            print("You do not have enough data points to use LM method to estimate the covariance matrix." \
                  "The covariance matrix is set to 0.0. please switch to MCMC if you need errors estimation.")

            fit_results = paczynski_parameters + self.find_fluxes(paczynski_parameters, self.model) + \
                          [differential_evolution_estimation['fun']]
            fit_covariance = np.zeros((len(paczynski_parameters) + 2 * len(self.event.telescopes),
                                       len(paczynski_parameters) + 2 * len(self.event.telescopes)))

        else:

            self.guess = paczynski_parameters + self.find_fluxes(paczynski_parameters, self.model)

            fit_results, fit_covariance, fit_time = self.lmarquardt()

        computation_time = python_time.time() - starting_time

        print(sys._getframe().f_code.co_name, ' : Differential evolution fit SUCCESS')
        return fit_results, fit_covariance, computation_time

    def chichi_differential_evolution(self, fit_process_parameters):
        """Return the chi^2 for the DE method.

        :param list fit_process_parameters: the model parameters ingested by the correpsonding
        fitting routine.

        :returns: the chi^2

        :rtype: float
        """
        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)

        chichi = 0.0

        prior = microlpriors.priors_on_models(pyLIMA_parameters, self.model, binary_regime=self.binary_regime)

        if prior != np.inf:
            chichi += prior
            for telescope in self.event.telescopes:
                # Find the residuals of telescope observation regarding the parameters and model
                residus = self.model_residuals(telescope, pyLIMA_parameters)

                chichi += (residus ** 2).sum()

        else:

            chichi = np.inf


        self.DE_population.append(fit_process_parameters.tolist() + [chichi])

        return chichi

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

           :return: a tuple containing (fit_results, covariance_matrix, computation_time)
           :rtype: tuple

           **WARNING**:
                     ftol (relative error desired in the sum of square) is set to 10^-6
                     maxfev (maximum number of function call) is set to 50000
                     These limits can avoid the fit to properly converge (expected to be rare :))
        """
        starting_time = python_time.time()

        # use the analytical Jacobian (faster) if no second order are present, else let the
        # algorithm find it.
        if self.guess == []:
            self.guess = self.initial_guess()
        n_data = 0
        for telescope in self.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        n_parameters = len(self.model.model_dictionnary)

        if self.model.Jacobian_flag == 'OK':
            lmarquardt_fit = scipy.optimize.leastsq(self.residuals_LM, self.guess, maxfev=50000,
                                                    Dfun=self.LM_Jacobian, col_deriv=0, full_output=1, ftol=10 ** -8,
                                                    xtol=10 ** -10, gtol=10 ** -10)
            fit_result = lmarquardt_fit[0].tolist()
            fit_result.append(microltoolbox.chichi(self.residuals_LM, lmarquardt_fit[0]))

            try:
            # Try to extract the covariance matrix from the lmarquard_fit output

                covariance_matrix = lmarquardt_fit[1] * fit_result[-1] / (n_data - n_parameters)

            except:

                covariance_matrix = np.zeros((len(self.model.model_dictionnary),
                                              len(self.model.model_dictionnary)))
        else:


            lmarquardt_fit = scipy.optimize.least_squares(self.residuals_LM, self.guess, method='lm', x_scale='jac', ftol=10 ** -10,
                                                          xtol=10 ** -10, gtol=10 ** -10,
                                                         )

            fit_result = lmarquardt_fit['x'].tolist()
            fit_result.append(microltoolbox.chichi(self.residuals_LM, lmarquardt_fit['x']))

            try:
                # Try to extract the covariance matrix from the lmarquard_fit output
                jacobian = lmarquardt_fit['jac']

                covariance_matrix = np.linalg.inv(np.dot(jacobian.T, jacobian))



            except:

                covariance_matrix = np.zeros((len(self.model.model_dictionnary),
                                              len(self.model.model_dictionnary)))


        computation_time = python_time.time() - starting_time


        # import pdb; pdb.set_trace()
        print(sys._getframe().f_code.co_name, ' : Levenberg_marquardt fit SUCCESS')
        print(fit_result)
        return fit_result, covariance_matrix, computation_time


    def residuals_LM(self, fit_process_parameters):
        """The normalized residuals associated to the model and parameters.

           :param list fit_process_parameters: the model parameters ingested by the correpsonding
                                               fitting routine.

           :return: a numpy array which represents the residuals_i for each telescope,
                    residuals_i=(data_i-model_i)/sigma_i
           :rtype: array_like
           The sum of square residuals gives chi^2.
        """

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)

       
        residuals = np.array([])

        for telescope in self.event.telescopes:
            # Find the residuals of telescope observation regarding the parameters and model
            residus = self.model_residuals(telescope, pyLIMA_parameters)

            residuals = np.append(residuals, residus)

        prior = microlpriors.priors_on_models(pyLIMA_parameters,self.model, binary_regime=self.binary_regime)

        if prior != np.inf:

           residuals += prior/len(residuals)

        else:

           residuals *= prior

        # print python_time.time()-start

        return residuals

    def LM_Jacobian(self, fit_process_parameters):
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
        for telescope in self.event.telescopes:

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

        for telescope in self.event.telescopes:
            derivative_fs = np.zeros((len(dresdfs)))
            derivative_g = np.zeros((len(dresdg)))
            index = np.arange(start_index, start_index + len(telescope.lightcurve_flux[:, 0]))
            derivative_fs[index] = dresdfs[index]
            derivative_g[index] = dresdg[index]
            jacobi = np.r_[jacobi, np.array([derivative_fs, derivative_g])]

            start_index = index[-1] + 1

        return jacobi.T

    def trust_region_reflective(self):

        starting_time = python_time.time()

        # use the analytical Jacobian (faster) if no second order are present, else let the
        # algorithm find it.
        if self.guess == []:
            self.guess = self.initial_guess()

        bounds_min = [i[0] for i in self.model.parameters_boundaries] + [0, -np.inf] * len(self.event.telescopes)
        bounds_max = [i[1] for i in self.model.parameters_boundaries] + [np.inf, np.inf] * len(self.event.telescopes)

        if self.model.Jacobian_flag == 'OK':
            trf_fit = scipy.optimize.least_squares(self.residuals_LM, self.guess, max_nfev=50000,
                                                   jac=self.LM_Jacobian, bounds=(bounds_min, bounds_max), ftol=10 ** -6,
                                                   xtol=10 ** -10, gtol=10 ** -5)
        else:

            trf_fit = scipy.optimize.least_squares(self.residuals_LM, self.guess, max_nfev=50000,
                                                   bounds=(bounds_min, bounds_max), ftol=10 ** -6,
                                                   xtol=10 ** -10, gtol=10 ** -5)
        computation_time = python_time.time() - starting_time

        fit_result = np.copy(trf_fit['x']).tolist()
        fit_result += [2 * trf_fit['cost']]

        try:

            jacobian = trf_fit['jac']

        except:

            jacobian = self.LM_Jacobian(fit_result)

        covariance_matrix = np.linalg.inv(np.dot(jacobian.T, jacobian))
        n_data = 0
        for telescope in self.event.telescopes:
            n_data = n_data + telescope.n_data('flux')

        n_parameters = len(self.model.model_dictionnary)
        covariance_matrix *= fit_result[-1] / (n_data - n_parameters)
        print(sys._getframe().f_code.co_name, ' : TRF fit SUCCESS')
        print(fit_result)
        return fit_result, covariance_matrix, computation_time

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

    def model_residuals(self, telescope, pyLIMA_parameters):
        """ Compute the residuals of a telescope lightcurve according to the model.

        :param object telescope: a telescope object. More details in telescopes module.
        :param object pyLIMA_parameters: object containing the model parameters, see microlmodels for more details

        :return: the residuals in flux, the priors
        :rtype: array_like, float
        """



        lightcurve = telescope.lightcurve_flux

        flux = lightcurve[:, 1]
        errflux = lightcurve[:, 2]

        microlensing_model = self.model.compute_the_microlensing_model(telescope,pyLIMA_parameters)

        residuals = (flux - microlensing_model[0]) / errflux

        return residuals

    def all_telescope_residuals(self, pyLIMA_parameters):
        """ Compute the residuals of all telescopes according to the model.

        :param object pyLIMA_parameters: object containing the model parameters, see microlmodels for more details

        :return: the residuals in flux,
        :rtype: list, a list of array of residuals in flux
        """

        residuals = []
        for telescope in self.event.telescopes:
            # Find the residuals of telescope observation regarding the parameters and model
            residus = self.model_residuals(telescope, pyLIMA_parameters)
            # no prior here
            residuals.append(residus)
        # print python_time.time()-start
        return residuals

    def find_fluxes(self, fit_process_parameters, model):
        """Find telescopes flux associated (fs,g) to the model. Used for initial_guess and LM
        method.

        :param list fit_process_parameters: the model parameters ingested by the correpsonding fitting
                                       routine.
        :param object model: a microlmodels which you want to compute the fs,g parameters.

        :return: a list of tuple with the (fs,g) telescopes flux parameters.
        :rtype: list
        """

        telescopes_fluxes = []
        pyLIMA_parameters = model.compute_pyLIMA_parameters(fit_process_parameters)

        for telescope in self.event.telescopes:

            flux = telescope.lightcurve_flux[:, 1]

            ml_model, f_source, f_blending = model.compute_the_microlensing_model(telescope, pyLIMA_parameters)

            # Prior here
            if f_source < 0:

                telescopes_fluxes.append(np.min(flux))
                telescopes_fluxes.append(0.0)
            else:
                telescopes_fluxes.append(f_source)
                telescopes_fluxes.append(f_blending)
        return telescopes_fluxes

    def grids(self):
        """ Compute models on a grid. ON CONSTRUCTION.
        """
        parameters_on_the_grid = []

        for parameter_name in self.fix_parameters_dictionnary:
            parameter_range = self.model.parameters_boundaries[self.model.model_dictionnary[parameter_name]]

            parameters_on_the_grid.append(
                np.linspace(parameter_range[0], parameter_range[1],
                            self.grid_resolution))

        hyper_grid = self.construct_the_hyper_grid(parameters_on_the_grid)

        self.new_parameters_boundaries = self.redefine_parameters_boundaries()

        if self.pool is not None:
            computational_map = self.pool.map

        else:
            computational_map = map

        grid_results = list(
            computational_map(emcee.ensemble._function_wrapper(self.optimization_on_grid_pixel, args=[], kwargs={}),
                              hyper_grid))

        return np.array(grid_results)

    def optimization_on_grid_pixel(self, grid_pixel_parameters):

        differential_evolution_estimation = scipy.optimize.differential_evolution(
            self.chichi_grids,
            bounds=self.new_parameters_boundaries, args=tuple(grid_pixel_parameters.tolist()),
            mutation=(0.8, 1.2), popsize=10, maxiter=1000,
            tol=0.0, atol=0.1, strategy='best1bin',
            recombination=0.2, polish=True,
            disp=True
        )

        best_parameters = self.reconstruct_fit_process_parameters(differential_evolution_estimation['x'],
                                                                  grid_pixel_parameters)

        best_parameters += [differential_evolution_estimation['fun']]

        print(sys._getframe().f_code.co_name, ' Grid step on ' + str(grid_pixel_parameters.tolist()).strip(
            '[]') + ' converge to f(x) = ' + str(differential_evolution_estimation['fun']))

        return best_parameters

    def chichi_grids(self, moving_parameters, *fix_parameters):
        """ Compute chi^2. ON CONSTRUCTION.
        """
        fit_process_parameters = self.reconstruct_fit_process_parameters(moving_parameters, fix_parameters)

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)

        chichi = 0.0
        for telescope in self.event.telescopes:
            # Find the residuals of telescope observation regarding the parameters and model

            residus = self.model_residuals(telescope, pyLIMA_parameters)

            chichi += (residus ** 2).sum()

        return chichi

    def reconstruct_fit_process_parameters(self, moving_parameters, fix_parameters):
        """ Reconstruc parameters. ON CONSTRUCTION.
        """
        fit_process_parameters = []

        for key in list(self.model.model_dictionnary.keys())[:len(self.model.parameters_boundaries)]:

            if key in self.moving_parameters_dictionnary:

                fit_process_parameters.append(moving_parameters[self.moving_parameters_dictionnary[key]])

            else:

                fit_process_parameters.append(fix_parameters[self.fix_parameters_dictionnary[key]])

        return fit_process_parameters

    def redefine_parameters_boundaries(self):
        """ Recompute the parameters boundaries. ON CONSTRUCTION.
        """
        parameters_boundaries = []
        self.moving_parameters_dictionnary = {}
        count = 0

        for indice, key in enumerate(list(self.model.model_dictionnary.keys())[:len(self.model.parameters_boundaries)]):

            if key not in self.fix_parameters_dictionnary.keys():
                parameters_boundaries.append(self.model.parameters_boundaries[indice])
                self.moving_parameters_dictionnary[key] = count
                count += 1
        return parameters_boundaries

    def construct_the_hyper_grid(self, parameters):
        """Define the grid. ON CONSTRUCTION.
        """
        params = map(np.asarray, parameters)
        grid = np.broadcast_arrays(*[x[(slice(None),) + (None,) * i] for i, x in enumerate(params)])

        reformate_grid = np.vstack(grid).reshape(len(parameters), -1).T
        return reformate_grid

    def produce_outputs(self):
        """ Produce the standard outputs for a fit.
        More details in microloutputs module.
        """

        outputs = microloutputs.fit_outputs(self)

        self.outputs = outputs

    def produce_fit_statistics(self):
        """ Produce the standard outputs for a fit.
        More details in microloutputs module.
        """

        stats_outputs = microloutputs.statistical_outputs(self)

        self.stats_outputs = stats_outputs

    def produce_pdf(self, output_directory='./'):
        """ ON CONSTRUCTION
        """
        microloutputs.pdf_output(self, output_directory)

    def produce_latex_table_results(self, output_directory='./'):
        """ ON CONSTRUCTION
        """
        microloutputs.latex_output(self, output_directory)
