# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:39:32 2015

@author: ebachelet
"""
from __future__ import division
import time as TIME

import numpy as np
from scipy.optimize import leastsq, differential_evolution
import scipy.signal as ss
import emcee

import microlmodels
import microloutputs
import microlguess

class MLFits(object):
    """
    ######## Fitter module ########
    @author: Etienne Bachelet

    This module fits the event with the selected attributes.
    WARNING: All fits (and so results) are made using data in flux.

    Attributes :

         event : the event object on which you perform the fit on. More details on the event module.
 
	 model : The microlensing model you want to fit. Has to be an object define in microlmodels module.
                 More details on the microlmodels module.

	 method : The fitting method you want to use for the fit.
                 
	 guess : The guess you can give to the fit or the guess return by the initial_guess function.

	 fit_results : the fit parameters returned by method LM and DE. 

	 fit_covariance : the fit parameters covariance matrix returned by method LM and DE.
	
	 fit_time : the time needed to fit.
	
	 MCMC_chains : the MCMC chains returns by the MCMC method
	 
	 MCMC_probabilities : the objective function computed for each chains of the MCMC method

    """

    def __init__(self, event):
        """The fit class has to be intialized with an event. """

        self.event = event

    def mlfit(self, model, method):
        """This function realize the requested microlensin fit, and set the according results attributes.
	

		:param model: The microlensing model you want to fit. Has to be an object define in microlmodels module.
	                      More details on the microlmodels module.

    		:param method: The fitting method you want to use. Has to be a string  in :

              		       'LM' --> Levenberg-Marquardt algorithm. Based on the scipy.optimize.leastsq routine.
                       	        WARNING : the parameter maxfev (number of maximum iterations) is set to 50000
                       		the parameter ftol (relative precision on the chi^2) is set to 0.00001
                       		your fit may not converge because of these limits.
                       		The starting points of this method are found using the initial_guess method.
                       		Obviously, this can fail. In this case, switch to method 'DE'.
                        
              			'DE' --> Differential evolution algoritm. Based on the scipy.optimize.differential_evolution.
                       		Look Storn & Price (1997) : "Differential Evolution – A Simple and Efficient Heuristic for 		               		global Optimization over Continuous Spaces"
                       		Because this method is heuristic, it is not 100% sure a satisfying solution is found. Just 					relaunch :)
                       		The result is then use as a starting point for the 'LM' method.
                       
                       
              			'MCMC' --> Monte-Carlo Markov Chain algorithm. Based on the emcee python package :  
                         	" emcee: The MCMC Hammer" (Foreman-Mackey et al. 2013).
                         	The inital population is computed around the best solution return by
                         	the 'DE' method.
       				
	Note that a sanity check is done post-fit to assess the fit quality with the check_fit 
	function.

	"""
        

	self.model = model
        self.method = method

        if self.method == 'LM':

            self.guess = self.initial_guess()

            self.fit_results, self.fit_covariance, self.fit_time = self.lmarquardt()

        if self.method == 'DE':

            self.fit_results, self.fit_covariance, self.fit_time = self.diff_evolution()
                       

        if self.method == 'MCMC':
            
            self.MCMC_chains,self.MCMC_probabilities=self.MCMC()
        
        fit_quality_flag = 'Good Fit'
        
        if self.method != 'MCMC' :
            
            fit_quality_flag = self.check_fit()

        if fit_quality_flag == 'Bad Fit':

           if self.method == 'LM':

                print 'We have to change method, this fit was unsuccessfull. We decided to switch ' \
                      '' \
                      'method to "DE"'

                self.method = 'DE'
                self.mlfit(self.model, self.method)

           else:

                print 'Unfortunately, this is too hard for pyLIMA :('

    def check_fit(self):
        """Check if the fit results and covariance make sens.
         0.0 terms or a negative term in the diagonal covariance matrix indicate the fit is not
         reliable.
         A negative source flux is also counted as a bad fit.
         A negative rho or rho> 0.1 is also consider as a bad fit

	
        """

        flag_quality = 'Good Fit'
        diago = np.diag(self.fit_covariance) < 0

        if (0.0 in self.fit_covariance) | (True in diago) | ( np.isnan(self.fit_covariance).any()) | (np.isinf(self.fit_covariance).any()):

            print 'Your fit probably wrong. Cause ==> bad covariance matrix'
            flag_quality = 'Bad Fit'
            return flag_quality

        for i in self.event.telescopes:

            if self.fit_results[self.model.model_dictionnary['fs_' + i.name]] < 0:

                print 'Your fit probably wrong. Cause ==> negative source flux for telescope ' + \
                      i.name
                flag_quality = 'Bad Fit'
                return flag_quality

        if 'rho' in self.model.model_dictionnary:

            if (self.fit_results[self.model.model_dictionnary['rho']] >0.1) |(self.fit_results[self.model.model_dictionnary['rho']] <0.0) :

                print 'Your fit probably wrong. Cause ==> bad rho ' 
                flag_quality = 'Bad Fit'
                return flag_quality

        return flag_quality

    def initial_guess(self):
        """Try to estimate the microlensing parameters. Only use for PSPL and FSPL
	models. More details on microlguess module.
        """
	#Estimate  the Paczynski parameters

        if self.model.paczynski_model == 'PSPL':

            parameters,fs = microlguess.initial_guess_PSPL(self.event)        
        
        if self.model.paczynski_model == 'FSPL':

            parameters,fs = microlguess.initial_guess_FSPL(self.event)

        #Estimate  the telescopes fluxes (flux_source + g_blending) parameters  

        fake_model = microlmodels.MLModels(self.event, 'PSPL')
        fluxes = self.find_fluxes(parameters, fake_model)
        fluxes[0] = fs
        fluxes[1] = 0.0

        if self.model.parallax_model[0] != 'None':

            parameters = parameters + [0.0, 0.0]

        if self.model.xallarap_model[0] != 'None':

            parameters = parameters + [0, 0]

        if self.model.orbital_motion_model[0] != 'None':

            parameters = parameters + [0, 0]

        if self.model.source_spots_model != 'None':

            parameters = parameters + [0]

        parameters = parameters + fluxes

        return parameters

    def MCMC(self) :
        """ The MCMC method. Construc starting points of the chains around
            the best solution found by the 'DE' method.
            The objective function is chichi_MCMC. Optimization
            is made on Paczynski parameters, fs and g are found using a linear fit (np.polyfit).
	    Based on the emcee python package :  
            " emcee: The MCMC Hammer" (Foreman-Mackey et al. 2013).
	    Have a look here : http://dan.iel.fm/emcee/current/
            Launch nwalkers = 100*number_of_paczynski_parameters chains with 100 links
        """
        differential_evolution_estimation = differential_evolution(self.chichi_differential,
                                   bounds=self.model.parameters_boundaries,mutation=(0.5,1), popsize=30,
                                   recombination=0.7,polish='None')
	# Best solution
        best_solution = differential_evolution_estimation['x']
        
	number_of_paczynski_parameters = len(best_solution)
        nwalkers = 100*number_of_paczynski_parameters
	#nwalkers = 100
	# Initialize the population of MCMC        
	population = []

        i=0
        while i < nwalkers:
	    # Construct an individual of the population around the best solution. THIS NEED A REWORK 
	    # TO TAKE ACCOUNT SECOND ORDER!
	
            individual = []
            for j in range(number_of_paczynski_parameters):

                if j==0:
                    
                    individual.append(best_solution[j]+np.random.uniform(-1,1))
                if j==1:
                    
                    individual.append(best_solution[j]*(np.random.uniform(0.1,3)))
                if j==2:
            
                    individual.append(best_solution[j]*(np.random.uniform(0.1,3)))
                
                if j==3:
                    
                    individual.append(best_solution[j]*(np.random.uniform(0.1,3)))

		if j==4:
                    
                    individual.append(best_solution[j]*(np.random.uniform(0.1,3)))

		
            
            chichi = self.chichi_MCMC(individual)
            if chichi != -np.inf :
                
                population.append(np.array(individual))
                i+=1
           
     

        sampler = emcee.EnsembleSampler(nwalkers, number_of_paczynski_parameters, self.chichi_MCMC)

	# First estimation using population as a starting points.

        final_positions, final_probabilities, state = sampler.run_mcmc(population, 100)
	print 'MCMC preburn done'
        sampler.reset()
	
	# Final estimation using the previous output.

        final_positions, final_probabilities, state = sampler.run_mcmc(final_positions, 100)

	
        MCMC_chains = sampler.chain
        MCMC_probabilities = sampler.lnprobability
            
        return MCMC_chains, MCMC_probabilities

    def diff_evolution(self) :
        """  The DE method. Differential evolution algoritm. 
	     Based on the scipy.optimize.differential_evolution.
             Look Storn & Price (1997) : 
	     "Differential Evolution – A Simple and Efficient Heuristic for 
	     global Optimization over Continuous Spaces"
	     WARNING : tol (relative standard deviation of the objective function) is set to 10^-6
                       popsize (the total number of individuals is : popsize*number_of_paczynski_parameters)
		       is set to 20
                       mutation is set to (0.5, 1.5) 
		       recombination is set to 0.6
                       These parameters can avoid the fit to properly converge (expected to be rare :)).
		       Just relaunch should be fine.

        """

        starting_time = TIME.time()
        differential_evolution_estimation = differential_evolution(self.chichi_differential,
                                    bounds=self.model.parameters_boundaries,
                                    mutation=0.6, popsize=20, tol=0.000001,
                                    recombination=0.6, polish='True', disp=True)
        
	paczynski_parameters = differential_evolution_estimation['x'].tolist()

	# Construct the guess for the LM method. In principle, guess and outputs of the LM 
	# method should be very close.

        self.guess = paczynski_parameters + self.find_fluxes(paczynski_parameters, self.model)

        fit_results, fit_covariance, fit_time = self.lmarquardt()

        computation_time = TIME.time() - starting_time

        return fit_results, fit_covariance, computation_time
              
    def lmarquardt(self):
        """The LM method. This is based on the Levenberg-Marquardt algorithm:

           "A Method for the Solution of Certain Problems in Least Squares"
            Levenberg, K. Quart. Appl. Math. 2, 1944, p. 164-168
           "An Algorithm for Least-Squares Estimation of Nonlinear Parameters"
            Marquardt, D. SIAM J. Appl. Math. 11, 1963, p. 431-441

           Based on scipy.optimize.leastsq python routine, which is based on MINPACK's lmdif and lmder
           algorithms (fortran based).

           The objective function (function to minimize) is LM_residuals
           The starting point parameters are self.guess.
           the Jacobian is given by the LM_Jacobian function

           The fit is performed on all parameters : Paczynski parameters and telescopes fluxes.
        
           WARNING : ftol (relative error desired in the sum of square) is set to 10^-6
                     maxfev (maximum number of function call) is set to 50000
                     These limits can avoid the fit to properly converge (expected to be rare :))
        """
        starting_time = TIME.time()
        
        #use the analytical Jacobian (faster) if no second order are present, else let the algorithm find it.
        ### NEED CHANGE ###
        #import pdb; pdb.set_trace()

        if self.model.parallax_model[0] == 'None':
            lmarquardt_fit = leastsq(self.LM_residuals, self.guess, maxfev=50000, Dfun=self.LM_Jacobian,
                                     col_deriv=1, full_output=1, ftol=10 ** -6, xtol=10 ** -10,
                                     gtol=10 ** -5)
        else:

            lmarquardt_fit = leastsq(self.LM_residuals, self.guess, maxfev=50000, full_output=1,
                                     ftol=10 ** -5, xtol=10 ** -5)

        computation_time = TIME.time() - starting_time

        fit_result = lmarquardt_fit[0].tolist()
        fit_result.append(self.chichi(lmarquardt_fit[0]))
        
	n_data = 0.0

        for telescope in self.event.telescopes:

            n_data = n_data + telescope.n_data('Flux')

        n_parameters = len(self.model.model_dictionnary)
        
	try:
	    # Try to extract the covariance matrix from the lmarquard_fit output

            if (True not in (lmarquardt_fit[1].diagonal() < 0)) & (lmarquardt_fit[1] is not None):
		
		# Normalise the output by the reduced chichi
                covariance_matrix = lmarquardt_fit[1] * fit_result[len(self.model.model_dictionnary)] / (
                    		    n_data - n_parameters)
            
	    # Try to do it "manually"
            else:

                print ' Attempt to construct a rough covariance matrix'
                jacobian = self.LM_Jacobian(fit_result)
		
		covariance_matrix = np.linalg.inv(np.dot( jacobian,  jacobian.T))
		# Normalise the output by the reduced chichi
                covariance_matrix = covariance_matrix * fit_res[
                    		    len(self.model.model_dictionnary)] / (n_data - n_parameters)

		# Construct a dummy covariance matrix
                if True in (covariance_matrix.diagonal() < 0):
                    print 'Bad covariance covariance matrix'
                    covariance_matrix = np.zeros((len(self.model.model_dictionnary),
                                       len(self.model.model_dictionnary)))
	
	# Construct a dummy covariance matrix        
	except:
            print 'Bad covariance covariance matrix'
            covariance_matrix = np.zeros((len(self.model.model_dictionnary),
                                len(self.model.model_dictionnary)))
        

        return fit_result, covariance_matrix, computation_time

    def LM_Jacobian(self, fit_process_parameters):
        
        """Return the analytical Jacobian matrix, if requested by method LM.
	   Available only for PSPL and FSPL without second_order. 
	   PROBABLY NEED REWORK
	"""

        if self.model.paczynski_model == 'PSPL':

	    # Derivatives of the LM_residuals objective function, PSPL version

            dresdto = np.array([])
            dresduo = np.array([])
            dresdtE = np.array([])
            dresdfs = np.array([])
            dresdeps = np.array([])


            for telescope in self.event.telescopes:

                lightcurve = telescope.lightcurve_flux

                time = lightcurve[:, 0]
                errflux = lightcurve[:, 2]
                gamma = telescope.gamma

		# Derivative of A = (u^2+2)/(u(u^2+4)^0.5). Amplification[0] is A(t). Amplification[1] is U(t).
		
                Amplification = self.model.magnification(fit_process_parameters, time, gamma)
                dAmplificationdU = (-8) / (Amplification[1] ** 2 * (Amplification[1] ** 2 + 4) ** 1.5)

		# Derivative of U = (uo^2+(t-to)^2/tE^2)^0.5
                dUdto = -(time - fit_process_parameters[self.model.model_dictionnary['to']]) / (
                    fit_process_parameters[self.model.model_dictionnary['tE']] ** 2 * Amplification[1])
                dUduo = fit_process_parameters[self.model.model_dictionnary['uo']] / Amplification[1]
                dUdtE = -(time - fit_process_parameters[self.model.model_dictionnary['to']]) ** 2 / (
                    fit_process_parameters[self.model.model_dictionnary['tE']] ** 3 * Amplification[1])


		# Derivative of the objective function
                dresdto = np.append(dresdto,
                                    -fit_process_parameters[self.model.model_dictionnary['fs_' + telescope.name]] *
                                    dAmplificationdU * dUdto / errflux)
                dresduo = np.append(dresduo,
                                    -fit_process_parameters[self.model.model_dictionnary['fs_' + telescope.name]] *
                                    dAmplificationdU * dUduo / errflux)
                dresdtE = np.append(dresdtE,
                                    -fit_process_parameters[self.model.model_dictionnary['fs_' + telescope.name]] *
                                    dAmplificationdU * dUdtE / errflux)
                dresdfs = np.append(dresdfs, -(
                    Amplification[0] + fit_process_parameters[self.model.model_dictionnary['g_' + telescope.name]]) / errflux)
                dresdeps = np.append(dresdeps, -fit_process_parameters[
                    self.model.model_dictionnary['fs_' + telescope.name]] / errflux)

            jacobi = np.array([dresdto, dresduo, dresdtE])

        if self.model.paczynski_model == 'FSPL':

	    # Derivatives of the LM_residuals objective function, FSPL version
            dresdto = np.array([])
            dresduo = np.array([])
            dresdtE = np.array([])
            dresdrho = np.array([])
            dresdfs = np.array([])
            dresdeps = np.array([])

            fake_model = microlmodels.MLModels(self.event, 'PSPL')
            fake_params = np.delete(fit_process_parameters, self.model.model_dictionnary['rho'])
            
            for telescope in self.event.telescopes:

                lightcurve = telescope.lightcurve_flux
                time = lightcurve[:, 0]
                errflux = lightcurve[:, 2]
                gamma = telescope.gamma

               
		# Derivative of A = Yoo et al (2004) method.
                Amplification_PSPL = fake_model.magnification(fake_params, time, gamma)
                dAmplification_PSPLdU = (-8) / (Amplification_PSPL[1] ** 2 * (Amplification_PSPL[1] ** 2 + 4) ** (1.5))

		# Z=U/rho
                Z = Amplification_PSPL[1] / fit_process_parameters[self.model.model_dictionnary['rho']]

		
                dadu = np.zeros(len(Amplification_PSPL[0]))
                dadrho = np.zeros(len(Amplification_PSPL[0]))
		
		# Far from the lens (Z>>1), then PSPL.	
                ind = np.where((Z > self.model.yoo_table[0][-1]))[0]
                dadu[ind] = dAmplification_PSPLdU[ind]
                dadrho[ind] = -0.0

		# Very close to the lens (Z<<1), then Witt&Mao limit.
                ind = np.where((Z < self.model.yoo_table[0][0]))[0]
                dadu[ind] = dAmplification_PSPLdU[ind] * (2 * Z[ind] - gamma * (2 - 3 * np.pi / 4) * Z[ind])
                dadrho[ind] = -Amplification_PSPL[0][ind] * Amplification_PSPL[1][ind] / fit_process_parameters[
                                                                   self.model.model_dictionnary[
                                                                       'rho']] ** 2 * (
                                  2 - gamma * (2 - 3 * np.pi / 4))

		# FSPL regime (Z~1), then Yoo et al derivatives
                ind = np.where(
                        (Z <= self.model.yoo_table[0][-1]) & (Z >= self.model.yoo_table[0][0]))[0]
                dadu[ind] = dAmplification_PSPLdU[ind] * (
                    self.model.yoo_table[1](Z[ind]) - gamma * self.model.yoo_table[2](
                        Z[ind])) + Amplification_PSPL[0][ind] * (
                    self.model.yoo_table[3](Z[ind]) - gamma * self.model.yoo_table[4](
                        Z[ind])) * 1 / fit_process_parameters[self.model.model_dictionnary['rho']]

                dadrho[ind] = -Amplification_PSPL[0][ind] * Amplification_PSPL[1][ind] / fit_process_parameters[
                                                                   self.model.model_dictionnary[
                                                                       'rho']] ** 2 * (
                                  self.model.yoo_table[3](Z[ind]) - gamma * self.model.yoo_table[4](
                                      Z[ind]))

                dUdto = -(time - fit_process_parameters[self.model.model_dictionnary['to']]) / (
                    fit_process_parameters[self.model.model_dictionnary['tE']] ** 2 * Amplification_PSPL[1])
                dUduo = fit_process_parameters[self.model.model_dictionnary['uo']] / Amplification_PSPL[1]
                dUdtE = -(time - fit_process_parameters[self.model.model_dictionnary['to']]) ** 2 / (
                    fit_process_parameters[self.model.model_dictionnary['tE']] ** 3 * Amplification_PSPL[1])
                

		# Derivative of the objective function
		dresdto = np.append(dresdto, -fit_process_parameters[
                    self.model.model_dictionnary['fs_' + telescope.name]] * dadu *
                                    dUdto / errflux)
                dresduo = np.append(dresduo, -fit_process_parameters[
                    self.model.model_dictionnary['fs_' + telescope.name]] * dadu *
                                    dUduo / errflux)
                dresdtE = np.append(dresdtE, -fit_process_parameters[
                    self.model.model_dictionnary['fs_' + telescope.name]] * dadu *
                                    dUdtE / errflux)

                dresdrho = np.append(dresdrho,
                                     -fit_process_parameters[self.model.model_dictionnary['fs_' + telescope.name]] *
                                     dadrho / errflux)

                

                Amplification_FSPL = self.model.magnification(fit_process_parameters, time, gamma)
                dresdfs = np.append(dresdfs, -(
                    Amplification_FSPL[0] + fit_process_parameters[self.model.model_dictionnary['g_' + telescope.name]]) / errflux)
                dresdeps = np.append(dresdeps, -fit_process_parameters[
                    self.model.model_dictionnary['fs_' + telescope.name]] / errflux)

            jacobi = np.array([dresdto, dresduo, dresdtE, dresdrho])

	# Split the fs and g derivatives in several columns correpsonding to 
	# each observatories 
        start_index = 0
	
        for telescope in self.event.telescopes:

            dFS = np.zeros((len(dresdto)))
            dG = np.zeros((len(dresdto)))
            index = np.arange(start_index, start_index + len(telescope.lightcurve_flux[:, 0]))
            dFS[index] = dresdfs[index]
            dG[index] = dresdeps[index]
            jacobi = np.vstack([jacobi, dFS])
            jacobi = np.vstack([jacobi, dG])

            start_index = index[-1] + 1

        

        return jacobi

    def LM_residuals(self, fit_process_parameters):
        """ The normalized residuals associated to the model and parameters.
        residuals_i=(data_i-model_i)/sigma_i
        The sum of square residuals gives chi^2.
        """
	
	# Construct an np.array with each telescope residuals
        residuals = np.array([])
       
        for telescope in self.event.telescopes:
           
            lightcurve = telescope.lightcurve_flux
            time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            gamma = telescope.gamma
           
	    # magnification according to the model. amplification[0] is A(t), amplification[1] is u(t)
            amplification = self.model.magnification(fit_process_parameters, time, gamma, telescope.deltas_positions)[0]
           

            residuals = np.append(residuals, (
                flux - amplification * fit_process_parameters[self.model.model_dictionnary['fs_' + telescope.name]] -
                (fit_process_parameters[self.model.model_dictionnary['fs_' + telescope.name]] * fit_process_parameters[
                    self.model.model_dictionnary['g_' +  telescope.name]])) / errflux)

            
            
        return residuals

    def chichi(self, fit_process_parameters):
        """Return the chi^2. """

        residuals = self.LM_residuals(fit_process_parameters)
        chichi = (residuals ** 2).sum()

        return chichi

    def chichi_telescopes(self, fit_process_parameters):
        """Return the chi^2 for each telescopes """

        residuals = self.LM_residuals(fit_process_parameters)
        CHICHI = []
        start_index = 0
        for telescope in self.event.telescopes:

            CHICHI.append((residuals[start:start + len(telescope.lightcurve_flux)] ** 2).sum())

            start_index += len(telescope.lightcurve_flux)

        return CHICHI

    def chichi_differential(self, fit_process_parameters):
        """Return the chi^2 for dirrential_evolution. fsi,fbi evaluated trough polyfit. """

        residuals = np.array([])
        
        for telescope in self.event.telescopes:
            lightcurve = telescope.lightcurve_flux
            time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            gamma = telescope.gamma
            
            try :
	       
		# magnification according to the model. amplification[0] is A(t), amplification[1] is u(t)
                amplification = self.model.magnification(fit_process_parameters, time, gamma,telescope.deltas_positions)[0]
                fs, fb = np.polyfit(amplification, flux, 1, w=1 / errflux)
            except :
                return np.inf
            #print i.name,fs
            if (fs < 0):
                # print fs
                return np.inf

            residuals = np.append(residuals, (flux - amplification * fs - fb) / errflux)
        # import pdb; pdb.set_trace()
        chichi = (residuals ** 2).sum()
        return chichi

    def chichi_MCMC(self, fit_process_parameters):
        """Return the chi^2 for MCMC. fsi,fbi evaluated trough polyfit. """

        residuals = np.array([])
       
        for telescope in self.event.telescopes:

            lightcurve = telescope.lightcurve_flux
            time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            gamma = telescope.gamma

            amplification = self.model.magnification(fit_process_parameters, time, gamma,telescope.deltas_positions)[0]

            fs, fb = np.polyfit(amplification, flux, 1, w=1 / errflux)

            # Little prior here
            if (fs < 0) | (fb/fs<-1.0):
                
                chichi = np.inf
                return -chichi
            
            else:

                residuals = np.append(residuals, (flux - amplification * fs - fb) / errflux)
                
                chichi = (residuals ** 2).sum()
                # Little prior here
                chichi += np.log(len(time))*1/(1+fb/fs)
                 
        return - (chichi)

    def find_fluxes(self, fit_process_parameters, model):
        """ Find telescopes flux associated (fs,g) to the model. Used for initial_guess and LM method.
        """
        telescopes_fluxes = []

        for telescope in self.event.telescopes:
            lightcurve = telescope.lightcurve_flux
            time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            gamma = telescope.gamma

            amplification = model.magnification(fit_process_parameters, time, gamma,telescope.deltas_positions)[0]
           
            fs, fb = np.polyfit(amplification, flux, 1, w=1 / errflux)
	    #Prior here
            if (fs < 0) :

                telescopes_fluxes.append(np.min(flux))
                telescopes_fluxes.append(0.0)
            else:
                telescopes_fluxes.append(fs)
                telescopes_fluxes.append(fb / fs)
        return telescopes_fluxes


    def produce_outputs(self) :
        """ Produce the standard outputs for a fit.
	    More details in microloutputs module.	
	"""
        
        if self.method != 'MCMC' :
            
            outputs = microloutputs.LM_outputs(self)
            
        else :
            
            outputs = microloutputs.MCMC_outputs(self)
        
        self.outputs = outputs
