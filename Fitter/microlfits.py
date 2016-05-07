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


class MLFits(object):
    """
    ######## Fitter module ########
    @author: Etienne Bachelet

    This module fits the event with the selected attributes.
    WARNING: All fits (and so results) are made using data in flux.

    Keyword arguments:

    model --> The microlensing model you want to fit. Has to be an object define in microlmodels module.
              More details on the microlmodels module.

    method --> The fitting method you want to use. Has to be a string :

              'LM' --> Levenberg-Marquardt algorithm. Based on the scipy.optimize.leastsq routine.
              WARNING : the parameter maxfev (number of maximum iterations) is set to 50000
                        the parameter ftol (relative precision on the chi^2) is set to 0.00001
                        your fit may not converge because of these limits.
                        The starting points of this method are found using the initial_guess method.
                        Obviously, this can fail. In this case, switch to method 'DE'.
                        
              'DE' --> Differential evolution algoritm. Based on the scipy.optimize.differential_evolution.
                       Look Storn & Price (1997) : "Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces"
                       Because this method is heuristic, it is not 100% sure a satisfying solution is found. Just relaunch :)
                       The result is then use as a starting point for the 'LM' method.
                       
                       
              'MCMC' --> Monte-Carlo Markov Chain algorithm. Based on the emcee python package :  
                         " emcee: The MCMC Hammer" (Foreman-Mackey et al. 2013).
                         The inital population is computed around the best solution return by
                         the 'DE' method.

    
    Return :

    Outputs depends on the method :
    
            'LM' and 'DE' --> the fit will have the new attributes fit_results, fit_covariance and fit_time results.
            'MCMC' -->  the fit will have the new attibutes MCMC_chains and MCMC_probabilities 
            
            """

    def __init__(self, event):
        """The fit class has to be intialized with an event. """
        self.event = event

    def mlfit(self, model, method):
        """This function realize the requested fit, and produce outputs accordingly.
            Note that a sanity check is done post-fit to assess the fit quality with the check_fit function."""
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
        flag = 'Good Fit'
        diago = np.diag(self.fit_covariance) < 0

        if (0.0 in self.fit_covariance) | (True in diago) | ( np.isnan(self.fit_covariance).any()) | (np.isinf(self.fit_covariance).any()):

            print 'Your fit probably wrong. Cause ==> bad covariance matrix'
            flag = 'Bad Fit'
            return flag

        for i in self.event.telescopes:

            if self.fit_results[self.model.model_dictionnary['fs_' + i.name]] < 0:

                print 'Your fit probably wrong. Cause ==> negative source flux for telescope ' + \
                      i.name
                flag = 'Bad Fit'
                return flag

        if 'rho' in self.model.model_dictionnary:

            if (self.fit_results[self.model.model_dictionnary['rho']] >0.1) |(self.fit_results[self.model.model_dictionnary['rho']] <0.0) :

                print 'Your fit probably wrong. Cause ==> bad rho ' 
                flag = 'Bad Fit'
                return flag
        return flag

    def initial_guess(self):
        """Function to find initial guess for Levenberg-Marquardt solver (method=='LM').
        Guess are made using the survey telescope for the Paczynski parameters (to,uo,tE).
        This assumes no blending.
        """

        # to estimation
        To = []
        Max_flux = []
        Std = []
        Errmag = []
        for i in self.event.telescopes:

            try:
                
                # only the best photometry
                good = np.where((i.lightcurve[:, 2] < max(0.1, np.mean(i.lightcurve[:, 2]))))[0]
                lightcurve_bis = i.lightcurve[good]
                mag = lightcurve_bis[:, 1]
                flux = 10 ** ((27.4 - mag) / 2.5)
                lightcurve_bis = lightcurve_bis[lightcurve_bis[:, 0].argsort(), :]
                mag_clean = ss.savgol_filter(mag, 3, 1)
                Time = lightcurve_bis[:, 0]
                flux_clean = 10 ** ((27.4 - mag_clean) / 2.5)

                errmag = lightcurve_bis[:, 2]

               

                fs = min(flux_clean)
                index = np.where(flux_clean > fs)[0]
                good = index

                while (np.std(Time[good]) > 5) | (len(good) > 100):

                    index = np.where((flux_clean[good] > np.median(flux_clean[good])) & (
                        errmag[good] <= max(0.1, 2.0 * np.mean(errmag[good]))))[0]

                    if len(index) < 1:

                        break

                    else:
                        good = good[index]
                       

                        gravity = (
                            np.median(Time[good]), np.median(flux_clean[good]),
                            np.mean(errmag[good]))
                       
                        distances = np.sqrt((Time[good] - gravity[0]) ** 2 / gravity[0] ** 2)
                      
                to = np.median(Time[good])
                max_flux = max(flux[good])
                std = np.std(lightcurve_bis[good, 0])
                To.append(to)
                Max_flux.append(max_flux)
                Errmag.append(np.mean(lightcurve_bis[good, 2]))
                if std == 0:

                    std = 0.1

                Std.append(std)

            except:

                Time = i.lightcurve[:, 0]
                flux = 10 ** ((27.4 - i.lightcurve[:, 1]) / 2.5)
                to = np.median(Time)
                max_flux = max(flux)
                To.append(to)
                Max_flux.append(max_flux)
                std = np.std(i.lightcurve[:, 0])
                if std == 0:

                    std = 0.1
                Std.append(std)
                Errmag.append(np.mean(i.lightcurve[:, 2]))




        to = sum(np.array(To) / np.array(Errmag) ** 2) / sum(1 / np.array(Errmag) ** 2)
        survey = self.event.telescopes[0]
        lightcurve = survey.lightcurve_flux
        lightcurve = lightcurve[lightcurve[:, 0].argsort(), :]
        Time = lightcurve[:, 0]
        flux = lightcurve[:, 1]
        errflux = lightcurve[:, 2]

        # fs, no blend

        baseline_flux_0 = np.min(flux)
        baseline_flux = np.median(flux)
        index = []

        while np.abs(baseline_flux_0 - baseline_flux) > 0.01 * baseline_flux:

            baseline_flux_0 = baseline_flux
            index = np.where((flux < baseline_flux))[0].tolist() + np.where(
                np.abs(flux - baseline_flux) < np.abs(errflux))[0].tolist()
            baseline_flux = np.median(flux[index])

            if len(index) < 100:

                print 'low'
                baseline_flux = np.median(flux[flux.argsort()[:100]])
                break

        fs = baseline_flux
        max_flux = Max_flux[0]
        Amax = max_flux / fs
        uo = np.sqrt(-2 + 2 * np.sqrt(1 - 1 / (1 - Amax ** 2)))


        # tE estimations
        flux_demi = 0.5 * fs * (Amax + 1)
        flux_tE = fs * (uo ** 2 + 3) / ((uo ** 2 + 1) ** 0.5 * np.sqrt(uo ** 2 + 5))
        index_plus = np.where((Time > to) & (flux < flux_demi))[0]
        index_moins = np.where((Time < to) & (flux < flux_demi))[0]
        B = 0.5 * (Amax + 1)
        if len(index_plus) != 0:

            if len(index_moins) != 0:

                ttE = (Time[index_plus[0]] - Time[index_moins[-1]])
                tE1 = ttE / (2 * np.sqrt(-2 + 2 * np.sqrt(1 + 1 / (B ** 2 - 1)) - uo ** 2))

            else:

                ttE = Time[index_plus[0]] - to
                tE1 = ttE / np.sqrt(-2 + 2 * np.sqrt(1 + 1 / (B ** 2 - 1)) - uo ** 2)
        else:

            ttE = to - Time[index_moins[-1]]
            tE1 = ttE / np.sqrt(-2 + 2 * np.sqrt(1 + 1 / (B ** 2 - 1)) - uo ** 2)

        indextEplus = np.where((flux < flux_tE) & (Time > to))[0]
        indextEmoins = np.where((flux < flux_tE) & (Time < to))[0]
        tEmoins = 0.0
        tEplus = 0.0

        if len(indextEmoins) != 0:

            indextEmoins = indextEmoins[-1]
            tEmoins = to - Time[indextEmoins]

        if len(indextEplus) != 0:

            indextEplus = indextEplus[0]
            tEplus = Time[indextEplus] - to

        indextEPlus = np.where((Time > to) & (np.abs(flux - fs) < np.abs(errflux)))[0]
        indextEMoins = np.where((Time < to) & (np.abs(flux - fs) < np.abs(errflux)))[0]
        tEPlus = 0.0
        tEMoins = 0.0

        if len(indextEPlus) != 0:

            tEPlus = Time[indextEPlus[0]] - to

        if len(indextEMoins) != 0:

            tEMoins = to - Time[indextEMoins[-1]]

        TE = np.array([tE1, tEplus, tEmoins, tEPlus, tEMoins])
        good = np.where(TE != 0.0)[0]
        tE = np.median(TE[good])

        if tE < 0.1:

            tE = 20.0
            
        fake_second_order = [['None', 0], ['None', 0], ['None', 0], 'None']
        fake_model = microlmodels.MLModels(self.event, 'PSPL', fake_second_order)

        #import pdb; pdb.set_trace()
        fluxes = self.find_fluxes([to, uo, tE], fake_model)
        fluxes[0] = fs
        fluxes[1] = 0.0

        parameters = [to, uo, tE]

        if self.model.paczynski_model == 'FSPL':

            parameters = parameters + [0.05]

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
            is made on Paczynski parameters, fs and g are found using a linear fit.
            Launch nwalkers chains with 100 links
            
            """
        
        
        AA=differential_evolution(self.chichi_differential,
                                   bounds=self.model.parameters_boundaries,mutation=(0.5,1), popsize=30,
                                   recombination=0.7,polish='None')
        res=AA['x']
        ndim = len(res)
        nwalkers = 300
        pp0 = []

        i=0
        while i < nwalkers:
            p1 = []
            for j in range(ndim):

                if j==0:
                    
                    p1.append(res[j]+np.random.uniform(-5,5))
                if j==1:
                    
                    p1.append(res[j]*(np.random.uniform(0,3)))
                if j==2:
            
                    p1.append(res[j]*(np.random.uniform(0,3)))
                
                if j==3:
                    
                    p1.append(res[j]*(np.random.uniform(0,1)))

            pp0.append(np.array(p1))
            i+=1
           
     

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.chichi_MCMC)

        pos, prob, state = sampler.run_mcmc(pp0, 100)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, 100)


        chains = sampler.chain
        probability = sampler.lnprobability
            
        return chains,probability

    def diff_evolution(self) :
        """ The DE method. Heuristic global optimizer. Optimization
             is made on Paczynski parameters inside self.model.parameters_boundaries,
             fs and g are found using a linear fit.
            
        """
        start = TIME.time()
        AA = differential_evolution(self.chichi_differential,
                                    bounds=self.model.parameters_boundaries,
                                    mutation=(0.5, 1.5), popsize=20, tol=0.000001,
                                    recombination=0.6, polish='True', disp=True)
        
        self.guess = AA['x'].tolist() + self.find_fluxes(AA['x'].tolist(), self.model)

        fit_res, cov,fit_time = self.lmarquardt()

        computation_time = TIME.time() - start
        return fit_res, cov, computation_time
        
        
    def lmarquardt(self):
        """Method LM of solver. This is based on the Levenberg-Marquardt algorithm:

        "A Method for the Solution of Certain Problems in Least Squares"
        Levenberg, K. Quart. Appl. Math. 2, 1944, p. 164-168
        "An Algorithm for Least-Squares Estimation of Nonlinear Parameters"
        Marquardt, D. SIAM J. Appl. Math. 11, 1963, p. 431-441

        Based scipy.optimize.leastsq python routine, which is based on MINPACK's lmdif and lmder
        algorithms (fortran based).

        The objective function (function to minimize) is residuals
        The starting point is found using the initial_guess function
        the Jacobian is given by the Jacobian function

        The fit is performed on all parameters : Paczynski parameters, second_order and telescopes fluxes.
        
        WARNING : ftol (relative error desired in the sum of square) is set to 10^-6
                  maxfev (maximum number of function call) is set to 50000
                  These limits can avoid the fit to properly converge (expected to be rare :))
        """
        start = TIME.time()
        
        #use the analytical Jacobian (faster) if no second order are present, else let the algorithm find it.
        ### NEED CHANGE ###
        if self.model.parallax_model[0] == 'None':
            lmarquardt_fit = leastsq(self.residuals, self.guess, maxfev=50000, Dfun=self.Jacobian,
                                     col_deriv=1, full_output=1, ftol=10 ** -6, xtol=10 ** -10,
                                     gtol=10 ** -5)
        else:

            lmarquardt_fit = leastsq(self.residuals, self.guess, maxfev=50000, full_output=1,
                                     ftol=10 ** -5, xtol=10 ** -5)

        computation_time = TIME.time() - start

        fit_res = lmarquardt_fit[0].tolist()
        fit_res.append(self.chichi(lmarquardt_fit[0]))
        n_data = 0.0

        for i in self.event.telescopes:

            n_data = n_data + i.n_data('Flux')
        n_parameters = len(self.model.model_dictionnary)
        try:

            if (True not in (lmarquardt_fit[1].diagonal() < 0)) & (lmarquardt_fit[1] is not None):

                cov = lmarquardt_fit[1] * fit_res[len(self.model.model_dictionnary)] / (
                    n_data - n_parameters)
                # import pdb; pdb.set_trace()

            else:

                print 'rough cov'
                jacky = self.Jacobian(fit_res)
                cov = np.linalg.inv(np.dot(jacky, jacky.T)) * fit_res[
                    len(self.model.model_dictionnary)] / (n_data - n_parameters)
                if True in (cov.diagonal() < 0):
                    print 'Bad rough covariance'
                    cov = np.zeros((len(self.model.model_dictionnary),
                                    len(self.model.model_dictionnary)))
        except:
            print 'hoho'
            cov = np.zeros((len(self.model.model_dictionnary),
                            len(self.model.model_dictionnary)))

        return fit_res, cov, computation_time

    def Jacobian(self, parameters):
        
        """Return the analytical Jacobian matrix, if requested by method LM. """

        if self.model.paczynski_model == 'PSPL':

            dresdto = np.array([])
            dresduo = np.array([])
            dresdtE = np.array([])
            dresdfs = np.array([])
            dresdeps = np.array([])

            to = parameters[0]
            uo = parameters[1]
            tE = parameters[2]

            for i in self.event.telescopes:

                lightcurve = i.lightcurve_flux
                Time = lightcurve[:, 0]
                errflux = lightcurve[:, 2]
                gamma = i.gamma


                ampli = self.model.magnification(parameters, Time, gamma)
                dAdU = (-8) / (ampli[1] ** 2 * (ampli[1] ** 2 + 4) ** 1.5)

                dUdto = -(Time - parameters[self.model.model_dictionnary['to']]) / (
                    parameters[self.model.model_dictionnary['tE']] ** 2 * ampli[1])
                dUduo = parameters[self.model.model_dictionnary['uo']] / ampli[1]
                dUdtE = -(Time - parameters[self.model.model_dictionnary['to']]) ** 2 / (
                    parameters[self.model.model_dictionnary['tE']] ** 3 * ampli[1])

                dresdto = np.append(dresdto,
                                    -parameters[self.model.model_dictionnary['fs_' + i.name]] *
                                    dAdU * dUdto / errflux)
                dresduo = np.append(dresduo,
                                    -parameters[self.model.model_dictionnary['fs_' + i.name]] *
                                    dAdU * dUduo / errflux)
                dresdtE = np.append(dresdtE,
                                    -parameters[self.model.model_dictionnary['fs_' + i.name]] *
                                    dAdU * dUdtE / errflux)
                dresdfs = np.append(dresdfs, -(
                    ampli[0] + parameters[self.model.model_dictionnary['g_' + i.name]]) / errflux)
                dresdeps = np.append(dresdeps, -parameters[
                    self.model.model_dictionnary['fs_' + i.name]] / errflux)

            jacobi = np.array([dresdto, dresduo, dresdtE])

        if self.model.paczynski_model == 'FSPL':

            dresdto = np.array([])
            dresduo = np.array([])
            dresdtE = np.array([])
            dresdrho = np.array([])
            dresdfs = np.array([])
            dresdeps = np.array([])

            fake_model = microlmodels.MLModels(self.event, 'PSPL', self.model.second_order)
            fake_params = np.delete(parameters, self.model.model_dictionnary['rho'])
            
            for i in self.event.telescopes:

                lightcurve = i.lightcurve_flux
                Time = lightcurve[:, 0]
                errflux = lightcurve[:, 2]
                gamma = i.gamma

               

                ampli = fake_model.magnification(fake_params, Time, gamma)
                dAdU = (-8) / (ampli[1] ** 2 * (ampli[1] ** 2 + 4) ** (1.5))

                Z = ampli[1] / parameters[self.model.model_dictionnary['rho']]

                dadu = np.zeros(len(ampli[0]))
                dadrho = np.zeros(len(ampli[0]))

                ind = np.where((Z > self.model.yoo_table[0][-1]))[0]
                dadu[ind] = dAdU[ind]
                dadrho[ind] = -0.0

                ind = np.where((Z < self.model.yoo_table[0][0]))[0]

                dadu[ind] = dAdU[ind] * (2 * Z[ind] - gamma * (2 - 3 * np.pi / 4) * Z[ind])
                dadrho[ind] = -ampli[0][ind] * ampli[1][ind] / parameters[
                                                                   self.model.model_dictionnary[
                                                                       'rho']] ** 2 * (
                                  2 - gamma * (2 - 3 * np.pi / 4))

                ind = \
                    np.where(
                        (Z <= self.model.yoo_table[0][-1]) & (Z >= self.model.yoo_table[0][0]))[0]

                dadu[ind] = dAdU[ind] * (
                    self.model.yoo_table[1](Z[ind]) - gamma * self.model.yoo_table[2](
                        Z[ind])) + ampli[0][ind] * (
                    self.model.yoo_table[3](Z[ind]) - gamma * self.model.yoo_table[4](
                        Z[ind])) * 1 / parameters[self.model.model_dictionnary['rho']]

                dadrho[ind] = -ampli[0][ind] * ampli[1][ind] / parameters[
                                                                   self.model.model_dictionnary[
                                                                       'rho']] ** 2 * (
                                  self.model.yoo_table[3](Z[ind]) - gamma * self.model.yoo_table[4](
                                      Z[ind]))

                dUdto = -(Time - parameters[self.model.model_dictionnary['to']]) / (
                    parameters[self.model.model_dictionnary['tE']] ** 2 * ampli[1])
                dUduo = parameters[self.model.model_dictionnary['uo']] / ampli[1]
                dUdtE = -(Time - parameters[self.model.model_dictionnary['to']]) ** 2 / (
                    parameters[self.model.model_dictionnary['tE']] ** 3 * ampli[1])
                dresdto = np.append(dresdto, -parameters[
                    self.model.model_dictionnary['fs_' + i.name]] * dadu *
                                    dUdto / errflux)
                dresduo = np.append(dresduo, -parameters[
                    self.model.model_dictionnary['fs_' + i.name]] * dadu *
                                    dUduo / errflux)
                dresdtE = np.append(dresdtE, -parameters[
                    self.model.model_dictionnary['fs_' + i.name]] * dadu *
                                    dUdtE / errflux)

                dresdrho = np.append(dresdrho,
                                     -parameters[self.model.model_dictionnary['fs_' + i.name]] *
                                     dadrho / errflux)

                

                ampli = self.model.magnification(parameters, Time, gamma)
                dresdfs = np.append(dresdfs, -(
                    ampli[0] + parameters[self.model.model_dictionnary['g_' + i.name]]) / errflux)
                dresdeps = np.append(dresdeps, -parameters[
                    self.model.model_dictionnary['fs_' + i.name]] / errflux)

            jacobi = np.array([dresdto, dresduo, dresdtE, dresdrho])

        start = 0

        for i in self.event.telescopes:

            dFS = np.zeros((len(dresdto)))
            dEPS = np.zeros((len(dresdto)))
            index = np.arange(start, start + len(i.lightcurve_flux[:, 0]))
            dFS[index] = dresdfs[index]
            dEPS[index] = dresdeps[index]
            jacobi = np.vstack([jacobi, dFS])
            jacobi = np.vstack([jacobi, dEPS])

            start = index[-1] + 1

        #import pdb; pdb.set_trace()

        return jacobi

    def residuals(self, parameters):
        """ The normalized residuals associated to the model and parameters.
        residuals_i=(y_i-model_i)/sigma_i
        The sum of square residuals gives chi^2.
        """
        errors = np.array([])
        count = 0
        
        for i in self.event.telescopes:
           
            lightcurve = i.lightcurve_flux
            Time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            gamma = i.gamma
           
            ampli = self.model.magnification(parameters, Time, gamma)[0]
           

            errors = np.append(errors, (
                flux - ampli * parameters[self.model.model_dictionnary['fs_' + i.name]] -
                (parameters[self.model.model_dictionnary['fs_' + i.name]] * parameters[
                    self.model.model_dictionnary['g_' + i.name]])) / errflux)

            
            count = count + 1
            # plt.show()
        return errors

    def chichi(self, parameters):
        """Return the chi^2. """
        errors = self.residuals(parameters)
        chichi = (errors ** 2).sum()

        return chichi

    def chichi_telescopes(self, parameters):
        """Return the chi^2 for each telescopes """
        errors = self.residuals(parameters)
        CHICHI = []
        start = 0
        for i in self.event.telescopes:

            CHICHI.append((errors[start:start + len(i.lightcurve_flux)] ** 2).sum())

            start = start + len(i.lightcurve_flux)

        return CHICHI

    def chichi_differential(self, parameters):
        """Return the chi^2 for dirrential_evolution. fsi,fbi evaluated trough polyfit. """
        errors = np.array([])
        
        for i in self.event.telescopes:
            lightcurve = i.lightcurve_flux
            Time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            gamma = i.gamma
            #import pdb; pdb.set_trace()
            

            try :
                ampli = self.model.magnification(parameters, Time, gamma)[0]
                fs, fb = np.polyfit(ampli, flux, 1, w=1 / errflux)
            except :
                return np.inf
            #print i.name,fs
            if (fs < 0):
                # print fs
                return np.inf

            errors = np.append(errors, (flux - ampli * fs - fb) / errflux)
        # import pdb; pdb.set_trace()
        chichi = (errors ** 2).sum()
        return chichi

    def chichi_MCMC(self, parameters):
        """Return the chi^2 for dirrential_evolution. fsi,fbi evaluated trough polyfit. """
        errors = np.array([])
       

       
        
        for i in self.event.telescopes:

            lightcurve = i.lightcurve_flux
            Time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            gamma = i.gamma

            ampli = self.model.magnification(parameters, Time, gamma)[0]

            fs, fb = np.polyfit(ampli, flux, 1, w=1 / errflux)

            # Little prior here
            if (fs < 0) | (fb/fs<-1.0):
                
                chichi = np.inf
                return -chichi
            
            else:

                errors = np.append(errors, (flux - ampli * fs - fb) / errflux)
                
                chichi = (errors ** 2).sum()
                # Little prior here
                chichi+=+np.log(len(Time))*1/(1+fb/fs)
                 
        return - (chichi)

    def find_fluxes(self, parameters, model):
        """ Find telescopes flux associated to the model. Note
        that in some case model differ from self.model (the requested model to fit)
        """
        fluxes = []

        for i in self.event.telescopes:
            lightcurve = i.lightcurve_flux
            Time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            gamma = i.gamma

            ampli = model.magnification(parameters, Time, gamma)[0]
           
            fs, fb = np.polyfit(ampli, flux, 1, w=1 / errflux)
            if (fs < 0) :

                fluxes.append(np.min(flux))
                fluxes.append(0.0)
            else:
                fluxes.append(fs)
                fluxes.append(fb / fs)
        return fluxes


    def produce_outputs(self) :
        """ Produce the standard outputs for a fit """
        
        if self.method != 'MCMC' :
            
            outputs = microloutputs.LM_outputs(self)
            
        else :
            
            outputs = microloutputs.MCMC_outputs(self)
        
        self.outputs=outputs