# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:39:32 2015

@author: ebachelet
"""
from __future__ import division
import numpy as np
from scipy.optimize import leastsq, differential_evolution
from scipy import interpolate
import time
from scipy.integrate import dblquad,nquad
import matplotlib.pyplot as plt

import microlmodels
import microlparallax
import microlmagnification


class MLFits(object):
    '''
    ######## Fitter module ########
    @author: Etienne Bachelet

    This module fits the event with the selected attributes.
    WARNING: All fits (and so results) are made using data in flux.

    Keyword arguments:

    model --> The microlensing model you want to fit. Has to be a string in the available_models parameter:

             'PSPL' --> Point Source Point Lens. The amplification is taken from :
             "Gravitational microlensing by the galactic halo" Paczynski,B. 1986ApJ...304....1P

             'FSPL' --> Finite Source Point Lens. The amplification is taken from :
             "OGLE-2003-BLG-262: Finite-Source Effects from a Point-Mass Lens' Yoo,J. et al.2004ApJ...603..139Y
              Note that the LINEAR LIMB-DARKENING is used, where the table b0b1.dat is interpolated
              to compute B0(z) and B1(z).

             'DSPL'  --> not available now
             'Binary' --> not available now
             'Triple' --> not available now


    method --> The fitting method you want to use. Has to be a integer in the available_methods parameter:.

              0 --> Levenberg-Marquardt algorithm. Based on the scipy.optimize.leastsq routine.
              WARNING : the parameter maxfev (number of maximum iterations) is set to 50000
                        the parameter ftol (relative precision on the chi^2) is set to 0.00001
                        your fit may not converge because of these limits

    second_order --> Second order effect : parallax, orbital_motion and source_spots . A list of list as :

                    [parallax,orbital_motion,source_spots]

                    parallax --> Parallax model you want to use. Has to be a list [parallax model, topar].
                    Parallax models are :

                    'Annual' --> Annual parallax
                    'Terrestrial' --> Terrestrial parallax
                    'Space' --> Space based parallax
                    'Full' --> combination of all previous.

                    topar --> time in HJD selected

                     More details in the microlparallax module

                    orbital_motion --> Orbital motion you want to use. Has to be a list [orbital model, toom].
                    Orbital models are:

                    'None' --> No orbital motion
                    '2D' --> Classical orbital motion
                    '3D' --> Full Keplerian orbital motion

                    toom --> a time in HJD choosed as the referenced time fot the orbital motion
                            (Often choose equal to topar)

                    More details in the microlomotion module

                    source_spots --> Consider spots on the source. Has to be a list in the
                                     available source_spots parameter :

                    'None' --> No source spots

                    More details in the microlsspots module

    survey --> Survey telescope linked to your event. Can be found using the find_survey function.

    number_of_parameters --> Number of parameters which are used for the magnification computation:
                             it is varying as an addition of model parameter and second_order parameter.

                             The PARAMETERS RULE is (quantity in brackets are optional):

                             [to,uo,tE,(rho),(s),(q),(alpha),(PiEN),(PiEE),(dsdt),(dalphadt),(source_spots)]
                             +Sum_i[fsi,fbi/fsi]

                             to --> time of maximum amplification in HJD
                             uo --> minimum impact parameter (for the time to)
                             tE --> angular Einstein ring crossing time in days
                             rho --> normalized angular source radius = theta_*/theta_E
                             s --> normalized projected angular speration between the two bodies
                             q --> mass ratio
                             alpha --> counterclockwise angle (in radians) between the source trajectory and the lenses axes
                             PiEN --> composant North (in the sky plane) of the parallax vector
                             PiEE --> composant East (in the sky plane) of the parallax vector
                             ds/dt --> s variations due to the lens movement
                             dalpha/dt --> angle variations due to the lens movement
                             source_spots --> ?????
                             fsi --> source flux in unit : m=27.4-2.5*np.log10(flux)
                             fbi/fsi --> blending flux ratio

                             As an example , if you choose an FSPL model with 'Annual' parallax and two telescopes 1 and 2
                             to fit, the parameters will look like :
                             [to,uo,tE,rho,PiEN,PiEE,fs1,fb1/fs1,fs2,fb2/fs2]
                             For this case, number_of_parameters will be 6.

    Return :

    fit_results --> A list containing the results of the fit:

                    [model, method, parameters,chi^2], parameters following the PARAMETERS RULE.

    fit_covariance --> A list containg the covariance matric of the fit:

                      [model, method, covariance]. The covariance matrix is a number_of_parameters*number_of_parameters square matrix.

    fit_time --> List of effective computational time (in seconds) of the requested fits in the form :
                            [model,method,time]
            '''


    def __init__(self, event):

        self.event = event


    def mlfit(self, model, method):

        self.model = model
        self.method = method

        if self.method == 0:

            self.guess = self.initial_guess()
            self.fit_results, self.fit_covariance, self.fit_time = self.lmarquardt()

        if self.method == 1:

            start=time.time()
            AA=differential_evolution(self.chi_differential,bounds=self.model.parameters_boundaries,mutation=[1.8,1.9],recombination=0.5,polish='None')
            print AA['fun']
            computation_time = time.time()-start
            self.guess=AA['x'].tolist()+self.find_fluxes(AA['x'].tolist(), self.model)
            self.fit_results, self.fit_covariance, self.fit_time = self.lmarquardt()



        fit_quality_flag = self.check_fit()


        if fit_quality_flag == 'Bad Fit':

            if self.method == 0 :
                
                print 'We have to change method, this fit was unsuccessfull. We decided to switch method to 1'
                self.method = 1
                self.mlfit(self.model, 1)

            else :
                
                print 'Unfortunately, this is too hard for pyLIMA :('
 
    def check_fit(self):
        '''Check if the fit results and covariance make sens.
         0.0 terms or a negative term in the diagonal covariance matrix indicate the fit is not reliable.
         A negative source flux is also counted as a bad fit.
        '''
        flag = 'Good Fit'
        diago = np.diag(self.fit_covariance) < 0

        if (0.0 in self.fit_covariance) or (True in diago):

            print 'Your fit probably wrong. Cause ==> bad covariance matrix'
            flag = 'Bad Fit'
            return flag


        for i in self.event.telescopes:

            if self.fit_results[self.model.model_dictionnary['fs_'+i.name]] < 0:

                print 'Your fit probably wrong. Cause ==> negative source flux for telescope '+i.name
                flag = 'Bad Fit'
                return flag
                
        if 'rho' in self.model.model_dictionnary:
            
                    if self.fit_results[self.model.model_dictionnary['rho']] < 0 :
                        
                                        print 'Your fit probably wrong. Cause ==> negative source flux for telescope '+i.name
                                        flag = 'Bad Fit'
                                        return flag
        return flag

    def initial_guess(self):
        '''Function to find initial guess for Levenberg-Marquardt solver (method==0).
        Guess are made using the survey telescope for the Paczynski parameters (to,uo,tE).
        This assumes no blending.
        '''
        To=[]
        Max_flux=[]
        for i in self.event.telescopes:


            lightcurve = i.lightcurve_flux
            lightcurve = lightcurve[lightcurve[:, 0].argsort(), :]
            Time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            fs = min(flux)
            index = np.where(flux > fs)[0]
            good = index
            while len(good) > 5:

                index = np.where(flux[good] > np.median(flux[good]))[0]

                if len(index) < 4:

                    break

                else:

                    gravity = (np.median(Time[good[index]]), np.median(flux[good[index]]))
                    distances = np.sqrt((Time[good[index]]-gravity[0])**2+(flux[good[index]]-gravity[1])**2)
                    index = index[distances.argsort()[:-1]]
                    good = good[index]

            to = Time[good[np.where(flux[good] == np.max(flux[good]))[0]]][0]
            To.append(to)
            Max_flux.append(np.max(flux[good]))

        to=np.median(To)
        survey = self.event.telescopes[0]
        lightcurve = survey.lightcurve_flux
        lightcurve = lightcurve[lightcurve[:, 0].argsort(), :]
        Time = lightcurve[:, 0]
        flux = lightcurve[:, 1]
        errflux = lightcurve[:, 2]

        #fs, no blend

        baseline_flux_0 = np.min(flux)
        baseline_flux = np.median(flux)
        index = []

        while np.abs(baseline_flux_0-baseline_flux) > 0.01*baseline_flux:

            baseline_flux_0 = baseline_flux
            index = np.where((flux < baseline_flux))[0].tolist()+np.where(
                np.abs(flux-baseline_flux) < np.abs(errflux))[0].tolist()
            baseline_flux = np.median(flux[index])

            if  len(index) < 100:

                print 'low'
                baseline_flux = np.median(flux[flux.argsort()[:100]])
                break

        
        fs=baseline_flux
        max_flux = Max_flux[0]
        Amax = max_flux/fs
        uo = np.sqrt(-2+2*np.sqrt(1-1/(1-Amax**2)))
        #import pdb; pdb.set_trace()
    
        if self.model.paczynski_model == 'FSPL':

            uo=uo/10.0

        flux_demi = 0.5*fs*(Amax+1)
        flux_tE = fs*(uo**2+3)/((uo**2+1)**0.5*np.sqrt(uo**2+5))
        index_plus = np.where((Time > to)&(flux < flux_demi))[0]
        index_moins = np.where((Time < to)&(flux < flux_demi))[0]
        B = 0.5*(Amax+1)
        if len(index_plus) != 0:

            if len(index_moins) != 0:

                ttE = (Time[index_plus[0]]-Time[index_moins[-1]])
                tE1 = ttE/(2*np.sqrt(-2+2*np.sqrt(1+1/(B**2-1))-uo**2))

            else:

                ttE = Time[index_plus[0]]-to
                tE1 = ttE/np.sqrt(-2+2*np.sqrt(1+1/(B**2-1))-uo**2)
        else:

            ttE = to-Time[index_moins[-1]]
            tE1 = ttE/np.sqrt(-2+2*np.sqrt(1+1/(B**2-1))-uo**2)

        indextEplus = np.where((flux < flux_tE)&(Time > to))[0]
        indextEmoins = np.where((flux < flux_tE)&(Time < to))[0]
        tEmoins = 0.0
        tEplus = 0.0

        if len(indextEmoins) != 0:

            indextEmoins = indextEmoins[-1]
            tEmoins = to-Time[indextEmoins]

        if len(indextEplus) != 0:

            indextEplus = indextEplus[0]
            tEplus = Time[indextEplus]-to

        indextEPlus = np.where((Time > to)&(np.abs(flux-fs) < np.abs(errflux)))[0]
        indextEMoins = np.where((Time < to)&(np.abs(flux-fs) < np.abs(errflux)))[0]
        tEPlus = 0.0
        tEMoins = 0.0

        if len(indextEPlus) != 0:

            tEPlus = Time[indextEPlus[0]]-to

        if len(indextEMoins) != 0:

            tEMoins = to-Time[indextEMoins[-1]]

        TE = np.array([tE1, tEplus, tEmoins, tEPlus, tEMoins])
        good = np.where(TE != 0.0)[0]
        tE = np.sum(TE[good])/len(good)

        if tE < 1:

            tE = 20.0

        fake_model = microlmodels.MLModels(self.event, 'PSPL', self.model.second_order)

        fluxes=self.find_fluxes([to, uo, tE], fake_model)
        fluxes[0]=fs
        fluxes[1]=0.0

        parameters = [to, uo, tE]

        if self.model.paczynski_model == 'FSPL':

            parameters = parameters+[0.05]

        if self.model.parallax_model[0] != 'None':

            parameters = parameters+[0,0]

        if self.model.xallarap_model[0] != 'None':

            parameters = parameters+[0,0]

        if self.model.orbital_motion_model[0] != 'None':

            parameters = parameters+[0,0]

        if self.model.source_spots_model != 'None':

            parameters = parameters+[0]

        parameters=parameters+fluxes

        return parameters

    def lmarquardt(self):
        '''Method 0 of solver. This is based on the Levenberg-Marquardt algorithm:

        "A Method for the Solution of Certain Problems in Least Squares" 
        Levenberg, K. Quart. Appl. Math. 2, 1944, p. 164-168
        "An Algorithm for Least-Squares Estimation of Nonlinear Parameters"
        Marquardt, D. SIAM J. Appl. Math. 11, 1963, p. 431-441

        Based scipy.optimize.leastsq python routine, which is based on MINPACK's lmdif and lmder algorithms (fortran based).

        The objective function (function to minimize) is residuals
        The starting point is found using the initial_guess function
        the Jacobian is given by the Jacobian function


        WARNING : ftol (relative error desired in the sum of square) is set to 10^-5
                  maxfev (maximum number of function call) is set to 50000
                  These limits can avoid the fit to properly converge (expected to be rare :))
        '''

        start = time.time()

#        lmarquardt_fit = leastsq(self.residuals, self.guess, maxfev=50000, Dfun=self.Jacobian,
#                                 col_deriv=1, full_output=1, ftol=0.00001)

        lmarquardt_fit=leastsq(self.residuals, self.guess, maxfev=50000, full_output=1, ftol=0.00001)

        computation_time = time.time()-start

        fit_res = lmarquardt_fit[0].tolist()
        fit_res.append(self.chichi(lmarquardt_fit[0]))
        ndata = 0.0

        for i in self.event.telescopes:

            ndata = ndata+i.n_data()

        try:

            if lmarquardt_fit[1] is not None:

                cov = lmarquardt_fit[1]*fit_res[len(self.model.model_dictionnary)]/ndata
                import pdb; pdb.set_trace()

            else:

                print 'rough cov'
                jacky = self.Jacobian(fit_res, self.model)
                cov = np.linalg.inv(jacky*jacky.T)*fit_res[len(self.model.model_dictionnary)]/ndata

        except:

            print 'hoho'
            cov = np.zeros((len(self.model.model_dictionnary),
                            len(self.model.model_dictionnary)))

        return fit_res, cov, computation_time

    def Jacobian(self, parameters):
        '''Return the analytical Jacobian matrix, requested by method 0.
        '''
        if self.model.paczynski_model == 'PSPL':

            dresdto = np.array([])
            dresduo = np.array([])
            dresdtE = np.array([])
            dresdfs = np.array([])
            dresdeps = np.array([])
           

            for i in self.event.telescopes:

                lightcurve = i.lightcurve_flux
                Time = lightcurve[:, 0]
                errflux = lightcurve[:, 2]
                gamma = i.gamma
                
                ampli = microlmagnification.amplification(self.model,Time, parameters, gamma)
                dAdU = (-8)/(ampli[1]**2*(ampli[1]**2+4)**1.5)

                dUdto = -(Time-parameters[self.model.model_dictionnary['to']])/(parameters[self.model.model_dictionnary['tE']]**2*ampli[1])
                dUduo = parameters[self.model.model_dictionnary['uo']]/ampli[1]
                dUdtE = -(Time-parameters[self.model.model_dictionnary['to']])**2/(parameters[self.model.model_dictionnary['tE']]**3*ampli[1])

                dresdto = np.append(dresdto, -parameters[self.model.model_dictionnary['fs_'+i.name]]*
                                    dAdU*dUdto/errflux)
                dresduo = np.append(dresduo, -parameters[self.model.model_dictionnary['fs_'+i.name]]*
                                    dAdU*dUduo/errflux)
                dresdtE = np.append(dresdtE, -parameters[self.model.model_dictionnary['fs_'+i.name]]*
                                    dAdU*dUdtE/errflux)
                dresdfs = np.append(dresdfs, -(ampli[0]+parameters[self.model.model_dictionnary['g_'+i.name]])/errflux)
                dresdeps = np.append(dresdeps, -parameters[self.model.model_dictionnary['fs_'+i.name]]/errflux)


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


                ampli = microlmagnification.amplification(fake_model,Time, fake_params, gamma)
                dAdU = (-8)/(ampli[1]**2*(ampli[1]**2+4)**(1.5))

                Z = ampli[1]/parameters[self.model.model_dictionnary['rho']]

                dadu = np.zeros(len(ampli[0]))
                dadrho = np.zeros(len(ampli[0]))
                #ind = np.where((Z > 10) | (Z < self.model.yoo_table[0][0]))[0]
                ind = np.where((Z > 19.9999) | (Z < self.model.yoo_table[0][0]))[0]
                #dadu[ind] = dAdU[ind]-ampli[0][ind]*1/parameters[self.model.model_dictionnary['rho']]*3*10**-5
                dadu[ind] = dAdU[ind]

                #dadrho[ind] = ampli[0][ind]*ampli[1][ind]/parameters[self.model.model_dictionnary['rho']]**2*3*10**-5
                dadrho[ind] = ampli[0][ind]*ampli[1][ind]/parameters[self.model.model_dictionnary['rho']]**2*3.1*10**-5
#                dadrho[ind] = -ampli[0][ind]*ampli[1][ind]/parameters[self.model.model_dictionnary['rho']]**2*(
#                                self.model.yoo_table[3](self.model.yoo_table[0][-1])-
#                                gamma*self.model.yoo_table[4](self.model.yoo_table[0][-1]))
#                dadrho[ind] = 0.0

                #ind = np.where((Z <= 10) & (Z >= self.model.yoo_table[0][0]))[0]
                ind = np.where((Z <= 19.9999) & (Z >= self.model.yoo_table[0][0]))[0]
                dadu[ind] = dAdU[ind]*(self.model.yoo_table[1](Z[ind])-gamma*self.model.yoo_table[2](
                            Z[ind]))+ampli[0][ind]*(self.model.yoo_table[3](Z[ind])-gamma*self.model.yoo_table[4](
                             Z[ind]))*1/parameters[self.model.model_dictionnary['rho']]

                dadrho[ind] = -ampli[0][ind]*ampli[1][ind]/parameters[self.model.model_dictionnary['rho']]**2*(
                              self.model.yoo_table[3](Z[ind])-gamma*self.model.yoo_table[4](Z[ind]))

                dUdto = -(Time-parameters[self.model.model_dictionnary['to']])/(
                        parameters[self.model.model_dictionnary['tE']]**2*ampli[1])

                dUduo = parameters[self.model.model_dictionnary['uo']]/ampli[1]
                dUdtE = -(Time-parameters[self.model.model_dictionnary['to']])**2/(
                        parameters[self.model.model_dictionnary['tE']]**3*ampli[1])

                dresdto = np.append(dresdto, -parameters[self.model.model_dictionnary['fs_'+i.name]]*dadu*
                                    dUdto/errflux)
                dresduo = np.append(dresduo, -parameters[self.model.model_dictionnary['fs_'+i.name]]*dadu*
                                    dUduo/errflux)
                dresdtE = np.append(dresdtE, -parameters[self.model.model_dictionnary['fs_'+i.name]]*dadu*
                                    dUdtE/errflux)
                dresdrho = np.append(dresdrho, -parameters[self.model.model_dictionnary['fs_'+i.name]]*
                                     dadrho/errflux)

                ampli = microlmagnification.amplification(self.model,Time, parameters, gamma)
                dresdfs = np.append(dresdfs, -(ampli[0]+parameters[self.model.model_dictionnary['g_'+i.name]])/errflux)
                dresdeps = np.append(dresdeps, -parameters[self.model.model_dictionnary['fs_'+i.name]]/errflux)

            jacobi = np.array([dresdto, dresduo, dresdtE, dresdrho])

        start = 0

        for i in self.event.telescopes:

            dFS = np.zeros((len(dresdto)))
            dEPS = np.zeros((len(dresdto)))

            index = np.arange(start, start+len(i.lightcurve_flux[:, 0]))
            dFS[index] = dresdfs[index]
            dEPS[index] = dresdeps[index]
            jacobi = np.vstack([jacobi, dFS])
            jacobi = np.vstack([jacobi, dEPS])

            start = start+index[-1]+1

        #import pdb; pdb.set_trace()

        return jacobi

    def residuals(self, parameters):
        ''' The normalized residuals associated to the model and parameters.
        residuals_i=(y_i-model_i)/sigma_i
        The sum of square residuals gives chi^2.
        '''
        errors = np.array([])
        count = 0

        for i in self.event.telescopes:

            lightcurve = i.lightcurve_flux
            Time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            gamma = i.gamma
            ampli = microlmagnification.amplification(self.model, Time, parameters, gamma)[0]
            errors = np.append(errors, (flux-ampli*parameters[self.model.model_dictionnary['fs_'+i.name]]-
                                       (parameters[self.model.model_dictionnary['fs_'+i.name]]*parameters[
                                       self.model.model_dictionnary['g_'+i.name]]))/errflux)
#            if 'rho' in self.model.model_dictionnary:
#                if parameters[self.model.model_dictionnary['rho']]<0 :
#                    errors=errors*10**10
            count = count+1

        return errors

    def chichi(self, parameters):
        '''Return the chi^2.
        '''
        errors = self.residuals(parameters)
        chichi = (errors**2).sum()

        return chichi

    def chi_differential(self, parameters) :
        
        '''Return the chi^2 for dirrential_evolution. fsi,fbi evaluated trough polyfit.
        '''
        errors = np.array([])

        for i in self.event.telescopes:

            lightcurve = i.lightcurve_flux
            Time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            gamma = i.gamma
            ampli = microlmagnification.amplification(self.model, Time, parameters, gamma)[0]
            fs, fb = np.polyfit(ampli, flux, 1, w=1/errflux)
            errors = np.append(errors, (flux-ampli*fs-fb)/errflux)

        chichi = (errors**2).sum()
        return chichi

    def find_fluxes(self, parameters, model):

        fluxes = []
        for i in self.event.telescopes:

            lightcurve = i.lightcurve_flux
            Time = lightcurve[:, 0]
            flux = lightcurve[:, 1]
            errflux = lightcurve[:, 2]
            gamma = i.gamma
            ampli = microlmagnification.amplification(model, Time, parameters, gamma)[0]
            fs, fb = np.polyfit(ampli, flux, 1, w=1/errflux)
            if (fs<0) :

                fluxes.append(np.abs(fs*(1+fb/fs)))
                fluxes.append(0.0)
            else:
                fluxes.append(fs)
                fluxes.append(fb/fs)
        return fluxes

