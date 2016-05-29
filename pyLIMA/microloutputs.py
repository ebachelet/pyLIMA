# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:38:14 2015

@author: ebachelet
"""
from __future__ import division
from datetime import datetime
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from astropy.time import Time
from scipy.stats.distributions import t as student
import collections

import microltoolbox

import copy




def LM_outputs(fit) :
    """ Standard 'LM' and 'DE' outputs (a named tuple python object) :
    
        - fit_parameters : a named tuple python object containing the fitted parameters. See microlmodels module for details.
        - fit_errors : a named tuple python object containing the error on fitted parameters. Square root of the variance from the covariance matrice.
        - fit_correlation_matrix : return a numpy array containing the correlation matrix.
        - figure_lightcurve : a data+model matplotlib.pyplot plot.
    """
    
    results = LM_parameters_result(fit)
    covariance_matrix = fit.fit_covariance
    errors = LM_fit_errors(fit)
    correlation_matrix = cov2corr(covariance_matrix)
    figure_lightcurve = LM_plot_lightcurves(fit)
    #figure_parameters = LM_plot_parameters(fit)
    key_outputs = ['fit_parameters','fit_errors','fit_correlation_matrix','figure_lightcurve']
    outputs=collections.namedtuple('Fit_outputs', key_outputs)        
     
    
    values_outputs = [results,errors,correlation_matrix,figure_lightcurve]
     
    count = 0
    for key in key_outputs :
         
         setattr(outputs,key,values_outputs[count])
         count += 1
     
 
    return outputs
def MCMC_outputs(fit) :  
    """ Standard 'MCMC' outputs (a named tuple python object) :
    
        - MCMC_chains : a numpy array containing the MCMC chains.
        - figure_lightcurve : a data+model matplotlib.pyplot plot. 35 models selected in the the MCMC chains, within 6 sigma lower than the maximum
          likelihood, are plotted.
        - figure_distributions : 6-sigma distributions of the MCMC_chains.
    """

    chains = fit.MCMC_chains
    probabilities = fit.MCMC_probabilities

    CHAINS = chains[:,:,0].ravel()
    for i in xrange(len(fit.model.parameters_boundaries)-1):
        i += 1
        CHAINS = np.c_[CHAINS,chains[:,:,i].ravel()]
    fluxes = MCMC_compute_fs_g(fit, CHAINS)
     
    CHAINS = np.c_[CHAINS,fluxes,probabilities.ravel()]
     
    best_proba = np.argmax(CHAINS[:,-1])
     
    #cut to 6 sigma for plots
    index=np.where(CHAINS[:,-1]>CHAINS[best_proba,-1]-36)[0]
    BEST = CHAINS[index]
    BEST=BEST[BEST[:,-1].argsort(),]
    
    covariance_matrix = MCMC_covariance(CHAINS)
    correlation_matrix = cov2corr(covariance_matrix)
    

    figure_lightcurve = MCMC_plot_lightcurves(fit,BEST)
    figure_distributions = MCMC_plot_parameters_distribution(fit,BEST)
    
    
    key_outputs = ['MCMC_chains','MCMC_correlations','figure_lightcurve','figure_distributions']
    outputs=collections.namedtuple('Fit_outputs', key_outputs) 
   
    values_outputs = [CHAINS, correlation_matrix, figure_lightcurve, figure_distributions]
    
    count = 0
    for key in key_outputs :
         
        setattr(outputs,key,values_outputs[count])
        count += 1
     
 
    
    return outputs
    
    
def MCMC_compute_fs_g(fit,CHAINS) :
    
    Fluxes=[]
    for chain in CHAINS :
        
        fluxes = fit.find_fluxes(chain, fit.model)
        Fluxes.append(fluxes)
    
    return np.array(Fluxes)


def  MCMC_plot_parameters_distribution(fit,BEST):
    
    dimensions = len(fit.model.parameters_boundaries)
    
    figure_distributions, axes2 = plt.subplots(dimensions, dimensions,sharex='col')
    #import pdb; pdb.set_trace()

    count_i = 0

    for key_i in fit.model.model_dictionnary.keys()[: dimensions] :
        axes2[count_i,0].set_ylabel(key_i)
        axes2[-1,count_i].set_xlabel(key_i)
        
        
        count_j = 0         
        for key_j in fit.model.model_dictionnary.keys()[: dimensions] :
            
            
            axes2[count_i,count_j].ticklabel_format(useOffset=False, style='plain')
            
            if count_i!=dimensions-1:
                
                plt.setp(axes2[count_i,count_j].get_xticklabels() , visible=False)
                
            if (count_j!=0) and (count_j!=count_i) :

                plt.setp(axes2[count_i,count_j].get_yticklabels() , visible=False)
            
            if count_i==count_j :
                
                axes2[count_i,count_j].hist(BEST[:,fit.model.model_dictionnary[key_i]], 100)
            
            else :
                
                if count_j<count_i :
                
                    axes2[count_i,count_j].scatter(BEST[:,fit.model.model_dictionnary[key_j]],BEST[:,fit.model.model_dictionnary[key_i]],c=BEST[:,-1],edgecolor='None')
                    
                else :
                    axes2[count_i,count_j].axis('off')     
                
            count_j += 1  
            
        count_i += 1
        
        
   
    return figure_distributions

def  MCMC_plot_lightcurves(fit,BEST):
    
     figure_lightcurves, axes = initialize_plot_figure(fit)  

     MCMC_plot_align_data(fit,BEST[0,len(fit.model.parameters_boundaries):-1], axes[0])
     
     
     index=np.linspace(0,len(BEST)-1,35).astype(int)
     norm=matplotlib.colors.Normalize(vmin=np.min(BEST[:,-1]),vmax=np.max(BEST[:,-1]))
     c_m = matplotlib.cm.jet

     s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
     s_m.set_array([])
     for indice in index :
         MCMC_plot_model(fit, BEST[indice], BEST[indice,-1],axes[0], s_m)

     
     plt.colorbar(s_m,ax=axes[0])
     axes[0].text(0.01,0.97,'provided by pyLIMA',style='italic',fontsize=10,transform=axes[0].transAxes)
     axes[0].invert_yaxis()   
     MCMC_plot_residuals(fit, BEST[0], axes[1])
     
     
     return figure_lightcurves

def MCMC_plot_model(fit, parameters, couleur, ax, s_m) :


    min_time = min([min(i.lightcurve[:,0]) for i in fit.event.telescopes])
    max_time = max([max(i.lightcurve[:,0]) for i in fit.event.telescopes])

    time = np.arange(min_time, max_time + 100, 0.01)
    
   
    gamma = reference_telescope.gamma
    fs_reference = parameters[fit.model.model_dictionnary['fs_'+reference_telescope.name]]
    g_reference = parameters[fit.model.model_dictionnary['g_'+reference_telescope.name]]
    
    ampli = fit.model.magnification(parameters, time, gamma,reference_telescope.deltas_positions)[0]
    
    flux = fs_reference*(ampli+g_reference)
    mag = microltoolbox.flux_to_magnitude(flux)
   

    ax.plot(time,mag,color=s_m.to_rgba(couleur), alpha=0.5)
    
    
  
    
    
def MCMC_plot_align_data(fit, fluxes, ax) :
    
    reference_telescope = fit.event.telescopes[0].name
    fs_reference = fluxes[0]
    g_reference = fluxes[1]

    count = 0
    for i in fit.event.telescopes :
        
        if i.name == reference_telescope :
            
            lightcurve = i.lightcurve
            
        else :
             
            fs_telescope = fluxes[count]
            g_telescope = fluxes[count+1]
            
            lightcurve = align_telescope_lightcurve(i.lightcurve,fs_reference,g_reference,fs_telescope,g_telescope)
        
        ax.errorbar(lightcurve[:,0], lightcurve[:,1], yerr=lightcurve[:,2],fmt='.',label=i.name)
        count += 2        
    ax.legend(numpoints=1)


def MCMC_plot_residuals(fit, parameters, ax):
    
   

    for i in fit.event.telescopes :
        
        fs_telescope = parameters[fit.model.model_dictionnary['fs_'+i.name]]
        g_telescope = parameters[fit.model.model_dictionnary['g_'+i.name]]
        
        gamma = i.gamma
        
        time = i.lightcurve[:,0]
        mag = i.lightcurve[:,1]
        flux = microltoolbox.magnitude_to_flux(mag)
        err_mag = i.lightcurve[:,2]

        ampli = fit.model.magnification(parameters, time, gamma,reference_telescope)[0]
        
        flux_model = fs_telescope*(ampli+g_telescope)
        
        residuals = 2.5*np.log10(flux_model/flux)
        ax.errorbar(time, residuals, yerr=err_mag,fmt='.')
    ax.set_ylim([-0.1,0.1])


def LM_parameters_result(fit) :
    
    
    parameters = collections.namedtuple('Parameters',fit.model.model_dictionnary.keys())
    
    for i in  fit.model.model_dictionnary.keys():
        
        setattr(parameters,i,fit.fit_results[fit.model.model_dictionnary[i]])
    
    setattr(parameters,'chichi',fit.fit_results[-1])
    return parameters

def MCMC_covariance(chains):
    
    esperances = []   
    for i in xrange(chains.shape[1]-1):
        
        esperances.append(chains[:,i]-np.median(chains[:,i]))   
       
    cov = np.zeros((chains.shape[1]-1,chains.shape[1]-1))
  
   
    for i in xrange(chains.shape[1]-1):
         for j in np.arange(i,chains.shape[1]-1):

            cov[i,j] = 1/(len(chains)-1)*np.sum(esperances[i]*esperances[j])
            cov[j,i] = 1/(len(chains)-1)*np.sum(esperances[i]*esperances[j])


    return cov
def LM_fit_errors(fit) :
    
    keys = ['err_'+i for i in fit.model.model_dictionnary.keys() ]
    parameters_errors = collections.namedtuple('Errors_Parameters',keys)
    errors = fit.fit_covariance.diagonal()**0.5
    for i in  fit.model.model_dictionnary.keys():
        
        setattr(parameters_errors,'err_'+i,errors[fit.model.model_dictionnary[i]])
    
    return parameters_errors

def cov2corr(A):
    """
    covariance matrix to correlation matrix.
    """

    d = np.sqrt(A.diagonal())
    correlation = ((A.T / d).T) / d

    return correlation

def LM_plot_lightcurves(fit) :
   
    figure,axes = initialize_plot_figure(fit)
    LM_plot_align_data(fit,axes[0])
    LM_plot_model(fit,axes[0])
    LM_plot_residuals(fit,axes[1])
    
    return figure

def LM_plot_parameters(fit) :
    
    figure,axes = initialize_plot_parameters()
   
    
    return figure
    
    
def initialize_plot_figure(fit):
    
    figure, axes = plt.subplots(2,1,sharex=True)
    axes[0].grid()
    axes[1].grid()
    figure.suptitle(fit.event.name,fontsize=30)
    
    return figure, axes

def initialize_plot_parameters(fit):

    dimension_y = np.floor(len(fit.fits_result)/3)
    dimension_x = len(fit.fits_result)-3*dimension_y
    
    figure, axes = plt.subplots(dimension_x,dimension_y)
    
    
    return figure, axes  
    
    
def LM_plot_model(fit, ax) :
    


    min_time = min([min(i.lightcurve[:,0]) for i in fit.event.telescopes])
    max_time = max([max(i.lightcurve[:,0]) for i in fit.event.telescopes])
	
    time = np.arange(min_time, max_time + 100, 0.01) 
    if fit.model.parallax_model !='None' :

	    reference_telescope = copy.copy(fit.event.telescopes[0])
	    reference_telescope.lightcurve = np.array([time,[0]*len(time),[0]*len(time)]).T
	    reference_telescope.compute_parallax(fit.event, fit.model.parallax_model)
    else :

	  reference_telescope = fit.event.telescopes[0] 
    gamma = reference_telescope.gamma
    fs_reference = fit.fit_results[fit.model.model_dictionnary['fs_'+reference_telescope.name]]
    g_reference = fit.fit_results[fit.model.model_dictionnary['g_'+reference_telescope.name]]
    
    ampli = fit.model.magnification(fit.fit_results, time, gamma,reference_telescope.deltas_positions)[0]
    
    flux = fs_reference*(ampli+g_reference)
    mag = microltoolbox.flux_to_magnitude(flux)
    
    ax.plot(time,mag,'r',lw=2)
    ax.set_ylim([min(mag)-0.1,max(mag)+0.1])
    ax.invert_yaxis()
    ax.text(0.01,0.97,'provided by pyLIMA',style='italic',fontsize=10,transform=ax.transAxes)
    
def LM_plot_residuals(fit,ax):
    
   

    for i in fit.event.telescopes :
        
        fs_telescope = fit.fit_results[fit.model.model_dictionnary['fs_'+i.name]]
        g_telescope = fit.fit_results[fit.model.model_dictionnary['g_'+i.name]]
        
        gamma = i.gamma
        
        time = i.lightcurve[:,0]
        mag = i.lightcurve[:,1]
        flux = microltoolbox.magnitude_to_flux(mag)
        err_mag = i.lightcurve[:,2]

        ampli = fit.model.magnification(fit.fit_results, time, gamma,i.deltas_positions)[0]
        
        flux_model = fs_telescope*(ampli+g_telescope)
        
        residuals = 2.5*np.log10(flux_model/flux)
        ax.errorbar(time, residuals, yerr=err_mag,fmt='.')
    ax.set_ylim([-0.1,0.1])
    ax.invert_yaxis()
    

        
    
    
def LM_plot_align_data(fit,ax) :
    
    reference_telescope = fit.event.telescopes[0].name
    fs_reference = fit.fit_results[fit.model.model_dictionnary['fs_'+reference_telescope]]
    g_reference = fit.fit_results[fit.model.model_dictionnary['g_'+reference_telescope]]

    for i in fit.event.telescopes :
        
        if i.name == reference_telescope :
            
            lightcurve = i.lightcurve
        
        else :
             
            fs_telescope = fit.fit_results[fit.model.model_dictionnary['fs_'+i.name]]
            g_telescope = fit.fit_results[fit.model.model_dictionnary['g_'+i.name]]
            
            lightcurve = align_telescope_lightcurve(i.lightcurve,fs_reference,g_reference,fs_telescope,g_telescope)

        ax.errorbar(lightcurve[:,0], lightcurve[:,1], yerr=lightcurve[:,2],fmt='.',label=i.name)
        
    ax.legend(numpoints=1)
    
    
    
def align_telescope_lightcurve(lightcurve_telescope_mag,fs_reference,g_reference,fs_telescope,g_telescope) :
    
    time = lightcurve_telescope_mag[:,0]
    mag = lightcurve_telescope_mag[:,1]
    err_mag = lightcurve_telescope_mag[:,2]

    flux = microltoolbox.magnitude_to_flux(mag)
    
    flux_normalised = (flux-(fs_telescope*g_telescope))/(fs_telescope)*fs_reference+fs_reference*g_reference
    
    mag_normalised = microltoolbox.flux_to_magnitude(flux_normalised)
    

    lightcurve_normalised = [time,mag_normalised,err_mag]
    
    lightcurve_mag_normalised = np.array(lightcurve_normalised).T
    
    return lightcurve_mag_normalised






### TO DO : some parts depreciated ####

def errors_on_fits(self, choice):

        if len(self.event.fits[choice].fit_covariance)==0:

            print 'There is no way to produce errors without covariance at this stage'

        else:

            self.event.fits[choice].fit_errors = np.sqrt(
                self.event.fits[choice].fit_covariance.diagonal())

def find_observables(self):

        count = 0
        self.observables_dictionnary = {'to': 0, 'Ao': 1, 'tE': 2, 'Anow': 3, 'Ibaseline': 4,
                                        'Ipeak': 5, 'Inow': 6}
        self.observables_dictionnary = OrderedDict(
            sorted(self.observables_dictionnary.items(), key=lambda x: x[1]))
        for i in self.event.fits_results:

            observables = []
            parameters = i[3]
            to = parameters[self.event.fits_models[count][2].model_dictionnary['to']]
            uo = parameters[self.event.fits_models[count][2].model_dictionnary['uo']]
            tE = parameters[self.event.fits_models[count][2].model_dictionnary['tE']]

            t = Time(datetime.utcnow())
            # tnow=t.jd1+t.jd2
            tnow = 150
            Ao = microlmagnification.amplification(self.event.fits_models[count][2], np.array([to]),
                                                   parameters, self.event.telescopes[0].gamma)[0][0]
            Anow = \
            microlmagnification.amplification(self.event.fits_models[count][2], np.array([tnow]),
                                              parameters, self.event.telescopes[0].gamma)[0][0]

            observables.append(to)
            observables.append(Ao)
            observables.append(tE)
            observables.append(Anow)

            Ibaseline = 27.4 - 2.5 * np.log10(
                parameters[self.event.fits_models[count][2].model_dictionnary[
                    'fs_' + self.event.telescopes[0].name]] * (
                1 + parameters[self.event.fits_models[count][2].model_dictionnary[
                    'g_' + self.event.telescopes[0].name]]))

            Ipeak = 27.4 - 2.5 * np.log10(
                parameters[self.event.fits_models[count][2].model_dictionnary[
                    'fs_' + self.event.telescopes[0].name]] * (
                Ao + parameters[self.event.fits_models[count][2].model_dictionnary[
                    'g_' + self.event.telescopes[0].name]]))

            Inow = 27.4 - 2.5 * np.log10(
                parameters[self.event.fits_models[count][2].model_dictionnary[
                    'fs_' + self.event.telescopes[0].name]] * (
                Anow + parameters[self.event.fits_models[count][2].model_dictionnary[
                    'g_' + self.event.telescopes[0].name]]))

            observables.append(Ibaseline)
            observables.append(Ipeak)
            observables.append(Inow)

            self.observables.append([i[0], i[1], i[2], observables])

def find_observables_errors(self):

        for i in xrange(len(self.event.fits_results)):

            parameters = self.observables[i][2]
            parameters_errors = self.error_parameters[i][2]

            to = self.event.fits_results[i][2][0]
            uo = self.event.fits_results[i][2][1]
            tE = self.event.fits_results[i][2][2]

            Ao = parameters[1]
            err_Ao = parameters_errors[1] * 8 / (
            parameters[1] ** 2 * (parameters[1] ** 2 + 4) ** 1.5)
            Anow = parameters[3]
            jd1, jd2 = Time(datetime.datetime.utcnow())
            tnow = jd1 + jd2
            unow = np.sqrt(uo ** 2 + (tnow - to) ** 2 / tE ** 2)
            err_Anow = (uo * parameters_errors[1] * np.abs((tnow - to)) / tE ** 3 * (
            tE * parameters_errors[0] + np.abs((tnow - to)) * parameters_errors[2])) / unow
            observables_errors = []
            observables = []
            observables_errors.append(parameters_errors[0])
            observables_errors.append(err_Ao)
            observables_errors.append(parameters_errors[2])
            observables_errors.append(err_Anow)

            start = len(parameters) - 2 * len(self.event.telescopes) - 1
            for j in xrange(len(self.event.telescopes)):

                Ibaseline = 27.4 - 2.5 * np.log10(parameters[start] * (1 + parameters[start]))
                Ipeak = 27.4 - 2.5 * np.log10(parameters[start] * (Ao + parameters[start]))

                observables.append(Ibaseline)
                observables.append(Ipeak)

                start += 2

            self.observables.append([i[0], i[1], observables])

def errors_on_observables(self):

        for i in self.event.fits_covariance:

            self.error_parameters.append([i[0], i[1], np.sqrt(i[2].diagonal)])



def student_errors(self):

        alpha = 0.05
        ndata = len(self.event.telescopes[0].lightcurve_flux)
        npar = 5
        dof = ndata - npar
        tval = student.ppf(1 - alpha / 2, dof)

        lower = []
        upper = []

        for i in xrange(len(self.event.fits_covariance[0][2].diagonal())):

            sigma = self.event.fits_covariance[0][2].diagonal()[i] ** 0.5
            lower.append(self.event.fits_results[0][2][i] - sigma * tval)
            upper.append(self.event.fits_results[0][2][i] + sigma * tval)

        self.upper = upper
        self.lower = lower

def K2_C9_outputs(self):
        import matplotlib.pyplot as plt

        # first produce aligned lightcurve#

        time = []
        mag = []
        err_mag = []
        groups = []

        time = time + self.event.telescopes[0].lightcurve[:, 0].tolist()
        mag = mag + self.event.telescopes[0].lightcurve[:, 1].tolist()
        err_mag = err_mag + self.event.telescopes[0].lightcurve[:, 2].tolist()
        groups = groups + [self.event.telescopes[0].name] * len(self.event.telescopes[0].lightcurve)

        for i in self.event.telescopes[1:]:

            time = time + i.lightcurve[:, 0].tolist()
            Mag = i.lightcurve[:, 1]
            flux = 10 ** ((27.4 - Mag) / 2.5)
            err_flux = np.abs(-i.lightcurve[:, 2] * flux / (2.5) * np.log(10))
            flux_normalised = self.event.fits[0].fit_results[
                                  self.event.fits[0].model.model_dictionnary[
                                      'fs_' + self.event.telescopes[0].name]] * ((
                                                                                     flux /
                                                                                     self.event.fits[
                                                                                         0].fit_results[
                                                                                         self.event.fits[
                                                                                             0].model.model_dictionnary[
                                                                                             'fs_' + i.name]] -
                                                                                     self.event.fits[
                                                                                         0].fit_results[
                                                                                         self.event.fits[
                                                                                             0].model.model_dictionnary[
                                                                                             'g_'
                                                                                             +
                                                                                             i.name]]) +
                                                                                 self.event.fits[
                                                                                     0].fit_results[
                                                                                     self.event.fits[
                                                                                         0].model.model_dictionnary[
                                                                                         'g_' +
                                                                                         self.event.telescopes[
                                                                                             0].name]])
            err_flux_norm = err_flux / flux * flux_normalised
            mag_norm = 27.4 - 2.5 * np.log10(flux_normalised)
            err_mag_norm = 2.5 * err_flux_norm / (flux_normalised * np.log(10))

            mag = mag + mag_norm.tolist()
            err_mag = err_mag + err_mag_norm.tolist()
            groups = groups + [i.name] * len(i.lightcurve)

        lightcurve_data = np.array([time, mag, err_mag, groups]).T

        # produce model lightcurve

        time = np.arange(min(self.event.telescopes[0].lightcurve[:, 0]), max(time) + 100, 0.01)
        ampli = microlmagnification.amplification(self.event.fits[0].model, time,
                                                  self.event.fits[0].fit_results, 0.5)[0]
        flux = self.event.fits[0].fit_results[self.event.fits[0].model.model_dictionnary[
            'fs_' + self.event.telescopes[0].name]] * (
                   ampli + self.event.fits[0].fit_results[
                       self.event.fits[0].model.model_dictionnary[
                           'g_' + self.event.telescopes[0].name]])
        mag = (27.4 - 2.5 * np.log10(flux)).tolist()
        err_mag = [0.001] * len(time)
        time = time.tolist()
        lightcurve_model = np.array([time, mag, err_mag]).T


        # produce parameters
        Parameters = []
        Names = []

        Uo = self.event.fits[0].fit_results[self.event.fits[0].model.model_dictionnary['uo']]
        Ao = (Uo ** 2 + 2) / (Uo * (Uo ** 2 + 4) ** 0.5)
        err_Ao = (8) / (Uo ** 2 * (Uo ** 2 + 4) ** 1.5) * \
                 (self.event.fits[0].fit_covariance.diagonal() ** 0.5)[1]

        Parameters.append(Ao)
        Parameters.append(err_Ao)

        Names.append('PYLIMA.AO')
        Names.append('PYLIMA.SIG_AO')

        names = ['TE', 'TO', 'UO']
        Official = ['tE', 'to', 'uo']

        for i in xrange(len(Official)):

            index = self.event.fits[0].model.model_dictionnary[Official[i]]
            Parameters.append(self.event.fits[0].fit_results[index])
            Parameters.append((self.event.fits[0].fit_covariance.diagonal() ** 0.5)[index])

            Names.append('PYLIMA.' + names[i])
            Names.append('PYLIMA.SIG_' + names[i])
        Parameters = np.array([Names, Parameters]).T
        count = 0
        for i in self.event.telescopes:
            index = np.where(lightcurve_data[:, 3] == i.name)[0]
            colors = np.random.uniform(0, 10)
            plt.scatter(lightcurve_data[index, 0].astype(float),
                        lightcurve_data[index, 1].astype(float), c=(
                np.random.randint(0, float(len(self.event.telescopes))) / float(
                    len(self.event.telescopes)),
                np.random.randint(0, float(len(self.event.telescopes))) / float(
                    len(self.event.telescopes)),
                np.random.randint(0, float(len(self.event.telescopes))) / float(
                    len(self.event.telescopes)), 1), label=i.name, s=25)
            count += 1
        plt.legend(scatterpoints=1)
        plt.plot(lightcurve_model[:, 0], lightcurve_model[:, 1], 'g')
        plt.show()

        return Parameters, lightcurve_model, lightcurve_data
