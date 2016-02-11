# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:38:14 2015

@author: ebachelet
"""
from __future__ import division
import numpy as np
from datetime import datetime
from astropy.time import Time
from scipy.stats.distributions import t as student
from collections import OrderedDict

import microlmagnification

class MLOutputs(object):
    
    
    def __init__(self, event):

        self.event = event
        self.fits_errors = []
        self.observables = [] 
        self.observables_errors = []
        self.plots= []
    
    def errors_on_fits(self,choice):
        
        if self.event.fits[choice].fit_covariance == None :

            print 'There is no way to produce errors without covariance at this stage'
        
        else:
                       
           self.event.fits[choice].fit_errors = np.sqrt(self.event.fits[choice].fit_covariance.diagonal()) 

    def find_observables(self):
        
        count = 0
        self.observables_dictionnary = {'to' : 0 , 'Ao' : 1, 'tE' : 2, 'Anow' : 3, 'Ibaseline' : 4, 'Ipeak' : 5, 'Inow' : 6 }
        self.observables_dictionnary = OrderedDict(sorted(self.observables_dictionnary.items(), key=lambda x: x[1]))
        for i in self.event.fits_results :
            
            observables = []
            parameters = i[3]
            to = parameters[self.event.fits_models[count][2].model_dictionnary['to']]
            uo = parameters[self.event.fits_models[count][2].model_dictionnary['uo']]
            tE = parameters[self.event.fits_models[count][2].model_dictionnary['tE']]

            t=Time(datetime.utcnow())
            #tnow=t.jd1+t.jd2
            tnow = 150     
            Ao=microlmagnification.amplification(self.event.fits_models[count][2], np.array([to]), parameters,self.event.telescopes[0].gamma )[0][0]
            Anow=microlmagnification.amplification(self.event.fits_models[count][2], np.array([tnow]), parameters,self.event.telescopes[0].gamma )[0][0] 
            
            observables.append(to)
            observables.append(Ao)
            observables.append(tE)
            observables.append(Anow)
            
           
         
            Ibaseline=27.4-2.5*np.log10(parameters[self.event.fits_models[count][2].model_dictionnary[
            'fs_'+self.event.telescopes[0].name]]*(1+parameters[self.event.fits_models[count][2].model_dictionnary[
            'g_'+self.event.telescopes[0].name]]))

            Ipeak=27.4-2.5*np.log10(parameters[self.event.fits_models[count][2].model_dictionnary[
            'fs_'+self.event.telescopes[0].name]]*(Ao+parameters[self.event.fits_models[count][2].model_dictionnary[
            'g_'+self.event.telescopes[0].name]]))
            
            Inow = 27.4-2.5*np.log10(parameters[self.event.fits_models[count][2].model_dictionnary[
            'fs_'+self.event.telescopes[0].name]]*(Anow+parameters[self.event.fits_models[count][2].model_dictionnary[
            'g_'+self.event.telescopes[0].name]]))        
            
            observables.append(Ibaseline)
            observables.append(Ipeak)
            observables.append(Inow)
            
               
            self.observables.append([i[0],i[1],i[2],observables])
            
    def find_observables_errors(self):
        
        
        for i in xrange(len(self.event.fits_results)) :
            
            observable_errors = []
            parameters = self.observables[i][2]
            parameters_errors = self.error_parameters[i][2]
          
            to=self.event.fits_results[i][2][0]
            uo=self.event.fits_results[i][2][1]
            tE=self.event.fits_results[i][2][2]
            
            Ao = parameters[1]
            err_Ao = parameters_errors[1]*8/(parameters[1]**2*(parameters[1]**2+4)**1.5)
            Anow = parameters[3]
            jd1,jd2=Time(datetime.datetime.utcnow())
            tnow=jd1+jd2
            unow=np.sqrt(uo**2+(tnow-to)**2/tE**2)
            err_Anow=(uo*parameters_errors[1]*np.abs((tnow-to))/tE**3*(tE*parameters_errors[0]+np.abs((tnow-to))*parameters_errors[2]))/unow                 
            
            observables_errors.append(parameters_errors[0])
            observables_errors.append(err_Ao)
            observables_errors.append(parameters_errors[2])
            observables_errors.append(err_Anow)
            
            start=len(parameters)-2*len(self.event.telescopes)-1
            for j in xrange(len(self.event.telescopes)):
                
                Ibaseline=27.4-2.5*np.log10(parameters[start]*(1+parameters[start]))
                Ipeak=27.4-2.5*np.log10(parameters[start]*(Ao+parameters[start]))
                
                observables.append(Ibaseline)
                observables.append(Ipeak)
                
                start=start+2
                
            self.observables.append([i[0],i[1],observables])        
   
            
    def errors_on_observables(self):
        
        
        for i in self.event.fits_covariance :
            
            self.error_parameters.append([i[0],i[1],np.sqrt(i[2].diagonal)]) 
            
    def cov2corr(self):
        """
        covariance matrix to correlation matrix.
        """
        self.correlations=[]
        for i in self.event.fits_covariance :
            
            A=i[3]    
            d = np.sqrt(A.diagonal())
            B = ((A.T/d).T)/d
            
            self.correlations.append([i[0],i[1],i[2],B])
     
    def student_errors(self):
        
        alpha=0.05
        ndata=len(self.event.telescopes[0].lightcurve_flux)
        npar=5
        dof=ndata-npar
        tval=student.ppf(1-alpha/2, dof)
        
        lower=[]
        upper=[]
        
        for i in xrange(len(self.event.fits_covariance[0][2].diagonal())):
            
            sigma=self.event.fits_covariance[0][2].diagonal()[i]**0.5
            lower.append(self.event.fits_results[0][2][i]-sigma*tval)
            upper.append(self.event.fits_results[0][2][i]+sigma*tval)

        self.upper=upper
        self.lower=lower
        
    def K2_C9_outputs(self):
        import matplotlib.pyplot as plt
        
        #first produce aligned lightcurve#
        
        time = []
        mag = []
        err_mag = []
        groups = []
        
        time = time + self.event.telescopes[0].lightcurve[:,0].tolist()
        mag = mag + self.event.telescopes[0].lightcurve[:,1].tolist()
        err_mag = err_mag + self.event.telescopes[0].lightcurve[:,2].tolist()
        groups = groups + [self.event.telescopes[0].name]*len(self.event.telescopes[0].lightcurve)
        
        for i in self.event.telescopes[1:] :
            
            time = time + i.lightcurve[:,0].tolist()
            Mag = i.lightcurve[:,1]
            flux = 10**((27.4-Mag)/2.5)
            err_flux = np.abs(-i.lightcurve[:, 2] * flux / (2.5) * np.log(10))            
            flux_normalised = self.event.fits[0].fit_results[self.event.fits[0].model.model_dictionnary['fs_'+self.event.telescopes[0].name]]*((
                              flux/self.event.fits[0].fit_results[self.event.fits[0].model.model_dictionnary['fs_'+i.name]]- 
                                self.event.fits[0].fit_results[self.event.fits[0].model.model_dictionnary['g_'+i.name]])+
                                self.event.fits[0].fit_results[self.event.fits[0].model.model_dictionnary['g_'+self.event.telescopes[0].name]])           
            err_flux_norm = err_flux/flux*flux_normalised
            mag_norm = 27.4-2.5*np.log10(flux_normalised)
            err_mag_norm = 2.5*err_flux_norm/(flux_normalised*np.log(10))
            
            mag = mag + mag_norm.tolist()
            err_mag = err_mag + err_mag_norm.tolist()
            groups = groups + [i.name] * len(i.lightcurve)
            
       

        lightcurve_data = np.array([time,mag,err_mag,groups]).T
        
        # produce model lightcurve
        
        time = np.arange(min(self.event.telescopes[0].lightcurve[:,0]),max(time)+100,0.01)
        ampli = microlmagnification.amplification(self.event.fits[0].model,  time,self.event.fits[0].fit_results,0.5 )[0]
        flux =  self.event.fits[0].fit_results[self.event.fits[0].model.model_dictionnary['fs_'+self.event.telescopes[0].name]]*(
                ampli+ self.event.fits[0].fit_results[self.event.fits[0].model.model_dictionnary['g_'+self.event.telescopes[0].name]])
        mag = (27.4-2.5*np.log10(flux)).tolist()
        err_mag = [0.001]*len(time)
        time = time.tolist()
        lightcurve_model =  np.array([time,mag,err_mag]).T
        

        #produce parameters
        Parameters = []
        Names = []

        Uo = self.event.fits[0].fit_results[self.event.fits[0].model.model_dictionnary['uo']]
        Ao = (Uo**2+2)/(Uo*(Uo**2+4)**0.5)
        err_Ao = (8)/(Uo**2*(Uo**2+4)**1.5)*(self.event.fits[0].fit_covariance.diagonal()**0.5)[1]
        
        Parameters.append(Ao)
        Parameters.append(err_Ao)
        
        Names.append('PYLIMA.AO')
        Names.append('PYLIMA.SIG_AO')

        
        names = ['TE','TO','UO']
        Official = ['tE','to','uo']
        
        for i in xrange(len(Official)) :

            index = self.event.fits[0].model.model_dictionnary[Official[i]]
            Parameters.append(self.event.fits[0].fit_results[index])
            Parameters.append((self.event.fits[0].fit_covariance.diagonal()**0.5)[index])
            
            
            Names.append('PYLIMA.'+names[i])
            Names.append('PYLIMA.SIG_'+names[i])
        Parameters = np.array([Names,Parameters]).T
        count=0
        for i in self.event.telescopes :
            index=np.where(lightcurve_data[:,3]==i.name)[0]
            colors = np.random.uniform(0,10)
            plt.scatter(lightcurve_data[index,0].astype(float),lightcurve_data[index,1].astype(float),c=(np.random.randint(0,float(len(self.event.telescopes))) / float(len(self.event.telescopes)), 
                        np.random.randint(0,float(len(self.event.telescopes))) / float(len(self.event.telescopes)), 
                        np.random.randint(0,float(len(self.event.telescopes))) / float(len(self.event.telescopes)), 1),label=i.name,s=25)
            count+=1
        plt.legend(scatterpoints=1)            
        plt.plot(lightcurve_model[:,0],lightcurve_model[:,1],'g')
        plt.show() 

        
        
        
        
        return Parameters,lightcurve_model,lightcurve_data