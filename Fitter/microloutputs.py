# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:38:14 2015

@author: ebachelet
"""
from __future__ import division
import numpy as np
import datetime
from astropy.time import Time

class MLOutputs(object):
    
    
    def __init__(self, event):

        self.event = event
        self.fits_errors = []
        self.observables = [] 
        self.observables_errors = []
        self.plots= []
    
    def errors_on_fits(self):
        
        
        for i in self.event.fits_covariance :
            import pdb; pdb.set_trace()
            self.error_parameters.append([i[0],i[1],np.sqrt(i[2].diagonal)] 

    def find_observables(self):
        
        
        for i in self.event.fits_results :
            
            observables = []
            parameters = i[2]
            to = parameters[0]
            uo = parameters[1]
            tE = parameters[2]
            
            jd1,jd2=Time(datetime.datetime.utcnow())
            tnow=jd1+jd2
            Ao=(uo**2+2)/(uo*np.sqrt(uo**2+4))
            unow=np.sqrt(uo**2+(tnow-to)**2/tE**2)
            Anow=(unow**2+2)/(unow*np.sqrt(unow**2+4))       
            
            observables.append(to)
            observables.append(Ao)
            observables.append(tE)
            observables.append(Anow)
            
            start=len(parameters)-2*len(self.event.telescopes)-1
            for j in xrange(len(self.event.telescopes)):
                
                Ibaseline=27.4-2.5*np.log10(parameters[start]*(1+parameters[start]))
                Ipeak=27.4-2.5*np.log10(parameters[start]*(Ao+parameters[start]))
                
                observables.append(Ibaseline)
                observables.append(Ipeak)
                
                start=start+2
                
            self.observables.append([i[0],i[1],observables])
            
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
            import pdb; pdb.set_trace()
            self.error_parameters.append([i[0],i[1],np.sqrt(i[2].diagonal)] 
            
     def cov2corr(self,A):
        """
        covariance matrix to correlation matrix.
        """

        d = np.sqrt(A.diagonal())
        B = ((A.T/d).T)/d
        #A[ np.diag_indices(A.shape[0]) ] = np.ones( A.shape[0] )
        return B