# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:38:14 2015

@author: ebachelet
"""
from __future__ import division
import numpy as np

class MLOutputs(object):
    
    
    def errors_on_observable(self):
        
        error_to=np.sqrt(np.diag(covariance[0,0]))
        error_Ao=8/(fits_results[1]**2*(fits_results[1]**2+4)**1.5)*np.sqrt(np.diag(covariance[1,1]))
        error_tE=np.sqrt(np.diag(covariance[2,2]))
        error_rho=np.sqrt(np.diag(covariance[3,3]))
        
        for i in self.telescopes:
            
            error_Is=np.sqrt(np.diag(covariance[4,4]))/fits_results[4]
            error_Ib=np.sqrt(fits_results[5]**2*np.diag(covariance[4,4])+fits_results[4]**2*np.diag(covariance[5,5])+2*fits_results[4]*fits_results[5]*np.diag(covariance[4,5]))
            error_Ibaseline=np.sqrt((1+fits_results[5])**2*np.diag(covariance[4,4])+fits_results[4]**2*np.diag(covariance[5,5])+2*fits_results[4]*(1+fits_results[5])*np.diag(covariance[4,5]))
            
     def cov2corr(self,A):
        """
        covariance matrix to correlation matrix.
        """

        d = np.sqrt(A.diagonal())
        B = ((A.T/d).T)/d
        #A[ np.diag_indices(A.shape[0]) ] = np.ones( A.shape[0] )
        return B