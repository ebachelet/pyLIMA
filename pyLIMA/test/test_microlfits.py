# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:46:17 2015

@author: ebachelet
"""
import mock
import numpy as np

from pyLIMA import microlfits
from pyLIMA import microltoolbox



def test_mlfits_without_data():
    
    event = mock.MagicMock()
    
    
    model = mock.MagicMock(event)
    
    fit = microlfits.MLFits(event)
    
    fit.mlfit(model,'titi')



def test_mlfits_LM():
    
    event = mock.MagicMock()
    event.telescopes= []
    telescope = mock.MagicMock()
    telescope.lightcurve_magnitude = np.array([[0.0,36.010025,-39420698921.705284],
                                             [1.0,56,-39420698921.705284],
                                             [3.0,46.010025,-39420698921.705284]]) 
                                             
    flux = microltoolbox.magnitude_to_flux(telescope.lightcurve_magnitude[:,1])
    err_flux = microltoolbox.error_magnitude_to_error_flux(telescope.lightcurve_magnitude[:,2],flux) 
    
    telescope.lightcurve_flux = np.array([telescope.lightcurve_magnitude[:,0],flux,err_flux]).T
   
                                   
    event.telescopes.append(telescope)
    
    model = mock.MagicMock(event)
    model.paczynski_model = 'PSPL'
    model.parallax_model = ['None',0]
    model.xallarap_model = ['None',0]
    model.orbital_motion_model = ['None',0]    
    model.source_spots_model = 'None'
    #model.magnification.return_value = np.array([1.0]*len(telescope.lightcurve_magnitude)),np.array([1.0]*len(telescope.lightcurve_magnitude))
    
    fit = microlfits.MLFits(event)
    
    fit.mlfit(model,'LM')

   