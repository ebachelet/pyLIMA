# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:19:27 2016

@author: ebachelet
"""
import numpy as np
import mock

import event


def test_telescopes_names():
    
    current_event = event.Event()    
    
    telescope1 = mock.MagicMock()
    telescope2 = mock.MagicMock()
        
    telescope1.name = 'telescope1'
    telescope2.name = 'telescope2'
    
    current_event.telescopes.append(telescope1)
    current_event.telescopes.append(telescope2)
    

    current_event.telescopes_names()
    
def test_find_survey():
    
    current_event = event.Event()    
    
    telescope1 = mock.MagicMock()
    telescope2 = mock.MagicMock()
        
    telescope1.name = 'telescope1'
    telescope2.name = 'telescope2'
    
    current_event.telescopes.append(telescope1)
    current_event.telescopes.append(telescope2)

    
    current_event.find_survey('telescope2')

    assert current_event.telescopes[0].name == 'telescope2'
    
def test_lightcurves_in_flux():
    
    current_event = event.Event()    
    
    telescope1 = mock.MagicMock()
    telescope2 = mock.MagicMock()
        
    telescope1.name = 'telescope1'
    telescope2.name = 'telescope2'
    
    telescope1.lightcurve = np.array([])
    telescope2.lightcurve = np.array([])
    
    current_event.telescopes.append(telescope1)
    current_event.telescopes.append(telescope2)
    
    current_event.lightcurves_in_flux()
    
    
