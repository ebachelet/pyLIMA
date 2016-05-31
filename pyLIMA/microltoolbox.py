# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:18:15 2016

@author: ebachelet
"""

import numpy as np


magnitude_constant = 27.4


def magnitude_to_flux(mag):
    
    flux = 10 ** ((magnitude_constant - mag) / 2.5)
    
    return flux
    
def flux_to_magnitude(flux):
    
    mag = magnitude_constant-2.5*np.log10(flux)
    
    return mag

def error_magnitude_to_error_flux(error_mag, flux):
    
    error_flux = error_mag * flux / (2.5) * np.log(10)
    
    return error_flux
