# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:18:15 2016

@author: ebachelet
"""

import numpy as np

# magnitude reference
magnitude_constant = 27.4


def magnitude_to_flux(mag):
    """ Transform the injected magnitude to the the corresponding flux."""

    flux = 10 ** ((magnitude_constant - mag) / 2.5)
    
    return flux
    
def flux_to_magnitude(flux):
    """ Transform the injected flux to the the corresponding magnitude."""

    mag = magnitude_constant-2.5*np.log10(flux)
    
    return mag

def error_magnitude_to_error_flux(error_mag, flux):
    """ Transform the injected magnitude error to the the corresponding error in flux."""

    error_flux = -error_mag * flux* np.log(10) / 2.5 
    
    return error_flux

def error_flux_to_error_magnitude(error_flux, flux):
    """ Transform the injected flux error to the the corresponding error in magnitude."""

    error_mag = -2.5 * error_flux  / (flux * np.log(10))
    
    return error_mag
