# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:09:52 2019

@author: rstreet
"""

from astropy import units as u
from astropy import constants
import numpy as np

ML = 1.0 * constants.M_sun
DL = 4e3 * constants.pc
DS = 8e3 * constants.pc
vT_min = 75e3 * (u.m/u.s)
vT_max = 120e3 * (u.m/u.s)

def calc_drel(DL,DS):
    return 1.0 / ( (1.0/DL) - (1.0/DS) )

def calc_proj_re(ML,Drel):
    return np.sqrt( (4.0 * constants.G * ML * Drel) / (constants.c*constants.c) )

def calc_RE(ML,DL,DS):
    
    DLS = DS-DL
    D = (DL*DLS)/DS
    
    return np.sqrt((4.0*constants.G*ML*D)/(constants.c*constants.c))

Drel = calc_drel(DL,DS)
proj_re = calc_proj_re(ML,Drel)
print(proj_re/constants.au)


ML = 5.0 * constants.M_sun
DL = 6e3 * constants.pc
DS = 8e3 * constants.pc

Drel = calc_drel(DL,DS)
proj_re = calc_proj_re(ML,Drel)
print(proj_re/constants.au)
