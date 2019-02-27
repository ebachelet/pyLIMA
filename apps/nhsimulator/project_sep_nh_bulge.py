# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:25:11 2019

@author: rstreet
"""

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants
import numpy as np

ML = 1.0 * constants.M_sun
DL = 4e3 * constants.pc
DS = 8e3 * constants.pc
DLS = DS - DL
vT_min = 75e3 * (u.m/u.s)
vT_max = 120e3 * (u.m/u.s)

def calc_RE(ML,DL,DS):
    
    DLS = DS-DL
    D = (DL*DLS)/DS
    
    return np.sqrt((4.0*constants.G*ML*D)/(constants.c*constants.c))
    
# Location of NH, RA, Dec on 2018-Feb-20 00:00
nh = SkyCoord('19 12 32.51 -20 27 48.1', frame='icrs', unit=(u.hourangle, u.deg))
drange = 41.4220703643889 * constants.au

# location of Bulge centre:
bulge = SkyCoord('17:57:34 -29:13:15', frame='icrs', unit=(u.hourangle, u.deg))

ang_sep = nh.separation(bulge)

print('Angular on-sky separation between New Horizons and the Bulge: '+str(ang_sep.to(u.rad)))

aproj = drange * np.sin(ang_sep.to(u.rad))

print('Projected distance between New Horizons and Earth, relative to the Bulge: '+str(aproj/constants.au)+'AU')


xL = (DLS * aproj)/DS

print('Distance traveled in lens plane, xL: '+str(xL)+' = '+str(xL/constants.au)+'AU')

RE = calc_RE(ML,DL,DS)

print('Lens Einstein radius: '+str(RE)+' = '+str(RE/constants.au)+'AU')

xi = 2.0*np.arctan(RE/xL)

prob_lens = xi.value / (2.0*np.pi)

print('Angle within which repeat lensing could happen: '+str(xi))

print('Probability of repeat lensing: '+str(prob_lens*100.0)+'%')

tgap_min = (xL/vT_min).value / (60*60*24*365.25)
tgap_max = (xL/vT_max).value / (60*60*24*365.25)

print('Assuming a transverse velocity of '+str(vT_min)+', tgap = '+str(tgap_min)+' years')
print('Assuming a transverse velocity of '+str(vT_max)+', tgap = '+str(tgap_max)+' years')
