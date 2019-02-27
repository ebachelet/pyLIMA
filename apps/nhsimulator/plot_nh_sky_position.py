# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:50:54 2018

@author: rstreet
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm, rcParams
from datetime import datetime
import os, sys
import jplhorizons_utils
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import SkyOffsetFrame, ICRS

def plot_nh_sky_position():
    """Function to plot the sky location (RA, Dec) of the New Horizons 
    spacecraft as a function of time, in comparison with Pluto's sky location
    """

    nh_positions = jplhorizons_utils.parse_JPL_Horizons_table(horizons_file_path='New_Horizons_positions.txt',
                                                              table_type='OBSERVER')
                                                              
    pluto_positions = jplhorizons_utils.parse_JPL_Horizons_table(horizons_file_path='Pluto_positions.txt',table_type='OBSERVER')

    GB_RA = 17.92
    GB_Dec = -29.0
    
    fig = plt.figure(3,(10,10))
    
    rcParams.update({'font.size': 18})
    
    plt.plot(nh_positions['RA']/15.0,nh_positions['Dec'],'b-',label='New Horizons')

    plt.scatter(nh_positions['RA'].data[::365]/15.0, nh_positions['Dec'].data[::365],color='b')
    
    add_date_labels(fig, nh_positions, 'b')
    
    plt.plot(pluto_positions['RA']/15.0,pluto_positions['Dec'],'m-', label='Pluto')

    plt.scatter(pluto_positions['RA'].data[::365]/15.0, pluto_positions['Dec'].data[::365],color='m')
    
    #add_date_labels(fig, pluto_positions)
    
    plt.plot([GB_RA],[GB_Dec],'k+',markersize=16)

    plt.text(GB_RA+0.2,GB_Dec,'Galactic Bulge')
    
    plt.xlabel('RA [hrs]')
    plt.ylabel('Dec [deg]')
    
    plt.grid()

    plt.legend(loc='upper right')
    
    plt.savefig('NH_sky_position.eps',bbox_inches='tight')

    plt.close(3)
    
def add_date_labels(fig, positions, text_color):
    """Function to plot a trajectory, with text markers indicating the position
    on Jan 1 of each year.
    """
    
    for i in range(0,len(positions['RA'].data[::365]),2):

        year = positions['Date'].data[::365][i].split('T')[0][0:4]

        if i%4 == 0:

            if year == '2006':
                doff = -2.0
            else:
                doff = -1.0

            roff = 0.0

        else:

            doff = 2.0
            roff = -0.25

        plt.text((positions['RA'].data[::365][i]/15.0)+roff, positions['Dec'].data[::365][i]+doff, 
           year,rotation=-75.0,color=text_color)
    
    
if __name__ == '__main__':
    
    plot_nh_sky_position()