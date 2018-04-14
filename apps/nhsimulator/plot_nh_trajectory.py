# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:24:30 2018

@author: rstreet
"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm, rcParams
from datetime import datetime
import os, sys
lib_path = os.path.abspath(os.path.join('/Users/rstreet/software/pyLIMA/'))
sys.path.append(lib_path)
import NH_data_simulator
import jplhorizons_utils
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import SkyOffsetFrame, ICRS


def plot_nh_trajectory():
    """Function to plot the trajectory of the New Horizon's spacecraft 
    through the Solar System since launch, relative to the orbits of 
    some planets based on data from JPL Horizons.  
    """

    (spacecraft_data, planet_data) = load_JPL_Horizons_tables()
    
    gal_bulge = get_location_gal_bulge()

    calc_intercept_angle_from_bulge(gal_bulge)

    plot_trajectory_through_solar_system(spacecraft_data, 
                                         planet_data,
                                         gal_bulge)

def calc_intercept_angle_from_bulge(gal_bulge):
    """Function to calculate the angular separation of Pluto from the Galactic 
    Bulge at the time when it was intercepted by New Horizons. 
    Pluto's location at this time was derived from a table from JPL Horizons.
    """
    
    # Coordinates from Heliocentric table
    pluto_intercept_location = SkyCoord(ra='19:00:17.68', dec='-20:47:33.3', 
                                        unit=(u.hourangle, u.deg), frame='icrs')

    dra, ddec = pluto_intercept_location.spherical_offsets_to(gal_bulge)
    print('Angular separation of Pluto from the Bulge at the time of New Horizons intercept: ')
    print(str(dra.value)+', '+str(ddec.value)+'deg')
    
def get_location_gal_bulge():
    """Function to return the location of the Galactic Bulge as a SkyCoord
    object"""
    
    gal_bulge = SkyCoord(ra=17.92*u.hourangle, dec=-29.0*u.degree, 
                     distance=(8500.0*u.pc).to(u.au), frame='icrs')
    print('Cartesian coordinates of the Galactic Bulge: ('+\
        str(gal_bulge.cartesian.x.value)+','+\
        str(gal_bulge.cartesian.y.value)+','+\
        str(gal_bulge.cartesian.z.value)+') AU')
    
    return gal_bulge    

def load_JPL_Horizons_tables():
    """Function to load pre-generated data tables from the JPL Horizons system 
    for the trajectories of the New Horizons and Spitzer spacecraft, and orbital 
    data for Earth and Pluto.
    """
    
    nh_trajectory = jplhorizons_utils.parse_JPL_Horizons_table(horizons_file_path='New_Horizons_trajectory.txt',table_type='VECTOR')
    spitzer_trajectory = jplhorizons_utils.parse_JPL_Horizons_table(horizons_file_path='Spitzer_trajectory.txt',table_type='VECTOR')
    earth_orbit = jplhorizons_utils.parse_JPL_Horizons_table(horizons_file_path='Earth_orbit_data.txt',table_type='VECTOR')
    pluto_orbit = jplhorizons_utils.parse_JPL_Horizons_table(horizons_file_path='Pluto_orbit_data.txt',table_type='VECTOR')

    spacecraft_data = { 'NH': nh_trajectory, 
                        'Spitzer': spitzer_trajectory }
                        
    planet_data = { 'Earth': earth_orbit,
                    'Pluto': pluto_orbit }
    
    return spacecraft_data, planet_data
    
def plot_trajectory_through_solar_system(spacecraft_data, 
                                         planet_data,
                                         gal_bulge):
    """Function to generate a 3D plot of the Solar System, with the
    orbits of selected planets shown relative to the trajectory of the
    New Horizons spacecraft and the orbit of the Spitzer spacecraft. """
    
    nh_trajectory = spacecraft_data['NH']
    spitzer_trajectory = spacecraft_data['Spitzer']
    earth_orbit = planet_data['Earth']
    pluto_orbit = planet_data['Pluto']
    
    rcParams.update({'font.size': 14})
    
    fig = plt.figure(2,(10,10),frameon=False)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(nh_trajectory['X'].data, nh_trajectory['Y'].data, nh_trajectory['Z'].data,label='New Horizons trajectory',color='b')
    ax.scatter(nh_trajectory['X'].data[::365], nh_trajectory['Y'].data[::365], nh_trajectory['Z'].data[::365])
    
    for i in range(0,len(nh_trajectory['X'].data[::365]),1):
        ax.text(nh_trajectory['X'].data[::365][i], nh_trajectory['Y'].data[::365][i], nh_trajectory['Z'].data[::365][i],
           nh_trajectory['DateTime'].data[::365][i].split('T')[0][0:4],color='b')
    
    ax.plot(spitzer_trajectory['X'].data, spitzer_trajectory['Y'].data, spitzer_trajectory['Z'].data,label='Spitzer trajectory')
        
    ax.plot(earth_orbit['X'].data, earth_orbit['Y'].data, earth_orbit['Z'].data, 'k-',label='Earth orbit')
    
    ax.plot(pluto_orbit['X'].data, pluto_orbit['Y'].data, pluto_orbit['Z'].data,'m-')
    ax.scatter(pluto_orbit['X'].data[::365], pluto_orbit['Y'].data[::365], pluto_orbit['Z'].data[::365],color='m')
    ax.text(pluto_orbit['X'].data[0], pluto_orbit['Y'].data[0]-0.0, pluto_orbit['Z'].data[0]-5.0,'Pluto orbit',color='m')
    for i in range(0,len(pluto_orbit['X'].data[::365]),1):
        ax.text(pluto_orbit['X'].data[::365][i], pluto_orbit['Y'].data[::365][i], pluto_orbit['Z'].data[::365][i],
           pluto_orbit['DateTime'].data[::365][i].split('T')[0][0:4],color='m')
    
    #ax.plot(gcx, gcy, gcz, 'r-')
    ax.quiver(0,0,0,
              gal_bulge.cartesian.x.value, 
              gal_bulge.cartesian.y.value,
              gal_bulge.cartesian.z.value,
              length=5e-9,arrow_length_ratio=0.15,color='r',
             label='Galactic Bulge direction')
    #ax.text(gcx[0],gcy[0]+2.0,gcz[0],'Galactic Bulge\ndirection',color='r')
    
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    
    ax.legend(frameon=False)
    
    #plt.show()
    plt.savefig('NH_orbital_trajectory.eps',bbox_inches='tight')
    plt.close(2)
    

if __name__ == '__main__':
    
    plot_nh_trajectory()
    