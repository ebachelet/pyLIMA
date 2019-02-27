# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:30:11 2018

@author: rstreet
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm, rcParams
from matplotlib.ticker import MaxNLocator

from datetime import datetime
import copy

import os, sys
lib_path = os.path.abspath(os.path.join('/Users/rstreet/software/pyLIMA/'))
sys.path.append(lib_path)

from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA import microlsimulator
from pyLIMA import microlmodels
from pyLIMA import microltoolbox
from pyLIMA import microlstats
from pyLIMA import microlcaustics
import NH_data_simulator
import jplhorizons_utils

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import SkyOffsetFrame, ICRS

from astropy.units import cds
cds.enable()
from astropy import constants

import logging

import NH_lensing

def bh_lensing():
    """Function to simulate lensing by a black hole as seen from Earth and
    New Horizons"""
    
    config_file = NH_lensing.get_args()
    
    (default_params,source_mag_range,tE_range) = NH_lensing.parse_config_file(config_file)
    
    log = NH_lensing.start_log( default_params['output_path'], 'sim_info' )
    
    NH_lensing.record_config(default_params,source_mag_range, tE_range,log)
    
    simulate_lensing(default_params,source_mag_range,tE_range,log)
    
    NH_lensing.stop_log(log)
    
def simulate_lensing(default_params,source_mag_range,tE_range,log):
    """Function to simulate lensing event including annual parallax
    as seen from Earth"""

    lc_keys = ['JD_start','JD_end']
    event_keys = ['name', 'ra', 'dec', 
                  'to','uo','rho','piEN', 'piEE', 'logs', 'logq', 'alpha',
                  'model_code']
    horizons_table = None
    
    if 'horizons_file' in default_params.keys():
        
        horizons_table = jplhorizons_utils.parse_JPL_Horizons_table(horizons_file_path=default_params['horizons_file'], 
                                                                   table_type='OBSERVER')
        
        log.info('Added spacecraft positions to pyLIMA tel object')
    
        log.info('Range of JDs in spacecraft positions table: '+
                    str(horizons_table['JD'].min())+' - '+
                    str(horizons_table['JD'].max()))
                    
        log.info('Event t0 requested: '+str(default_params['to']))

        log.info('Range of JDs requested for simulated lightcurve: '+
                    str(default_params['JD_start'])+' - '+
                    str(default_params['JD_end']))
                        
        spacecraft_positions = jplhorizons_utils.calc_spacecraft_positions_observer_table(horizons_table,default_params['to'])
        
    for j in range(0,len(source_mag_range),1):
        
        mag = source_mag_range[j]
        
        for i in range(0,len(tE_range),1):
            
            tE = tE_range[i]
            
            log.info('Simulating grid point mag='+str(mag)+'mag and tE='+str(tE)+'days')
            
            (lc_params, event_params_no_parallax, event_params_parallax) = NH_lensing.make_param_dicts(default_params,mag,tE,log)
            
            baseline_lc = NH_data_simulator.generate_LORRI_lightcurve(lc_params,log)
            
            # Using the parallax parameters here so that pyLIMA calculates
            # the position of the Earth
            (lc_earth,sim_e_earth) = NH_data_simulator.add_event_to_lightcurve_geocentric('Earth','Earth',baseline_lc,
                                                                       event_params_parallax,
                                                                       lc_params,log,default_params['output_path'],
                                                                       output_lc=True)
                                                                       
            (lc_space,sim_e_space) = NH_data_simulator.add_event_to_lightcurve_geocentric('NH_LORRI','Space',baseline_lc,
                                                                    event_params_parallax,
                                                                    lc_params,log,default_params['output_path'],
                                                                    spacecraft_positions=spacecraft_positions,
                                                                    output_lc=True)
                                                                    
            log.info('Completed model generation for grid point mag='+str(mag)+'mag and tE='+str(tE)+'days')
            
            fit_earth = NH_lensing.FitParams()
            fit_space = NH_lensing.FitParams()
            e_earth = None
            e_space = None
            
            file_path = os.path.join(default_params['output_path'],
                                        'sim_lightcurves_'+str(round(mag,1))+
                                        '_'+str(round(tE,0))+'.eps')

            plot_lightcurves(lc_earth,lc_space,file_path)
            
            plot_file_path = os.path.join(default_params['output_path'],
                                        'sim_lens_plane_'+str(round(mag,1))+
                                        '_'+str(round(tE,0))+'.eps')
            txt_file_path = os.path.join(default_params['output_path'],
                                        'sim_lens_plane_'+str(round(mag,1))+
                                        '_'+str(round(tE,0))+'.txt')
                                        
            NH_lensing.plot_lens_plane_trajectories(default_params['model_type'],
                                                    sim_e_earth,sim_e_space,
                                                    'Earth','Space',
                                                    plot_file_path,txt_file_path)
                                 
def plot_lightcurves(lc_earth,lc_space,file_path):
    """Function to plot lightcurves and model fits for both with- and without
    parallax models"""
    
    plot_models = False
    
    dt = float(int(lc_earth[0,0]))
    
    ts_no_parallax = lc_earth[:,0] - dt
    ts_parallax = lc_space[:,0] - dt
    
    fig = plt.figure(6,(10,5))

    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.95,
                wspace=None, hspace=None)
                
    plt.plot(ts_no_parallax,lc_earth[:,1],color='#8c6931',linestyle='solid',
                label='Earth')
                
    plt.errorbar(ts_no_parallax,lc_earth[:,1],
                 yerr=lc_earth[:,2],alpha=1.0,color='#8c6931',fmt='none')
    
    plt.plot(ts_parallax,lc_space[:,1],color='#2b8c85',linestyle='solid',
                label='New Horizons')
                
    plt.errorbar(ts_parallax,lc_space[:,1],
                 yerr=lc_space[:,2],alpha=1.0,color='#2b8c85',fmt='none')
    
                
    plt.xlabel('HJD - '+str(dt), fontsize=18)

    plt.ylabel('Magnitude', fontsize=18)
    
    plt.legend(loc=2, fontsize=16)
    
    plt.grid()
    
    (xmin,xmax,ymin,ymax) = plt.axis()
    plt.axis([xmin,xmax,ymax,ymin])

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    
    plt.savefig(file_path, bbox_inches='tight')

    plt.close(6)

if __name__ == '__main__':
    
    bh_lensing()