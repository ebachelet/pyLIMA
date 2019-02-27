# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:50:26 2019

@author: rstreet
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def compare_sim_measured_lc():
    """Program to compare the simulated lightcurve used to generate the 
    imaging data with the photometry measured from the difference images."""
    
    params = get_args()
    
    sim_lc = read_sim_lc_file(params['sim_lc_file'])
    diff_lc = read_measured_lc_file(params['diff_lc_file'])

    fs_t = diff_lc[:,1]+params['source_flux']
    
    (meas_mag, meas_mag_error) = flux_to_mag(fs_t,
                                            diff_lc[:,2])
    
    fig = plt.figure(1,(10,10))

    dt = 2450000.0
    
    plt.plot(sim_lc[:,1]-dt, sim_lc[:,2],'k.',
             label='Simulated lightcurve')
             
    plt.errorbar(diff_lc[:,0]-dt, meas_mag+10.0, yerr=meas_mag_error,
                 alpha=1.0,color='#2b8c85',ecolor='#2b8c85',
                 ls='none',fmt='+',elinewidth=2.0,
                 label='Measured lightcurve')
                 
    plt.xlabel('HJD - '+str(dt), fontsize=18)

    plt.ylabel('Magnitude', fontsize=18)
    
    plt.legend(loc=1, fontsize=18)
    
    plt.grid()
    
    (xmin,xmax,ymin,ymax) = plt.axis()
    plt.axis([8020.0,8080.0,ymax,ymin])

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    
    plt.savefig(os.path.join(params['output'],'measured_sim_lc.png'),
                bbox_inches='tight')

    plt.close(1)

def flux_to_mag(flux, flux_error):
    
    ZP = 17.7
    
    mag = ZP - 2.5 * np.log10(flux)
    
    mag_error = np.abs(-2.5 * flux_error / (flux * np.log(10)))
    
    return mag, mag_error
    
def read_measured_lc_file(file_path):
    """Function to read the measured lightcurve file in the format produced
    by the simple_photometry code"""
    
    if not os.path.isfile(file_path):
        
        print('ERROR: Cannot find lightcurve input file '+file_path)
        
        sys.exit()
    
    f = open(file_path,'r')
    file_lines = f.readlines()
    f.close()
    
    data = []
    
    for line in file_lines:
        
        if line[0:1] != '#':
            
            entries = line.replace('\n','').split()
            
            row = []
            
            for e in entries[1:]:
                
                row.append(float(e))
                
            data.append( row )
            
    data = np.array(data)
    
    return data
    
def read_sim_lc_file(file_path):
    """Function to read in a lightcurve file in the format output by the
    NH lensing simulation"""
    
    if not os.path.isfile(file_path):
        
        print('ERROR: Cannot find lightcurve input file '+file_path)
        
        sys.exit()
    
    f = open(file_path,'r')
    file_lines = f.readlines()
    f.close()
    
    data = []
    
    for line in file_lines:
        
        if line[0:1] != '#':
            
            entries = line.replace('\n','').split()
            
            row = []
            
            for e in entries:
                
                row.append(float(e))
                
            data.append( row )
            
    data = np.array(data)
    
    return data
    
def get_args():
    
    params = {}
    
    if len(sys.argv) == 1:
        
        print('Please enter the parameters of the simulation to plot: ')
        params['sim_lc_file'] = input('Path to the simulated lightcurve file: ')
        params['diff_lc_file'] = input('Path to the measured difference photometry file: ')
        params['source_flux'] = float(input('Measured source flux: '))
        params['blend_flux'] = float(input('Blend flux: '))
        params['output'] = input('Output directory path: ')
    
    else:
        
        params['sim_lc_file'] = sys.argv[1]
        params['diff_lc_file'] = sys.argv[2]
        params['source_flux'] = float(sys.argv[3])
        params['blend_flux'] = float(sys.argv[4])
        params['output'] = sys.argv[5]
        
    return params


if __name__ == '__main__':
    
    compare_sim_measured_lc()
    