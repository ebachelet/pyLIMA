# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 10:55:45 2018

@author: rstreet
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def compare_lightcurves():
    """Function to compare simulated lightcurves and models"""
    
    (dir_path, mag_base, tE) =  get_args()

    flc1 = os.path.join(dir_path,'sim_lightcurve_'+str(round(mag_base,1))+
                        '_'+str(round(tE,1))+'_no_parallax.txt')
    flc2 = os.path.join(dir_path,'sim_lightcurve_'+str(round(mag_base,1))+
                        '_'+str(round(tE,1))+'_parallax.txt')
    mlc1 = os.path.join(dir_path,'sim_lightcurve_'+str(round(mag_base,1))+
                        '_'+str(round(tE,1))+'_no_parallax_model.txt')
    mlc2 = os.path.join(dir_path,'sim_lightcurve_'+str(round(mag_base,1))+
                        '_'+str(round(tE,1))+'_parallax_model.txt')
    foutput = os.path.join(dir_path,'compare_lightcurves_'+str(round(mag_base,1))+
                        '_'+str(round(tE,1))+'.png')
                        
    lc_no_parallax = read_lc_file(flc1)
    lc_parallax = read_lc_file(flc2)
    model_no_parallax = read_lc_file(mlc1)
    model_parallax = read_lc_file(mlc2)

    plot_fitted_lightcurves(lc_no_parallax,lc_parallax,
                            model_no_parallax,model_parallax,
                            foutput)
                            
def plot_fitted_lightcurves(lc_no_parallax,lc_parallax,
                            model_no_parallax,model_parallax,
                            file_path):
    """Function to plot lightcurves and model fits for both with- and without
    parallax models"""
    
    dt = float(int(lc_no_parallax[0,0]))
    
    ts_no_parallax = lc_no_parallax[:,0] - dt
    ts_parallax = lc_parallax[:,0] - dt
    
    fig = plt.figure(6,(10,10))

    plt.errorbar(ts_no_parallax,lc_no_parallax[:,1],
                 yerr=lc_no_parallax[:,2],alpha=0.4,color='#8c6931')
                 
    plt.errorbar(ts_parallax,lc_parallax[:,1],
                 yerr=lc_parallax[:,2],alpha=0.4,color='#2b8c85')
    
    plt.plot(ts_no_parallax,model_no_parallax[:,1],linestyle='dashed',
                 color='#4c1377',label='No parallax model')
     
    plt.plot(ts_parallax,model_parallax[:,1],linestyle='solid',
                 color='black',label='Parallax model')
    
    plt.xlabel('HJD - '+str(dt), fontsize=18)

    plt.ylabel('Magnitude', fontsize=18)
    
    plt.legend(loc=1, fontsize=18)
    
    plt.grid()
    
    (xmin,xmax,ymin,ymax) = plt.axis()
    plt.axis([xmin,xmax,ymax,ymin])

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    
    plt.savefig(file_path)

    plt.close(6)


def read_lc_file(file_path):
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
    
    if len(sys.argv) == 1:
        
        print('Please enter the parameters of the simulation to plot: ')
        dir_path = raw_input('Path to the output directory: ')
        mag_base = float(raw_input('Baseline magnitude: '))
        tE = float(raw_input('Einstein crossing time: '))
    
    else:
        
        dir_path = sys.argv[1]
        mag_base = float(sys.argv[2])
        tE = float(sys.argv[3])
        
    return dir_path, mag_base, tE


if __name__ == '__main__':
    
    compare_lightcurves()