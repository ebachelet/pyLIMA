# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:55:33 2019

@author: rstreet
"""

from sys import argv
from os import path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import optimize

def measure_blending():

    params = get_params()
    
    (difflc, modellc) = load_data(params)
    
    lc = match_lc_entries(difflc,modellc)
    
    fit = fit_blending(lc)
    
    fig = plt.figure(1,(10,10))

    plt.rcParams.update({'font.size': 16})
    
    plt.plot(lc[:,4], lc[:,1], 'r.')
    
    ydata = measured_flux(fit[0], lc[:,4])
    
    label = 'fs = '+str(fit[0][0])+'\n fb = '+str(fit[0][1])
    
    plt.plot(lc[:,4], ydata, 'k-', label=label)
    
    plt.legend()
    
    plt.xlabel('Magnification')
    plt.ylabel('Difference flux')
    
    plt.yticks(rotation=30)
    plt.ticklabel_format(style='scientific', scilimits=(0,3))
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    plt.axis([1.2,40.0,0,0.2e7])
    
    plt.savefig(path.join(params['data_dir'],'blend_analysis.png'), 
                bbox_inches='tight')
    #plt.show()
    
def get_params():
    
    params = {}
    
    if len(argv) == 1:
        params['diff_file'] = input('Please enter the path to the difference photometry file: ')
        params['lc_file'] = input('Please enter the path to the lightcurve file: ')
        params['data_dir'] = input('Please enter the path for output files: ')
    else:
        params['diff_file'] = argv[1]
        params['lc_file'] = argv[2]
        params['data_dir'] = argv[3]

    return params

def load_data(params):
    
    modellc = np.loadtxt(params['lc_file'])
    
    flines = open(params['diff_file'],'r').readlines()
    
    data = []
    
    for l in flines:
        
        if '#' not in l:
            entries = l.replace('\n','').split()
            
            row = [ float(entries[1]), float(entries[2]), float(entries[3]) ]
            
            data.append(row)
    
    difflc = np.array(data)
    
    return difflc, modellc
    
def match_lc_entries(difflc,modellc):
    
    lc = np.zeros((len(difflc),5))
    
    for i in range(0,len(difflc),1):
        
        ts = difflc[i,0]
        
        idx = np.where(modellc[:,1] == ts)[0]
        
        lc[i,0] = ts
        lc[i,1] = difflc[i,1]
        lc[i,2] = difflc[i,2]
        lc[i,3] = modellc[idx,2]
        lc[i,4] = modellc[idx,4]
    
    return lc


def fit_blending(lc):

    init_pars = [ 1.0, 0.0 ]
    
    fit = optimize.leastsq(magnification_residuals, init_pars, args=(
        lc[:,4], lc[:,1]), full_output=1)

    fs = fit[0][0]
    fb = fit[0][1]
    print('Extracted source flux = '+str(fs))
    print('Extracted blend flux = '+str(fb))
    
    return fit
    
def magnification_residuals(coeffs, x, y):
    
    return y - measured_flux(coeffs, x)

def measured_flux(coeffs, x):
    
    return coeffs[0] * x + coeffs[1]
        
    
if __name__ == '__main__':
    
    measure_blending()