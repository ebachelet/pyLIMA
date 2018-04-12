# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:43:55 2018

@author: rstreet
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm, rcParams
from datetime import datetime
from astropy import visualization

def analyze_simulation_results():
    """Function to analyze the results of simulating microlensing
    events as observed by New Horizons"""
    
    (sim_data,log_dir) = read_simulation_output()

    fitted_data = read_fitted_model_parameters(log_dir, 'parallax')
    
    plot_dbic_matrix(sim_data,log_dir)
    
    plot_dchi_matrix(sim_data,log_dir)

    plot_pi_matrix(sim_data,log_dir)

    plot_tE_matrix(fitted_data,log_dir)

    plot_sn_matrix(sim_data,log_dir)
    
def plot_sn_matrix(sim_data,log_dir):
    """Function to plot the signal to noise as a colour-coded grid in (tE,mag) 
    parameter space."""
    
    (mag_range,tE_range,max_res) = load_col_sim_data_as_2d(sim_data,13)
    
    max_res = np.flip(max_res,0)

    fig = plt.figure(1,(8,3))
    plt.subplots_adjust(top=0.90,left=0.1,right=0.98)
                
    norm = visualization.ImageNormalize(max_res, 
                                        interval=visualization.MinMaxInterval(),
                                        stretch=visualization.SqrtStretch())
    
    plt.imshow(max_res, origin='lower', cmap=plt.cm.viridis, norm=norm)
    
    plt.colorbar(orientation='horizontal',pad=0.35)
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    
    (xticks,xlabels,yticks,ylabels) = calc_axis_ranges(tE_range, mag_range, 
                                                        xmin,xmax,ymin,ymax)
    plt.xticks(xticks,xlabels, rotation=45.0)
    plt.yticks(yticks,ylabels)
    
    plt.xlabel('tE [days]')
    plt.ylabel('Magnitude')
    
    plt.title('Signal to noise')
    plt.savefig(os.path.join(log_dir,'sn_matrix.png'))
    
    plt.close(1)

def plot_tE_matrix(fitted_data,log_dir):
    """Function to plot the tE squared as a colour-coded grid in (tE,mag) 
    parameter space."""
    
    (mag_range,tE_range,fitted_tE) = load_col_sim_data_as_2d(fitted_data,2)
    
    for row in range(0,len(fitted_tE),1):
        fitted_tE[row,:] = fitted_tE[row,:] - tE_range
        
    fitted_tE = np.flip(fitted_tE,0)
    
    fig = plt.figure(1)
    
    norm = visualization.ImageNormalize(fitted_tE, 
                                        interval=visualization.MinMaxInterval(),
                                        stretch=visualization.SqrtStretch())
    
    plt.imshow(fitted_tE, origin='lower', cmap=plt.cm.viridis, norm=norm)
    
    plt.colorbar(orientation='horizontal')
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    
    (xticks,xlabels,yticks,ylabels) = calc_axis_ranges(tE_range, mag_range, 
                                                        xmin,xmax,ymin,ymax)
    
    plt.xticks(xticks,xlabels)
    plt.yticks(yticks,ylabels)
    
    plt.xlabel('tE [days]')
    plt.ylabel('Magnitude')
    
    plt.title('Fitted model $t_{E}$')
    plt.savefig(os.path.join(log_dir,'tE_matrix.png'))
    
    plt.close(1)

def plot_pi_matrix(sim_data,log_dir):
    """Function to plot the parallax as a colour-coded grid in (tE,mag) 
    parameter space."""
    
    (mag_range,tE_range,piEN) = load_colerr_sim_data_as_2d(sim_data,4,5)
    (mag_range,tE_range,piEE) = load_colerr_sim_data_as_2d(sim_data,6,7)
    
    piEN = np.log10(np.abs(piEN))
    piEE = np.log10(np.abs(piEE))
    
    
    piEN = np.flip(piEN,0)
    piEE = np.flip(piEE,0)
    
    fig = plt.figure(1)
    
    a=fig.add_subplot(2,1,1)
    
    plt.imshow(piEN, origin='lower', cmap=plt.cm.viridis)
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    
    (xticks,xlabels,yticks,ylabels) = calc_axis_ranges(tE_range, mag_range, 
                                                        xmin,xmax,ymin,ymax)
    
    plt.xticks(xticks,xlabels)
    plt.yticks(yticks,ylabels)
    
    plt.title('$log_{10}$ of fractional error in $\pi_{E,N}$')
    plt.xlabel('tE [days]')
    plt.ylabel('Magnitude')
    
    plt.colorbar(orientation ='horizontal')
    
    a=fig.add_subplot(2,1,2)

    plt.imshow(piEE, origin='lower', cmap=plt.cm.viridis)
    
    plt.colorbar(orientation='horizontal')
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    
    (xticks,xlabels,yticks,ylabels) = calc_axis_ranges(tE_range, mag_range, 
                                                        xmin,xmax,ymin,ymax)
    
    plt.xticks(xticks,xlabels)
    plt.yticks(yticks,ylabels)
    
    plt.title('$log_{10}$ of fractional error in $\pi_{E,E}$')
    plt.xlabel('tE [days]')
    plt.ylabel('Magnitude')
    
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                wspace=0.2, hspace=0.3)
                
    plt.savefig(os.path.join(log_dir,'pi_matrix.png'))
    
    plt.close(1)


def plot_dchi_matrix(sim_data,log_dir):
    """Function to plot the delta chi squared as a colour-coded grid in (tE,mag) 
    parameter space."""
    
    (mag_range,tE_range,dchi2) = load_col_sim_data_as_2d(sim_data,10)
    
    dchi2 = np.flip(dchi2,0)

    fig = plt.figure(1)
    
    norm = visualization.ImageNormalize(dchi2, 
                                        interval=visualization.MinMaxInterval(),
                                        stretch=visualization.SqrtStretch())
    
    plt.imshow(dchi2, origin='lower', cmap=plt.cm.viridis, norm=norm)
    
    plt.colorbar(orientation='horizontal')
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    
    (xticks,xlabels,yticks,ylabels) = calc_axis_ranges(tE_range, mag_range, 
                                                        xmin,xmax,ymin,ymax)
    
    plt.xticks(xticks,xlabels)
    plt.yticks(yticks,ylabels)
    
    plt.xlabel('tE [days]')
    plt.ylabel('Magnitude')
    
    plt.title('$\Delta \chi^{2}$')
    plt.savefig(os.path.join(log_dir,'chi2_matrix.png'))
    
    plt.close(1)

def plot_dbic_matrix(sim_data,log_dir):
    """Function to plot the delta BIC as a colour-coded grid in (tE,mag) 
    parameter space."""
    
    (mag_range,tE_range,dbic) = load_col_sim_data_as_2d(sim_data,11)
    
    #dbic = np.log10(dbic)
    dbic = np.flip(dbic,0)

    fig = plt.figure(1)
    
    plt.imshow(dbic, origin='lower', cmap=plt.cm.viridis)
    
    plt.colorbar(orientation='horizontal')
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    
    (xticks,xlabels,yticks,ylabels) = calc_axis_ranges(tE_range, mag_range, 
                                                        xmin,xmax,ymin,ymax)
    
    plt.xticks(xticks,xlabels)
    plt.yticks(yticks,ylabels)
    
    plt.xlabel('tE [days]')
    plt.ylabel('Magnitude')
    
    #plt.title('$log_{10}(\Delta BIC)$')
    plt.title('\Delta BIC')
    plt.savefig(os.path.join(log_dir,'dbic_matrix.png'))
    
    plt.close(1)

def load_col_sim_data_as_2d(sim_data,col_num):
    """Function to load a column of data from the simulated data table, 
    and reshape it into a 2D array according to the tE, mag ranges used"""
    
    mag_range = np.unique(sim_data[:,0])
    
    tE_range = np.unique(sim_data[:,1])
    
    data = []
    
    for m in range(0,len(mag_range),1):
        
        row = []
        
        for t in range(0,len(tE_range),1):
            
            row.append(sim_data[(t+(m*len(tE_range))),col_num])
        
        data.append(row)
    
    data = np.array(data)
   
    return mag_range, tE_range, data

def load_colerr_sim_data_as_2d(sim_data,col_num,err_col_num):
    """Function to load a column of data from the simulated data table, 
    and reshape it into a 2D array according to the tE, mag ranges used"""
    
    mag_range = np.unique(sim_data[:,0])
    
    tE_range = np.unique(sim_data[:,1])
    
    data = []
    
    for m in range(0,len(mag_range),1):
        
        row = []
        
        mag = mag_range[m]
        
        for t in range(0,len(tE_range),1):
            
            te = tE_range[t]
            
            value = sim_data[(t+(m*len(tE_range))),col_num]
            error = sim_data[(t+(m*len(tE_range))),err_col_num]
            
            if error == 0.0:
                error = np.nan
            
            row.append(error/value)
            
        data.append(row)
    
    data = np.array(data)
   
    return mag_range, tE_range, data
    

def calc_axis_ranges(tE_range, mag_range, xmin,xmax,ymin,ymax):
    """Function to calculate the ranges in the physical parameters corresponding
    to the plotted image, and return appropriate axis labels"""
    
    xint = (xmax-xmin)/len(tE_range)
    yint = (ymax-ymin)/len(mag_range)
    tEint = (tE_range.max() - tE_range.min())/len(tE_range)
    magint = (mag_range.max() - mag_range.min())/len(mag_range)
    
    xticks = np.arange(xmin,xmax,(2.0*xint)).tolist()
    yticks = np.arange(ymin,ymax,(2.0*yint)).tolist()
    
    xlabels = []
    for x in range(0,len(xticks),1):
        
        xlabels.append(str(round( (tE_range.min()+x*2.0*tEint), 0) ))

    ylabels = []
    for y in range(0,len(yticks),1):
        
        ylabels.append(str(round( (mag_range.max()-y*2.0*magint), 0) ))
    
    return xticks,xlabels,yticks,ylabels

def read_simulation_output():
    """Function to read the output file produced by the simulator."""
    
    if len(sys.argv) == 1:
        
        log_dir = raw_input('Please enter the path to the output directory: ')
    
    else:
        
        log_dir = sys.argv[1]
        
    file_path = os.path.join(log_dir,'lensing_statistics.txt')
    
    if not os.path.isfile(file_path):
        print('ERROR: Cannot find simulations output file '+file_path)
        sys.exit()
        
    file_lines = open(file_path,'r').readlines()
    
    data = []
    
    for line in file_lines:
        
        if line[0:1] != '#':
            
            entries = line.replace('\n','').replace('|','').split()
            
            line_data = []
            
            for i in range(0,len(entries),1):
                
                if '+/-' not in entries[i]:
                    
                    if 'None' in entries[i]:
                        
                        line_data.append( 0.0 )
                        
                    else:
                        
                        line_data.append( float(entries[i]) )
                
            data.append(line_data)
            
    sim_data = np.array(data)
    
    return sim_data, log_dir
    
def read_fitted_model_parameters(log_dir, model_type):
    """Function to read the parameters of the fitted models"
    
    Inputs:
        :param path log_dir: Path to input directory
        :param string model_type: One of {parallax, no_parallax}
    """
    
    file_path = os.path.join(log_dir,'fitted_model_parameters.txt')
    
    if not os.path.isfile(file_path):
        print('ERROR: Cannot find fitted parameters output file '+file_path)
        sys.exit()
        
    file_lines = open(file_path,'r').readlines()
    
    data = []
    
    for line in file_lines:
        
        if line[0:1] != '#' and model_type == line.split()[2]:
            
            entries = line.replace('\n','').replace('|','').split()
            
            line_data = []
            
            for i in range(0,len(entries),1):
                
                if '+/-' not in entries[i] and i != 2 and i != 12 and i != 14:
                    
                    if 'None' not in entries[i] and 'parallax' not in entries[i]:
                        
                        line_data.append( float(entries[i]) )
                        
                    else:
                        
                        if 'None' not in entries[i]:
                            line_data.append(entries[i])
                        else:
                            line_data.append(0.0)
                            
            data.append(line_data)
            
    fitted_data = np.array(data)
    
    return fitted_data

if __name__ == '__main__':
    
    analyze_simulation_results()
    
    