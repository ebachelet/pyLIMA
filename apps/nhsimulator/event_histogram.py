# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:34:12 2018

@author: rstreet
"""

import sys, os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import rcParams
import numpy as np

def calc_percent(value,position):
    """Function to convert input data to percent, based on Matplotlib demo"""
    
    svalue = str(100 * value)

    if rcParams['text.usetex'] is True:
        return svalue+r'$\%$'
    else:
        return svalue+'%'

def plot_event_histogram():
    
    catalog_file = get_args()
    
    data = read_events_param_from_catalog(catalog_file,13,survey='ogle')
    
    nbins = 20.0
    hmax = 24.0
    hmin = 10.0
    
    fig = plt.figure(1)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.95,
                wspace=None, hspace=None)
    
    plt.hist(data,int(nbins),range=(hmin,hmax), normed=True)
    
    formatter = FuncFormatter(calc_percent)
    
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xlabel('Baseline I [mag]')

    plt.ylabel('Cumulative percentage frequency')

    ([xmin,xmax,ymin,ymax]) = plt.axis()
    plt.axis([25.0,10.0,ymin,ymax])
    
    y = np.linspace(ymin,ymax,10)
    x = np.zeros(len(y))
    x.fill(17.0)
    plt.plot(x,y,'k--')
    
    plt.savefig('hist_events_baseline_brightness.png')

    mlimit = 17.0
    n = len(np.where(data <= mlimit)[0])
    print str(n)+' events brigher than '+str(mlimit)+'mag detected'
    print str(round( ((float(n)/float(len(data)))*100.0), 1 ))+'% of total'
    print str(round((float(n)/9.0),0))+' events per year'
    
    plt.close(1)

def read_events_param_from_catalog(catalog_file,col_num,survey='all'):
    
    if not os.path.isfile(catalog_file):
        
        print 'ERROR: Cannot find input catalogue '+catalog_file
    
        sys.exit()
    
    f = open(catalog_file,'r')
    file_lines = f.readlines()
    f.close()
    
    data = []
    
    for line in file_lines:
        
        if line[0:1] != '#':
            
            entries = line.replace('\n','').split()
            
            selected = False
            
            if survey.lower() == 'all':
                
                selected = True
                
            elif survey.lower() == 'ogle' and 'None' not in entries[0]:
                
                selected = True
            
            elif survey.lower() == 'moa' and 'None' not in entries[1]:
                
                selected = True
            
            if selected:
                
                if col_num in [4, 6, 8, 9, 10, 11, 12, 13]:
                    
                    data.append(float(entries[col_num]))
                    
                else:
                    
                    data.append(entries[col_num])
                
    data = np.array(data)
    
    return data
    

def get_args():
        
    if len(sys.argv) > 1:
        
        catalog_file = sys.argv[1]
    
    else:

        catalog_file = raw_input('Please enter the path to the events catalog file: ')

    return catalog_file


if __name__ == '__main__':
    
    plot_event_histogram()
    