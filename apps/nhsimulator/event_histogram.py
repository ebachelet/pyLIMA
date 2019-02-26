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
from astropy.coordinates import SkyCoord
from astropy import units as u

def calc_percent(value,position):
    """Function to convert input data to percent, based on Matplotlib demo"""
    
    svalue = str(100 * value)

    if rcParams['text.usetex'] is True:
        return svalue+r'$\%$'
    else:
        return svalue+'%'

def plot_event_histogram(source='combined_catalog'):
    
    (catalog_file,combined_catalog) = get_args()
    
    if combined_catalog:
        (data,event_names) = read_events_param_from_catalog(catalog_file,13,survey='ogle')
    else:
        (data,event_names) = read_events_from_lenses_par(catalog_file)
    
    nbins = 20.0
    hmax = 24.0
    hmin = 10.0
    temin = 50.0
    temax = 650.0
    
    year_list = []
    for year in data[:,3]:
        if year not in year_list:
            year_list.append(year)
    
    for year in year_list:
        
        kdx = np.where(data[:,3] == year)[0]
        
        year_data = data[kdx,:]
        year_names = event_names[kdx]
        
        idx1 = np.where(year_data[:,1] >= temin)[0]
        idx2 = np.where(year_data[:,1] <= temax)[0]
        idx = list(set(idx1).intersection(set(idx2)))
        
        fig = plt.figure(1)
        
        rcParams.update({'font.size': 14})
        
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95,
                    wspace=None, hspace=None)
        
        plt.hist(year_data[idx,0],int(nbins),range=(hmin,hmax), normed=True)
        
        ([xmin,xmax,ymin,ymax]) = plt.axis()
        
        yticks = np.arange(ymin,ymax,0.05)
        ylab = []
        for y in yticks:
            ylab.append( str(round(y*100.0,1))+'%' )
        
        plt.yticks(yticks,ylab)
        
        plt.xlabel('Baseline I [mag]')
    
        plt.ylabel('Percentage frequency')
    
        ([xmin,xmax,ymin,ymax]) = plt.axis()
        plt.axis([25.0,10.0,ymin,ymax])
        
        mlimit = 16.5
        y = np.linspace(ymin,ymax,10)
        x = np.zeros(len(y))
        x.fill(mlimit)
        plt.plot(x,y,'k--')
        
        plt.savefig('hist_events_baseline_brightness_'+str(year)+'.eps')
    
        n = len(np.where(year_data[idx,0] <= mlimit)[0])
        
        print(str(len(idx))+' events (out of '+str(len(year_data))+') have tE >= '+str(temin)+' and <= '+str(temax))
        print(str(n)+' events brigher than '+str(mlimit)+'mag detected')
        print(str(round( ((float(n)/float(len(year_data)))*100.0), 1 ))+'% of total')
        print(str(round(float(n),0))+' events through the duration of the input catalog')
        
        plt.close(1)
        
        f = open('event_dump_'+str(year)+'.dat','w')
        dups = []
        for i in idx:
            f.write(str(year_data[i,0])+' '+str(year_data[i,1])+' '+str(year_data[i,2])+' '+year_names[i]+'\n')
        f.close()
        
def read_events_param_from_catalog(catalog_file,col_num,survey='all'):
    
    if not os.path.isfile(catalog_file):
        
        print('ERROR: Cannot find input catalogue '+catalog_file)
    
        sys.exit()
    
    f = open(catalog_file,'r')
    file_lines = f.readlines()
    f.close()
    
    data = []
    event_names = []
    event_positions = []
    
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
                
                try:
                    year = int(entries[0].split('-')[1])
                    data.append([float(entries[col_num]), float(entries[9]), year])
                    event_names.append(entries[0])
                    
                except ValueError:
                    pass
                
    data = np.array(data)
    event_names = np.array(event_names)
    
    return data, event_names
    
def read_events_from_lenses_par(catalog_file):
    
    if not os.path.isfile(catalog_file):
        
        print('ERROR: Cannot find input catalogue '+catalog_file)
    
        sys.exit()
    
    f = open(catalog_file,'r')
    file_lines = f.readlines()
    f.close()
    
    data = []
    event_names = []
    event_coords = []
    
    for line in file_lines:
        
        if line[0:1] != '#':
            
            try:
                entries = line.replace('\n','').split()
                
                year = int(entries[0].split('-')[0])
                
                ra = entries[3]
                dec = entries[4]
                
                data.append([float(entries[12]), float(entries[7]), float(entries[9]), year])    # I_0, tE, Amax, year
                event_names.append(entries[0])
                event_coords.append(ra+' '+dec)
                
            except ValueError:
                pass
            
    data = np.array(data)
    event_names = np.array(event_names)
    
    data1 = []
    event_names1 = []
    event_coords1 = []
    
    for j in range(0,len(data),1):
        
        if len(event_coords1) > 1:
            other_star_coords = event_coords1.copy()
            
            catalog = SkyCoord(other_star_coords,frame='icrs', unit=(u.hourangle, u.deg))
            
            c = SkyCoord(event_coords[j],frame='icrs', unit=(u.hourangle, u.deg))
            
            (match,min_separation,d3d) = c.match_to_catalog_3d(catalog)
            
            if min_separation.value > 0.0003:
                data1.append(data[j,:])
                event_names1.append(event_names[j])
                event_coords1.append(event_coords[j])
                
                print('Added to catalog star '+str(j))
            else:
                print('Identified duplicate: ',event_names[j],event_coords[j], \
                                event_names1[match], event_coords1[match], min_separation)
        
        else:
            data1.append(data[j,:])
            event_names1.append(event_names[j])
            event_coords1.append(event_coords[j])
                
    print('Original number of entries: '+str(len(data)))
    print('After de-duplication: '+str(len(data1)))
    
    data = np.array(data1)
    event_names = np.array(event_names1)
    
    return data, event_names
    
    
def get_args():
        
    if len(sys.argv) > 1:
        
        catalog_file = sys.argv[1]
        combined_catalog = sys.argv[2]
    
    else:

        catalog_file = input('Please enter the path to the events catalog file: ')
        combined_catalog = input('Is this a multi-year combined catalog?  Y or n [default=no]: ')
    
    if 'Y' in combined_catalog.upper():
        combined_catalog = True
    else:
        combined_catalog = False
        
    return catalog_file, combined_catalog


if __name__ == '__main__':
    
    plot_event_histogram()