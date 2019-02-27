# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 10:48:07 2019

@author: rstreet
"""

from os import path
from sys import argv
from astropy.io import fits
import numpy as np
from datetime import datetime, timedelta

def edit_sim_image_headers(input_file):
    """Function to add header keywords to simulated image data"""
    
    params = parse_input_file(input_file)
    
    hdr_keys = ['object', 'RA', 'DEC', 'obstype', 'bias', 'dark', 'flat', 
                'filter1', 'filter2', 'filter3', 'l1median', 'exptime']

    ts = np.arange(params['tstart'],params['tend'],params['cadence'])
    
    ddate = timedelta(days=params['cadence'])
    
    for i,t in enumerate(ts):
        
        f = path.join(params['data_dir'],
                      params['root_image_name']+'_'+str(i)+'.fits')
        fout = path.join(params['data_dir'],
                      params['root_image_name']+'_'+str(i)+'_new.fits')
        
        if path.isfile(f):              
            image = fits.open(f)
            
            for key in hdr_keys:
                image[0].header.append( (key.upper(), params[key]) )
            
            d = params['date_start'] + (i-1)*ddate
            
            image[0].header.append( ('DATE-OBS', d.strftime("%Y-%m-%dT%H:%M:%S")) )
            image[0].header.append( ('UTSTART', d.strftime("%H:%M:%S")) )
            image[0].header.append( ('HJD', str(t)) )
                    
            image.writeto(fout,overwrite=True)
        
def parse_input_file(input_file):
    
    if path.isfile(input_file) == False:
        raise IOError('Cannot find configuration file '+input_file)
        exit()
        
    flines = open(input_file,'r').readlines()
    
    params = {}
    for l in flines:
        (key, value) = l.replace('\n','').split()
        params[key] = value
    
    for key in ['tstart','tend','cadence','l1median','exptime']:
        params[key] = float(params[key])
    
    params['date_start'] = datetime.strptime(params['date_start'],
                                            "%Y-%m-%dT%H:%M:%S")
    
    return params


if __name__ == '__main__':
    
    if len(argv) == 1:
        input_file = raw_input('Please enter the path to the input file: ')
    else:
        input_file = argv[1]
    
    edit_sim_image_headers(input_file)