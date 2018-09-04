# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:24:45 2017

@author: rstreet
"""

import os
import sys
from astropy.table import Table
import datetime
from astropy.coordinates import SkyCoord
from astropy.coordinates import SkyOffsetFrame, ICRS
from astropy.time import Time
import numpy as np
from scipy import interpolate

def parse_JPL_Horizons_table(horizons_file_path = None, table_type = None):
    """Function to parse a file containing tabular output from JPL Horizons
    produced in the VECTOR format.
    
    :param str horizons_file_path: File path to the text-format file produced
                                    by JPL Horizons in download/save mode
    :param str table_type: Horizons table format; supports VECTOR and OBSERVER
    
    Returns:
    :param astropy Table data: Table object of the tabular contents
    
    None is returned if an issue arises reading the file.
    """
    
    if horizons_file_path == None:
        
        (horizons_file_path,table_type) = get_params()
    
    if os.path.isfile(horizons_file_path) == False:
        
        return None
    
    
    if table_type == 'VECTOR':
        
        file_lines = open(horizons_file_path,'r').readlines()[0].split('\r')
        
        horizons_data = parse_vector_table_data(file_lines)
    
    else:
        
        file_lines = open(horizons_file_path,'r').readlines()
        
        horizons_data = parse_observer_table_data(file_lines)
        
    return horizons_data

def parse_file_header(file_lines):
    """Function to identify the column names of the table
    
    :param list of str file_lines: Contents of the JPL Horizons file
    
    Returns:
    
    :param list of str col_names:   Column header names
    :param int idata:               Line index in file of the start of the
                                    data table.
    """
    
    col_names = []

    soe = 0
    i = 0
    while i < len(file_lines):

        if '$$SOE' in file_lines[i]:
            
            soe = i
            i = len(file_lines) + 1
        
        i += 1
    
    soh = 0
    i = soe - 2
    while i > 0:

        if '*******' in file_lines[i]:
            
            soh = i
            i = -1
        
        i -= 1
        
    for line in file_lines[soh:soe]:
        
        if '****' not in line:
            
            entries = line.replace('\n','').split()
            
            for item in entries:
                
                col_names.append(item)
                
    col_names.append('DateTime')
    
    return col_names, (soe + 1)

def parse_vector_table_data(file_lines):
    """Function to parse the tabulated contents of the JPL Horizons file
    in VECTOR format
    
    :param list of str file_lines:  Contents of the JPL Horizons file
    
    Returns:
    
    :param Table horizons_data: Data table
    """
    
    def parse_entry(text, entry_type='float'):
        
        search_str = ['X', 'Y', 'Z', 'V', 'VX', 'VY', 'VZ', 'LT', 'RG', 'RR',
                      'A.D', 'TDB' ]
        
        for c in search_str:
            text = text.replace(c,'')
        
        text = text.replace(' ','').replace('\t','')
        
        if entry_type == 'float':
            
            value = float(text)
        
        elif entry_type == 'timestamp':
            
            date = text[1:12]
            time = text[12:]
            value = date+'T'+time
            
        return value
        
    (col_names, idata) = parse_file_header(file_lines)
    
    data = []
    
    i = idata
    while i <= len(file_lines):
        
        line = ' '.join(file_lines[i:i+4])
        
        if '*****' in line:
            
            i = len(file_lines) + 1
        
        else:
        
            entries = line.replace('\n','').split('=')
            jd = parse_entry(entries[0])
            ts = parse_entry(entries[1], entry_type='timestamp')
            x = parse_entry(entries[2])
            y = parse_entry(entries[3])
            z = parse_entry(entries[4])
            vx = parse_entry(entries[5])
            vy = parse_entry(entries[6])
            vz = parse_entry(entries[7])
            lt = parse_entry(entries[8])
            rg = parse_entry(entries[9])
            rr = parse_entry(entries[10])
            
            data.append( (jd, x, y, z, vx, vy, vz, lt, rg, rr, ts) )
            
            i += 4
    
    col_dtypes = ('float', 'float', 'float','float', 'float', 'float', 
                  'float', 'float', 'float','float', 'string')
    
    horizons_data = Table(rows=data, names=tuple(col_names), dtype=col_dtypes)
    
    return horizons_data

def parse_observer_table_data(file_lines):
    """Function to parse the tabulated contents of the JPL Horizons file
    in OBSERVER format
    
    :param list of str file_lines:  Contents of the JPL Horizons file
    
    Returns:
    
    :param Table horizons_data: Data table
    """
    
    def parse_timestamp(date,time):
        
        value = date+'T'+time
        
        dt = datetime.datetime.strptime(value,'%Y-%b-%dT%H:%M')

        t = Time(dt.strftime('%Y-%m-%d %H:%M'), format='iso', scale='utc')
        t.format = 'jd'
        
        return value, t.value
    
    col_names = ['Date', 'RA', 'Dec', 'Delta', 'Deldot', 'JD']
    data = []
    
    iend = -1
        
    for i,line in enumerate(file_lines):
                
        if '$$SOE' in line:
            
            istart = i + 1
            
        elif '$$EOE' in line:
            
            iend = i - 1
        
    if iend == -1:
        
        iend = len(file_lines) - 1
    
    
    for i in range(istart,iend,1):
        
        line = ' '.join(file_lines[i:i+4])
        
        if '*****' in line:
            
            i = len(file_lines) + 1
        
        else:
            
            if len(line.replace('\n','')) > 0:
                entries = line.replace('\L','').split()
                (ts,jd) = parse_timestamp(entries[0],entries[1])
                ra_hr = entries[2]
                ra_min = entries[3]
                ra_sec = entries[4]
                dec_deg = entries[5]
                dec_min = entries[6]
                dec_sec = entries[7]
                delta = float(entries[8])
                deldot = float(entries[9])
                ra = float(ra_hr) + (float(ra_min)/60.0) + (float(ra_sec)/3600.0)
                ra = ra*15.0
                sign = 1.0
                if dec_deg[0:1] == '-':
                    sign = -1.0
                    dec_deg = dec_deg[1:]
                dec = float(dec_deg) + (float(dec_min)/60.0) + (float(dec_sec)/3600.0)
                dec = dec*sign
                
                data.append( (ts,ra,dec,delta,deldot,jd) )
                
            i += 1
    
    col_dtypes = ('S16', 'float', 'float', 'float', 'float', 'float')

    horizons_data = Table(rows=data, names=tuple(col_names), dtype=col_dtypes)
    
    return horizons_data

def get_params():
    """Function to acquire necessary commandline arguments"""
    
    if len(sys.argv) == 1:
        
        horizons_file_path = raw_input('Please enter the path to the JPL Horizons table file: ')
        
        table_type = raw_input('Please enter the type of the table [VECTOR,OBSERVER]: ')
        
    else:
        
        horizons_file_path = sys.argv[1]
        
        table_type = sys.argv[2]
    
    return horizons_file_path, table_type

def calc_norm_spacecraft_positions(horizons_data,t0):
    """Function to calculate the position of the spacecraft as a function of
    time, normalized to the position at t0"""
    
    dates = horizons_data['JD'].tolist()
    ras = horizons_data['RA'].tolist()
    decs = horizons_data['Dec'].tolist()
    deltas = horizons_data['Delta'].tolist()

    interpolated_delta = interpolate.interp1d(np.array(dates), 
                                                 np.array(deltas))
    
    delta_t0 = interpolated_delta(t0)
    
    deltas = deltas - delta_t0
    
    positions = []
    
    for i in range(0,len(dates),1):
        
        positions.append( [dates[i], ras[1], decs[i], deltas[i]] )
                    
    return positions
    
def extract_spacecraft_positions(horizons_data,t0):
    """Function to calculate the position of the spacecraft as a function of
    time, normalized to the position at t0"""
    
    dates = horizons_data['JD'].tolist()
    ras = horizons_data['RA'].tolist()
    decs = horizons_data['Dec'].tolist()
    deltas = horizons_data['Delta'].tolist()

    positions = []
    
    for i in range(0,len(dates),1):
        
        positions.append( [dates[i], ras[1], decs[i], deltas[i]] )
                    
    return positions
    
if __name__ == '__main__':
    
    horizons_data = parse_JPL_Horizons_table()
    tE = 2457125.0
    index = np.argmin(abs(horizons_data['JD']-tE))
    print horizons_data['JD'][index]
    print horizons_data