# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 14:58:25 2019

@author: rstreet
"""

from sys import argv
from os import path
import glob
from astropy.io import fits
import numpy as np

def aperture_photometry(input_file):
    
    params = parse_input_file(input_file)
    
    image_list = glob.glob(path.join(params['data_dir'],
                                     params['root_image_name']+'*.fits'))
    image_list.sort()
    
    output = open(path.join(params['data_dir'],'diff_photometry.txt'),'w')
    output.write('# Image   HJD    dflux[e-]   dflux_err[e-]   sky[e-]    sky_err[e-]  dmag   dmag_err\n')
    
    for image_file in image_list:
        
        hdr = fits.getheader(image_file)
        
        (flux, sig_flux, sky_bkgd, sig_sky, mag, mag_err) = photometer_star(image_file,params)
        
        output.write(path.basename(image_file)+' '+str(hdr['HJD'])+' '+\
                            str(flux)+' '+str(sig_flux)+' '+\
                                    str(sky_bkgd)+' '+str(sig_sky)+' '+\
                                    str(mag)+' '+str(mag_err)+'\n')
                                    
def photometer_star(image_file,params):
    
    zp = 17.7
    base_flux = 10**((zp-params['baseline_mag'])/-2.5)
    
    try:
        
        image = fits.open(image_file)
        
        data = image[0].data
        idx = np.where(data <= 0.0)
        data[idx] = 0.0
        
        xmax = image[0].header['NAXIS2']
        ymax = image[0].header['NAXIS1']
                
        dx = np.arange(0,xmax,1) - params['star_x']
        dy = np.arange(0,ymax,1) - params['star_y']
        
        (dxx,dyy) = np.meshgrid(dx,dy)
        
        sep = np.sqrt(dxx*dxx + dyy*dyy)
        
        idx1 = np.where(sep <= params['sky_radius_out'])
        out_pix = zip( idx1[0].tolist(), idx1[1].tolist() )
        
        idx2 = np.where(sep >= params['sky_radius_in'])
        in_pix = zip( idx2[0].tolist(), idx2[1].tolist() )
        
        idx = set(in_pix).intersection(set(out_pix))
        idx = zip(*idx)
        idx = [np.array(idx[0]), np.array(idx[1])]
        
        naper = np.pi * params['aperture_radius']**2
        
        # Sky background in annulus in e-
        sky_bkgd = (data[idx].mean() * params['gain'])
        sig_sky = (data[idx].std() * params['gain'])
        
        # Star flux in aperture in e-
        idx = np.where(sep <= params['aperture_radius'])
        star_flux = data[idx].sum() * params['gain']

        if star_flux < 0.0:
            star_flux = 0.0
            sig_star_flux = 0.0
        else:
            sig_star_flux = np.sqrt(star_flux)

        if star_flux > 0.0:
            flux = star_flux - sky_bkgd*naper
            
            if flux > 0.0:
                flux = flux/params['exptime'] + base_flux
                sig_flux = np.sqrt( sig_star_flux**2 + \
                                sig_sky**2 + \
                                    params['ron']**2 )
                                    
                (mag, mag_err) = flux_to_mag(flux, sig_flux)
                
            else:

                flux = 0.0
                sig_flux = 0.0
                mag = -99.999
                mag_err = -9.999
                
        else:
            
            flux = 0.0
            sig_flux = 0.0
            mag = -99.999
            mag_err = -9.999

        sky_bkgd = sky_bkgd/params['exptime']
        sig_sky = sig_sky/params['exptime']
        
        return flux, sig_flux, sky_bkgd, sig_sky, mag, mag_err
        
    except IOError:
        
        raise IOError('Cannot find image '+image_file)
        exit()
    
def parse_input_file(input_file):
    
    if path.isfile(input_file) == False:
        raise IOError('Cannot find configuration file '+input_file)
        exit()
        
    flines = open(input_file,'r').readlines()
    
    params = {}
    for l in flines:
        (key, value) = l.replace('\n','').split()
        params[key] = value
    
    for key in ['star_x', 'star_y', 'aperture_radius', 
                'sky_radius_in', 'sky_radius_out',
                'gain', 'ron', 'baseline_mag', 'exptime']:
        
        params[key] = float(params[key])
    
    return params

def flux_to_mag(flux, flux_err):
    
    zp = 17.7
    
    mag = zp - 2.5*np.log10(flux)
    
    logfactor = 2.5 * (1.0 / flux) * np.log10(np.exp(1.0))

    mag_err = flux_err * logfactor
    
    return mag, mag_err
    
if __name__ == '__main__':
    
    if len(argv) == 1:
        input_file = raw_input('Please enter the path to the input file: ')
    else:
        input_file = argv[1]
        
    aperture_photometry(input_file)