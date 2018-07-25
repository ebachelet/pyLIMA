# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:44:40 2018

@author: rstreet
"""

import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import os, sys
lib_path = os.path.abspath(os.path.join('../'))
sys.path.append(lib_path)
from scipy import interpolate
import jplhorizons_utils
import vizier_tools
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
from astropy import table
from astropy.io import fits

class LORRIImage():
    """Class describing the essential parameters of the LORRI imager 
    onboard New Horizons.
    All time stamps are in seconds
    Exposure time must be in integer seconds
    gain and readnoise are in e-/DN and e- respectively, 
    aperture is in m, 
    pixel scape is in arcsec/pixel
    drift rate is in arcsec/sec
    field of view radius is in arcsec
    half psf is the half-width of the PSF in pixels
    Parameters are taken from Cheng, A.F. et al. 2007, 
    https://arxiv.org/pdf/0709.4278.pdf
    """
    
    def __init__(self,field_center=None):
        
        self.aperture = 0.208 
        self.read_time = 1.0
        self.zp = 20.5
        self.gain = 22.0
        self.read_noise = 10.0
        self.max_exp_time = 30          # Must be an integer
        self.pixel_scale = 1.02
        self.half_psf = 100.0
        self.psf_radius = 10.0
        #self.xdrift = 5.16
        self.xdrift = 0.0
        self.ydrift = 0.0
        self.naxis1 = 1024
        self.naxis2 = 1024
        self.fov = (self.naxis1 * self.pixel_scale)
        self.data = np.zeros([1,1])
        self.wcs = None
        self.field_center = None
        
        if field_center != None:
            self.field_center = field_center
    
    def generate_wcs(self):
        
        if self.field_center != None:
            
            pixscale = (self.pixel_scale/3600.0)

            w = wcs.WCS(naxis=2)

            w.wcs.crpix = [int(self.naxis1/2.0),int(self.naxis2/2.0)]
            w.wcs.cdelt = np.array([pixscale, pixscale])
            w.wcs.crval = [self.field_center.ra.deg, self.field_center.dec.deg]
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            
            self.wcs = w
        
        else:
            
            print('ERROR: Cannot generate a WCS without a field center pointing SkyCoord')
        
    def simulate_image(self):
        
        image = np.random.rand(self.naxis1,self.naxis2)
        self.data = image*self.read_noise
        
        self.generate_wcs()
    
    def add_stars(self,stars):
        
        image_limits = (1, self.naxis2, 1, self.naxis1)
        
        for j in range(0,len(stars),1):

            if np.isnan(stars[j,2]) == False:
                
                peak_flux = (10**( (stars[j,2] - self.zp)/-2.5 ))*self.gain
                
                xstart = stars[j,0]
                ystart = stars[j,1]
            
                if xstart > 0.0 and xstart < self.naxis2 and \
                   ystart > 0.0 and ystart < self.naxis1:
                    
                    star = StellarPSF({'xstart':xstart, 'ystart':ystart, 
                                      'intensity':peak_flux, 
                                      'gamma':4.0, 
                                      'alpha':3.3,
                                      'radius': self.psf_radius})
                                      
                    star.calc_trail(self.xdrift,self.ydrift,
                                    self.max_exp_time,image_limits)
                    
                    if len(star.xcenter) > 0 and len(star.ycenter) > 0:
                        
                        model = star.psf_model()
                        
                        idx = np.argwhere(np.isnan(model))
                        
                        if len(idx) == 0:
                            
                            star.output_psf_model('psf.fits')
                                                        
                            try:
                                self.data[star.Y_data,star.X_data] += model
                                
                                idx = np.argwhere(np.isnan(self.data[star.Y_data,star.X_data]))
                                
                                if len(idx) > 0:
                                    
                                    print 'NaN instance: ', model, self.data[star.Y_data,star.X_data]
                                    
                            except IndexError:
                                
                                print 'ERROR adding star at: ',star.X_data, star.Y_data, \
                                                                xmin, xmax, ymin, ymax, \
                                                                x, y

    def output_image(self,file_path):
        
        hdr = fits.Header()
        
        hdr['CRPIX1'] = self.wcs.wcs.crpix[0]
        hdr['CRPIX2'] = self.wcs.wcs.crpix[1]
        hdr['CDELT1'] = self.wcs.wcs.cdelt[0]
        hdr['CDELT2'] = self.wcs.wcs.cdelt[1]
        hdr['CRVAL1'] = self.wcs.wcs.crval[0]
        hdr['CRVAL2'] = self.wcs.wcs.crval[1]
        hdr['CTYPE1'] = self.wcs.wcs.ctype[0]
        hdr['CTYPE2'] = self.wcs.wcs.ctype[1]
        
        hdu = fits.PrimaryHDU(self.data, header=hdr)
        
        hdu.writeto(file_path,overwrite=True)

class StellarPSF():
    """Class descrbing the Point Spread Function of a stellar, point source"""
    
    def __init__(self,kwargs):
        
        self.intensity = None
        self.xstart = None
        self.xend = None
        self.ystart = None
        self.yend = None
        self.xcenter = None
        self.ycenter = None
        self.radius = None
        self.psf_bounds = []
        self.gamma = None
        self.alpha = None
        self.model = np.zeros(1)
        
        for key, value in kwargs.items():
            setattr(self,key,value)
    
    def calc_trail(self,xdrift,ydrift,exp_time,image_limits):
        """Function to calculate the range of X,Y pixel positions intersected
        by a star over the course of an exposure where it trails across the
        image.
        
        Note: it was originally considered to use this trail to calculate
        the PSF model directly, rather than accumulating flux in a loop.  
        However, this risks underestimating the total flux, since the star
        does not necessarily drift precisely one pixel per second.         
        """
        
        # Calculate the center of the star as a function of time
        # Performed as a loop rather than linspace etc to ensure that there
        # are always entries in the arrays even if the drift is zero
        self.xcenter = []
        self.ycenter = []
        
        for t in range(0,exp_time,1):
            self.xcenter.append( (self.xstart + t*xdrift) )
            self.ycenter.append( (self.ystart + t*ydrift) )
            
        self.xend = self.xcenter[-1]
        self.yend = self.ycenter[-1]
        
        self.xcenter = np.array(self.xcenter)
        self.ycenter = np.array(self.ycenter)
        
        # Curtail the PSF trail if it exceeds the image boundaries
        idx = set(np.where( self.xcenter > 1.0 )[0])
        jdx = set(np.where(self.xcenter < image_limits[1])[0])
        idx = list(idx.intersection(jdx))
        self.xcenter = self.xcenter[idx]

        idx = set(np.where( self.ycenter > 1.0 )[0])
        jdx = set(np.where(self.ycenter < image_limits[3])[0])
        idx = list(idx.intersection(jdx))
        self.ycenter = self.ycenter[idx]
        
        # Calculate the boundaries of the image section containing this star's
        # PSF, taking into account the PSF radius at the start and end of the
        # PSF trail. 
        if len(self.xcenter) > 0 and len(self.ycenter) > 0:
            
            xmin = max( 0, round((self.xcenter.min() - self.radius),0) )
            xmax = min( round((self.xcenter.max() + self.radius),0), image_limits[1] )
            ymin = max( 0, round((self.ycenter.min() - self.radius),0) )
            ymax = min( round((self.ycenter.max() + self.radius),0), image_limits[3] )
            
            self.psf_bounds = [xmin, xmax, ymin, ymax]
            
            nx = int(xmax - xmin)
            ny = int(ymax - ymin)
            
            # For each pixel in the PSF image section, calculate the separation
            # of the pixel from the nearest position of the star in X, Y at 
            # any point in the exposure
            self.deltax = np.zeros( [nx,ny,exp_time] )
            self.deltay = np.zeros( [nx,ny,exp_time] )
            
            (self.X_data, self.Y_data) = np.indices( (nx,ny) )
            
            self.X_data += int(xmin)
            self.Y_data += int(ymin)
            
            for t in range(0,len(self.xcenter),1):
                x = self.xcenter[t]
                y = self.ycenter[t]
                
                self.deltax[:,:,t] = self.X_data - x
                self.deltay[:,:,t] = self.Y_data - y
        
        else:
            
            self.psf_bounds = [0, 0, 0, 0]
            self.deltax = np.zeros([1,1,1])
            self.deltay = np.zeros([1,1,1])
            
    def psf_model(self):
        
        self.model = self.intensity * (1 + ((self.deltax)**2 +\
                                       (self.deltay)**2) \
                                       / self.gamma**2)**(-self.alpha)
        
        self.model = self.model.sum(axis=2)
        
        return self.model

    def output_psf_model(self, file_path):
        
        hdu = fits.PrimaryHDU(self.model)
        
        hdu.writeto(file_path,overwrite=True)
        
def simulate_image_dataset():
    """Function to simulate the set of images that LORRI would obtain 
    for a given sky pointing."""
    
    params = get_args()
    
    field_center = SkyCoord(params['field_ra']+' '+params['field_dec'],unit=(u.hourangle, u.deg))
    
    image = LORRIImage(field_center=field_center)
    
    vphas_catalog = vizier_tools.search_vizier_for_sources(params['field_ra'], 
                                                     params['field_dec'], 
                                                     (image.fov/60.0), 
                                                     'VPHAS+', 
                                                     row_limit=-1)
    
    image.simulate_image()
    
    stars = calc_star_image_positions(params, vphas_catalog, image)
    
    image.add_stars(stars)
    
    image.output_image('test_image.fits')
    
    
def calc_star_image_positions(params, catalog, image):
    """Function to calculate the x,y pixel positions of stars in the 
    catalog"""
    
    field_center = SkyCoord(params['field_ra'],params['field_dec'],unit=(u.hourangle, u.deg))
    
    image_catalog = []
        
    stars_world = np.zeros([len(catalog),2])
    stars_world[:,0] = catalog['_RAJ2000'].tolist()
    stars_world[:,1] = catalog['_DEJ2000'].tolist()
    
    stars_pixels = image.wcs.wcs_world2pix(stars_world, 1)
    
    xmin = np.where(stars_pixels[:,0] > 0.0)[0]
    xmax = np.where(stars_pixels[:,0] < image.naxis2)[0]
    idx = set(xmin).intersection(xmax)
    ymin = np.where(stars_pixels[:,1] > 0.0)[0]
    ymax = np.where(stars_pixels[:,1] < image.naxis1)[0]
    idy = set(ymin).intersection(ymax)
    idx = list(idx.intersection(idy))
    
    stars = np.zeros([len(stars_pixels[idx,:]),3])
    stars[:,0] = stars_pixels[idx,0]
    stars[:,1] = stars_pixels[idx,1]
    stars[:,2] = catalog['rmag'][idx]

    return stars
    
def get_args():
    """Function to harvest the necessary simulation parameters"""
    
    params = {}
    
    if len(argv) < 3:
        
        params['field_ra'] = raw_input('Please enter the field center RA (J2000.0, sexigesimal): ')
        params['field_dec'] = raw_input('Please enter the field center Dec (J2000.0, sexigesimal): ')
        
    else:

        params['field_ra'] = argv[1]
        params['field_dec'] = argv[2]
    
    return params

def plot_image(image):
    """Function to plot a simulated LORRI image"""
    
    fig = plt.figure(1,(10,10))
    
    plt.imshow(image1)
    
    plt.show()
    
    plt.close(1)


if __name__ == '__main__':
    
    simulate_image_dataset()