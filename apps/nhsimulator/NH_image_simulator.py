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

ZP = 17.7

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
    and
    Zemcov, M. et al. 2017, arXiv:1704.02989
    """
    
    def __init__(self,field_center=None):
        
        self.aperture = 0.208 
        self.read_time = 1.0
        self.zp = ZP                 # Limiting mag in 30s exposure
        self.gain = 22.0
        self.read_noise = 10.0
        self.max_exp_time = 30          # Must be an integer
        self.pixel_scale = 1.02
        self.half_psf = 20.0
        self.psf_radius = 40.0          # pixels 
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
        
        print('-> Generated empty image')
        
    def add_stars(self,stars,params):
        
        image_limits = (1, self.naxis2, 1, self.naxis1)
        
        for j in range(0,len(stars),1):
            
            if (j%10000) == 0:
                print(' -> Working on star '+str(j)+' out of '+str(len(stars)))
                
            if np.isnan(stars[j,2]) == False:
                
                peak_flux = (10**( (stars[j,2] - self.zp)/-2.5 ))*self.gain
                
                xstart = stars[j,0]
                ystart = stars[j,1]
            
                if xstart > 0.0 and xstart < self.naxis2 and \
                   ystart > 0.0 and ystart < self.naxis1:
                    
                    star = StellarPSF({'xstart':xstart, 'ystart':ystart, 
                                      'intensity':peak_flux, 
                                      'gamma':4.765, 
                                      'alpha':1.933,
                                      'radius': self.psf_radius})
                                      
                    star.calc_trail(self.xdrift,self.ydrift,
                                    self.max_exp_time,image_limits)
                    
                    if len(star.xcenter) > 0 and len(star.ycenter) > 0:
                        
                        model = star.psf_model()
                        
                        idx = np.argwhere(np.isnan(model))
                        
                        if len(idx) == 0:
                            
                            #star.output_psf_model(os.path.join(params['output'],
                            #                                'psf.fits'))
                            
                            try:
                                self.data[star.Y_data,star.X_data] += model
                                
                                idx = np.argwhere(np.isnan(self.data[star.Y_data,star.X_data]))
                                
                                if len(idx) > 0:
                                    
                                    print 'NaN instance: ', model, self.data[star.Y_data,star.X_data]
                                    
                            except IndexError:
                                
                                print 'ERROR adding star at: ',star.X_data, star.Y_data, \
                                                                xmin, xmax, ymin, ymax, \
                                                                x, y
        
        print('-> Added stars to the image')
        
    def output_image(self,file_path, ts=None):
        
        hdr = fits.Header()
        
        hdr['CRPIX1'] = self.wcs.wcs.crpix[0]
        hdr['CRPIX2'] = self.wcs.wcs.crpix[1]
        hdr['CDELT1'] = self.wcs.wcs.cdelt[0]
        hdr['CDELT2'] = self.wcs.wcs.cdelt[1]
        hdr['CRVAL1'] = self.wcs.wcs.crval[0]
        hdr['CRVAL2'] = self.wcs.wcs.crval[1]
        hdr['CTYPE1'] = self.wcs.wcs.ctype[0]
        hdr['CTYPE2'] = self.wcs.wcs.ctype[1]
        
        if ts != None:
            hdr['HJD'] = ts
            
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
    
    field_center = SkyCoord(params['field_ra']+' '+params['field_dec'],
                            unit=(u.hourangle, u.deg))
    
    image = LORRIImage(field_center=field_center)
    
    if 'none' in params['catalog'].lower():
        vphas_catalog = vizier_tools.search_vizier_for_sources(params['field_ra'], 
                                                     params['field_dec'], 
                                                     (image.fov/60.0), 
                                                     'VPHAS+', 
                                                     row_limit=-1)
                                                     
    else:
        query_data = np.loadtxt(params['catalog'])
        
        data = [ table.Column(name='_RAJ2000', data=query_data[:,0]),
                 table.Column(name='_DEJ2000', data=query_data[:,1]),
                 table.Column(name='rmag', data=query_data[:,4]) ]
                 
        vphas_catalog = table.Table(data=data)
    
    jstar = find_target_star_in_catalog(params,vphas_catalog)
    
    lightcurve = simulate_lensing_lightcurve(params,vphas_catalog,jstar)
    
    for i,t in enumerate(lightcurve[:,0]):
        
        image.simulate_image()
        
        (stars,jprime) = calc_stars_on_image(params, vphas_catalog, image, 
                                            jstar, lightcurve[i,1])
    
        image.add_stars(stars,params)
        
        image.output_image(os.path.join(params['output'],
                                      'sim_image_'+str(i)+'.fits'), ts=t)
    
        print('-> Simulated image '+str(i)+' for timestamp '+str(t)+\
                ' and target magnitude '+str(lightcurve[i,1]))
    
def calc_stars_on_image(params, catalog, image, jstar, target_mag):
    """Function to calculate the x,y pixel positions of stars in the 
    catalog"""
    
    field_center = SkyCoord(params['field_ra'],params['field_dec'],
                            unit=(u.hourangle, u.deg))
    
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
    
    if jstar in idx:
        jprime = np.where(idx == jstar)[0][0]
        
        stars[jprime,2] = target_mag
    
        print(' -> Target at x,y='+str(stars_pixels[jstar,0])+', '+\
                                   str(stars_pixels[jstar,1]))

    print('-> Calculated stars on image')
    
    return stars, jprime

def find_target_star_in_catalog(params,vphas_catalog):
    """Function to identify the closest star to the given coordinates in the
    catalogue."""
    
    star = SkyCoord(params['field_ra'],params['field_dec'],
                            unit=(u.hourangle, u.deg))
    pointA = (star.ra.value, star.dec.value)
    
    seps = separation_from_point(pointA, vphas_catalog['_RAJ2000'], 
                                         vphas_catalog['_DEJ2000'])

    idx = seps.argsort()
    
    return idx[0]

def simulate_lensing_lightcurve(params,vphas_catalog,jstar):
    """Function to simulate a microlensing lightcurve for the selected object"""
    
    ts = np.arange(params['tstart'], params['tend'],1.0)
    
    ut = np.sqrt(params['u0']**2 + ((ts - params['t0'])/params['tE'])**2) 
    
    A = (ut*ut + 2) / ( ut*np.sqrt(ut + 4) )
    
    lightcurve = np.zeros([len(ts),2])

    lightcurve[:,0] = ts
    lightcurve[:,1].fill(vphas_catalog['rmag'][jstar])
    lightcurve[:,1] += -2.5*np.log10(A)
    
    fig = plt.figure(1,(10,10))
    
    plt.plot(lightcurve[:,0], lightcurve[:,1], 'r-')

    plt.xlabel('HJD')
    plt.ylabel('Mag')

    [xmin,xmax,ymin,ymax] = plt.axis()
    plt.axis([xmin,xmax,ymax,ymin])

    plt.savefig(os.path.join(params['output'],'sim_lightcurve.png'))
    
    f = open(os.path.join(params['output'],'sim_lightcurve.txt'),'w')
    f.write('# Image_index  HJD  model_mag  model_flux  magnification\n')
    
    for i in range(0,len(lightcurve),1):

        model_flux = mag_to_flux(lightcurve[i,1])

        f.write(str(i)+' '+str(lightcurve[i,0])+' '+str(lightcurve[i,1])+\
                    ' '+str(model_flux)+' '+str(A[i])+'\n')
    f.close()
    
    return lightcurve
    
def separation_from_point(pointA, ra_array, dec_array):
    """Function to calculate the separation between a points on the sky, and
    an array of positions.
    Input are tuples of (RA, Dec) for each point in decimal degrees.
    Output is the arclength between them in decimal degrees.
    This function uses the full formula for angular separation, and should be applicable
    at arbitrarily large distances."""
    
    # Convert to radians because numpy requires them:
    pA = ( d2r(pointA[0]), d2r(pointA[1]) )
    raB = d2r(ra_array)
    decB = d2r(dec_array)
    
    half_pi = np.pi/2.0
    
    d1 = half_pi - pA[1]
    d2 = half_pi - decB
    dra = pA[0] - raB
    
    cos_gamma = ( np.cos(d1) * np.cos(d2) ) + \
                        ( np.sin(d1) * np.sin(d2) * np.cos(dra) )
                        
    gamma = np.arccos(cos_gamma)
    
    gamma = r2d( gamma )
    
    return gamma

def d2r( angle_deg ):
    """Function to convert an angle in degrees to radians"""
    
    angle_rad = ( np.pi * angle_deg ) / 180.0
    return angle_rad

def r2d( angle_rad ):
    """Function to convert an angle in radians to degrees"""
    
    angle_deg = ( 180.0 * angle_rad ) / np.pi
    return angle_deg

def mag_to_flux(mag):
    
    return 10**((mag-ZP)/-2.5)
    
def get_args():
    """Function to harvest the necessary simulation parameters"""
    
    params = {}
    
    if len(argv) < 8:
        
        params['field_ra'] = raw_input('Please enter the field center RA (J2000.0, sexigesimal): ')
        params['field_dec'] = raw_input('Please enter the field center Dec (J2000.0, sexigesimal): ')
        params['catalog'] = raw_input('Please enter the path to the input catalog or None: ')
        params['tstart'] = raw_input('Please enter the start date of the simulated lightcurve in HJD: ')
        params['tend'] = raw_input('Please enter the start date of the simulated lightcurve in HJD: ')
        params['t0'] = raw_input('Please enter the time of lensing closest approach (t0) in HJD: ')
        params['u0'] = raw_input('Please enter the minimum lensing impact parameter (u0): ')
        params['tE'] = raw_input('Please enter the Einstein crossing time (tE): ')
        params['output'] = raw_input('Please enter the path to the output directory: ')
        
    else:

        params['field_ra'] = argv[1]
        params['field_dec'] = argv[2]
        params['catalog'] = argv[3]
        params['tstart'] = argv[4]
        params['tend'] = argv[5]
        params['t0'] = argv[6]
        params['u0'] = argv[7]
        params['tE'] = argv[8]
        params['output'] = argv[9]
    
    for key in ['tstart', 'tend', 't0', 'u0', 'tE']:
        params[key] = float(params[key])
        
    return params

def plot_image(image):
    """Function to plot a simulated LORRI image"""
    
    fig = plt.figure(1,(10,10))
    
    plt.imshow(image1)
    
    plt.show()
    
    plt.close(1)


if __name__ == '__main__':
    
    simulate_image_dataset()