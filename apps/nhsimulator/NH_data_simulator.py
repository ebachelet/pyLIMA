# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:33:59 2017

@author: rstreet
"""

import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import os, sys
lib_path = os.path.abspath(os.path.join('../'))
sys.path.append(lib_path)
from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA import microlsimulator
from pyLIMA import microlmodels
from pyLIMA import microlfits
from scipy import interpolate
import jplhorizons_utils

class LORRIParams():
    """Class describing the essential parameters of the LORRI telescope 
    onboard New Horizons.
    All time stamps are in seconds, gain and readnoise are in e-/DN and e-
    respectively, aperture is in m.
    Parameters are taken from Cheng, A.F. et al. 2007, 
    https://arxiv.org/pdf/0709.4278.pdf
    """
    
    def __init__(self):
        
        self.aperture = 0.208 
        self.read_time = 1.0
        self.ZP = 17.7
        self.gain = 22.0
        self.read_noise = 10.0
        self.max_exp_time = 30.0
    
    def mag_to_flux(self, mag):
        """m2 - m1 = -2.5*log(f2/f1)
        f2 = 10**[(m2-m1)/-2.5]"""
        
        flux = ( 10**( (mag-self.ZP)/-2.5 ) ) * self.gain
        
        return flux
    
    def calc_flux_uncertainty(self, flux):
        """Method to calculate the expected photometric uncertainty for a given
        photometric measurement in flux units.
        
        :param float mag: Magnitude of star
        """
    
        aperradius = self.aperture/2.0
        
        npix_aper = np.pi*aperradius*aperradius
        
        read_noise = np.sqrt(self.read_noise*self.read_noise*npix_aper)
        
        possion_noise = np.sqrt(flux)
        
        total_noise = np.sqrt(read_noise*read_noise + possion_noise*possion_noise )
        
        return total_noise, read_noise, possion_noise

    def flux_to_mag(self, flux, flux_err):
        """Function to convert the flux of a star from its fitted PSF model 
        and its uncertainty onto the magnitude scale.
        
        :param float flux: Total star flux
        :param float flux_err: Uncertainty in star flux
        
        Returns:
        
        :param float mag: Measured star magnitude
        :param float flux_mag: Uncertainty in measured magnitude
        """
        
        f = flux / self.gain
        
        if flux < 0.0 or flux_err < 0.0:
            
            mag = 0.0
            mag_err = 0.0
    
        else:        
    
            mag = self.ZP - 2.5 * np.log10(f)
            
            mag_err = (2.5/np.log(10.0))*flux_err/f
            
        return mag, mag_err
    
    def convert_lc_flux_to_mag(self, flux_lc, flux_lc_err):
        
        mag_lc = np.zeros(len(flux_lc))
        mag_lc_err = np.zeros(len(flux_lc))

        for i in range(0,len(flux_lc),1):

            (mag_lc[i],mag_lc_err[i]) = self.flux_to_mag(flux_lc[i],flux_lc_err[i])
        
        return mag_lc, mag_lc_err
        
def generate_LORRI_lightcurve(params,log):
    """Function to generate a lightcurve with the noise characteristics of the
    LORRI telescope onboard New Horizons
    
    :param dict params: Parameter dictionary containing:
            :param float JD_start: JD of the start of the lightcurve    
            :param float JD_end: JD of the end of the lightcurve  
            :param float baseline_mag: Mean magnitude of star at baseline
    :param logger log: Logging object
    
    Returned:
    :param array lightcurve: 3-col array containing timestamps, magnitude, 
                                and magnitude error
    """
    lorri = LORRIParams()
    cadence = params['cadence_days'] * 24.0 * 60.0 * 60.0
    
    ts_incr = (lorri.max_exp_time + lorri.read_time + cadence) / ( 60.0 * 60.0 * 24.0 )
    
    ts = np.arange(params['JD_start'], params['JD_end'], ts_incr)
    
    flux_base = lorri.mag_to_flux(params['baseline_mag'])
    
    (sig_flux, read_noise, possion_noise) = calc_flux_uncertainty(lorri,flux_base)
    
    flux_lc = sig_flux * np.random.randn(len(ts)) + flux_base
    
    (flux_lc_err, read_noise, possion_noise) = calc_flux_uncertainty(lorri,flux_lc)

    (mag, mag_err) = lorri.convert_lc_flux_to_mag(flux_lc, flux_lc_err)
    
    lightcurve = np.zeros([len(ts),3])

    lightcurve[:,0] = ts
    lightcurve[:,1] = mag
    lightcurve[:,2] = mag_err

    log.info('Generated lightcurve')
    
    return lightcurve

def get_parameter_order(model_type,parallax):
    """Function to provide a list of the microlensing event parameters, 
    in the order which they should be appended to the list of model parameters
    in pyLIMA
    
    Inputs:
    :param string model_type: Type of lensing model, one of {PSPL, FSPL, USBL}
    :param Bool parallax: Switch to include parallax
    
    Returns:
    :param list porder: List of parameter names, in the pyLIMA convention
    """
    
    porder = ['to', 'uo', 'tE']
    
    if 'FS' in model_type:
        porder.append('rho')
    
    if 'BL' in model_type:
        porder.append('logs')
        porder.append('logq')
        porder.append('alpha')
    
    if parallax:
        porder.append('piEN')
        porder.append('piEE')
        
    return porder
    

def add_event_to_lightcurve(lightcurve,event_params,lc_params,log,
                            output_path,
                            spacecraft_positions=None,output_lc=False):
    """Function to inject the signal of a microlensing event into an 
    existing LORRI lightcurve, adjusting the photometric uncertainties 
    appropriately.
    
    :param dict event_params: [Optional] Microlensing model parameters containing:
            :param str name: Name of event
            :param float ra: RA in decimal degrees
            :param float dec: Dec in decimal degrees
            :param float to: Event t0 in JD
            :param float uo: Event u0
            :param float tE: Event tE in days
            :param float piEN: Event parallax component
            :param float piEE: Event parallax component
            :param float logs: Binary event mass separation, log10
            :param float logq: Binary event mass ratio, log10
            :param float alpha: Binary event angle of trajectory
            :param boolean log: Logger object
    :param dict lc_params: Lightcurve parameters
    :param logger log: Debugging log object
    :param string output_path: Path for all output files
    :param list spacecraft_positions: [Optional] Spacecraft positions from JPL observer table
    :param Boolean output_lc: [Optional] Switch to turn on lightcurve text file
                                output
    
    Returned:
    :param array lightcurve: 3-col array containing timestamps, magnitude, 
                                and magnitude error
    """

    log.info('Adding event model to lightcurve')
        
    if spacecraft_positions != None:
        log.info('Using horizons data provided')
        
    lorri = LORRIParams()
    
    lens = event.Event()
    lens.name = event_params['name']
    lens.ra = event_params['ra']
    lens.dec = event_params['dec']                    
    
    tel = telescopes.Telescope(name='NH_LORRI', camera_filter='I', 
                               spacecraft_name = 'New Horizons',
                               location='NewHorizon',
                               light_curve_magnitude=lightcurve)
                               
    log.info('Created telescope object with lightcurve of length = '+\
    str(len(lightcurve)))
        
    if spacecraft_positions != None and \
        'piEN' in event_params.keys() and 'piEE' in event_params.keys():
        
        log.info('Assigning spacecraft position data')
        tel.spacecraft_positions = spacecraft_positions
        
    lens.telescopes.append(tel)
    
    lens.find_survey('NH_LORRI')
    
    lens.check_event()
    
    porder_parallax = get_parameter_order(event_params['model_type'],True)
    porder_no_parallax = get_parameter_order(event_params['model_type'],False)
    
    if 'piEN' in event_params.keys() and 'piEE' in event_params.keys():
        
        log.info(' -> Model with parallax parameters: '+\
                    str(event_params['piEN'])+\
                    ' '+str(event_params['piEE']))
                    
        model = microlmodels.create_model(event_params['model_type'], lens, 
                                      parallax=['Full', event_params['to']])
        
        model_params = []
        
        for key in porder_parallax:                
            model_params.append(event_params[key])
        
    else:
        
        log.info(' -> No parallax parameters')
                    
        model = microlmodels.create_model(event_params['model_type'], lens, 
                                      parallax=['None', event_params['to']])
        
        model_params = []
        
        for key in porder_no_parallax:                
            model_params.append(event_params[key])
            
    model.define_model_parameters()
        
    if 'piEN' in event_params.keys() and 'piEE' in event_params.keys():
        log.info('Computing parallax for all telescopes, parllax_model = '+\
                    repr(model.parallax_model))
        model.event.compute_parallax_all_telescopes(model.parallax_model)
    
    tel = model.event.telescopes[0]
    
    f = microlfits.MLFits(lens)
    f.model = model
    f.fit_results = model_params
    lens.fits.append(f)
    
    pyLIMA_parameters = f.model.compute_pyLIMA_parameters(f.fit_results)
    
    param_sequence_check(event_params,model_params,pyLIMA_parameters,log)
    
    pylima_params = model.compute_pyLIMA_parameters(model_params)
    
    A = model.model_magnification(tel,pylima_params)
    
    lightcurve = lens.telescopes[0].lightcurve_magnitude
    
    lightcurve[:,1] = lightcurve[:,1] + -2.5*np.log10(A)
    (lightcurve[:,2],read_noise,poisson_noise) = calc_phot_uncertainty(lorri,
                                                            lightcurve[:,1])
    
    if output_lc:
        
        if 'piEN' in event_params.keys() and 'piEE' in event_params.keys():
            
            file_path = os.path.join(output_path,
                                 'sim_lightcurve_'+\
                                 str(round(lc_params['baseline_mag'],1))+'_'+\
                                 str(round(event_params['tE'],0))+'_parallax.txt')
        else:
            
                file_path = os.path.join(output_path,
                                 'sim_lightcurve_'+\
                                 str(round(lc_params['baseline_mag'],1))+'_'+\
                                 str(round(event_params['tE'],0))+'_no_parallax.txt')
                
        f = open(file_path,'w')

        for i in range(0,len(lightcurve),1):

            f.write(str(lightcurve[i,0])+' '+str(lightcurve[i,1])+' '+\
                    str(lightcurve[i,2])+'\n')

        f.close()
    
    return lightcurve,lens

def param_sequence_check(event_params,model_params,pylima_params,log):
    """Function to sanity check the sequence of event model parameters"""
    
    if 'piEN' in event_params.keys():
        
        porder = get_parameter_order(event_params['model_type'],True)
    
    else:
    
        porder = get_parameter_order(event_params['model_type'],False)
    
    log.info('Sanity checking event model parameters: '+repr(porder))
    
    for i,key in enumerate(porder):
        
        if model_params[i] == getattr(pylima_params,key):
            log.info(porder[i]+': Expected '+str(model_params[i])+', got '+\
                        str(getattr(pylima_params,key))+' -> OK')
        
        else:
            log.info(porder[i]+': Expected '+str(model_params[i])+', got '+\
                        str(getattr(pylima_params,key))+' -> ERROR')
            sys.exit()
    
    log.info('Model parameters are correct')
    
def calc_phot_uncertainty(lorri, mag):
    """Function to calculate the expected photometric uncertainty for a given
    photometric measurement.
    
    :param float mag: Magnitude of star
    """
    
    flux = (10**((mag-lorri.ZP)/-2.5)) * lorri.gain
    
    logfactor = 2.5 * (1.0 / flux) * np.log10(np.exp(1.0))
    
    aperradius = lorri.aperture/2.0
    
    npix_aper = np.pi*aperradius*aperradius
    
    read_noise = np.sqrt(lorri.read_noise*lorri.read_noise*npix_aper)*logfactor
    
    possion_noise = np.sqrt(flux)*logfactor
    
    total_noise = np.sqrt(read_noise*read_noise + possion_noise*possion_noise )
    
    return total_noise, read_noise, possion_noise

def calc_flux_uncertainty(lorri, flux):
    """Function to calculate the expected photometric uncertainty for a given
    photometric measurement in flux units.
    
    :param float mag: Magnitude of star
    """

    aperradius = lorri.aperture/2.0
    
    npix_aper = np.pi*aperradius*aperradius
    
    read_noise = np.sqrt(lorri.read_noise*lorri.read_noise*npix_aper)
    
    possion_noise = np.sqrt(flux)
    
    total_noise = np.sqrt(read_noise*read_noise + possion_noise*possion_noise )
    
    return total_noise, read_noise, possion_noise
    
def plot_LORRI_lightcurve():
    """Function to plot a lightcurve from LORRI"""
    
    params = get_args()
    
    event_params = { 'name': 'Simulated event',
                     'ra': 268.75, 'dec': -29.0,
                     'to': (params['JD_start'] + (params['JD_end']-params['JD_start'])/2.0),
                     'uo': params['uo'],
                     'tE': params['tE'],
                     'rho': 0.001,
                     'piEN': 0.1, 'piEE': 0.1,
                     }
                     
    lightcurve = generate_LORRI_lightcurve(params, event_params=event_params)

    fig = plt.figure(1)
    
    plt.errorbar(lightcurve[:,0], lightcurve[:,1], yerr=lightcurve[:,2])
    #plt.plot(lightcurve[:,0], lightcurve[:,1],'b.')
    
    (xmin,xmax,ymin,ymax) = plt.axis()
    plt.axis([xmin,xmax,ymax,ymin])
    
    plt.xlabel('JD')
    plt.ylabel('Magnitude')
    
    plt.savefig('lorri_lightcurve.png')
    
    plt.close(1)
    

def get_args():
    """Function to gather commandline arguments, if any"""
    
    params = { 'baseline_mag': None, 'JD_start': None, 'JD_end': None}
    
    if len(argv) > 1:
        
        params['baseline_mag'] = float(argv[1])
        params['JD_start'] = float(argv[2])
        params['JD_end'] = float(argv[3])
        params['uo'] = float(argv[4])
        params['tE'] = float(argv[5])
    
    else:
        
        params['baseline_mag'] = float(raw_input('Please enter the baseline magnitude for the star: '))
        params['JD_start'] = float(raw_input('Please enter the JD at the start of the lightcurve: '))
        params['JD_end'] = float(raw_input('Please enter the JD at the start of the lightcurve: '))
        params['uo'] = float(raw_input('Please enter the u0 of the lensing event: '))
        params['tE'] = float(raw_input('Please enter the tE of the lensing event: '))
    
    return params

if __name__ == '__main__':
    
    plot_LORRI_lightcurve()
    