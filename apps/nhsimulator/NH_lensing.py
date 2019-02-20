# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:05:37 2018

@author: rstreet
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm, rcParams
from matplotlib.ticker import MaxNLocator

from datetime import datetime
import copy

import os, sys

from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA import microlsimulator
from pyLIMA import microlmodels
from pyLIMA import microltoolbox
from pyLIMA import microlstats
from pyLIMA import microlcaustics
import NH_data_simulator
import jplhorizons_utils

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import SkyOffsetFrame, ICRS

from astropy.units import cds
cds.enable()
from astropy import constants

import logging

class FitParams():
    """Object containing the fitted parameters and errors of a model fitted
    to a microlensing lightcurve"""
    
    def __init__(self):
        
        self.to = None
        self.toerr = None
        self.uo = None
        self.uoerr = None
        self.tE= None
        self.tEerr = None
        self.rho = None
        self.rhoerr = None
        self.piEN = None
        self.piENerr = None
        self.piEE= None
        self.piEEerr = None
        self.s = None
        self.serr = None
        self.q = None
        self.qerr = None
        self.logs = None
        self.logserr = None
        self.logq = None
        self.logqerr = None
        self.alpha = None
        self.alphaerr = None
        self.fs = None
        self.fserr = None
        self.fb = None
        self.fberr = None
        self.chichi = None
        self.bic = None
        self.nparam = 0
        self.type = None

    def summary(self):
        
        line = 'tE = '+str(self.tE)+'+/-'+str(self.tEerr)+'\n'+\
                'to = '+str(self.to)+'+/-'+str(self.toerr)+'\n'+\
                'uo = '+str(self.uo)+'+/-'+str(self.uoerr)+'\n'+\
                'rho = '+str(self.rho)+'+/-'+str(self.rhoerr)+'\n'+\
                'piEN = '+str(self.piEN)+'+/-'+str(self.piENerr)+'\n'+\
                'piEE = '+str(self.piEE)+'+/-'+str(self.piEEerr)+'\n'+\
                'fs = '+str(self.fs)+'+/-'+str(self.fserr)+'\n'+\
                'fb = '+str(self.fb)+'+/-'+str(self.fberr)+'\n'+\
                'chi^2 = '+str(self.chichi)+'\n'+\
                'BIC = '+str(self.bic)
                
        return line

def simulate_NH_lensing():
    """Driver function to simulate microlensing as seen by New Horizons."""

    config_file = get_args()
    
    (default_params,source_mag_range,tE_range) = parse_config_file(config_file)
    
    log = start_log( default_params['output_path'], 'sim_info' )
    
    record_config(default_params,source_mag_range, tE_range,log)
    
    simulate_model_grid(default_params,source_mag_range,tE_range,log)
    
    stop_log(log)
    
def simulate_model_grid(default_params,source_mag_range,tE_range,log):
    """Function to simulate a grid of microlensing events with ranges of
    values of source star baseline magnitude, tE, with all other parameters 
    taking the provided defaults.
    """
    
    lc_keys = ['JD_start','JD_end']
    event_keys = ['name', 'ra', 'dec', 
                  'to','uo','rho','piEN', 'piEE', 'logs', 'logq', 'alpha',
                  'model_code']
    horizons_table = None
    table_type = 'VECTOR'
    
    if 'horizons_file' in default_params.keys():
        
        horizons_table = jplhorizons_utils.parse_JPL_Horizons_table(horizons_file_path=default_params['horizons_file'], 
                                                                   table_type=table_type)

        log.info('Added spacecraft positions to pyLIMA tel object')
    
        log.info('Range of JDs in spacecraft positions table: '+
                    str(horizons_table['JD'].min())+' - '+
                    str(horizons_table['JD'].max()))
                    
        log.info('Event t0 requested: '+str(default_params['to']))

        log.info('Range of JDs requested for simulated lightcurve: '+
                    str(default_params['JD_start'])+' - '+
                    str(default_params['JD_end']))
        
        if table_type == 'OBSERVER':
        
            spacecraft_positions = jplhorizons_utils.calc_spacecraft_positions_observer_table(horizons_table,default_params['to'])
            
        else:
            
            spacecraft_positions = jplhorizons_utils.calc_spacecraft_positions_vector_table(horizons_table,default_params['to'])
        
    fit_data = np.zeros([len(source_mag_range),len(tE_range),3])
    
    output_path = os.path.join(default_params['output_path'],'lensing_statistics.txt')
    
    stats_file = open(output_path,'w')
    stats_file.write('#                           |   No parallax model    |   Parallax model          |                        | \n')
    stats_file.write('# Baseline mag   tE [days]  |   Chi^2  BIC           | piEN    piEE    Chi^2 BIC |  Delta_Chi^2 Delta_BIC | Max_residual[mag] S/N\n')
    
    output_path = os.path.join(default_params['output_path'],'fitted_model_parameters.txt')
    
    model_file = open(output_path,'w')
    model_file.write('# Baseline mag  tE_input   Modeltype  tE[days]   t0[days]  u0   rho  pi_EN   pi_EE   fs   fb   chi^2   BIC\n')
    
    for j in range(0,len(source_mag_range),1):
        
        mag = source_mag_range[j]
        
        for i in range(0,len(tE_range),1):
            
            tE = tE_range[i]
            
            log.info('Simulating grid point mag='+str(mag)+'mag and tE='+str(tE)+'days')
            
            (lc_params, event_params_no_parallax, event_params_parallax) = make_param_dicts(default_params,mag,tE,log)
            
            baseline_lc = NH_data_simulator.generate_LORRI_lightcurve(lc_params,log)
            
            (lc_no_parallax,sim_e_no_parallax) = NH_data_simulator.add_event_to_lightcurve(baseline_lc,
                                                                       event_params_no_parallax,
                                                                       lc_params,log,default_params['output_path'],
                                                                       spacecraft_positions=None,
                                                                       output_lc=True)
            
            (lc_parallax,sim_e_parallax) = NH_data_simulator.add_event_to_lightcurve(baseline_lc,
                                                                    event_params_parallax,
                                                                    lc_params,log,default_params['output_path'],
                                                                    spacecraft_positions=spacecraft_positions,
                                                                    output_lc=True)
            
            (max_res, S2N) = calc_lc_signal_to_noise(lc_no_parallax,lc_parallax)
            
            log.info('Completed data simulation for grid point mag='+str(mag)+'mag and tE='+str(tE)+'days')
                
            if default_params['fit_models']:
                (fit_no_parallax,e_no_parallax) = fit_microlensing_model(lc_parallax, event_params_no_parallax,
                                                                    lc_params, log,default_params['output_path'],
                                                                    spacecraft_positions=spacecraft_positions,
                                                                    output_lc=True)
            
                (fit_parallax,e_parallax) = fit_microlensing_model(lc_parallax, event_params_parallax,
                                                                    lc_params, log,default_params['output_path'],
                                                                    spacecraft_positions=spacecraft_positions,
                                                                    output_lc=True)
            
            else:
                fit_no_parallax = FitParams()
                fit_parallax = FitParams()
                e_no_parallax = None
                e_parallax = None
                
            log.info('Completed model fitting for grid point mag='+str(mag)+'mag and tE='+str(tE)+'days')
                
            output_plots(default_params,mag,tE,lc_no_parallax,lc_parallax,
                 e_no_parallax, e_parallax,sim_e_no_parallax, sim_e_parallax,
                 log)
             
            if default_params['fit_models']:
                dchichi = fit_no_parallax.chichi - fit_parallax.chichi
                dbic = abs(fit_no_parallax.bic - fit_parallax.bic)
            else:
                dchichi = 0.0
                dbic= 0.0
                
            fit_data[j,i,0] = mag
            fit_data[j,i,1] = tE
            fit_data[j,i,2] = dbic
            
            output_metrics(mag,tE,fit_no_parallax,fit_parallax,
                           dchichi,dbic,max_res,S2N,stats_file,model_file,log)

    stats_file.close()
    
    model_file.close()
    
    log.info('Completed simulated grid')

def output_metrics(mag,tE,fit_no_parallax,fit_parallax,dchichi,dbic,max_res,S2N,
                   stats_file,model_file,log):
    """Function to output the computed metrics to file"""
    
    stats_file.write(str(mag)+'  '+str(tE)+' | '+\
                        str(fit_no_parallax.chichi)+'  '+str(fit_no_parallax.bic)+' | '+\
                        str(fit_parallax.piEN)+' +/- '+str(fit_parallax.piENerr)+'  '+\
                        str(fit_parallax.piEE)+' +/- '+str(fit_parallax.piEEerr)+'  '+\
                        str(fit_parallax.chichi)+'  '+str(fit_parallax.bic)+' | '+\
                        str(dchichi)+'  '+str(dbic)+' | '+str(max_res)+' '+str(S2N)+'\n')
    stats_file.flush()
            
    model_file.write(str(mag)+' '+str(tE)+'  no_parallax   '+\
                    str(fit_no_parallax.tE)+' +/- '+str(fit_no_parallax.tEerr)+'  '+\
                    str(fit_no_parallax.to)+' +/- '+str(fit_no_parallax.toerr)+'  '+\
                    str(fit_no_parallax.uo)+' +/- '+str(fit_no_parallax.uoerr)+'  '+\
                    str(fit_no_parallax.rho)+' +/- '+str(fit_no_parallax.rhoerr)+'  '+\
                    'None +/- None  '+\
                    'None +/- None  '+\
                    str(fit_no_parallax.fs)+' +/- '+str(fit_no_parallax.fserr)+'  '+\
                    str(fit_no_parallax.fb)+' +/- '+str(fit_no_parallax.fberr)+'  '+\
                    str(fit_no_parallax.chichi)+'  '+str(fit_no_parallax.bic)+'\n')
                    
    model_file.write(str(mag)+' '+str(tE)+'  parallax   '+\
                    str(fit_parallax.tE)+' +/- '+str(fit_parallax.tEerr)+'  '+\
                    str(fit_parallax.to)+' +/- '+str(fit_parallax.toerr)+'  '+\
                    str(fit_parallax.uo)+' +/- '+str(fit_parallax.uoerr)+'  '+\
                    str(fit_parallax.rho)+' +/- '+str(fit_parallax.rhoerr)+'  '+\
                    str(fit_parallax.piEN)+' +/- '+str(fit_parallax.piENerr)+'  '+\
                    str(fit_parallax.piEE)+' +/- '+str(fit_parallax.piEEerr)+'  '+\
                    str(fit_parallax.fs)+' +/- '+str(fit_parallax.fserr)+'  '+\
                    str(fit_parallax.fb)+' +/- '+str(fit_parallax.fberr)+'  '+\
                    str(fit_parallax.chichi)+'  '+str(fit_parallax.bic)+'\n')
    model_file.flush()

    log.info('Completed output of results for grid point mag='+str(mag)+'mag and tE='+str(tE)+'days\n')

def output_plots(default_params,mag,tE,lc_no_parallax,lc_parallax,
                 e_no_parallax, e_parallax,
                 sim_e_no_parallax, sim_e_parallax,log):
    """Function to output, on user request, plots of the lightcurve models and 
    lens plane plots.  
    
    If the user has requested that models be fitted to the simulated lightcurves, 
    then the plots represent the results of the models fit.  Otherwise, the plots
    represent the output from a theoretical event of the same parameters as used
    to generate the simulated lightcurve.
    """
    
    if default_params['plots']:
        
        if default_params['fit_models']:
        
            log.info('Outputting lightcurve plots')
            
            lc_plot_file = os.path.join(default_params['output_path'],
                                        'fitted_lightcurves_'+str(round(mag,1))+
                                        '_'+str(round(tE,0))+'.eps')
        
            plot_fitted_lightcurves(lc_no_parallax,lc_parallax,e_no_parallax,e_parallax,
                        lc_plot_file)
        
            log.info('Outputting lens plane plots')
            
            lens_plane_plot_file = os.path.join(default_params['output_path'],
                                                'lens_plane_'+str(round(mag,1))+
                                                '_'+str(round(tE,0))+'.eps')
            lens_plane_txt_file = os.path.join(default_params['output_path'],
                                                'lens_plane_'+str(round(mag,1))+
                                                '_'+str(round(tE,0))+'.txt')
                                                
            plot_lens_plane_trajectories(default_params['model_type'],e_no_parallax,e_parallax,
                                     'No parallax','Parallax',
                                     lens_plane_plot_file)
    
        if not default_params['fit_models']:
            
            log.info('Outputting simulated lightcurve plots')
            
            lc_plot_file = os.path.join(default_params['output_path'],
                                        'sim_lightcurves_'+str(round(mag,1))+
                                        '_'+str(round(tE,0))+'.eps')
        
            plot_fitted_lightcurves(lc_no_parallax,lc_parallax,sim_e_no_parallax,sim_e_parallax,
                        lc_plot_file)
        
            log.info('Outputting simulated lens plane plots')
            
            lens_plane_plot_file = os.path.join(default_params['output_path'],
                                                'sim_lens_plane_'+str(round(mag,1))+
                                                '_'+str(round(tE,0))+'.eps')
            lens_plane_txt_file = os.path.join(default_params['output_path'],
                                                'sim_lens_plane_'+str(round(mag,1))+
                                                '_'+str(round(tE,0))+'.txt')
                                                
            plot_lens_plane_trajectories(default_params['model_type'],sim_e_no_parallax,sim_e_parallax,
                                     'No parallax','Parallax',
                                     lens_plane_plot_file,lens_plane_txt_file)
    
        log.info('Completed plotting output')

    else:
        
        log.info('Plotting output switched off in configuration')
        
def calc_lc_signal_to_noise(lc_no_parallax,lc_parallax):
    """Function to calculate the signal-to-noise of the parallax signature
    in the microlensing lightcurve, by comparing the lightcurves observed
    with and without parallax included in the model.
    
    S/N = var(signal) / var(phot noise) = [amplitude(signal)/amplitude(noise)]^2
    """
    
    lc_residual = lc_no_parallax[:,1] - lc_parallax[:,1]
    
    var_residuals = (lc_residual*lc_residual).sum() / float(len(lc_residual))
    
    var_phot_noise = (lc_no_parallax[:,2]*lc_no_parallax[:,2]).sum() / float(len(lc_no_parallax))
    
    max_res = abs(lc_residual).max()
    
    S2N = var_residuals / var_phot_noise
    
    return max_res, S2N
    
def make_param_dicts(default_params,mag,tE,log):
    """Function to repackage and combine the parameter dictionaries required 
    for the simulation functions.
    """

    lc_keys = ['JD_start','JD_end','cadence_days']
    event_keys = ['name', 'ra', 'dec', 
                  'to','uo','rho','piEN', 'piEE','logs', 'logq', 'alpha',
                  'model_type']
    
    lc_params = {}
        
    for key in lc_keys:
        
        lc_params[key] = default_params[key]
    
    lc_params['baseline_mag'] = mag
    
    event_params_parallax = {}
    event_params_no_parallax = {}
    
    for key in event_keys:

        if 'pi' not in key:
        
            event_params_no_parallax[key] = default_params[key]
        
        event_params_parallax[key] = default_params[key]
        
    event_params_parallax['tE'] = tE
    event_params_no_parallax['tE'] = tE

    event_params_no_parallax['model_code'] = default_params['model_type']
    event_params_parallax['model_code'] = default_params['model_type']+'para'

    log.info('Lightcurve parameters: '+repr(lc_params))

    return lc_params, event_params_no_parallax, event_params_parallax
    
    
def fit_microlensing_model(lightcurve, event_params, lc_params, log, 
                           output_path,
                           spacecraft_positions=None,
                           produce_plots=False,output_lc=True):
    """Function to simulate microlensing events as seen
    from New Horizons and to fit a corresponding model to the simulated data.
    
    Inputs:
       lightcurve   array Lightcurve data to be fitted
       event_params dict Parameters of the microlensing event
       lc_params dict Lightcurve parameters
       produce_plots  bool   Switch on or off plotting output
       
    Outputs:
       fit_params FitParams obj  Fitted parameters and statistics
    """
    
    log.info('\n *** STARTING MODEL FITTING ROUTINE ***')
    
    e = event.Event()
    e.name = event_params['name']
    e.ra = event_params['ra']
    e.dec = event_params['dec']
    
    nh_tel = telescopes.Telescope(name='NH', camera_filter='I', 
                                  spacecraft_name = 'New Horizons',
                                  location='space', 
                                  light_curve_magnitude=lightcurve)
    log.info('LIGHTCURVE: '+str(lightcurve[:,0].min())+' '+str(lightcurve[:,0].max()))
    
    if spacecraft_positions != None:
        
        nh_tel.spacecraft_positions = spacecraft_positions
        
    e.telescopes.append(nh_tel)
    
    e.find_survey('NH')

    e.check_event()

    fit_method = 'LM'
    if 'BL' in event_params['model_type']:
        fit_method = 'DE'
        
    output = 'Fitting a '+event_params['model_type']+' model'
    
    if 'para' in event_params['model_code']:
        
        log.info('Fitting model with parallax')
        log.info('LC time range: '+str(e.telescopes[0].lightcurve_flux[:,0].min())+\
                    ' '+str(e.telescopes[0].lightcurve_flux[:,0].max()))
        log.info('With parameters '+repr(event_params))
            
        model = microlmodels.create_model(event_params['model_type'], e, 
                                          parallax=['Full', event_params['t0']], 
                                          blend_flux_ratio=False)
        output += ' with parallax'
        
    else:
        
        log.info('No parallax in fitting')
        log.info('LC time range: '+str(e.telescopes[0].lightcurve_flux[:,0].min())+\
                    ' '+str(e.telescopes[0].lightcurve_flux[:,0].max()))
        log.info('With parameters '+repr(event_params))
        
        model = microlmodels.create_model(event_params['model_type'], e, 
                                          blend_flux_ratio=False)
    
    log.info(output)
    
    e.fit(model,fit_method)
    fit_flag = e.fits[-1].check_fit()
    log.info('Fit flag: '+repr(fit_flag))
    log.info('Telescopes: '+str(len(e.telescopes)))
    log.info('Fit results: '+repr(e.fits[-1].fit_results))
    log.info('LIGHTCURVE: '+str(e.telescopes[-1].lightcurve_flux[:,0].min())+' '+\
                        str(e.telescopes[-1].lightcurve_flux[:,0].max()))
    log.info('LIGHTCURVE fit: '+str(e.fits[-1].event.telescopes[-1].lightcurve_flux[:,0].min())+' '+\
                        str(e.fits[-1].event.telescopes[-1].lightcurve_flux[:,0].max()))
    log.info('LIGHTCURVE fit: '+str(e.fits[-1].event.telescopes[-1].lightcurve_flux[:,0].min())+' '+\
                        str(e.fits[-1].event.telescopes[-1].lightcurve_flux[:,0].max()))
                        
    try:
        e.fits[-1].produce_outputs()
    
        # Close plot objects opened automatically by pyLIMA to avoid 
        # later plots overplotting the same axes.  Both close statements are
        # necessary. 
        plt.close()
        plt.close()
    
        log.info('-> Completed model fit, output parameters: ')    
        
        fit_params = get_fit_params(e,len(lightcurve))
        
        log.info(fit_params.summary())
        
        log.info('Fitted parameters: '+repr(fit_params.summary()))
    
        if produce_plots:
            fig = plt.figure(5)
        
            plt.errorbar(lightcurve[:,0],lightcurve[:,1],
                         yerr=lightcurve[:,2],alpha=0.2,color='b')
        
            model_lc = generate_model_lightcurve(e)
            
            plt.plot(lightcurve[:,0],model_lc,'k-')
            
            plt.xlabel('HJD')
        
            plt.ylabel('Magnitude')
        
            (xmin,xmax,ymin,ymax) = plt.axis()
            plt.axis([xmin,xmax,ymax,ymin])
        
            plt.savefig(os.path.join(output_path),'simulated_lc.png')
        
            plt.close(5)
        
        if output_lc:
            
            model_lc = generate_model_lightcurve(e)
            
            if 'para' in event_params['model_code']:
                
                file_path = os.path.join(output_path,
                                'sim_lightcurve_'+\
                                str(round(lc_params['baseline_mag'],1))+'_'+\
                                str(round(event_params['tE'],0))+'_parallax_model.txt')
            else:
                
                file_path = os.path.join(output_path,
                                'sim_lightcurve_'+\
                                str(round(lc_params['baseline_mag'],1))+'_'+\
                                str(round(event_params['tE'],0))+'_no_parallax_model.txt')
    
            f = open(file_path,'w')
    
            for i in range(0,len(lightcurve),1):
    
                f.write(str(lightcurve[i,0])+' '+str(model_lc[i])+'\n')
    
            f.close()
    
    except ValueError:
        
        log.info('ERROR in fitting process, inconsistent timestamps')
        
        fit_params = get_fit_params(None, 0)
        
    log.info('\n *** END MODEL FITTING ROUTINE ***\n')
        
    return fit_params, e

def get_fit_params(fitted_event,ndata):
    """Function to decant the parameters of the model fitted to a lightcurve
    from a pyLIMA Event object.
    """

    fit = FitParams()
    
    if fitted_event != None:

        fit.nparam = 0
        
        fit.tE = fitted_event.fits[-1].outputs.fit_parameters.tE
        fit.tEerr = fitted_event.fits[-1].outputs.fit_errors.err_tE
        fit.nparam += 1
        
        fit.t0 = fitted_event.fits[-1].outputs.fit_parameters.to
        fit.t0err = fitted_event.fits[-1].outputs.fit_errors.err_to
        fit.nparam += 1
        
        fit.u0 = fitted_event.fits[-1].outputs.fit_parameters.uo
        fit.u0err = fitted_event.fits[-1].outputs.fit_errors.err_uo
        fit.nparam += 1
        
        fit.fs = fitted_event.fits[-1].outputs.fit_parameters.fs_NH
        fit.fserr = fitted_event.fits[-1].outputs.fit_errors.err_fs_NH
        fit.nparam += 1
        
        fit.fb = fitted_event.fits[-1].outputs.fit_parameters.fb_NH
        fit.fberr = fitted_event.fits[-1].outputs.fit_errors.err_fb_NH
        fit.nparam += 1
    
        try:
            fit.rho = fitted_event.fits[-1].outputs.fit_parameters.rho
            fit.rhoerr = fitted_event.fits[-1].outputs.fit_errors.err_rho
            fit.nparam += 1
    
        except AttributeError:
            pass
        
    
        try:
            fit.piEN = fitted_event.fits[-1].outputs.fit_parameters.piEN
            fit.piENerr = fitted_event.fits[-1].outputs.fit_errors.err_piEN
            fit.nparam += 1
    
        except AttributeError:
            pass
        
        try:
            fit.piEE = fitted_event.fits[-1].outputs.fit_parameters.piEE
            fit.piEEerr = fitted_event.fits[-1].outputs.fit_errors.err_piEE
            fit.nparam += 1
    
        except AttributeError:
            pass
        
        try:
            fit.logs = fitted_event.fits[-1].outputs.fit_parameters.logs
            fit.logq = fitted_event.fits[-1].outputs.fit_parameters.logq
            fit.alpha = fitted_event.fits[-1].outputs.fit_parameters.alpha
            fit.logserr = fitted_event.fits[-1].outputs.fit_errors.err_logs
            fit.logqerr = fitted_event.fits[-1].outputs.fit_errors.err_logq
            fit.alphaerr = fitted_event.fits[-1].outputs.fit_errors.err_alpha

        except AttributeError:
            pass
        
        fit.chichi = fitted_event.fits[-1].outputs.fit_parameters.chichi
        
        fit.bic = microlstats.Bayesian_Information_Criterion(fit.chichi,
                                                              ndata,
                                                              fit.nparam)
    
    return fit


def plot_fitted_lightcurves(lc_no_parallax,lc_parallax,e_no_parallax,e_parallax,
                            file_path):
    """Function to plot lightcurves and model fits for both with- and without
    parallax models"""
    
    plot_models = False
    
    dt = float(int(lc_no_parallax[0,0]))
    
    ts_no_parallax = lc_no_parallax[:,0] - dt
    ts_parallax = lc_parallax[:,0] - dt
    
    fig = plt.figure(6,(10,10))
    plt.subplot(2,1,1)

    plt.errorbar(ts_no_parallax,lc_no_parallax[:,1],
                 yerr=lc_no_parallax[:,2],alpha=1.0,color='#8c6931',
                label='No parallax lightcurve')
                 
    plt.errorbar(ts_parallax,lc_parallax[:,1],
                 yerr=lc_parallax[:,2],alpha=1.0,color='#2b8c85',
                label='Parallax lightcurve')

    if plot_models:
        model_lc_no_parallax = generate_model_lightcurve(e_no_parallax)
        
        model_lc_parallax = generate_model_lightcurve(e_parallax)
        
        plt.plot(ts_no_parallax,model_lc_no_parallax,linestyle='dashed',
                     color='#4c1377',label='No parallax model')
         
        plt.plot(ts_parallax,model_lc_parallax,linestyle='solid',
                     color='black',label='Parallax model')
        
    plt.xlabel('HJD - '+str(dt), fontsize=18)

    plt.ylabel('Magnitude', fontsize=18)
    
    plt.legend(loc=1, fontsize=16)
    
    plt.grid()
    
    (xmin,xmax,ymin,ymax) = plt.axis()
    plt.axis([xmin,xmax,ymax,ymin])

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    
    plt.subplot(2,1,2)
    
    comb_err = np.sqrt( lc_no_parallax[:,2]*lc_no_parallax[:,2] + lc_parallax[:,2]*lc_parallax[:,2] )
    
#    plt.errorbar(ts_no_parallax,(lc_parallax[:,1]-lc_no_parallax[:,1]),
#                 yerr=comb_err,color='black',alpha=0.4)
    
    dmag = lc_parallax[:,1]-lc_no_parallax[:,1]
    plt.plot(ts_no_parallax,dmag,color='black',markersize=2)
    
    plt.xlabel('HJD - '+str(dt), fontsize=18)

    plt.ylabel('Magnitude', fontsize=18)
        
    plt.grid()
    
    (xmin2,xmax2,ymin2,ymax2) = plt.axis()
    ymin2 = dmag.min()*1.05
    ymax2 = dmag.max()*1.05
    
    plt.axis([xmin,xmax,ymax2, ymin2])

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    
    plt.savefig(file_path)

    plt.close(6)

def plot_lens_plane_trajectories(model_type,event_no_parallax,event_parallax,
                                 label1,label2,
                                 plot_file_path,txt_file_path):
    """
    Function to plot the geometry of the lens plane showing the source 
    relative trajectory for two different fitted models
    
    Adapted from the pyLIMA.microloutputs.plot_LM_ML_geometry routine
    by E. Bachelet.
    
    Inputs:
    :param Event event_parallax: First pyLIMA event object with fitted model
    :param Event event_no_parallax: Second pyLIMA event object with fitted model
    :param string label1: Label for event 1
    :param string label2: Label for event 2
    :param string plot_file_path: Path to output plot file
    :param string txt_file_path: Path to output text file for trajectories
    """
    
    txt_file_no_para = txt_file_path.split('.txt')[0]+'_no_parallax.txt'
    txt_file_para = txt_file_path.split('.txt')[0]+'_parallax.txt'
    
    figure_trajectory_xlimit = 1.5
    figure_trajectory_ylimit = 1.5

    fig = plt.figure(7,figsize=(10,10),dpi=75)
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.95,
                wspace=None, hspace=None)
    ax = fig.add_subplot(111, aspect=1)
    
    fit1 = event_no_parallax.fits[-1]
    fit2 = event_parallax.fits[-1]
    
    tmin = min([min(i.lightcurve_magnitude[:, 0]) for i in fit1.event.telescopes])
    tmax = max([max(i.lightcurve_magnitude[:, 0]) for i in fit1.event.telescopes])
    times = np.linspace(tmin, tmax + 100, 3000)
    
    (fig,ax) = plot_rel_trajectory(fit1,times,fig,ax,label1,txt_file_no_para,
                                    trajectory_color='#8c6931')
                                    
    tmin = min([min(i.lightcurve_magnitude[:, 0]) for i in fit2.event.telescopes])
    tmax = max([max(i.lightcurve_magnitude[:, 0]) for i in fit2.event.telescopes])
    times = np.linspace(tmin, tmax + 100, 3000)
                                    
    (fig,ax) = plot_rel_trajectory(fit2,times,fig,ax,label2,txt_file_para,
                                    trajectory_color='#2b8c85')

    ax.axis( [- figure_trajectory_xlimit, figure_trajectory_xlimit, 
              - figure_trajectory_ylimit, figure_trajectory_ylimit] )

    if 'PL' in model_type:
        ax.scatter(0, 0, s=10, c='k')
        
    else:
        pyLIMA_parameters = event_parallax.fits[-1].model.compute_pyLIMA_parameters(event_parallax.fits[-1].fit_results)
        s = 10**(pyLIMA_parameters.logs)
        ax.scatter((s/2.0), 0.0, s=10, c='k')
        ax.scatter(-(s/2.0), 0.0, s=10, c='k')
        
    if 'PL' in model_type:
        einstein_ring = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--')
        ax.add_artist(einstein_ring)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.get_major_ticks()[0].draw = lambda *args: None
    ax.yaxis.get_major_ticks()[0].draw = lambda *args: None
    ax.xaxis.get_major_ticks()[-1].draw = lambda *args: None
    ax.yaxis.get_major_ticks()[-1].draw = lambda *args: None

    ax.set_xlabel(r'$x(\theta_E)$', fontsize=25)
    ax.set_ylabel(r'$y(\theta_E)$', fontsize=25)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)

    plt.legend(fontsize=18)
    
    plt.savefig(plot_file_path)

    plt.close(7)
    
def plot_rel_trajectory(fit,times,fig,ax,label,txt_file_path,trajectory_color='r'):
    """Function to calculate the source's trajectory relative to a lens in 
    the lens plane from a pyLIMA Event with a fitted model.
    
    Adapted from the pyLIMA.microloutputs.plot_LM_ML_geometry routine
    by E. Bachelet.
    """
 
    tel = copy.copy(fit.event.telescopes[0])
    
    tel.lightcurve_magnitude = np.array(
        [times, [0] * len(times), [0] * len(times)]).T

    tel.lightcurve_flux = np.array(
        [times, [0] * len(times), [0] * len(times)]).T

    if fit.model.parallax_model[0] != 'None':

        tel.compute_parallax(fit.event, fit.model.parallax_model)
        
    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(fit.fit_results)
    
    (trajectory_x, trajectory_y) = fit.model.source_trajectory(tel, 
                                     pyLIMA_parameters.to, pyLIMA_parameters.uo,
                                     pyLIMA_parameters.tE, pyLIMA_parameters)
    
    ax.plot(trajectory_x, trajectory_y, trajectory_color,label=label)
        
    if 'PS' not in fit.model.model_type:
        index_source  = np.argmin((trajectory_x ** 2 + trajectory_y ** 2) ** 0.5)
        source_disk = plt.Circle((trajectory_x[index_source], trajectory_y[index_source]), 0.02,
                                 color=trajectory_color)
        ax.add_artist(source_disk)

    if 'BL' in fit.model.model_type:
        (regime, caustics, critical_curves) = microlcaustics.find_2_lenses_caustics_and_critical_curves(10**pyLIMA_parameters.logs,
                                                                                         10** pyLIMA_parameters.logq,
                                                                                         resolution=5000)
        for ic, caustic in enumerate(caustics):
            
                    try:
                        plt.plot(caustic.real, caustic.imag,lw=1, c='r')
                        plt.plot(critical_curves[ic].real, critical_curves[ic].imag, c=trajectory_color, linestyle='--')
                        
                    except AttributeError:
                        pass

    index_trajectory_limits = np.where((np.abs(times - pyLIMA_parameters.to) < 50))[0]

    if len(index_trajectory_limits) >= 3:
        
        middle = int(len(index_trajectory_limits) / 2)
        index_t0  = np.argmin(times - pyLIMA_parameters.to)
        
        ax.arrow(trajectory_x[index_trajectory_limits[index_t0]], trajectory_y[index_trajectory_limits[index_t0]],
                          trajectory_x[index_trajectory_limits[index_t0 + 1]] - trajectory_x[index_trajectory_limits[index_t0]],
                          trajectory_y[index_trajectory_limits[index_t0 + 1]] - trajectory_y[index_trajectory_limits[index_t0]],
                          head_width=0.04, head_length=0.04, 
                          color=trajectory_color)
    
    
    f = open(txt_file_path,'w')
    
    for key in NH_data_simulator.get_parameter_order(fit.model.model_type,True):
        try:
            f.write('# '+key+' :: '+str(getattr(pyLIMA_parameters,key))+'\n')
        except AttributeError:
            pass
        
    f.write('# Time lens_plane_x, lens_plane_y\n')

    for i in range(0,len(tel.lightcurve_flux[:,0]),1):

        f.write(str(tel.lightcurve_flux[i,0])+' '+str(trajectory_x[i])+' '+str(trajectory_y[i])+'\n')

    f.close()
    
    return fig,ax
    

def generate_model_lightcurve(e):
    """Function to produce a model lightcurve based on a parameter set
    fitted by pyLIMA
    
    Inputs:
    e  Event object
    
    """
    
    lc = e.telescopes[0].lightcurve_magnitude
    
    fit_params = e.fits[-1].model.compute_pyLIMA_parameters(e.fits[-1].fit_results)
    
    ts = np.linspace(lc[:,0].min(), lc[:,0].max(), len(lc[:,0]))

    reference_telescope = copy.copy(e.fits[-1].event.telescopes[0])
    
    reference_telescope.lightcurve_magnitude = np.array([ts, [0] * len(ts), [0] * len(ts)]).T
    
    reference_telescope.lightcurve_flux = reference_telescope.lightcurve_in_flux()

    if e.fits[-1].model.parallax_model[0] != 'None':
        
        reference_telescope.compute_parallax(e.fits[-1].event, e.fits[-1].model.parallax_model)

    flux_model = e.fits[-1].model.compute_the_microlensing_model(reference_telescope, fit_params)[0]
    
    mag_model = microltoolbox.flux_to_magnitude(flux_model)

    return mag_model

def start_log( log_dir, log_name, version=None ):
    """Function to initialize a log file for a single stage of pyDANDIA.  
    
    The naming convention for the file is [stage_name].log.  
    
    The new file will automatically overwrite any previously-existing logfile
    for the given reduction.  

    This function also configures the log file to provide timestamps for 
    all entries.  
    
    Parameters:
        log_dir   string        Path to log file.
        log_name  string        Name of log object and file
        version   string        [optional] Stage code version string
    Returns:
        log       open logger object
    """
    
    # Console output not captured, though code remains for testing purposes
    console = False

    log_path = os.path.join(log_dir, log_name+'.log')
    if os.path.isfile(log_path) == True:
        os.remove(log_path)
        
    # To capture the logging stream from the whole script, create
    # a log instance together with a console handler.  
    # Set formatting as appropriate.
    log = logging.getLogger( log_name )
    
    if len(log.handlers) == 0:
        log.setLevel( logging.INFO )
        file_handler = logging.FileHandler( log_path )
        file_handler.setLevel( logging.INFO )
        
        if console == True:
            console_handler = logging.StreamHandler()
            console_handler.setLevel( logging.INFO )
    
        formatter = logging.Formatter( fmt='%(asctime)s %(message)s', \
                                    datefmt='%Y-%m-%dT%H:%M:%S' )
        file_handler.setFormatter( formatter )

        if console == True:        
            console_handler.setFormatter( formatter )
    
        log.addHandler( file_handler )
        if console == True:            
            log.addHandler( console_handler )
    
    log.info( 'Started simulation run\n')
    if version != None:
        log.info('  Software version: '+version+'\n')
        
    return log
    

def stop_log(log):
    """Function to cleanly shutdown logging functions with a final timestamped
    entry.
    Parameters:
        log     logger Object
    Returns:
        None
    """
    
    if log:
        
        log.info( 'Processing complete\n' )

        logging.shutdown()

def get_new_fig_num():
    """Function to return a currently-unused python figure index, 
    to avoid overlapping with default pyLIMA products"""
    
    plt_num = -1
    
    i = 1
    
    while i <= 11 and plt_num < 0:
        
        if plt.fignum_exists(i):

            i += 1

        else:

            plt_num = i
            
    return plt_num

def get_args():
    """Function to request or read the required commandline arguments"""
    
    if len(sys.argv) > 1:
        
        config_file = sys.argv[1]
    
    else:
        
        config_file = raw_input('Please enter the path to the configuration file: ')
    
    return config_file

def parse_config_file(config_file):
    """Function to parse the contents of the configuration file"""
    
    if os.path.isfile(config_file) == False:
        
        msg = 'ERROR: Cannot find input file '+config_file
        print(msg)
        
        sys.exit()
    
    default_params = {}
    
    file_lines = open(config_file).readlines()
    
    for line in file_lines:
        
        if line[0:1] != '#':
            
            entries = line.replace('\n','').split('::')
            
            key = entries[0].replace(' ','').replace('\t','')
            
            value = ''.join(entries[1:])
            value = value.strip()

            if not key in ['name', 'model_type', 'horizons_file', 
                           'output_path', 'fit_models', 'plots']:
                value = float(value)
            
            if key in ['fit_models', 'plots']:
                value = bool(int(value))
                
            default_params[key] = value

    if 's' in default_params.keys():
        
        default_params['logs'] = np.log10(default_params['s'])

    if 'q' in default_params.keys():
        
        default_params['logq'] = np.log10(default_params['q'])
  
    source_mag_range = np.arange(default_params['source_mag_bright'], 
                                 default_params['source_mag_faint'],
                                 default_params['source_mag_incr'])
    
    tE_range = np.arange(default_params['tE_min_days'], 
                         default_params['tE_max_days'],
                         default_params['tE_incr_days'])
        
    return default_params, source_mag_range, tE_range

def record_config(default_params,source_mag_range, tE_range,log):
    """Function to ensure a written record in the log of the original
    parameters of the simulation"""
    
    log.info('Parameters input:')
    for key, value in default_params.items():
        log.info(key+': '+repr(value))

    log.info('Simulating for range of parameter grid:')
    log.info('Source mag range: '+repr(source_mag_range))
    log.info('Einstein crossing time: '+repr(tE_range))



if __name__ == '__main__':

    # Model_type options are: PSPL, FSPL, USBL
    simulate_NH_lensing()
