# -*- coding: utf-8 -*-
"""
Created on Wed 30 Sep 2020

@author: martindominik 
"""

from __future__ import division

import sys
import copy
import numpy as np

# TBF: DUMMY function
def calculate_flux(event,time,tel_idx):
    return 100.0
    

def raise_flags(deviations, DEV_SIG_OLD, DEV_SIG, THRESH_CHISQ):
    """
    Raise deviation assessment flags according to the deviations and return as list
    
    deviations is list of _Deviation objects with the calculated deviations with respect to the old model and the new model
    
    flags[0]:   |sigscaled[0]| > DEV_SIG_OLD (old model)
    flags[1]:   |sigscaled[1]| > DEV_SIG  (new model)
    flags[2]:   Delta chi^2 > THRESH_CHISQ  (this option is currently not implemented)
    flags[3]:   |sigscaled[0]| > crit_dev[0] (old model)
    flags[4]:   |sigscaled[1]| > crit_dev[1] (new model)
    """
    
    flags = np.zeros(5,dtype='Bool')
    
    if abs(deviations[0].sigscaled) > DEV_SIG_OLD:
        flags[0] = True
    
    if abs(deviations[1].sigscaled) > DEV_SIG:
        flags[1] = True
         
    # if delta_chisq > THRESH_CHISQ:
    #     flags[2] = True
    
    if abs(deviations[0].sigscaled) > deviations[0].crit_dev:
        flags[3] = True
    
    if abs(deviations[1].sigscaled) > deviations[1].crit_dev:
        flags[4] = True
         
    return(flags)
    
    
def make_decision(flags, DEV_CRIT, DEV_MEASURE):
    """
    Flag up anomaly based on flagging criteria
    
    Returns anomaly flag as boolean
    """
    
    if DEV_CRIT == 'new_only':
        if DEV_MEASURE == 'percentile':
            if flags[4]:
                return True
        elif DEV_MEASURE == 'median_sigma':
            if flags[1]:
                return True
        else:
            if flags[1] and flags[4]:
                return True
    elif DEV_CRIT == 'rattenbury':
        if DEV_MEASURE == 'percentile':
            if flags[3] and (flags[4] or flags[2]):
                return True
        elif DEV_MEASURE == 'median_sigma':
            if flags[0] and (flags[1] or flags[2]):
                return True
        else:
            if (flags[0] and (flags[1] or flags[2])) and (flags[3] and (flags[4] or flags[2])):
                return True
    
    return False
    

class _Deviation(object):
    """
    Groups evaluation of deviation of a point
    
    FLOAT anomaly_level;   (replaces "del", which is difference in magnification)
    FLOAT sig_anomaly_level;
    FLOAT sig;
    FLOAT sigscaled;
    FLOAT crit_dev;
    FLOAT reject_dev;
    """
    
    def __init__(self, anomaly_level, sig_anomaly_level, sig, sigscaled, crit_dev, crit_rej):
        self.anomaly_level = anomaly_level
        self.sig_anomaly_level = sig_anomaly_level
        self.sig = sig
        self.sigscaled = sigscaled
        self.crit_dev = crit_dev
        self.crit_rej = crit_rej
        

class _DataPoint(object):
    """
    Holds information on a specific data point, corresponds to struct ANOMALY_LIST in signalmen.c
    
    Grouped in data_flux:
        FLOAT time;
        FLOAT mag;
        FLOAT err;
        FLOAT seeing;  (not implemented in pyLIMA)
        FLOAT backgnd; (not implemented in pyLIMA)
        FLOAT airmass; (not implemented in pyLIMA)
        FLOAT exptime; (not implemented in pyLIMA)
        int type;      (not implemented in pyLIMA)
        
    tel_idx:
        int arch_index;
        
    tel_name:
        char extsite[256];
        
    char filter;
 
    Grouped in deviation
        FLOAT anomaly_level;
        FLOAT sig_anomaly_level;
        FLOAT sig;
        FLOAT sigscaled;
        FLOAT crit_dev;
        FLOAT crit_rej;
    """
    
    def __init__(self,anom_status,data_idx,deviations=[]):
        self.data_idx = data_idx
        self.data_flux = anom_status._alldata[data_idx,0:3]
        self.tel_idx = anom_status._alldata[data_idx,5].astype(int)
        self.tel_name = anom_status.event.telescopes[self.tel_idx].name
        self.filter = anom_status.event.telescopes[self.tel_idx].filter
        self.deviations = deviations
        # TBF: Question: Should we store fluxes or magnitudes here -> see "signalmen.c" for details


class AnomalyStatus(object):
    """
    ######## AnomalyStatus class ########

    This class defines an AnomalyStatus object holding the assessment of an event for an ongoing anomaly 
    and providing methods to carry out the assessment
    
    It contains the complete SIGNALMEN management
    and builds on the Event class for fitting models to event data.

    Attributes :

         filename : specifies master file name for various input/output files to control SIGNALMEN
                      and log assessment

         alldata : numpy array of full photometric data sorted in time sequence

         event : instance of Event class with all data (old and new)

         old_event : instance of Event class with previous data and model

         eventNew : instance of Event class with data and model corresponding to current assessment step

         status : SIGNALMEN status (0 = ordinary, 1 = check, 2 = anomaly) after assessment

         prev_status : SIGNALMEN status (0 = ordinary, 1 = check, 2 = anomaly) before assessment 
 
	 time_prev : epoch of previous assessment (new assessment starts after this epoch)

	 [Some other attributes for management required, maybe bundle in one or several classes, e.g.
             list of anomalous points
             list of specific points to be excluded for modelling]

	 [For specific fits, data has to be copied from "alldata" to the telescopes attribute of "event_curr"
             taking into account the time range and further exclusion criteria]
  
         ["alldata" initially has to be populated with data stored in "event_all" and sorted in time sequence]
         
     unsigned long ndata; /* data to be considered for fit (0..ndata-1) */
     unsigned long totdata; /* total # data */
     unsigned long olddata; /* total # data previous run */
     
     ANOMALY_LIST_PTR exclude_list;
     boolean include_modelling;
     unsigned long include_Idx;
     

         
    """

    def __init__(self,event):
        """ Initialization of the attributes described above. """

        self.event = event
        # build array of all data, containing time, flux, flux_err, mag, mag_err, telescope_index, [deltas_north, deltas_east]
        # include [deltas_north, deltas_east] only if these are provided in the deltas_position array of the telescope object
        alldata_list = []
        tel_idx = 0
        if isinstance(self.event.telescopes[0].deltas_positions,list):
            par_deltas_exist = self.event.telescopes[0].deltas_positions
        else:
            par_deltas_exist = self.event.telescopes[0].deltas_positions.size != 0
             # check whether parallax offsets have been calculated according to the model type
        for telescope in self.event.telescopes:
            dlen = len(telescope.lightcurve_flux)
            if par_deltas_exist:
                lightcurve = np.c_[telescope.lightcurve_flux,telescope.lightcurve_magnitude[:,1:3],np.full(dlen,tel_idx),
                    telescope.deltas_positions[0,:],telescope.deltas_positions[1,:]]
            else:
                lightcurve = np.c_[telescope.lightcurve_flux,telescope.lightcurve_magnitude[:,1:3],np.full(dlen,tel_idx)]
            alldata_list.append(lightcurve)
            tel_idx += 1
        self._alldata = np.concatenate(alldata_list)
        # sort array with all data in time sequence
        # do not use quicksort given that array is partially presorted
        self._alldata = self._alldata[self._alldata[:,0].argsort(kind='heapsort')]
        self._totdata = len(self._alldata)  # Total number of data points
        
        
    def _filter_data(self, this_event, ndata, exclude_list):
        """ Filter event data from self._alldata to this_event.telescopes
        
        (please explain parameters)
        
        """
        
        # extract data indices from exclude_list
        if exclude_list:
            exclude_indices = [datapoint.data_idx for datapoint in exclude_list]
            mask = np.ones(len(self._alldata),dtype=bool)
            mask[exclude_indices] = False
            selected_data = self._alldata[mask][0:ndata,:]
        else:
            selected_data = self._alldata[0:ndata,:]
            
        if isinstance(self.event.telescopes[0].deltas_positions,list):
            par_deltas_exist = self.event.telescopes[0].deltas_positions
        else:
            par_deltas_exist = self.event.telescopes[0].deltas_positions.size != 0
             # check whether parallax offsets have been calculated according to the model type
        tel_idx = 0
        for telescope in this_event.telescopes:
            selection = selected_data[selected_data[:,5].astype(int) == tel_idx]
            telescope.lightcurve_flux = selection[:,0:3]
            if par_deltas_exist:
                telescope.lightcurve_magnitude = selection[:,np.array([True,False,False,True,True,False,False,False])]
                telescope.deltas_positions = np.array([selection[:,6],selection[:,7]])
            else:
                telescope.lightcurve_magnitude = selection[:,np.array([True,False,False,True,True,False])]
            tel_idx += 1


    def test_point(self, data_idx, event_list, DEV_PERC, REJECT_PERC):
        """ determine deviation of data point with index "data_idx" with respect to models provided in "event_list"
        """
        
        deviation_list = []
        for event in event_list:
            time = self._alldata[data_idx,0]
            flux = self._alldata[data_idx,1]
            errflux = self._alldata[data_idx,2]
            tel_idx = self._alldata[data_idx,5].astype(int)
            flux_model = calculate_flux(event,time,tel_idx)  # TBF: specify model rather than event ???? -> most recent model ???
                # TBF: calculate_flux function is missing !!!!
            
            delta = flux-flux_model
            sig = delta/errflux
            
            # get f_source from model parameters
            pyLIMA_parameters = self.event.fits[-1].model.compute_pyLIMA_parameters(self.event.fits[-1].fit_results[:-1])
            f_source = getattr(pyLIMA_parameters, 'fs_' + self.new_event.telescopes[tel_idx].name)
            if self.event.fits[-1].model.blend_flux_ratio:
                f_blend = f_source * getattr(pyLIMA_parameters, 'g_' + self.new_event.telescopes[tel_idx].name)
            else:
                f_blend = getattr(pyLIMA_parameters, 'fb_' + self.new_event.telescopes[tel_idx].name)
            
            anomaly_level = delta / (flux_model-f_blend)
                # this is equal to the relative difference in magnification [A_i - A(t_i)]/A(t_i)
                
            sig_anomaly_level = errflux / (flux_model-f_blend)
            
            # calculate median scatter and percentiles
            percentiles = event.fits[-1].medscattdev_tel(event.telescopes[tel_idx], [DEV_PERC, REJECT_PERC])
            median = percentiles[0]
            crit_dev = percentiles[1]
            crit_rej = percentiles[2]
            
            sig_scaled = sig
            if (median > 1.0):
                sig_scaled /= median
                sig_anomaly_level *= median
                
            deviation = _Deviation(anomaly_level, sig_anomaly_level, sig, sig_scaled, crit_dev, crit_rej)
            deviation_list.append(deviation)
            
        return deviation_list


    def assess(self, model, method, start_time, DE_population_size=10, flux_estimation_MCMC='MCMC', fix_parameters_dictionnary=None,
            grid_resolution=10, computational_pool=None, binary_regime=None,
            robust=True, DEV_SIG=2.0, DEV_PERC=0.05, REJECT_SIG=2.0, REJECT_PERC=0.05, DEV_SIG_OLD=3.0,  THRESH_CHISQ=20.0,
            DEV_CRIT='new_only', DEV_MEASURE='median_and_percentile', REJECT_MEASURE='median_sigma',
            MIN_DATA_FIT=3, MIN_DATA_TEST=6, MIN_NIGHTS=2, MINPTS_REJECT=4, MINPTS_ANOMALY=5, MINPTS_ALL_ANOM=3,
            MAX_DAYS_BACK_ASSESS=1.5):
        """ Method to assess an event for an ongoing anomaly
        
        start_time  Start assessment after start_time
        
        MAX_DAYS_BACK_ASSESS    max number of days stepped back for assessment
        MIN_DATA_FIT            number of data points required for fit
        MIN_DATA_TEST           number of data points required for test
        MIN_NIGHTS              number of nights required
        MINPTS_REJECT           number of non-anomalous points that lead to rejection of anomaly
        MINPTS_ANOMALY          number of deviant points that create anomaly alert
        MINPTS_ALL_ANOM=3       number of deviant points for all-data model
        
        please see Dominik et al. 2007, MNRAS 380, 792 a description of the SIGNALMEN assessment

        [add details...]

        """
        
        self.old_event = copy.deepcopy(self.event)  # copy from Event object provided as argument

        # get index of first data point AFTER start_time, self._alldata is sorted in time order
        next_idx = self._alldata[:,0].searchsorted(start_time, side='right')
        
        # TBF: properly establish old model, current code is PLACEHOLDER only...
        self._filter_data(self.old_event, next_idx-1, [])  # TBF: check range
        old_model = copy.copy(model)
        old_model.event = self.old_event
        
        self.old_event.fit(old_model, method, robust=robust)
           
        # initialise new_event and use old model as starting point
        self.new_event = copy.deepcopy(self.old_event)
           
        while next_idx < self._totdata:   # main loop to step through data points
            
            # TBF: skip data points for assessment if there is an insufficient number for this telescope
            
            # include new data point(s) in determining new model
            self._filter_data(self.new_event, next_idx,[])
            new_model = copy.copy(old_model)    # copies in particular previous fit parameters
            new_model.event = self.new_event
            
            self.new_event.fit(new_model, method, use_last_as_guess=True, robust=robust)
            
            deviations = self.test_point(next_idx,[self.old_event,self.new_event], DEV_PERC, REJECT_PERC)
                # assess data points with regard to either model
                
            flags = raise_flags(deviations, DEV_SIG_OLD, DEV_SIG, THRESH_CHISQ)
            anomaly_flag = make_decision(flags, DEV_CRIT, DEV_MEASURE)
            
            if anomaly_flag:
            
                # new anomaly sequence has been detected
                
                # create list of anomalous data points
                this_anomaly = _DataPoint(self,next_idx, deviations)
                this_anomaly_list = [this_anomaly]
            
            next_idx += 1
            
            break
            
        # Note: pyLIMA needs to be enabled to support fits with "insufficient" data -> provide defined result
        #           [see SIGNALMEN recipe]
        #       check both the inclusion of data in fit, and providing well-defined fit result


