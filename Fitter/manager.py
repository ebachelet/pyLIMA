# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:17:30 2015

@author: ebachelet
"""

###############################################################################

#General code manager

###############################################################################

import numpy as np
import os
import time
import matplotlib.pyplot as plt

import event
import telescopes

Location='FSPL'
Model='FSPL'
second_order=[['None',2457164.6365],['None',0],'None']
def main(path):
    Events_path = path

    
    Events_names=[i for i in os.listdir(Events_path) if 'Survey' in i]
    #EEvents_names=[i for i in os.listdir(Events_path) if '.phot' in i]
    Events=[]
    start=time.time()
    Results=[]
    Errors=[]
    Source=[]
    Blend=[]
    Source_err=[]
    Blend_err=[]
    Models=[]
    time_fit=[]
    for i in Events_names[0:] :
        #i='OGLE-2015-BLG-0542.phot'
        #i='Lightcurve_51.dat'
        name=i.replace('Survey.dat','')
        #name=i.replace('.dat','')
        #name='.phot'
        #name=i
        Event=event.Event()
        Event.name=name
        Event.ra=270.65404166666667
        Event.dec=-27.721305555555553
        tels=[j for j in os.listdir(Events_path) if name in j]
        #tels=['OGLE-2015-BLG-1577.phot','MOA-2015-BLG-363.phot']
        #tels=['MOA-2015-BLG-363.phot']
        for j in tels :
            #import pdb; pdb.set_trace()
            #import pdb; pdb.set_trace()

            k=j.partition(name)
            Tel=telescopes.Telescope()
            Tel.name=k[-1][:-4]
            #Tel.name=k[1]
            #Tel.name=j[:4]
            #Tel.name=k[0]
            Tel.lightcurve=np.genfromtxt(Events_path+j,usecols = (0,1,2))
            
            
            Tel.lightcurve_in_flux()
            Tel.filter='I'
            Tel.find_gamma(5300.0,4.5)
            Event.telescopes.append(Tel)
   
        Events.append(Event)
        print 'Start;',Event.name
        import pdb; pdb.set_trace()
        #Event.check_event()
        Event.find_survey('Survey')
        Event.check_event()
        Event.fit(Model,0,second_order)
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        if Model=='PSPL' :
            
            Results.append([Event.name,Event.fits_results[0][1],Event.fits_results[0][2][0],Event.fits_results[0][2][1],Event.fits_results[0][2][2],Event.fits_results[0][2][-1],Event.fits_time[0][2]])
        
            Errors.append([Event.name]+np.sqrt(np.diagonal( Event.fits_covariance[0][2]))[:3].tolist())
            Source.append([Event.name,Event.fits_results[0][2][3]])
            Blend.append([Event.name,Event.fits_results[0][2][4]])
            Source_err.append([Event.name,np.sqrt(np.diagonal(Event.fits_covariance[0][2]))[3]])
            Blend_err.append([Event.name,np.sqrt(np.diagonal(Event.fits_covariance[0][2]))[4]])
        
        if Model=='FSPL':

            
            Results.append([Event.name,Event.fits_results[0][1],Event.fits_results[0][2][0],Event.fits_results[0][2][1],Event.fits_results[0][2][2],Event.fits_results[0][2][3],Event.fits_results[0][2][-1],Event.fits_time[0][2]])
        
            Errors.append([Event.name]+np.sqrt(np.diagonal( Event.fits_covariance[0][2]))[:4].tolist())
            Source.append([Event.name,Event.fits_results[0][2][4]])
            Blend.append([Event.name,Event.fits_results[0][2][5]])
            Source_err.append([Event.name,np.sqrt(np.diagonal(Event.fits_covariance[0][2]))[4]])
            Blend_err.append([Event.name,np.sqrt(np.diagonal(Event.fits_covariance[0][2]))[5]])
        
    end=time.time()
    print end-start
    Reresults=np.array(Results) 
    EErrors=np.array(Errors) 
    SSource=np.array(Source) 
    BBlend=np.array(Blend) 
    ESource=np.array(Source_err)
    EBlend=np.array(Blend_err)
    TTime=np.array(time_fit)
    import pdb; pdb.set_trace()

    np.savetxt('/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/'+Location+'/Fits_'+Location+'.txt',Reresults,fmt="%s")
    np.savetxt('/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/'+Location+'/Fits_'+Location+'_Error.txt',EErrors,fmt="%s")
    np.savetxt('/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/'+Location+'/Fits_'+Location+'_Source.txt',SSource,fmt="%s")
    np.savetxt('/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/'+Location+'/Fits_'+Location+'_Blend.txt',BBlend,fmt="%s")
    np.savetxt('/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/'+Location+'/Fits_'+Location+'_Source_errors.txt',ESource,fmt="%s")
    np.savetxt('/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/'+Location+'/Fits_'+Location+'_Blend_errors.txt',EBlend,fmt="%s")
if __name__=='__main__':
    path=  '/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Lightcurves_'+Location+'/Lightcurves/'
    main(path)