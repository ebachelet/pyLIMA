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
import Event
import Telescopes

Location='OGLE_2015'
Model='FSPL'
def main(path):
    Events_path = path

    
    Events_names=[i for i in os.listdir(Events_path) if '.phot' in i]
 
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
    #import pdb; pdb.set_trace()
    for i in Events_names[0:] :
        i='OGLE-2015-BLG-0058.phot'
        #i='Lightcurve_41_Survey.dat'
        name=i.replace('Survey.dat','')
        event=Event.Event()
        event.name=name
        tels=[i for i in os.listdir(Events_path) if name in i]
        for j in tels :
           
            k=j.partition(name)
            Tel=Telescopes.Telescope()
            Tel.name=k[-1][:-4]
            Tel.lightcurve=np.genfromtxt(Events_path+j,usecols = (0,1,2))
            Tel.clean_data()
            Tel.filter='I'
            Tel.find_gamma(5300.0,4.5)
            event.telescopes.append(Tel)
       
        Events.append(event)
        print 'Start;',event.name
        event.fit(Model,0)
        #import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()
        if Model=='PSPL' :
            
            Results.append([event.name,event.fit_results[1],event.fit_results[2][0],event.fit_results[2][1],event.fit_results[2][2],event.fit_results[2][-1],event.fit_time[1]])
        
            Errors.append([event.name]+np.sqrt(np.diagonal(event.fit_covariance[2]))[:3].tolist())
            Source.append([event.name,event.fit_results[2][3]])
            Blend.append([event.name,event.fit_results[2][4]])
            Source_err.append([event.name,np.sqrt(np.diagonal(event.fit_covariance[2]))[3]])
            Blend_err.append([event.name,np.sqrt(np.diagonal(event.fit_covariance[2]))[4]])
        
        if Model=='FSPL':

            
            Results.append([event.name,event.fit_results[1],event.fit_results[2][0],event.fit_results[2][1],event.fit_results[2][2],event.fit_results[2][3],event.fit_results[2][-1],event.fit_time[1]])
        
            Errors.append([event.name]+np.sqrt(np.diagonal(event.fit_covariance[2]))[:4].tolist())
            Source.append([event.name,event.fit_results[2][4],event.fit_results[2][6]])
            Blend.append([event.name,event.fit_results[2][5],event.fit_results[2][5]])
            Source_err.append([event.name,np.sqrt(np.diagonal(event.fit_covariance[2]))[4],np.sqrt(np.diagonal(event.fit_covariance[2]))[6]])
            Blend_err.append([event.name,np.sqrt(np.diagonal(event.fit_covariance[2]))[5],np.sqrt(np.diagonal(event.fit_covariance[2]))[7]])
        
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

    np.savetxt('./'+Location+'/Fits_'+Location+'.txt',Reresults,fmt="%s")
    np.savetxt('./'+Location+'/Fits_'+Location+'_Error.txt',EErrors,fmt="%s")
    np.savetxt('./'+Location+'/Fits_'+Location+'_Source.txt',SSource,fmt="%s")
    np.savetxt('./'+Location+'/Fits_'+Location+'_Blend.txt',BBlend,fmt="%s")
    np.savetxt('./'+Location+'/Fits_'+Location+'_Source_errors.txt',ESource,fmt="%s")
    np.savetxt('./'+Location+'/Fits_'+Location+'_Blend_errors.txt',EBlend,fmt="%s")
if __name__=='__main__':
    path=  '/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Lightcurves_'+Location+'/Lightcurves/'
    main(path)