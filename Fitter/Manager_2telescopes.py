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


def main():
   

    #Events_path=  '/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Lightcurves_Ground/Lightcurves/'
    Events_path='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Lightcurves_Ground_2telescopes/Lightcurves/'
    Events_names=[i for i in os.listdir(Events_path) if 'Lightcurve_0_0.dat' in i]
 
    Events=[]
    start=time.time()
    Results=[]
    Errors=[]
    
    for i in Events_names :
        
        event=Event.Event()
        event.name=i
        
        
        for j in xrange(2):
            
            Tel=Telescopes.Telescope()
            if j==0:
                
                Tel.name='Simulation_'+str(j)+''
                Tel.lightcurve=np.loadtxt(Events_path+i)
                event.telescopes.append(Tel)
            else :
                Tel.name='Simulation_'+str(j)+''
                Tel.lightcurve=np.loadtxt(Events_path+'Lightcurve_0_1.dat')
                event.telescopes.append(Tel)
             
        Events.append(event)
        event.fit('PSPL',0)
        

        import pdb; pdb.set_trace()
        Results.append([event.name]+event.fit_results[0][2])
        Errors.append([event.name]+event.fit_errors[0][2])
        
    end=time.time()
    print end-start
    Reresults=np.array(Results) 
    EErrors=np.array(Errors) 
    np.savetxt('Fits_Ground.txt',Reresults,fmt="%s")
    np.savetxt('Fits_Error_Ground.txt',EErrors,fmt="%s")
if __name__=='__main__':
    
    main()