# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:17:30 2015

@author: ebachelet
"""

###############################################################################

# General code manager

###############################################################################

import os
import time
import glob
import matplotlib.pyplot as plt

import numpy as np

import event
import telescopes
import microlmodels



def main(command_line):
   
    events_names=[event_name for event_name in os.listdir(command_line.input_directory) if ('.dat'  in event_name) and ('Follow' not in event_name)]
   
    events = []
    start = time.time()
    results = []
    errors = []
   
    
    for event_name in events_names[0:]:

        name='Lightcurve_'+str(8366)+'_'
      
        current_event = event.Event()
        current_event.name = name
        current_event.ra = 269.39166666666665 
        current_event.dec = -29.22083333333333
        event_telescopes = [i for i in os.listdir(command_line.input_directory) if name  in i]
        #import pdb; pdb.set_trace()

        count=0
     
        start=time.time()
        for event_telescope in event_telescopes:
            try :
               raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope, usecols=(0, 1, 2))
               
               lightcurve=np.array([raw_light_curve[:,0],raw_light_curve[:,1],raw_light_curve[:,2]]).T
               if lightcurve[0,0]>2450000 :
                   lightcurve[:,0] = lightcurve[:,0]-2450000
            except :
                pass
            telescope = telescopes.Telescope(name=event_telescope[2:-4], camera_filter='I', light_curve=lightcurve)
            telescope.gamma=0.5
            
            current_event.telescopes.append(telescope)
            count+=1
           
        print 'Start;', current_event.name
       
        current_event.find_survey('Survey')
   
        current_event.check_event()
       
        Model = microlmodels.MLModels(current_event, command_line.model, command_line.second_order)
        
        current_event.fit(Model,'MCMC')
        import pdb; pdb.set_trace()

        current_event.fits[0].produce_outputs()
       
        print time.time()-start

    end = time.time()

    print end - start

    all_results = [('Fits.txt', results),
                   ('Fits_Error.txt', errors)]

    for file_name, values in all_results:
        np.savetxt(os.path.join(command_line.output_directory, file_name), np.array(values), fmt="%s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='FSPL')
    parser.add_argument('-so', '--second_order', default=[['None', 0], ['None', 0], ['None', 0], 'None'])
    parser.add_argument('-i', '--input_directory', default='/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Lightcurves_FSPL/Lightcurves/') 
    parser.add_argument('-o', '--output_directory', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/FSPL/')
    parser.add_argument('-c', '--claret', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Claret2011/J_A+A_529_A75/')
    arguments = parser.parse_args()

    model = arguments.model

    main(arguments)
