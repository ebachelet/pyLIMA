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

# location = 'Space'
# model = 'PSPL'
second_order = [['None', 2457164.6365], ['None', 0], ['None', 0], 'None']


def main(command_line):
    events_names = [os.path.split(x)[1] for x in glob.glob(command_line.input_directory + '/*.dat')]
    #print 'event_names = ', events_names

    # events_names=[event_name for event_name in os.listdir(events_path) if '.dat' in event_name]
    # EEvents_names=[event_name for event_name in os.listdir(events_path) if '.phot' in event_name]
    events = []
    start = time.time()
    results = []
    errors = []
    source = []
    blend = []
    source_error = []
    blend_error = []
    models = []
    time_fit = []
    events_names=[i for i in events_names if '.dat' in i]
    for event_name in events_names[8191:]:
       
        name=event_name[:-4]

        current_event = event.Event()
        current_event.name = name
        current_event.ra = 270.65404166666667
        current_event.dec = -27.721305555555553
        event_telescopes = [event_telescope for event_telescope in os.listdir(command_line.input_directory) if
                            name in event_telescope]
      
        filters = ['I','J']    
        count=0
        for event_telescope in event_telescopes:

            
            raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope, usecols=(0, 1, 2))
            telescope = telescopes.Telescope(name=event_name[:-4], camera_filter='I', light_curve=raw_light_curve)
            # telescope.name=k[-1][:-4]
            # telescope.name=k[1]
            # telescope.name=event_telescope[:4]
            # telescope.name=k[0]
            telescope.filter = filters[count]            
            telescope.find_gamma(5300.0, 4.5, command_line.claret)
            current_event.telescopes.append(telescope)
            count = count+1
        events.append(current_event)
        print 'Start;', current_event.name
        # import pdb; pdb.set_trace()
        # current_event.check_event()
        current_event.find_survey(current_event.telescopes[0].name)
        current_event.check_event()

        current_event.fit(command_line.model, second_order,2)
        
       
        current_event.produce_outputs(0)
#        import pdb; pdb.set_trace()
        Results = []
        Errors = []
        
        Results.append(current_event.name)
        Errors.append(current_event.name)
        Results.append(current_event.fits[0].method)
        Errors.append(current_event.fits[0].method)
        for i in current_event.fits[0].model.model_dictionnary.keys() :
            
            Results.append(current_event.fits[0].fit_results[current_event.fits[0].model.model_dictionnary[i]])
            Errors.append(current_event.fits[0].fit_errors[current_event.fits[0].model.model_dictionnary[i]])
            
        Results.append(current_event.fits[0].fit_results[-1])
        Results.append(current_event.fits[0].fit_time) 
        results.append(Results)
        errors.append(Errors)
       
    end = time.time()

    print end - start

    all_results = [('Fits.txt', results),
                   ('Fits_Error.txt', errors)]

    for file_name, values in all_results:
        np.savetxt(os.path.join(command_line.output_directory, file_name), np.array(values), fmt="%s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='PSPL')
    parser.add_argument('-i', '--input_directory', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Lightcurves_Ground/Lightcurves/')
    parser.add_argument('-o', '--output_directory', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/Ground/')
    parser.add_argument('-c', '--claret', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Claret2011/J_A+A_529_A75/')
    arguments = parser.parse_args()

    model = arguments.model

    main(arguments)
