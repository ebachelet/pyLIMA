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

import numpy as np
from Fitter import microlmodels

import event
import telescopes

second_order = [['None', 2457027], ['None', 0], ['None', 0], 'None']


def main(command_line):
    events_names = [event_name for event_name in os.listdir(command_line.input_directory) if
                    '.dat' in event_name]
    events_names2 = [event_name for event_name in events_names if 'Survey' in event_name]

    start = time.time()
    results = []
    errors = []
    for event_name in events_names2[:]:
        name = event_name[:-10]
        current_event = event.Event()
        current_event.name = name
        current_event.ra = 269.39166666666665
        current_event.ra = 79.17125
        current_event.dec = -29.22083333333333
        event_telescopes = [event_telescope for event_telescope in events_names if
                            name in event_telescope]

        for event_telescope in event_telescopes:
            try:
                raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope,
                                                usecols=(0, 1, 2))
            except:
                import pdb;
                pdb.set_trace()

            telescope = telescopes.Telescope(name=event_telescope[-10:-4], camera_filter='I',
                                             light_curve=raw_light_curve)

            telescope.gamma = 0.5
            if 'swift' in event_telescope:
                telescope.kind = 'Space'

            else:

                current_event.telescopes.append(telescope)
        print 'Start;', current_event.name

        current_event.check_event()
        current_event.find_survey('Survey')

        Model = microlmodels.MLModels(current_event, model, second_order)

        if second_order[0][0]!='None':
            current_event.compute_parallax(second_order)
        current_event.fit(Model, 1)

        Results = []
        Errors = []

        Results.append(current_event.name)
        Errors.append(current_event.name)
        Results.append(current_event.fits[0].method)
        Errors.append(current_event.fits[0].method)

        Results += current_event.fits[0].fit_results
        Errors += (current_event.fits[0].fit_covariance.diagonal() ** 0.5).tolist()
        Results.append(current_event.fits[0].fit_time)
        results.append(Results)
        errors.append(Errors)
    import pdb;
    pdb.set_trace()
    end = time.time()

    print end - start

    all_results = [('Fits.txt', results),
                   ('Fits_Error.txt', errors)]

    for file_name, values in all_results:
        np.savetxt(os.path.join(command_line.output_directory, file_name), np.array(values),
                   fmt="%s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='FSPL')
    parser.add_argument('-i', '--input_directory',
                        default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing'
                                '/OpenSourceProject/SimulationML/Lightcurves_FSPL/Lightcurves/')
    parser.add_argument('-o', '--output_directory',
                        default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing'
                                '/OpenSourceProject/Developement/Fitter/FSPL/')
    parser.add_argument('-c', '--claret',
                        default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing'
                                '/OpenSourceProject/Claret2011/J_A+A_529_A75/')
    arguments = parser.parse_args()

    model = arguments.model

    main(arguments)
