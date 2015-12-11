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
    events_names=[i for i in events_names if 'Survey' in i]
    for event_name in events_names[7958:10000]:
        # event_name='OGLE-2015-BLG-3851.phot'
        # event_name='Lightcurve_3016.dat'
        name = event_name.replace('Survey.dat', '')
        # name=event_name.replace('.dat','')
        # name='.phot'
        # name=event_name

        current_event = event.Event()
        current_event.name = name
        current_event.ra = 270.65404166666667
        current_event.dec = -27.721305555555553
        event_telescopes = [event_telescope for event_telescope in os.listdir(command_line.input_directory) if
                            name in event_telescope]
        # event_telescopes=['OGLE-2015-BLG-1577.phot','MOA-2015-BLG-363.phot']
        # event_telescopes=['MOA-2015-BLG-363.phot']
                    
        for event_telescope in event_telescopes:

            k = event_telescope.partition(name)
            raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope, usecols=(0, 1, 2))
            telescope = telescopes.Telescope(name=k[-1], camera_filter='I', light_curve=raw_light_curve)
            # telescope.name=k[-1][:-4]
            # telescope.name=k[1]
            # telescope.name=event_telescope[:4]
            # telescope.name=k[0]
            telescope.find_gamma(5300.0, 4.5, command_line.claret)
            current_event.telescopes.append(telescope)

        events.append(current_event)
        print 'Start;', current_event.name
        # import pdb; pdb.set_trace()
        # current_event.check_event()
        current_event.find_survey('Survey')
        current_event.check_event()

        current_event.fit(command_line.model, second_order,0)
        import pdb; pdb.set_trace()
        current_event.produce_outputs()
        current_event.output.student_errors()
        current_event.plot_data('Mag')
        current_event.plot_model('PSPL', second_order, 'Mag')
        # tt=np.arange(min(current_event.telescopes[0].lightcurve[:,0]),max(current_event.telescopes[0].lightcurve[:,0]),0.01)
        # par=current_event.output.lower
        # uu=(par[1]**2+(tt-par[0])**2/par[2]**2)**0.5
        # aa=(uu**2+2)/(uu*(uu**2+4)**0.5)
        # plt.plot(tt,27.4-2.5*np.log10(par[3]*(aa+par[4])),'k--',lw=2)

        # par=current_event.output.upper
        # uu=(par[1]**2+(tt-par[0])**2/par[2]**2)**0.5
        # aa=(uu**2+2)/(uu*(uu**2+4)**0.5)
        # plt.plot(tt,27.4-2.5*np.log10(par[3]*(aa+par[4])),'k--',lw=2)

        # plt.show()
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        if command_line.model == 'PSPL':
            results.append(
                [current_event.name, current_event.fits_results[0][1], current_event.fits_results[0][2][0],
                 current_event.fits_results[0][2][1],
                 current_event.fits_results[0][2][2], current_event.fits_results[0][2][-1],
                 current_event.fits_time[0][2]])

            errors.append([current_event.name] + np.sqrt(np.diagonal(current_event.fits_covariance[0][2]))[:3].tolist())
            source.append([current_event.name, current_event.fits_results[0][2][3]])
            blend.append([current_event.name, current_event.fits_results[0][2][4]])
            source_error.append([current_event.name, np.sqrt(np.diagonal(current_event.fits_covariance[0][2]))[3]])
            blend_error.append([current_event.name, np.sqrt(np.diagonal(current_event.fits_covariance[0][2]))[4]])

        if command_line.model == 'FSPL':
            results.append(
                [current_event.name, current_event.fits_results[0][1], current_event.fits_results[0][2][0],
                 current_event.fits_results[0][2][1],
                 current_event.fits_results[0][2][2], current_event.fits_results[0][2][3],
                 current_event.fits_results[0][2][-1],
                 current_event.fits_time[0][2]])

            errors.append([current_event.name] + np.sqrt(np.diagonal(current_event.fits_covariance[0][2]))[:4].tolist())
            source.append([current_event.name, current_event.fits_results[0][2][4]])
            blend.append([current_event.name, current_event.fits_results[0][2][5]])
            source_error.append([current_event.name, np.sqrt(np.diagonal(current_event.fits_covariance[0][2]))[4]])
            blend_error.append([current_event.name, np.sqrt(np.diagonal(current_event.fits_covariance[0][2]))[5]])

    end = time.time()

    print end - start

    all_results = [('Fits.txt', results),
                   ('Fits_Error.txt', errors),
                   ('Fits_Source.txt', source),
                   ('Fits_Blend.txt', blend),
                   ('Fits_Source_errors.txt', source_error),
                   ('Fits_Blend_errors.txt', blend_error), ]

    for file_name, values in all_results:
        np.savetxt(os.path.join(command_line.output_directory, file_name), np.array(values), fmt="%s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='FSPL')
    parser.add_argument('-i', '--input_directory', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Lightcurves_FSPL/Lightcurves/')
    parser.add_argument('-o', '--output_directory', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/FSPL/')
    parser.add_argument('-c', '--claret', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Claret2011/J_A+A_529_A75/')
    arguments = parser.parse_args()

    model = arguments.model

    main(arguments)
