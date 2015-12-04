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

import event
import telescopes

location = 'Space'
model = 'PSPL'
second_order = [['None', 2457164.6365], ['None', 0], 'None']


def main(events_path, command_line):

    events_names = [event_name for event_name in os.listdir(events_path) if '.dat' in event_name]

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

    for event_name in events_names[6:10]:
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
        event_telescopes = [event_telescope for event_telescope in os.listdir(events_path) if name in event_telescope]
        # event_telescopes=['OGLE-2015-BLG-1577.phot','MOA-2015-BLG-363.phot']
        # event_telescopes=['MOA-2015-BLG-363.phot']
        for event_telescope in event_telescopes:
            # import pdb; pdb.set_trace()

            k = event_telescope.partition(name)
            telescope = telescopes.Telescope()
            # telescope.name=k[-1][:-4]
            # telescope.name=k[1]
            # telescope.name=event_telescope[:4]
            # telescope.name=k[0]
            telescope.name = k[-1]
            telescope.lightcurve = np.genfromtxt(events_path + event_telescope, usecols=(0, 1, 2))

            telescope.lightcurve_in_flux()
            telescope.filter = 'I'
            telescope.find_gamma(5300.0, 4.5)
            current_event.telescopes.append(telescope)

        events.append(current_event)
        print 'Start;', current_event.name
        # import pdb; pdb.set_trace()
        # current_event.check_event()
        current_event.find_survey('Survey')
        current_event.check_event()
        current_event.fit(model, 0, second_order)
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
        if model == 'PSPL':
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

        if model == 'FSPL':
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
    reresults = np.array(results)
    eerrors = np.array(errors)
    ssource = np.array(source)
    bblend = np.array(blend)
    esource = np.array(source_error)
    eblend = np.array(blend_error)
    TTime = np.array(time_fit)
    # import pdb; pdb.set_trace()

    np.savetxt('/home/mnorbury/Microlensing/Fits_' + location + '.txt', reresults, fmt="%s")
    np.savetxt('/home/mnorbury/Microlensing/Fits_' + location + '_Error.txt', eerrors, fmt="%s")
    np.savetxt('/home/mnorbury/Microlensing/Fits_' + location + '_Source.txt', ssource, fmt="%s")
    np.savetxt('/home/mnorbury/Microlensing/Fits_' + location + '_Blend.txt', bblend, fmt="%s")
    np.savetxt('/home/mnorbury/Microlensing/Fits_' + location + '_Source_errors.txt', esource, fmt="%s")
    np.savetxt('/home/mnorbury/Microlensing/Fits_' + location + '_Blend_errors.txt', eblend, fmt="%s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--location', default='FSPL')
    parser.add_argument('-m', '--model', default='PSPL')
    parser.add_argument('-i', '--input_directory',
                        default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Lightcurves_')
    parser.add_argument('-o', '--output_directory',
                        default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/')
    arguments = parser.parse_args()

    location = arguments.location
    model = arguments.model
    input_directory = arguments.input_directory

    path = input_directory + location + '/Lightcurves/'
    main(path, arguments)
