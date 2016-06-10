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

import matplotlib.pyplot as plt

import numpy as np

from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA import microlmodels


def main(command_line):
    events_names = [event_name for event_name in os.listdir(command_line.input_directory) if
                    ('Lightcurve_' in event_name) and ('Follow' not in event_name)]
    events_names2 = [event_name for event_name in os.listdir(command_line.input_directory) if
                     ('Lightcurve_' in event_name) and ('~' not in event_name)]

    import pdb
    pdb.set_trace()

    start = time.time()
    results = []
    errors = []

    for event_name in events_names[0:]:

        # name='Lightcurve_'+str(9975)+'_'
        name = event_name[:-10]
        # name = 'Lightcurve_1'
        current_event = event.Event()
        current_event.name = name

        event_telescopes = [i for i in events_names2 if name in i]
        # event_telescopes = ['OGLE-2016-BLG-0676.dat','MOA-2016-BLG-215_MOA_transformed.dat',
        # 'MOA-2016-BLG-215_transformed.dat']
        # event_telescopes = ['MOA-2016-BLG-215_transformed.dat']
        # Names = ['OGLE','Kepler']
        # Locations = ['Earth','Space']

        current_event.ra = 269.8865416666667
        current_event.dec = -28.407416666666666
        Names = ['Survey', 'Followr']
        Locations = ['Earth', 'Earth']
        # event_telescopes = ['Lightcurve_1_Survey.dat','Lightcurve_1_Follow.dat']
        # event_telescopes = ['MOA2016BLG0221_flux.dat','MOA2016BLG0221_K2_flux.dat']
        # event_telescopes = ['MOA2016BLG0233_flux.dat','MOA2016BLG0233_K2_flux.dat']
        # event_telescopes = ['OGLE2016BLG0548.dat','OGLE20160548_K2_flux.dat']
        # event_telescopes = ['MOA2016BLG0286_flux.dat']
        # event_telescopes = ['MOA2016BLG0307_flux.dat']
        count = 0

        start = time.time()
        for event_telescope in event_telescopes:
            try:
                raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope,
                                                usecols=(0, 1, 2))
                # good = np.where(raw_light_curve[:,1]<24)[0]
                good = np.where(raw_light_curve[:, 0] > -1)[0]
                raw_light_curve = raw_light_curve[good]
                lightcurve = np.array(
                    [raw_light_curve[:, 0], raw_light_curve[:, 1], raw_light_curve[:, 2]]).T
                if lightcurve[0, 0] > 2450000:
                    lightcurve[:, 0] = lightcurve[:, 0]
            except:
                pass

            if Names[count] == 'Kepler':
                telescope = telescopes.Telescope(name=Names[count], camera_filter='I',
                                                 light_curve_flux=lightcurve)
            else:
                telescope = telescopes.Telescope(name=Names[count], camera_filter='I',
                                                 light_curve_magnitude=lightcurve)
            telescope.gamma = 0.5
            telescope.location = Locations[count]
            current_event.telescopes.append(telescope)
            count += 1

        print 'Start;', current_event.name

        current_event.find_survey('Survey')
        current_event.dec = 98
        current_event.check_event()

        Model = microlmodels.MLModels(current_event, command_line.model,
                                      parallax=['None', 2457510.0])

        current_event.fit(Model, 'MCMC')

        current_event.fits[0].produce_outputs()
        # print current_event.fits[0].fit_results
        plt.show()

    import pdb
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
                        default='/nethome/Desktop/Microlensing/OpenSourceProject/'
                                'SimulationML/Lightcurves_FSPL/Lightcurves/')
    parser.add_argument('-o', '--output_directory', default='/nethome/Desktop/Microlensing/'
                        'OpenSourceProject/Developement/Fitter/FSPL/')
    parser.add_argument('-c', '--claret',
                        default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/'
                        'OpenSourceProject/Claret2011/J_A+A_529_A75/')
    arguments = parser.parse_args()

    model = arguments.model

    main(arguments)
