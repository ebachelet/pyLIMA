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

from pyLIMA import microlsimulator


def main(command_line):
    events_names = [event_name for event_name in os.listdir(command_line.input_directory) if
                    ('.dat' in event_name) and ('Follow' not in event_name)]
    events_names2 = [event_name for event_name in os.listdir(command_line.input_directory) if
                     ('.dat' in event_name)]

    start = time.time()
    results = []
    errors = []

    for event_name in events_names:

        # name='Lightcurve_'+str(17)+'_'
        name = 'OB160241'

        # name = 'Lightcurve_'+str(event_name)+'.'
        current_event = event.Event()
        current_event.name = name

        event_telescopes = [i for i in events_names2]
        # event_telescopes = ['OGLE-2016-BLG-0676.dat','MOA-2016-BLG-215_MOA_transformed.dat',
        # 'MOA-2016-BLG-215_transformed.dat']
        # event_telescopes = ['MOA-2016-BLG-215_transformed.dat']
        # Names = ['OGLE','Kepler']
        # Locations = ['Earth','Space']

        current_event.ra = 271.0010
        current_event.dec = -28.155111
        count = 0
        import pdb;
        pdb.set_trace()
        start = time.time()
        for event_telescope in event_telescopes:
            try:
                lightcurve = np.loadtxt(command_line.input_directory+event_telescope,dtype=str)

                if 'OGLE' in event_telescope:
                    lightcurve = lightcurve[:,[0,1,2]].astype(float)

                if 'COJA' in event_telescope:
                    lightcurve = lightcurve[:,[0,1,3]].astype(float)
                    lightcurve[:,2] *= 10

                if 'MOA' in event_telescope:
                    lightcurve = lightcurve[:, [0, 1, 2]].astype(float)
                    lightcurve[:, 2] *= 1
                if 'Kepler' in event_telescope:
                    lightcurve = lightcurve[:, [0, 1]].astype(float)
                    lightcurve = np.c_[lightcurve,[50]*len(lightcurve)]
                    #lightcurve[:, 0] += 2450000
                    #lightcurve[:, 1] *= 10
                    #lightcurve[:,2] *= 50
            except:
                pass
            #index = np.where((lightcurve[:,0]>2457500) | ((lightcurve[:,0]<2457487)))[0]
            #lightcurve = lightcurve[index]
            if ('MOA_R'  in event_telescope) | ('Kepler' in event_telescope):
                telescope = telescopes.Telescope(name=event_telescope[:-6],
                                                 camera_filter=event_telescope[-5],
                                                 light_curve_flux=lightcurve,
                                                 light_curve_flux_dictionnary={'time': 0, 'flux': 1, 'err_flux': 2},
                                                 reference_flux=5000.0)
                telescope.location = 'Earth'
            else:

                telescope = telescopes.Telescope(name=event_telescope[:-4], camera_filter=event_telescope[-5],
                                                 light_curve_magnitude=lightcurve,
                                                 light_curve_magnitude_dictionnary={'time': 0, 'mag': 1, 'err_mag': 2})
                telescope.location = 'Earth'
            if telescope.name == 'OGLE_I':

                telescope.gamma = 0.44
            else:
                telescope.gamma = 0.5
            if telescope.name == 'Kepler':
                telescope.location = 'Space'
            current_event.telescopes.append(telescope)
            count += 1

        print 'Start;', current_event.name
        import pdb;
        pdb.set_trace()
        current_event.find_survey('OGLE_I')
        current_event.check_event()
        print [i.name for i in current_event.telescopes]
        # Model = microlmodels.MLModels(current_event, command_line.model,
        #                              parallax=['None', 50.0])

        Model = microlmodels.create_model('PSPL', current_event, parallax=['Annual', 2457512],
                                          orbital_motion=['None', 2457143])
        #Model.USBL_windows = [2456850, 2457350]

        #Model.parameters_guess = [2457228.6691167313, 0.5504378287491769, 119.57049960671444, 0.12837357112315345, 0.10480898512419343]
        #                          0.007759720542532279, 0.2082247217878811, -0.0894830786973533, -2.810468587634137,
        #                          0.2, -0.1
        #                          ]

        # Model.parameters_guess = [2457118.2589892996, 0.04693224313041394, 97.82343956853856, 0.008179766610627337, 0.19677292474954153,
        # -0.027979987829886924, -2.7820828889654297, 0.10258095275259316, -0.060878335472263845]
        # Model.parameters_guess = [2457124.8648720165, 0.19, 91.73208043047485, 0.0076081975458405365,
        #                        0.21238283769339375, -0.10463980545672134, -2.8938338520787865, 0.23125922595225648,
        #                       -0.08570629277441434, -0.0003936674943793056, -0.0003000986621273718]

        # Model.parameters_guess = [2457123.7393666035, 0.2059, 100.19567145132673,
        #                          0.00781922126907861, 0.1891170236894218, -0.08522691762768343, -2.751077942426451,
        #                         ]
        #Model.parameters_boundaries[0] = (18.07, 24)
        #Model.parameters_boundaries[1] = (-2.04, -1.25)
        #Model.parameters_boundaries[2] = (99.93, 102)
        ##Model.parameters_boundaries[3] = (9.540325464407533804e-03, 0.008)
        #Model.parameters_boundaries[4] = (0.166666, 0.25)
        #Model.parameters_boundaries[5] = (-0.22222, 0.0)
        #Model.parameters_boundaries[6] = (-2.6080, -2.5)

        # Model.parameters_boundaries[6] = (-0.8, -0.6)
        # Model.parameters_boundaries[3] = (20, 40)
        #Model.fancy_to_pyLIMA_dictionnary = {'eps': 'to', 'loguo': 'uo'}

        #Model.pyLIMA_to_fancy = {'eps': lambda parameters: 2457143 - parameters.to,
        #                         'loguo': lambda parameters: np.log10(parameters.uo)}
        #Model.fancy_to_pyLIMA = {'to': lambda parameters: 2457143 - parameters.eps,
        #                         'uo': lambda parameters: 10 ** parameters.loguo}
        current_event.fit(Model, 'LM',flux_estimation_MCMC='polyfit')
        import pdb;
        pdb.set_trace()
        current_event.fits[0].produce_outputs()
        # current_event.fits[0].produce_pdf(command_line.input_directory)
        # print current_event.fits[0].fit_results
        plt.show()
        import pdb;
        pdb.set_trace()
        end = time.time()

        import pdb;
        pdb.set_trace()
        print end - start

        all_results = [('Fits.txt', results),
                       ('Fits_Error.txt', errors)]

        for file_name, values in all_results:
            np.savetxt(os.path.join(command_line.output_directory, file_name), np.array(values),
                       fmt="%s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='PSPL')
    parser.add_argument('-i', '--input_directory',
                        default='/nethome/ebachelet/Desktop/Microlensing/'
                                                            'OpenSourceProject/SimulationML/OB160795/')
    parser.add_argument('-o', '--output_directory', default='/nethome/ebachelet/Desktop/Microlensing/'
                                                            'OpenSourceProject/Developement/Fitter/FSPL/')
    parser.add_argument('-c', '--claret',
                        default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/'
                                'OpenSourceProject/Claret2011/J_A+A_529_A75/')
    arguments = parser.parse_args()

    model = arguments.model

    main(arguments)
