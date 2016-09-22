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
                    ('OGLE' in event_name) and ('Follow' not in event_name)]
    events_names2 = [event_name for event_name in os.listdir(command_line.input_directory) if
                     ('.dat' in event_name) and ('~' not in event_name)]

    start = time.time()
    results = []
    errors = []

    for event_name in events_names[0:]:

        # name='Lightcurve_'+str(17)+'_'
        name = 'OB160813'
        # name = 'Lightcurve_9_'
        current_event = event.Event()
        current_event.name = name

        event_telescopes = [i for i in events_names2]
        # event_telescopes = ['OGLE-2016-BLG-0676.dat','MOA-2016-BLG-215_MOA_transformed.dat',
        # 'MOA-2016-BLG-215_transformed.dat']
        # event_telescopes = ['MOA-2016-BLG-215_transformed.dat']
        # Names = ['OGLE','Kepler']
        # Locations = ['Earth','Space']

        current_event.ra = 269.88654166
        current_event.dec = -28.40741666

        count = 0
        import pdb;
        pdb.set_trace()
        start = time.time()
        for event_telescope in event_telescopes:
            try:
                raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope,
                                                usecols=(0, 1, 2))



                lightcurve = np.array(
                    [raw_light_curve[:, 0], raw_light_curve[:,1], raw_light_curve[:, 2]]).T

                if lightcurve[0, 0] < 2450000:
                    lightcurve[:, 0] += 2450000
            except:
                pass

            if 'Kepler_R' in event_telescope:
                telescope = telescopes.Telescope(name='Kepler', camera_filter=event_telescope[-5],
                                                 light_curve_flux=lightcurve,
                                                 light_curve_flux_dictionnary={'time': 0, 'flux': 1, 'err_flux': 2}, reference_flux=2000)
                telescope.location = 'Space'
            else:

                telescope = telescopes.Telescope(name=event_telescope[0:-4], camera_filter=event_telescope[-5],
                                                 light_curve_magnitude=lightcurve,
                                                 light_curve_magnitude_dictionnary={'time': 0, 'mag': 1, 'err_mag': 2})
                telescope.location = 'Earth'
            if telescope.name == 'OGLE_I':

                telescope.gamma = 0.44
            else :
                telescope.gamma = 0.53
            current_event.telescopes.append(telescope)
            count += 1

        print 'Start;', current_event.name

        current_event.find_survey('OGLE_I')
        # current_event.check_event()

        # Model = microlmodels.MLModels(current_event, command_line.model,
        #                              parallax=['None', 50.0])

        Model = microlmodels.create_model('PSPL', current_event, parallax=['None', 2457511])
        Model.parameters_guess = [2457511, 0.06267,6.38]
        # Model.parameters_boundaries[3] = (-5.0, -1.0)

        # Model.fancy_to_pyLIMA_dictionnary = {'logrho': 'rho'}
        # Model.pyLIMA_to_fancy = {'logrho': lambda parameters: np.log10(parameters.rho)}

        # Model.fancy_to_pyLIMA = {'rho': lambda parameters: 10 ** parameters.logrho}
        current_event.fit(Model, 'LM', flux_estimation_MCMC='polyfit')

        import pdb;
        pdb.set_trace()
        current_event.fits[0].produce_outputs()
        # print current_event.fits[0].fit_results
        plt.show()

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
                        default='/nethome/ebachelet/Desktop/Microlensing/OpenSourceProject/'
                                'SimulationML/OB160813/')
    parser.add_argument('-o', '--output_directory', default='/nethome/ebachelet/Desktop/Microlensing/'
                                                            'OpenSourceProject/Developement/Fitter/FSPL/')
    parser.add_argument('-c', '--claret',
                        default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/'
                                'OpenSourceProject/Claret2011/J_A+A_529_A75/')
    arguments = parser.parse_args()

    model = arguments.model

    main(arguments)
