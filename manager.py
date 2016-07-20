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
                    ('Lightcurve' in event_name) and ('Follow' not in event_name)]
    events_names2 = [event_name for event_name in os.listdir(command_line.input_directory) if
                     ('Lightcurve' in event_name) and ('~' not in event_name)]

    start = time.time()
    results = []
    errors = []

    for event_name in events_names[0:]:

        name='Lightcurve_'+str(17)+'_'
       # name = 'KB120486'
        #name = 'Lightcurve_9_'
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
        Names = ['Survey', 'Follow']
        Locations = ['Earth', 'Earth']
        # event_telescopes = ['Lightcurve_1_Survey.dat','Lightcurve_1_Follow.dat']
        # event_telescopes = ['MOA2016BLG0221_flux.dat','MOA2016BLG0221_K2_flux.dat']
        # event_telescopes = ['MOA2016BLG0233_flux.dat','MOA2016BLG0233_K2_flux.dat']
        # event_telescopes = ['OGLE2016BLG0548.dat','OGLE20160548_K2_flux.dat']
        # event_telescopes = ['MOA2016BLG0286_flux.dat']
        # event_telescopes = ['MOA2016BLG0307_flux.dat']
        count = 0
        import pdb;
        pdb.set_trace()
        start = time.time()
        for event_telescope in event_telescopes:
            try:
                raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope,
                                                usecols=(0, 1, 2))
                # good = np.where(raw_light_curve[:,1]<24)[0]
                #good = np.where(raw_light_curve[:, 0] > -1)[0]
                #raw_light_curve = raw_light_curve[good]
                lightcurve = np.array(
                    [raw_light_curve[:, 0], raw_light_curve[:, 1], raw_light_curve[:, 2]]).T



            except:
                pass

            if (event_telescope[0]+event_telescope[-5] == 'KR') & (event_telescope[0]+event_telescope[-5] == 'AI'):
                if lightcurve[0,0] <2450000:
                    lightcurve[:, 0] = lightcurve[:, 0] + 2450000
                telescope = telescopes.Telescope(name=event_telescope[:-5], camera_filter=event_telescope[-5],
                                                 light_curve_magnitude=lightcurve)
            else:
                #if lightcurve[0,0] <2450000:
                    #lightcurve[:, 0] = lightcurve[:, 0] + 2450000
                telescope = telescopes.Telescope(name=Names[count], camera_filter=event_telescope[-5],
                                                 light_curve_magnitude=lightcurve,light_curve_magnitude_dictionnary={'time':0 ,'mag': 1, 'err_mag': 2})
            telescope.gamma = 0.5
            telescope.location = 'Earth'
            current_event.telescopes.append(telescope)
            count += 1

        print 'Start;', current_event.name

        current_event.find_survey('Survey')
        #current_event.check_event()

        #Model = microlmodels.MLModels(current_event, command_line.model,
        #                              parallax=['None', 50.0])

        Model = microlmodels.create_model('FSPL', current_event)
        #Model.parameters_guess = [2456154.24, 0.077, 2456137.20, 0.027,
        #                      78.1, 0.08118, 0.125,
        #                      0.163,0.128,-0.37,0.08]
        #Model.parameters_boundaries[3] = (-5.0, -1.0)

        #Model.fancy_to_pyLIMA_dictionnary = {'logrho': 'rho'}
        #Model.pyLIMA_to_fancy = {'logrho': lambda parameters: np.log10(parameters.rho)}

        #Model.fancy_to_pyLIMA = {'rho': lambda parameters: 10 ** parameters.logrho}
        current_event.fit(Model, 'MCMC', flux_estimation_MCMC='MCMC')




        current_event.fits[0].produce_outputs()
        # print current_event.fits[0].fit_results
        plt.show()
        results.append(current_event.fits[0].fit_results+[current_event.fits[0].fit_time])
        errors.append(current_event.fits[0].fit_covariance.diagonal()**0.5)
    end = time.time()
    mcmc_unique = mcmc_best
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
                                'SimulationML/Lightcurves_FSPL/Lightcurves/')
    parser.add_argument('-o', '--output_directory', default='/nethome/ebachelet/Desktop/Microlensing/'
                                                            'OpenSourceProject/Developement/Fitter/FSPL/')
    parser.add_argument('-c', '--claret',
                        default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/'
                                'OpenSourceProject/Claret2011/J_A+A_529_A75/')
    arguments = parser.parse_args()

    model = arguments.model

    main(arguments)
