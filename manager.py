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
                    ('OO' in event_name) and ('Follow' not in event_name)]
    events_names2 = [event_name for event_name in os.listdir(command_line.input_directory) if
                     ('.dat' in event_name)]

    start = time.time()
    results = []
    errors = []

    for event_name in events_names:

        # name='Lightcurve_'+str(17)+'_'
        name = 'OB150196'

        # name = 'Lightcurve_'+str(event_name)+'.'
        current_event = event.Event()
        current_event.name = name

        event_telescopes = [i for i in events_names2]
        # event_telescopes = ['OGLE-2016-BLG-0676.dat','MOA-2016-BLG-215_MOA_transformed.dat',
        # 'MOA-2016-BLG-215_transformed.dat']
        # event_telescopes = ['MOA-2016-BLG-215_transformed.dat']
        # Names = ['OGLE','Kepler']
        # Locations = ['Earth','Space']

        current_event.ra = 268.36254166
        current_event.dec = -29.7928055
        count = 0
        import pdb;
        pdb.set_trace()
        start = time.time()
        for event_telescope in event_telescopes:
            try:
                tel_name = event_telescope[14:-4]
                if ('OOB' in event_telescope):

                    raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope,
                                                    usecols=(0, 1, 2))

                    lightcurve = np.array(
                        [raw_light_curve[:, 0], raw_light_curve[:, 1],
                         1.0 * (raw_light_curve[:, 2] ** 2 + 0.00 ** 2) ** 0.5]).T
                elif ('Spitzer' in event_telescope):

                    raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope,
                                                    usecols=(0, 1, 2))

                    lightcurve = np.array(
                        [raw_light_curve[:, 0], raw_light_curve[:, 1],
                         0.4 * (raw_light_curve[:, 2] ** 2 + 0.02 ** 2) ** 0.5]).T

                else:
                    raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope,
                                                    usecols=(0, 1, 2))

                    lightcurve = np.array(
                        [raw_light_curve[:, 2], raw_light_curve[:, 0], raw_light_curve[:, 1]]).T
                print event_telescope

                if lightcurve[0, 0] < 2450000:
                    lightcurve[:, 0] += 2450000
                START = 2446906
                END = 2488998
                good = np.where((lightcurve[:, 0] < END) & (lightcurve[:, 0] > START))[0]
                # good = np.where((lightcurve[:,0]>2457400))[0]


                lightcurve = lightcurve[good]
            except:
                pass

            if 'MOA_R' in event_telescope:
                telescope = telescopes.Telescope(name=event_telescope[-9:-4] + '_' + event_telescope[-5],
                                                 camera_filter=event_telescope[-5],
                                                 light_curve_flux=lightcurve,
                                                 light_curve_flux_dictionnary={'time': 0, 'flux': 1, 'err_flux': 2},
                                                 reference_flux=0.0)
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
            if telescope.name == 'Spitzer':
                telescope.location = 'Space'
            current_event.telescopes.append(telescope)
            count += 1

        print 'Start;', current_event.name
        import pdb;
        pdb.set_trace()
        current_event.find_survey('OOB150196I')
        current_event.check_event()
        print [i.name for i in current_event.telescopes]
        # Model = microlmodels.MLModels(current_event, command_line.model,
        #                              parallax=['None', 50.0])

        Model = microlmodels.create_model('USBL', current_event, parallax=['Annual', 2457120],
                                          orbital_motion=['None', 2457120])
        Model.USBL_windows = [2456900, 2457300]

        Model.parameters_guess = [2457123.7393666035, 0.07377374912968027, 100.19567145132673,
                                  0.00781922126907861, 0.1891170236894218, -0.08522691762768343, -2.751077942426451,
                                  0.1,
                                  -0.1]
        #Model.parameters_guess = [2457121.718802689, 0.044874912769544036, 92.42670084982296, 0.007699897220068091,
        #                          0.21064050918430152, -0.06865864981640608, -2.888118768836607, 0,0]
        # Model.parameters_guess = [2457123.339606512, 0.07127476937284467, 98.29935489284918, 0.007789733755782033,
        #                          0.19294303151201372, -0.07806259861571774, -2.7967866875047314, -0.04708553491145748,
        #                          -0.011182419093513155, 2.6840603557907263e-05, 0.0012152762867827978]

        # Model.parameters_guess = [2457124.8648720165, 0.05087890409693893, 91.73208043047485, 0.0076081975458405365,
        #                          0.21238283769339375, -0.10463980545672134, -2.8938338520787865, 0.23125922595225648,
        #                          -0.08570629277441434, -0.0003936674943793056, -0.0003000986621273718]

        Model.parameters_boundaries[0] = (2457120, 2457125)
        Model.parameters_boundaries[1] = (0.05, 0.1)
        Model.parameters_boundaries[2] = (90, 110)
        Model.parameters_boundaries[3] = (0.005, 0.01)
        Model.parameters_boundaries[4] = (0.15, 0.25)
        Model.parameters_boundaries[5] = (-0.2, 0.0)
        Model.parameters_boundaries[6] = (-3.0, -2.5)

        # Model.parameters_boundaries[6] = (-0.8, -0.6)
        # Model.parameters_boundaries[3] = (20, 40)
        # Model.fancy_to_pyLIMA_dictionnary = {'logrho': 'rho'}
        # Model.pyLIMA_to_fancy = {'logrho': lambda parameters: np.log10(parameters.rho)}
        # Model.parameters_boundaries[3] = (-5.0, -1.0)
        # Model.fancy_to_pyLIMA = {'rho': lambda parameters: 10 ** parameters.logrho}
        current_event.fit(Model, 'LM')
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
                        default='/nethome/ebachelet/Desktop/Microlensing/OpenSourceProject/SimulationML/OB150196/')
    parser.add_argument('-o', '--output_directory', default='/nethome/ebachelet/Desktop/Microlensing/'
                                                            'OpenSourceProject/Developement/Fitter/FSPL/')
    parser.add_argument('-c', '--claret',
                        default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/'
                                'OpenSourceProject/Claret2011/J_A+A_529_A75/')
    arguments = parser.parse_args()

    model = arguments.model

    main(arguments)
