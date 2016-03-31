# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:32:13 2015

@author: ebachelet
"""

from __future__ import division
from collections import OrderedDict

import numpy as np
from scipy import interpolate, misc

try:

    # yoo_table = np.loadtxt('b0b1.dat')
    yoo_table = np.loadtxt('Yoo_B0B1_3.dat')
except:

    print 'ERROR : No b0b1.dat file found, please check!'

b0b1 = yoo_table
zz = b0b1[:, 0]
b0 = b0b1[:, 1]
b1 = b0b1[:, 2]
# db0 = b0b1[:,4]
# db1 = b0b1[:, 5]
interpol_b0 = interpolate.interp1d(zz, b0, kind='linear')
interpol_b1 = interpolate.interp1d(zz, b1, kind='linear')
# import pdb; pdb.set_trace()

dB0 = misc.derivative(lambda x: interpol_b0(x), zz[1:-1], dx=10 ** -4, order=3)
dB1 = misc.derivative(lambda x: interpol_b1(x), zz[1:-1], dx=10 ** -4, order=3)
dB0 = np.append(2.0, dB0)
dB0 = np.concatenate([dB0, [dB0[-1]]])
dB1 = np.append((2.0 - 3 * np.pi / 4), dB1)
dB1 = np.concatenate([dB1, [dB1[-1]]])
interpol_db0 = interpolate.interp1d(zz, dB0, kind='linear')
interpol_db1 = interpolate.interp1d(zz, dB1, kind='linear')


class MLModels(object):
    def __init__(self, event, model='PSPL',
                 second_order=[['None', 0.0], ['None', 0.0], ['None', 0.0], 'None']):
        """ Initialization of the attributes described above. """

        self.event = event
        self.paczynski_model = model
        self.second_order = second_order
        self.parallax_model = second_order[0]
        self.xallarap_model = second_order[1]
        self.orbital_motion_model = second_order[2]
        self.source_spots_model = second_order[3]

        self.yoo_table = [zz, interpol_b0, interpol_b1, interpol_db0, interpol_db1]
        self.define_parameters()

    def f_derivative(x, function):
        import pdb;
        pdb.set_trace()

        return function(x)

    def define_parameters(self):
        """Provide the number of parameters on which depend the magnification computation.(
        Paczynski parameters+second_order)
        """

        self.model_dictionnary = {'to': 0, 'uo': 1, 'tE': 2}

        if self.paczynski_model == 'FSPL':

            self.model_dictionnary['rho'] = len(self.model_dictionnary)

        if self.parallax_model[0] != 'None':

            self.model_dictionnary['piEN'] = len(self.model_dictionnary)
            self.model_dictionnary['piEE'] = len(self.model_dictionnary)

        if self.xallarap_model[0] != 'None':

            self.model_dictionnary['XiEN'] = len(self.model_dictionnary)
            self.model_dictionnary['XiEE'] = len(self.model_dictionnary)

        if self.orbital_motion_model[0] != 'None':

            self.model_dictionnary['dsdt'] = len(self.model_dictionnary)
            self.model_dictionnary['dalphadt'] = len(self.model_dictionnary)

        if self.source_spots_model != 'None':

            self.model_dictionnary['spot'] = len(self.model_dictionnary) + 1

        model_paczynski_boundaries = {'PSPL': [(min(self.event.telescopes[0].lightcurve[:, 0])-300,
                                                max(self.event.telescopes[0].lightcurve[:, 0])+300),
                                               (-2.0, 2.0), (1.0, 300)], 'FSPL': [
            (min(self.event.telescopes[0].lightcurve[:, 0])-300,
             max(self.event.telescopes[0].lightcurve[:, 0])+300),
            (0.00001, 2.0), (1.0, 300), (0.0001, 0.05)]}

        model_parallax_boundaries = {'None': [], 'Annual': [(-2.0, 2.0), (-2.0, 2.0)],
                                     'Terrestrial': [(-2.0, 2.0), (-2.0, 2.0)], 'Full':
                                         [(-2.0, 2.0), (-2.0, 2.0)]}

        model_xallarap_boundaries = {'None': [], 'True': [(-2.0, 2.0), (-2.0, 2.0)]}

        model_orbital_motion_boundaries = {'None': [], '2D': [], '3D': []}

        model_source_spots_boundaries = {'None': []}

        self.parameters_boundaries = model_paczynski_boundaries[self.paczynski_model] + \
                                     model_parallax_boundaries[
                                         self.parallax_model[0]] + model_xallarap_boundaries[
                                         self.xallarap_model[0]] + model_orbital_motion_boundaries[
                                         self.orbital_motion_model[0]] + \
                                     model_source_spots_boundaries[
                                         self.source_spots_model]

        for i in self.event.telescopes:

            self.model_dictionnary['fs_' + i.name] = len(self.model_dictionnary)
            self.model_dictionnary['g_' + i.name] = len(self.model_dictionnary)

        self.model_dictionnary = OrderedDict(
            sorted(self.model_dictionnary.items(), key=lambda x: x[1]))
