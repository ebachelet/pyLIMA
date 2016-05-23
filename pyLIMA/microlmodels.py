# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:32:13 2015

@author: ebachelet
"""

from __future__ import division
from collections import OrderedDict

import numpy as np
from scipy import interpolate, misc
import os.path 

import microlmagnification
import microlparallax

full_path = os.path.abspath(__file__)
directory, filename = os.path.split(full_path)


### THIS NEED TO BE SORTED ####
try:

    # yoo_table = np.loadtxt('b0b1.dat')
    yoo_table = np.loadtxt(os.path.join(directory, 'data/Yoo_B0B1.dat'))
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
yoo_table = [zz, interpol_b0, interpol_b1, interpol_db0, interpol_db1]

class MLModels(object):
    
    """
    ######## MLModels module ########
  


    Keyword arguments:
    
    event --> A event class which describe your event that you want to model. See the event module.

    model --> The microlensing model you want. Has to be a string :

             'PSPL' --> Point Source Point Lens. The amplification is taken from :
             "Gravitational microlensing by the galactic halo" Paczynski,B. 1986ApJ...304....1P

             'FSPL' --> Finite Source Point Lens. The amplification is taken from :
             "OGLE-2003-BLG-262: Finite-Source Effects from a Point-Mass Lens' Yoo,
             J. et al.2004ApJ...603..139Y
              Note that the LINEAR LIMB-DARKENING is used, where the table b0b1.dat is interpolated
              to compute B0(z) and B1(z).

             'DSPL'  --> not available now
             'Binary' --> not available now
             'Triple' --> not available now
             
    
                          
    
    second_order --> Second order effect : parallax, orbital_motion and source_spots . A list
        of string as :

            [parallax,orbital_motion,source_spots]
            Example : [['Annual',2456876.2],['2D',2456876.2],'None']

            parallax --> Parallax model you want to use for the Earth types telescopes.
                         Has to be a list containing the model in the available_parallax
                         parameter and
                         the value of topar.

                         'Annual' --> Annual parallax
                         'Terrestrial' --> Terrestrial parallax
                         'Full' --> combination of previous

                         topar --> a time in HJD choosed as the referenced time fot the parallax

                         If you have some Spacecraft types telescopes, the space based parallax
                         is computed.

                         More details in the microlparallax module

            orbital_motion --> Orbital motion you want to use. Has to be a list containing the model
                               in the available_orbital_motion parameter and the value of toom:

                'None' --> No orbital motion
                '2D' --> Classical orbital motion
                '3D' --> Full Keplerian orbital motion

                toom --> a time in HJD choosed as the referenced time fot the orbital motion
                        (Often choose equal to topar)

                More details in the microlomotion module

            source_spots --> Consider spots on the source. Has to be a string in the
            available_source_spots parameter :

                'None' --> No source spots

                More details in the microlsspots module
                
     Parameters description. The PARAMETERS RULE is (quantity in brackets are optional):

            [to,uo,tE,(rho),(s),(q),(alpha),(PiEN),(PiEE),(dsdt),(dalphadt),(source_spots)]+Sum_i[fsi,fbi/fsi]

            to --> time of maximum amplification in HJD
            uo --> minimum impact parameter (for the time to)
            tE --> angular Einstein ring crossing time in days
            rho --> normalized angular source radius = theta_*/theta_E
            s --> normalized projected angular speration between the two bodies
            q --> mass ratio
            alpha --> counterclockwise angle (in radians) between the source
            trajectory and the lenses axes
            PiEN --> composant North (in the sky plane) of the parallax vector
            PiEE --> composant East (in the sky plane) of the parallax vector
            ds/dt --> s variations due to the lens movement
            dalpha/dt --> angle variations due to the lens movement
            source_spots --> ?????
            fsi --> source flux in unit : m=27.4-2.5*np.log10(flux)
            fbi/fsi --> blending flux ratio

            As an example , if you choose an FSPL model with 'Annual' parallax and two telescopes 1 and 2
            to fit, the parameters will look like :
                             
            [to,uo,tE,rho,PiEN,PiEE,fs1,fb1/fs1,fs2,fb2/fs2]
        
             """
             
             
    def __init__(self, event, model='PSPL', parallax = ['None', 0.0], xallarap = ['None', 0.0], orbital_motion = ['None', 0.0],
                 source_spots = 'None'):
        """ Initialization of the attributes described above. """

        self.event = event
        self.paczynski_model = model
        self.parallax_model = parallax
        self.xallarap_model = xallarap
        self.orbital_motion_model = orbital_motion
        self.source_spots_model =  source_spots

        self.yoo_table = yoo_table
        self.define_parameters()

    def f_derivative(x, function):
        import pdb;
        pdb.set_trace()

        return function(x)

    def define_parameters(self):
        """ Create the model_dictionnary which explain to the different modules which parameter is what (
        Paczynski parameters+second_order+fluxes)
        Also defines the parameters_boundaries requested by method 'DE' and 'MCMC'
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


    def magnification(self, parameters , time, gamma = 0, delta_positions = 0) :
        
        """ Compute the according magnification """
        
        to = parameters[self.model_dictionnary['to']]
        uo = parameters[self.model_dictionnary['uo']]
        tE = parameters[self.model_dictionnary['tE']]        
        
        tau = (time - to) / tE
        
        
        
        dtau = 0
        duo = 0
        
        if self.parallax_model[0] != 'None' :
            piE = np.array([parameters[self.model_dictionnary['piEN']],
                                parameters[self.model_dictionnary['piEE']]])
            dTau,dUo = self.compute_parallax_curvature(self, piE, delta_positions)
            dtau += dtau
            duo += dUo
        
                
        tau += dtau
        uo += duo
        
        if self.paczynski_model == 'PSPL':
            
            amplification, u = microlmagnification.amplification_PSPL(tau, uo)
            return amplification, u
            
        if self.paczynski_model == 'FSPL':
            
            rho = parameters[self.model_dictionnary['rho']]
            amplification, u = microlmagnification.amplification_FSPL(tau, uo, rho, gamma, self.yoo_table)
            return amplification, u   
            
    
    def compute_parallax(self, second_order):
         """ Compute the parallax for all the telescopes, if this is desired in
         the second order parameter."""
         telescopes = []
         for i in self.event.telescopes:
  
            if len(i.deltas_positions) == 0:
                telescopes.append(i)

         para = microlparallax.MLParallaxes(self.event, second_order[0])
         para.parallax_combination(telescopes)        
            
    def compute_parallax_curvature(self, piE, delta_positions) :
        """ Compute the curvature induce by the parallax of from
        deltas_positions of a telescope """
                     
        delta_tau = -np.dot(piE, delta_positions)
        delta_u = -np.cross(piE, delta_positions.T)
           
        return delta_tau,delta_u
    
    
        
        