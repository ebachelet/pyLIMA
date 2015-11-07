# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:49:44 2015

@author: ebachelet
"""
from __future__ import division
import numpy as np
from pyslalib import slalib
from astropy import constants as const



class Parallaxes(object):

    def __init__(self, event, model):
        ''' Initialization of the attributes described above.
        '''
        self.AU = const.au.value
        self.speed_of_light = const.c.value
        self.Earth_radius=const.R_earth.value
        self.event = event
        self.model= model[0]
        self.topar = model[1]
        self.delta_tau = []
        self.delta_u = []
        self.target_angles=[self.event.ra*np.pi/180,self.event.dec*np.pi/180]


    def N_E_vectors_target(self):

        target_angles=self.target_angles
        Target=np.array([np.cos(target_angles[1])*np.cos(target_angles[0]),np.cos(target_angles[1])*np.sin(target_angles[0]),np.sin(target_angles[1])])

        self.East=np.array([-np.sin(target_angles[0]),np.cos(target_angles[0]),0.0])
        self.North=np.cross(Target,self.East)

    def HJD_to_JD(self, t):

        AU=self.AU
        light_speed=self.speed_of_light

        time_correction=[]
        #DTT=[]
        t=t

        for i in t :

            count=0
            jd=np.copy(i)

            while count<3:

                Earth_position=slalib.sla_epv(jd)
                Sun_position=-Earth_position[0]

                Sun_angles=slalib.sla_dcc2s(Sun_position)
                target_angles=self.target_angles

                t_correction=np.sqrt(Sun_position[0]**2+Sun_position[1]**2+Sun_position[2]**2)*AU/light_speed*(np.sin(Sun_angles[1])*np.sin(target_angles[1])+np.cos(Sun_angles[1])*np.cos(target_angles[1])*np.cos(target_angles[0]-Sun_angles[0]))/(3600*24.0)
                count=count+1

        #DTT.append(slalib.sla_dtt(jd)/(3600*24))
        time_correction.append(t_correction)   

        JD=t+np.array(time_correction)

        return JD

    def parallax_combination(self):

        self.N_E_vectors_target()
        delta_position_North = np.array([])
        delta_position_East = np.array([])

        for i in self.event.telescopes:

            kind = i.kind
            t = self.HJD_to_JD(i.lightcurve_flux[:,0])
            delta_North = np.array([])
            delta_East = np.array([])

            if kind == 'Earth':

                if (self.model == 'Annual') or (self.model == 'Full'):


                    positions=self.annual_parallax(t)
                    delta_North = np.append(delta_North, positions[0])
                    delta_East = np.append(delta_East, positions[1])

                if (self.model == 'Terrestrial') or (self.model == 'Full'):

                    altitude=i.altitude
                    longitude=i.longitude
                    latitude=i.latitude

                    positions=self.terrestrial_parallax(t, altitude, longitude, latitutde)
                    delta_North = np.append(delta_North, positions[0])
                    delta_East = np.append(delta_East, positions[1])


            else:

                name=i.name

                positions=self.space_parallax(t, name)
                delta_North = np.append(delta_North, positions[0])
                delta_East = np.append(delta_East, positions[1])


            delta_position_North = np.append(delta_position_North, delta_North)
            delta_position_East = np.append(delta_position_East, delta_East)

        self.delta_position = -np.array([delta_position_North,delta_position_East])


    def annual_parallax(self, t):

        topar=self.HJD_to_JD(np.array([self.topar]))-2400000.5

        Earth_position_ref=slalib.sla_epv(topar)
        Sun_position_ref=-Earth_position_ref[0]
        Sun_speed_ref=-Earth_position_ref[1]
        delta_Sun=[]

        for i in t :

            tt=i-2400000.5
            
            Earth_position=slalib.sla_epv(tt)
            Sun_position=-Earth_position[0]
            delta_sun= Sun_position-(tt-topar)*Sun_speed_ref-Sun_position_ref
            delta_Sun.append(delta_sun.tolist())

        delta_Sun=np.array(delta_Sun)
        delta_Sun_proj=-np.array([np.dot(delta_Sun, self.North),np.dot(delta_Sun, self.East)])

        return delta_Sun_proj

    def terrestrial_parallax(t,model,target):

       return 'hello'

    def space_parallax(t,model,target):
    
        return 'Hello'

    def parallax_outputs(self, PiE):

        piE=np.array(PiE)
        delta_tau = np.dot(piE,self.delta_position)
        delta_u = np.cross(piE,self.delta_position.T)

        return delta_tau, delta_u
