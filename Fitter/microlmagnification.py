# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:37:33 2015

@author: ebachelet
"""

from __future__ import division
import numpy as np

def amplification(model, t, parameters, gamma):
    ''' The magnification associated to the model, at time t using parameters and gamma.
        The formula change regarding the requested model :
        PSPL' --> Point Source Point Lens. The amplification is taken from :
        "Gravitational microlensing by the galactic halo" Paczynski,B. 1986ApJ...304....1P
        A=(u^2+2)/[u*(u^2+4)^0.5]

        'FSPL' --> Finite Source Point Lens. The amplification is taken from :
        "OGLE-2003-BLG-262: Finite-Source Effects from a Point-Mass Lens' Yoo,J. et al.2004ApJ...603..139Y
        Note that the LINEAR LIMB-DARKENING is used, where the table b0b1.dat is interpolated
        to compute B0(z) and B1(z).

        'DSPL'  --> not available now
        'Binary' --> not available now
        'Triple' --> not available now
        '''
#        X,Y = self.source_trajectory(t, parameters)
    u = (parameters[model.model_dictionnary['uo']]**2+(t-parameters[model.model_dictionnary['to']])**2/parameters[model.model_dictionnary['tE']]**2)**0.5
    u2 = u**2
    ampli = (u2+2)/(u*(u2+4)**0.5)
#   if model.paczynski_model == 'PSPL' or model.paczynski_model == 'FSPL' :

#            u = (X**2+Y**2)**0.5
#            u2 = u**2
#            ampli = (u2+2)/(u*(u2+4)**0.5)
#             pass
    if model.paczynski_model == 'FSPL':

        Z = u/parameters[model.model_dictionnary['rho']]

        ampli_fspl = np.zeros(len(ampli))
        ind = np.where((Z > 10) | (Z < model.yoo_table[0][0]))[0]
        ampli_fspl[ind] = ampli[ind]
        ind = np.where((Z <= 10) & (Z >= model.yoo_table[0][0]))[0]
        ampli_fspl[ind] = ampli[ind]*(model.yoo_table[1](Z[ind])-gamma*model.yoo_table[2](Z[ind]))
        ampli = ampli_fspl


#        if model.paczynski_model == 'FSPL':
#            Ampli=[]
#            start=time.time()

#            for j in u :
#                print j
#                Ampli.append(2/(np.pi*parameters[model.model_dictionnary['rho']]**2)*nquad(
#                self.function_LEE,[[0,np.pi],[lambda x : self.LEE_1(x,j,parameters[model.model_dictionnary['rho']],gamma),lambda x : self.LEE_2(
#                x,j,parameters[model.model_dictionnary['rho']],gamma)]],args=(
#                j,parameters[model.model_dictionnary['rho']],gamma),opts=[{'limit'=10}])[0])
#            print start-time.time()

#            import pdb; pdb.set_trace()

#            ampli=np.array(Ampli)
    return ampli, u


def source_trajectory(model, t, parameters):

    tau = (t-parameters[model.model_dictionnary['to']])/parameters[model.model_dictionnary['tE']]

    if model.paczynski_model is not 'Binary':

        alpha = 0.0

    x = tau*np.cos(alpha)-np.sin(alpha)*parameters[model.model_dictionnary['uo']]
    y = tau*np.sin(alpha)+np.cos(alpha)*parameters[model.model_dictionnary['uo']]

    return x,y

def function_LEE(r,v,u,rho,gamma):

    if r==0 :
        LEE=0
    else :
        LEE = (r**2+2)/((r**2+4)**0.5)*(1-gamma*(1-1.5*(1-(r**2-2*u*r*np.cos(v)+u**2)/rho**2)**0.5))

        return LEE

def LEE_1(v,u,rho,gamma):

    if u<=rho:
        limit_1=0.0
    else :
        if v<=np.arcsin(rho/u) :

            limit_1=u*np.cos(v)-(rho**2-u**2*np.sin(v)**2)**0.5
        else :
            limit_1=0.00

    return limit_1

def LEE_2(v,u,rho,gamma):

    if u<=rho:
        limit_2=u*np.cos(v)+(rho**2-u**2*np.sin(v)**2)**0.5
    else :
        if v<=np.arcsin(rho/u) :

            limit_2=u*np.cos(v)+(rho**2-u**2*np.sin(v)**2)**0.5
        else :
            limit_2=0.0

    return limit_2