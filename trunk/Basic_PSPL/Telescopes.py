# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:39:32 2015

@author: ebachelet
"""
import numpy as np
import astropy.io.fits as fits
class Telescope:
    
    
    def __init__(self):
        
        self.name='None'
        self.kind='None'
        self.filter='I' #Claret2011 convention
        self.lightcurve=[]
        self.altitude='None'
        self.longitude='None'
        self.latitude='None'
        self.gamma=0.5
        
    def Ndata(self):
        
        
        return len(self.lightcurve[:,0])
    
    def find_gamma(self,Teff,logg):
        #assumption   Microturbulent velocity =2km/s, metallicity= 0.0 (Sun value) Claret2011 convention        
        VT=2.0
        metal=0.0
       

        claret_path='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Claret2011/J_A+A_529_A75/'
        claret=fits.open(claret_path+'Claret2011.fits')
        claret=np.array([claret[1].data['log g'],claret[1].data['Teff (K)'],claret[1].data['Z (Sun)'],claret[1].data['Xi (km/s)'],claret[1].data['u'],claret[1].data['filter']]).T
        index_filter=np.where(claret[:,5]==self.filter)[0]
        
        claret_reduce=claret[index_filter,:-1].astype(float)
        
        coeff_index=np.sqrt((claret_reduce[:,0]-logg)**2+(claret_reduce[:,1]-Teff)**2+(claret_reduce[:,2]-metal)**2+(claret_reduce[:,3]-VT)**2).argmin()
              
        U=claret_reduce[coeff_index,-1]
        self.gamma=2*U/(3-U)
        self.gamma=0.5
        
    def clean_data(self):
        
            index=np.where(np.abs(self.lightcurve[:,1]-np.median(self.lightcurve[:,1]))>10)[0]
            for i in index :
                print self.name+' point at '+str(self.lightcurve[i,0])+' is consider as outlier, rejected from this fit'
            index=np.where(np.abs(self.lightcurve[:,1]-np.median(self.lightcurve[:,1]))<10)[0]
            self.lightcurve=self.lightcurve[index]
            #import pdb; pdb.set_trace()
