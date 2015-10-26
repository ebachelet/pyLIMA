# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:46:45 2015

@author: ebachelet
"""

import numpy as np
import matplotlib.pyplot as plt

Location='OGLE_2015'

Fits=np.loadtxt('./'+Location+'/Fits_'+Location+'.txt',dtype='string')
Names=Fits[:,0]
Fits=Fits[:,1:].astype(float)

Errors=np.loadtxt('./'+Location+'/Fits_'+Location+'_Error.txt',dtype='string')
Errors=Errors[:,1:].astype(float)

Source=np.loadtxt('./'+Location+'/Fits_'+Location+'_Source.txt',dtype='string')
Source=Source[:,1:].astype(float)

Source_err=np.loadtxt('./'+Location+'/Fits_'+Location+'_Source_errors.txt',dtype='string')
Source_err=Source_err[:,1].astype(float)

Blend=np.loadtxt('./'+Location+'/Fits_'+Location+'_Blend.txt',dtype='string')
Blend=Blend[:,1:].astype(float)

Blend_err=np.loadtxt('./'+Location+'/Fits_'+Location+'_Blend_errors.txt',dtype='string')
Blend_err=Blend_err[:,1].astype(float)

Model=np.loadtxt('/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Lightcurves_'+Location+'/Models.txt',dtype='string')
Model=Model[:,0:].astype(float)

Sigmato=[]
Sigmauo=[]
Sigmate=[]
SigmaSource=[]
SigmaBlend=[]
DChi2=[]
snr=[]
ffb1=[]
ffb2=[]
count=0
#import pdb; pdb.set_trace()

Events_path=  '/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Lightcurves_'+Location+'/Lightcurves/'
for i in xrange(len(Fits)):
        #print Names[i]
        #data=np.loadtxt(Events_path+Names[i])
        #fbaseline=Model[i,3]*(1+Model[i,4])
        #errf=np.sqrt(fbaseline)
        #Uo=Model[i,1]
        #Amax=(Uo**2+2)/(Uo*np.sqrt(Uo**2+4))
        #SNR=(Model[i,3]*(Amax-1))/errf
        #import pdb; pdb.set_trace()

        if (np.isnan(Errors[i]).any()) and (0.0 in Errors[i]):
            
           count=count+1
           print 'blabla'
           #plt.subplot(311)
           #plt.scatter(Model[i,0],Model[i,1])
           #plt.subplot(312)
           #plt.scatter(Model[i,2],Model[i,3])
           #plt.subplot(313)
           #plt.scatter(Model[i,4],SNR)
        else :
           

            #if Fits[i,3]>Model[i,6] :
                #plt.subplot(311)
                #plt.scatter(Model[i,0],Model[i,1],color='r')
                #plt.subplot(312)
                #plt.scatter(Model[i,2],Model[i,3],color='r')
                #plt.subplot(313)
                #plt.scatter(Model[i,4],SNR,color='r')
                #snr.append(SNR)
                #ffb1.append(Blend[i,0])
                #ffb2.append(Model[i,4])
                #count=count+1
            Sigmato.append((Fits[i,1]-Model[i,0])/Errors[i,0])
            Sigmauo.append((Fits[i,2]-Model[i,1])/Errors[i,1])
            Sigmate.append((Fits[i,3]-Model[i,2])/Errors[i,2])
            #SigmaSource.append((Source[i,0]-Model[i,3])/Source_err[i])
            #SigmaBlend.append((Blend[i,0]-Model[i,4])/Blend_err[i])
            #DChi2.append(Fits[i,4]-Model[i,6])
          
            if Fits[i,0]==1:
                #import pdb; pdb.set_trace()
               
                print Names[i]
                print Fits[i]
                #print (Fits[i,4]-Model[i,6])
                #print Model[i,5]
                #print Fits[i,0]
               # print Fits[i,1]
               # print Fits[i,2]
               # print Fits[i,4]/len(data)
                
                data=np.genfromtxt(Events_path+Names[i],usecols = (0,1,2))
                plt.errorbar(data[:,0]-2450000,data[:,1],yerr=data[:,2],linestyle='none')
                tt=np.arange(data[0,0],data[-1,0],0.01)-2450000
                u=np.sqrt(Fits[i,2]**2+(tt-(Fits[i,1]-2450000))**2/Fits[i,3]**2)
                A=(u**2+2)/(u*np.sqrt(u**2+4))
                fs=Source[i,0]
                fb=Blend[i,0]*fs
                plt.plot(tt,27.4-2.5*np.log10(fs*A+fb),'r')
                plt.gca().invert_yaxis()
                plt.xlabel('Time')
                plt.ylabel('I')
                plt.show()
              #  count=count+1
                #import pdb; pdb.set_trace()

            #Sigmato.append((Fits[i,0]-Model[i,0]))
            #Sigmauo.append((Fits[i,1]-Model[i,1]))
            #Sigmate.append((Fits[i,2]-Model[i,2]))
            #DChi2.append(Fits[i,5]-Model[i,5])
#plt.show()
print count
criteria=1
plt.subplot(321)
plt.hist(Sigmato,np.arange(-3,3,0.1))
index=np.where(np.abs(Sigmato)<criteria)[0]
plt.xlabel('To',fontsize=20)
print float(len(index))/len(Sigmato)*100

plt.subplot(322)
plt.hist(Sigmauo,np.arange(-3,3,0.1))
index=np.where(np.abs(Sigmauo)<criteria)[0]
plt.xlabel('Uo',fontsize=20)
print float(len(index))/len(Sigmato)*100

plt.subplot(323)
plt.hist(Sigmate,np.arange(-3,3,0.1))
index=np.where(np.abs(Sigmate)<criteria)[0]
plt.xlabel('tE',fontsize=20)
print float(len(index))/len(Sigmato)*100



#plt.subplot(324)
#plt.hist(SigmaSource,np.arange(-3,3,0.1))
#index=np.where(np.abs(SigmaSource)<criteria)[0]
#plt.xlabel('fs',fontsize=20)
#print float(len(index))/len(Sigmato)*100

#plt.subplot(325)
#plt.hist(SigmaBlend,np.arange(-3,3,0.1))
#index=np.where(np.abs(SigmaBlend)<criteria)[0]
#plt.xlabel('fb/fs',fontsize=20)
#print float(len(index))/len(Sigmato)*100



#plt.subplot(326)


#plt.hist(DChi2,np.arange(-100,100,10))
#plt.xlabel('DChi2',fontsize=20)
#plt.suptitle(''+Location+'',fontsize=50)
plt.show()
import pdb; pdb.set_trace()

plt.subplot(211)
index=np.where(np.array(DChi2)>0)[0]
plt.scatter(Model[index,5],np.array(DChi2)[index])
plt.xlabel('S/N',fontsize=20)
plt.ylabel('DChi2',fontsize=20)
plt.suptitle(''+Location+'',fontsize=50)
plt.subplot(212)
index=np.where(Blend<0)[0]
plt.scatter(Model[index,5],Blend[index])
plt.xlabel('S/N',fontsize=20)
plt.ylabel('fb/fs',fontsize=20)
plt.suptitle(''+Location+'',fontsize=50)
plt.show()
import pdb; pdb.set_trace()
