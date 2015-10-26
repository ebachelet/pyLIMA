# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:25:00 2015

@author: ebachelet
"""

##############################################################################

#Microlensing fitter#
from __future__ import division

import numpy as np
from scipy.optimize import leastsq,minimize
import scipy.stats
import matplotlib.pyplot as plt
import time
from pyslalib import slalib

class ML_Fits:
    
    
    def __init__(self,event,option,method):
        
        
        self.model=[option]
        self.method=method
        self.parallax='None'
        self.survey='None'
        
        if self.model[0]=='FSPL' :
                
            self.model.append(np.loadtxt('b0b1.dat'))
            self.survey=find_survey(self,event)
            self.guess=initial_guess(self,event)
            
            if not self.method : 
                
                self.fit_results,self.fit_covariance,self.fit_time=LMarquardt(self,event)
                Flag=check_fit(self,event)
                
                if Flag=='Bad Fit' :
                    print 'We have to change method, this fit was unsuccessfull'
                   
                    #import pdb; pdb.set_trace()
                    self.method=1
                   
                    #Max_flux=[10**((27.4-i.lightcurve[np.where(i.lightcurve[:,1]==min(i.lightcurve[:,1]))[0],1])/2.5) for i in event.telescopes]
                    #Max_time=[i.lightcurve[np.where(i.lightcurve[:,1]==min(i.lightcurve[:,1]))[0],0] for i in event.telescopes]
                    #boundaries=[(np.median(Max_time)-10,np.median(Max_time)+10),(10**-5,1.0),(0.1,300),]
                    #boundaries=[(min(event.telescopes[0].lightcurve[:,0]),max(event.telescopes[0].lightcurve[:,0])),(10**-5,0.05),(0.1,300),(10**-5,0.05)]
                    #for i in xrange(len(event.telescopes)):
                        #boundaries=boundaries+[(1.0,Max_flux[i]),(0.0,1000.0)]
                    #AA=scipy.optimize.differential_evolution(chichi,bounds=boundaries,args=(self.model,event))
                    #import pdb; pdb.set_trace()
                   
                    #self.fit_results=AA['x'].tolist()
                    #self.fit_results.append(AA['fun'])
                    #Jacky=Jacobian(self.fit_results,self.model,event)
                    #try :
                        #self.fit_covariance=np.linalg.inv(Jacky*Jacky.T)
                    
                    #except :

                        #print 'Something strange with this event'
                        
                        #self.fit_covariance=np.zeros((len(self.fit_results[:-1])+2*len(event.telescopes),len(self.fit_results[:-1])+2*len(event.telescopes)))
                    self.fit_results=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                    self.fit_covariance=np.zeros((len(self.fit_results[:-1])+2*len(event.telescopes),len(self.fit_results[:-1])+2*len(event.telescopes)))
                    self.fit_time=0.0
        if self.model[0]=='PSPL':
            
            #print 'you will start a microlensing fit!'
            self.survey=find_survey(self,event)
            self.guess=initial_guess(self,event)
            
            if self.method==0 : 
                
                self.fit_results,self.fit_covariance,self.fit_time=LMarquardt(self,event)
                #import pdb; pdb.set_trace()
                Flag=check_fit(self,event)
                #print event.name
                #print self.results
                #print self.errors
                #Flag='Bad Fit'
                if Flag=='Bad Fit' :
                    print 'We have to change method, this fit was unsuccessfull'
                   
                    #import pdb; pdb.set_trace()
                    self.method=1
                    Max_flux=[10**((27.4-i.lightcurve[np.where(i.lightcurve[:,1]==min(i.lightcurve[:,1]))[0],1])/2.5) for i in event.telescopes]
                    Max_time=[i.lightcurve[np.where(i.lightcurve[:,1]==min(i.lightcurve[:,1]))[0],0] for i in event.telescopes]
                    #boundaries=[(np.median(Max_time)-10,np.median(Max_time)+10),(10**-5,1.0),(0.1,300),]
                    boundaries=[(min(event.telescopes[0].lightcurve[:,0]),max(event.telescopes[0].lightcurve[:,0])),(10**-5,1.0),(0.1,300)]
                    for i in xrange(len(event.telescopes)):
                        boundaries=boundaries+[(1.0,Max_flux[i]),(0.0,1000.0)]
                    AA=scipy.optimize.differential_evolution(chichi,bounds=boundaries,args=(self.model,event))
                    import pdb; pdb.set_trace()
                    self.fit_results=AA['x'].tolist()
                    self.fit_results.append(AA['fun'])
                    Jacky=Jacobian(self.fit_results,self.model,event)
                    try :
                        self.fit_covariance=np.linalg.inv(Jacky*Jacky.T)
                    
                    except :

                        print 'Something strange with this event'
                        
                        self.fit_covariance=np.zeros((len(self.fit_results[:-1])+2*len(event.telescopes),len(self.fit_results[:-1])+2*len(event.telescopes)))
                    
                    self.fit_time=0.0
                    
def find_survey(self,target):
    
            self.survey='Survey'
            names=np.array([i.name for i in target.telescopes])
            index=np.where(self.survey==names)[0]
            sorting=np.arange(0,len(target.telescopes))

            sorting=np.delete(sorting,index)
            sorting=np.insert(sorting,0,index)
            target.telescopes=np.array(target.telescopes)[sorting.tolist()].tolist()
            
def check_fit(self,target) :
    
        istart={'PSPL':3,'FSPL':4}
        flag='Good Fit'
        diago=np.diag(self.fit_covariance)<0
        
        if 0.0 in self.fit_covariance or True in diago :
            
            print 'Your fit probably wrong'
            flag='Bad Fit'
        
        for i in xrange(len(target.telescopes)):
            
            if self.fit_results[istart[self.model[0]]+2*i]<0 :
                 print 'Your fit probably wrong'
                 flag='Bad Fit'   
                 break
        return flag     
        
def initial_guess(self,target):
    
    #found intial guess for the PSPL model.
    
    Telescopes=target.telescopes
    
    Survey=Telescopes[0]

    Lightcurve=Survey.lightcurve
    Lightcurve=Lightcurve[Lightcurve[:,0].argsort(),:]
    Time=Lightcurve[:,0]
    flux=10**((27.4-Lightcurve[:,1])/2.5)
    errflux=np.abs(-Lightcurve[:,2]*flux/(2.5)*np.log(10))
    
    #fs, no blend
    lower_mag=np.min(Lightcurve[:,1])
    index_lower=np.argmin(Lightcurve[:,1])
    baseline_mag_0=lower_mag
    baseline_mag=np.median(Lightcurve[:,1])
        
   
    
    index=[]
    start=time.time()
    while np.abs(baseline_mag_0-baseline_mag)>0.01 :
        baseline_mag_0=baseline_mag
        index=np.where((Lightcurve[:,1]>baseline_mag))[0].tolist()+np.where(np.abs(Lightcurve[:,1]-baseline_mag)<Lightcurve[:,2])[0].tolist()
        baseline_mag=np.median(Lightcurve[index,1])
        
        if  len(index)<100:
            print 'low'
            baseline_mag= np.median(Lightcurve[Lightcurve[:,1].argsort()[-100:],1])
            break
    #print time.time()-start
    fs=10**((27.4-baseline_mag)/2.5)
    
    #found to,Uo
    index=np.where(flux>fs)[0]
    good=index
    
    

    while len(good)>5 :
       
       
        index=np.where(flux[good]>np.median(flux[good]))[0]
        if len(index)<2 :
            break
        else :
            
            gravity=(np.median(Time[good[index]]),np.median(flux[good[index]]))
            dd=np.sqrt((Time[good[index]]-gravity[0])**2)
            index=index[dd.argsort()[:-1]]
            good=good[index]
        
   
    #import pdb; pdb.set_trace()
        

    to=Time[good[np.where(flux[good]==np.max(flux[good]))[0]]][0]
   

    
    
   
    Max_flux=np.max(flux[good])    
   
  
    
    Amax=Max_flux/fs
    uo=np.sqrt(-2+2*np.sqrt(1-1/(1-Amax**2)))
   
    
    #found tE
 
    if self.model[0]=='FSPL':
        if np.abs(uo)>0.05 :
            
            uo=0.05

    Flux_demi=0.5*fs*(Amax+1)
    Flux_tE=fs*(uo**2+3)/((uo**2+1)**0.5*np.sqrt(uo**2+5))
        
     
    index_plus=np.where((Time>to)&(flux<Flux_demi))[0]
    index_moins=np.where((Time<to)&(flux<Flux_demi))[0]
    B=0.5*(Amax+1)
    #import pdb; pdb.set_trace()
    if len(index_plus) !=0:

         if len(index_moins)!=0 :
            
            ttE=(Time[index_plus[0]]-Time[index_moins[-1]])
            tE1=ttE/(2*np.sqrt(-2+2*np.sqrt(1+1/(B**2-1))-uo**2))
        
         else :
 
            ttE=Time[index_plus[0]]-to
            tE1=ttE/np.sqrt(-2+2*np.sqrt(1+1/(B**2-1))-uo**2)
    else :
        
            ttE=to-Time[index_moins[-1]]
            tE1=ttE/np.sqrt(-2+2*np.sqrt(1+1/(B**2-1))-uo**2)



 
    indextEplus=np.where((flux<Flux_tE)&(Time>to))[0]
    indextEmoins=np.where((flux<Flux_tE)&(Time<to))[0]
 
    tEmoins=0.0
    tEplus=0.0
    if len(indextEmoins)!=0:
         indextEmoins=indextEmoins[-1]
         tEmoins=to-Time[indextEmoins]
         
    if len(indextEplus)!=0:
        indextEplus=indextEplus[0]
        tEplus=Time[indextEplus]-to
    
    indextEPlus=np.where((Time>to)&(np.abs(flux-fs)<errflux))[0]
    indextEMoins=np.where((Time<to)&(np.abs(flux-fs)<errflux))[0]

    tEPlus=0.0
    tEMoins=0.0

    
    if len(indextEPlus)!=0:
        
        tEPlus=Time[indextEPlus[0]]-to
        
    if len(indextEMoins)!=0:
            
           
        tEMoins=to-Time[indextEMoins[-1]]
           
    
            
    TE=np.array([tE1,tEplus,tEmoins,tEPlus,tEMoins])
    good=np.where(TE!=0.0)[0]
    tE=np.sum(TE[good])/len(good)
    #import pdb; pdb.set_trace()

    #import pdb; pdb.set_trace()
    if tE<1:
        tE=20.0
   
    #tE=20
    Fluxes=[]
    #import pdb; pdb.set_trace()

    for i in xrange(len(Telescopes)):
        
        ampli=amplification([to,uo,tE,uo],Telescopes[i].lightcurve[:,0],['PSPL'],target,target.telescopes[i].gamma)[0]
        flux=10**((27.4-Telescopes[i].lightcurve[:,1])/2.5)
        errflux=-Telescopes[i].lightcurve[:,2]*flux*np.log(10)/2.5
        Fs,Fb=np.polyfit(ampli,flux,1,w=errflux)
        
        
      
        if i==0:
            Fluxes.append(fs)
            Fluxes.append(0.0)
        else :
            if (Fs<0) or (Fb/Fs<0) :
                 Fluxes.append(min(flux))
                 Fluxes.append(0.0)
            else :
                
                Fluxes.append(Fs)
                Fluxes.append(Fb/Fs)
            #Fluxes.append(Fb)
            #Fluxes.append(Fs/(Fs+Fb))
            #Fluxes.append(Fs+Fb)
            #Fluxes.append(Fs)
            #Fluxes.append(Fb+Fs)            
            
            
            
        #Fluxes.append(Fb)
   
    if self.model[0]=='PSPL':
    
        Parameters=[to,uo,tE]+Fluxes

    if self.model[0]=='FSPL':
    
        Parameters=[to,uo,tE,2*uo]+Fluxes
    return Parameters
    
def LMarquardt(self,target):
    
    # Levenberg Marquardt routine. 
    #import pdb; pdb.set_trace()
    guess=self.guess
    #print guess
    istart={'PSPL':3,'FSPL':4}
    
    
    start=time.time()
    #import pdb; pdb.set_trace()
    LMarquardt_fit=leastsq(residuals,guess,args=(self.model,target),maxfev=5000,Dfun=Jacobian,col_deriv=1,full_output=1,ftol=0.00001)
    #LMarquardt_fit=leastsq(residuals,guess,args=(self.model,target),maxfev=5000,full_output=1,ftol=0.00001)
    Time=time.time()-start
         
       
    fit_res=LMarquardt_fit[0].tolist()
    fit_res.append(chichi(LMarquardt_fit[0],self.model,target))
    #target.plot_lightcurve()
    #target.plot_model(fit_res)
    #plt.show()
    #import pdb; pdb.set_trace()
    if fit_res[4]<-0.25 :
        print '!!!!!!!!'
    Ndata=0.0
    for i in xrange(len(target.telescopes)):
        Ndata=Ndata+target.telescopes[i].Ndata()
       
    try :
        
        if LMarquardt_fit[1]!=None :

            cov=LMarquardt_fit[1]*fit_res[istart[self.model[0]]+2*len(target.telescopes)]/Ndata
        
        else :
            print 'rough cov'
          

            Jacky=Jacobian(fit_res,self.model,target)
            cov=np.linalg.inv(Jacky*Jacky.T)*fit_res[istart[self.model[0]]+2*len(target.telescopes)]/Ndata
            
                
            
    except:
        print 'hoho'
        
        cov=np.zeros((istart[self.model[0]]+2*len(target.telescopes),istart[self.model[0]]+2*len(target.telescopes)))
           
    return fit_res,cov,Time


def Jacobian(P,model,target) :
    
    istart={'PSPL':3,'FSPL':4}
    
    if model[0]=='PSPL':
        
        dresdto=np.array([])
        dresduo=np.array([])
        dresdtE=np.array([])
        dresdfs=np.array([])
        dresdeps=np.array([])
    
   
        for i in xrange(len(target.telescopes)):
    
            Lightcurve=target.telescopes[i].lightcurve
            flux=10**((27.4-Lightcurve[:,1])/2.5)
            errflux=-Lightcurve[:,2]*flux*np.log(10)/2.5
        
        
            Ampli=amplification(P,Lightcurve[:,0],model,target,target.telescopes[i].gamma)
        
            dAdU=(-8)/(Ampli[1]**2*(Ampli[1]**2+4)**(1.5))
        
            dUdto=-(Lightcurve[:,0]-P[0])/(P[2]**2*Ampli[1])
            dUduo=P[1]/Ampli[1]
            dUdtE=-(Lightcurve[:,0]-P[0])**2/(P[2]**3*Ampli[1])
        
        
            dresdto=np.append(dresdto,-P[istart[model[0]]+2*i]*dAdU*dUdto/errflux)
            dresduo=np.append(dresduo,-P[istart[model[0]]+2*i]*dAdU*dUduo/errflux)
            dresdtE=np.append(dresdtE,-P[istart[model[0]]+2*i]*dAdU*dUdtE/errflux)
            dresdfs=np.append(dresdfs,-(Ampli[0]+P[istart[model[0]]+2*i+1])/errflux)
            dresdeps=np.append(dresdeps,-P[istart[model[0]]+2*i]/errflux)
            #dresdfs=np.append(dresdfs,-(Ampli[0])/errflux)
            #dresdeps=np.append(dresdeps,-1.0/errflux)
            #dresdfs=np.append(dresdfs,-(Ampli[0]-1)/errflux)
            #dresdeps=np.append(dresdeps,-1.0/errflux)
            #dresdfs=np.append(dresdfs,-P[istart[model[0]]+2*i+1]*(Ampli[0]*-1)/errflux)
            #dresdeps=np.append(dresdeps,-(P[istart[model[0]]+2*i]*(Ampli[0]*-1)+1)/errflux)
            #dresdfs=np.append(dresdfs,-1/(1+P[istart[model[0]]+2*i+1])*(Ampli[0]+P[istart[model[0]]+2*i+1])/errflux)
            #dresdeps=np.append(dresdeps,+(P[istart[model[0]]+2*i]/(1+P[istart[model[0]]+2*i+1])**2*(Ampli[0]*-1))/errflux)
            #dresdfs=np.append(dresdfs,-(Ampli[0])/errflux)
            #dresdeps=np.append(dresdeps,-1/errflux)
        jacobi=np.array([dresdto,dresduo,dresdtE])
        start=0
        for i in target.telescopes:
            dFS=np.zeros((len(dresdto)))
            dEPS=np.zeros((len(dresdto)))
                        
            index=np.arange(start,start+len(i.lightcurve[:,0]))
            dFS[index]=dresdfs[index]
            dEPS[index]=dresdeps[index]
            start=start+index[-1]+1                                        
            
            jacobi=np.vstack([jacobi,dFS])
            jacobi=np.vstack([jacobi,dEPS])
        
    
    if model[0]=='FSPL':
        
        dresdto=np.array([])
        dresduo=np.array([])
        dresdtE=np.array([])
        dresdrho=np.array([])
        dresdfs=np.array([])
        dresdeps=np.array([])
    
        b0b1=model[1]
        b0=b0b1[:,1]
        b1=b0b1[:,2]
        zz=b0b1[:,0]
        db0=b0b1[:,3]
        db1=b0b1[:,4]
        
        for i in xrange(len(target.telescopes)):
           
            Lightcurve=target.telescopes[i].lightcurve
            flux=10**((27.4-Lightcurve[:,1])/2.5)
            errflux=-Lightcurve[:,2]*flux*np.log(10)/2.5
        
        
            Ampli=amplification(P,Lightcurve[:,0],['PSPL'],target,target.telescopes[i].gamma)
            
            dAdU=(-8)/(Ampli[1]**2*(Ampli[1]**2+4)**(1.5))
            
            Z=Ampli[1]/P[3]
            dadu=[]
            dadrho=[]
            count=0
            
            for j in Z :
                
                if j < 10.0 :
                  
                    index=np.abs(zz-j).argmin()
              
                    if (j-zz[index]<0) :

                        b01=b0[index-1]
                        b11=b1[index-1]
                        db01=db0[index-1]
                        db11=db1[index-1]
                        Z1=zz[index-1]
                    
                        b02=b0[index]
                        b12=b1[index]
                        db02=db0[index]
                        db12=db1[index]
                        Z2=zz[index]
			
                    else :

                        b01=b0[index]
                        b11=b1[index]
                        db01=db0[index]
                        db11=db1[index]
                        Z1=zz[index]
			
                        b02=b0[index+1]
                        b12=b1[index+1]
                        db02=db0[index+1]
                        db12=db1[index+1]
                        Z2=zz[index+1]
			
				
	   	
		
                    b0f=(Z2-j)/(Z2-Z1)*b01+(j-Z1)/(Z2-Z1)*b02
                    b1f=(Z2-j)/(Z2-Z1)*b11+(j-Z1)/(Z2-Z1)*b12
                    db0f=(Z2-j)/(Z2-Z1)*db01+(j-Z1)/(Z2-Z1)*db02
                    db1f=(Z2-j)/(Z2-Z1)*db11+(j-Z1)/(Z2-Z1)*db12
                
                   
                    dadu.append(dAdU[count]*(b0f-target.telescopes[i].gamma*b1f)+Ampli[0][count]*1/P[3]*(db0f-target.telescopes[i].gamma*db1f))
                    dadrho.append(-Ampli[0][count]*Ampli[1][count]/P[3]**2*(db0f-target.telescopes[i].gamma*db1f))
                    
                else :
 
                   dadu.append(dAdU[count])
                   dadrho.append(0)

                count=count+1
           
            dadu=np.array(dadu)
            dadrho=np.array(dadrho)
            
            dUdto=-(Lightcurve[:,0]-P[0])/(P[2]**2*Ampli[1])
            dUduo=P[1]/Ampli[1]
            dUdtE=-(Lightcurve[:,0]-P[0])**2/(P[2]**3*Ampli[1])
            
         
            
            
            dresdto=np.append(dresdto,-P[istart[model[0]]+2*i]*dadu*dUdto/errflux)
            dresduo=np.append(dresduo,-P[istart[model[0]]+2*i]*dadu*dUduo/errflux)
            dresdtE=np.append(dresdtE,-P[istart[model[0]]+2*i]*dadu*dUdtE/errflux)
            dresdrho=np.append(dresdrho,-P[istart[model[0]]+2*i]*dadrho/errflux)            
            
            
            
            
            Ampli=amplification(P,Lightcurve[:,0],model,target,target.telescopes[i].gamma)
            dresdfs=np.append(dresdfs,-(Ampli[0]+P[istart[model[0]]+2*i+1])/errflux)
            dresdeps=np.append(dresdeps,-P[istart[model[0]]+2*i]/errflux)
            
        jacobi=np.array([dresdto,dresduo,dresdtE,dresdrho])
        start=0
        for i in target.telescopes:
            dFS=np.zeros((len(dresdto)))
            dEPS=np.zeros((len(dresdto)))
                        
            index=np.arange(start,start+len(i.lightcurve[:,0]))
            dFS[index]=dresdfs[index]
            dEPS[index]=dresdeps[index]
            start=start+index[-1]+1                                        
            
            jacobi=np.vstack([jacobi,dFS])
            jacobi=np.vstack([jacobi,dEPS])
        
    #import pdb; pdb.set_trace()
    return np.matrix(jacobi) 


#######################
#from MicroLFits import amplification
#def test_amplification():
   # P = 
    #t = 
    #model = 
    #target = 
    #gamma =
    
    #result = amplification(P,t,model,target,gamma)
    #assert result == [3,2,1]
    #assert_almost_equal(result, 3.12, sig_fig=2)
    
def amplification(P,t,model,target,gamma):
    '''
    Takes model parameters and calculates the tangential amplification of the frobniz, based on
    the approach of Bacehelet et.al 2013.

    Note that if you pass an unnormalised value for the doowitz, then the result is formally undefined.
    :params model - the model (choose from PSPL, FSPL or woob)    
    '''
    
    if model[0]=='PSPL':
        
        U=(P[1]**2+(t-P[0])**2/P[2]**2)**0.5
        U2=U**2
    
        amplI=(U2+2)/(U*(U2+4)**0.5)
        ampli=np.copy(amplI)
        
    if model[0]=='FSPL' :
        
        b0b1=model[1]
        b0=b0b1[:,1]
        b1=b0b1[:,2]
        zz=b0b1[:,0]

        U=np.sqrt(P[1]**2+(t-P[0])**2/P[2]**2)
 
        U2=U**2
    
        amplI=(U2+2)/(U*(U2+4)**0.5)
    
        
        
        Z=U/P[3]
        Amp=[]
        count=0
        for i in Z :
        
            if i<10.0:
                
                index=np.abs(zz-i).argmin()
              
                if (i-zz[index]<0) :

                    b01=b0[index-1]
                    b11=b1[index-1]
                    Z1=zz[index-1]
                    
                    b02=b0[index]
                    b12=b1[index]
                    Z2=zz[index]
			
                else :

                    b01=b0[index]
                    b11=b1[index]
                    Z1=zz[index]
			
                    b02=b0[index+1]
                    b12=b1[index+1]
                    Z2=zz[index+1]
			
				
	   	
		
                b0f=(Z2-i)/(Z2-Z1)*b01+(i-Z1)/(Z2-Z1)*b02
                b1f=(Z2-i)/(Z2-Z1)*b11+(i-Z1)/(Z2-Z1)*b12
		
                ampf=amplI[count]*(b0f-gamma*b1f)
          
            else:
           
               ampf=amplI[count]
        
            Amp.append(ampf)
            count=count+1
       
        ampli=np.array(Amp)
      
    return ampli,U   
    

    
    
def residuals(P,model,target): 
    
    istart={'PSPL':3,'FSPL':4}
    
    
    
    error=np.array([])  
    for i in xrange(len(target.telescopes)):
                
        
        
        Lightcurve=target.telescopes[i].lightcurve
        flux=10**((27.4-Lightcurve[:,1])/2.5)
        errflux=-Lightcurve[:,2]*flux*np.log(10)/2.5
        ampli=amplification(P,Lightcurve[:,0],model,target,target.telescopes[i].gamma)[0]
       
        
        error=np.append(error,((flux-ampli*P[istart[model[0]]+2*i]-P[istart[model[0]]+2*i+1]*P[istart[model[0]]+2*i])/errflux))
        
        #error=np.append(error,((flux-ampli*P[istart[model[0]]+2*i]-P[istart[model[0]]+2*i+1])/errflux))
        #error=np.append(error,((flux-ampli*P[istart[model[0]]+2*i]-(P[istart[model[0]]+2*i+1]-P[istart[model[0]]+2*i]))/errflux)) 
        #error=np.append(error,((flux-P[istart[model[0]]+2*i]*P[istart[model[0]]+2*i+1]*(ampli-1)-P[istart[model[0]]+2*i+1])/errflux)) 
        #error=np.append(error,((flux-P[istart[model[0]]+2*i]/(1+P[istart[model[0]]+2*i+1])*(ampli+P[istart[model[0]]+2*i+1]))/errflux))
        #error=np.append(error,((flux-P[istart[model[0]]+2*i]*ampli-(P[istart[model[0]]+2*i+1]-P[istart[model[0]]+2*i]))/errflux))
        
        
        #deno=sum((ampli)**2/(errflux)**2)*sum(1/errflux**2)-sum(ampli/errflux**2)**2
        #FFs=(sum(ampli*flux/errflux**2)*sum(1/errflux**2)-sum(ampli/errflux**2)*sum(flux/errflux**2))/deno
        #FFb=(sum((ampli)**2/(errflux)**2)*sum(flux/errflux**2)-sum(ampli/errflux**2)*sum(flux*ampli/errflux**2))/deno
    #print P,np.sum(error**2)
    #plt.scatter(Lightcurve[:,0],flux)
    #plt.plot(Lightcurve[:,0],P[istart[model[0]]+2*i]*(ampli+P[istart[model[0]]+2*i+1]),'r')
    #plt.show()
    
            

    return error
    
def chichi(P,model,target): 
    istart={'PSPL':3,'FSPL':4}
    
    error=np.array([])  
    for i in xrange(len(target.telescopes)):
        
        Lightcurve=target.telescopes[i].lightcurve
        flux=10**((27.4-Lightcurve[:,1])/2.5)
        errflux=-Lightcurve[:,2]*flux*np.log(10)/2.5
        ampli=amplification(P,Lightcurve[:,0],model,target,target.telescopes[i].gamma)[0]
        error=np.append(error,(flux-ampli*P[istart[model[0]]+2*i]-P[istart[model[0]]+2*i+1]*P[istart[model[0]]+2*i])/errflux)
        #error=np.append(error,(flux-ampli*P[istart[model[0]]+2*i]-P[istart[model[0]]+2*i+1])/errflux)
        #error=np.append(error,((flux-ampli*P[istart[model[0]]+2*i]-(P[istart[model[0]]+2*i+1]-P[istart[model[0]]+2*i]))/errflux)) 
        #error=np.append(error,((flux-ampli*P[istart[model[0]]+2*i]*P[istart[model[0]]+2*i+1]-P[istart[model[0]]+2*i+1]*(1-P[istart[model[0]]+2*i]))/errflux))
        #error=np.append(error,((flux-P[istart[model[0]]+2*i]/(1+P[istart[model[0]]+2*i+1])*(ampli+P[istart[model[0]]+2*i+1]))/errflux))
        #error=np.append(error,((flux-P[istart[model[0]]+2*i]*ampli-(P[istart[model[0]]+2*i+1]-P[istart[model[0]]+2*i]))/errflux))
    #print P,np.sum(error**2)
    chi=(error**2).sum()
    #import pdb; pdb.set_trace()
    return chi

def chichi_penalty(P,model,target): 
    
    istart={'PSPL':3,'FSPL':4}
    
    error=residuals(P,target,model)
    chi=(error**2).sum()
    
    return chi
 
 
def annual_parallax(model,target):
    
    topar=model[1][1]
    
    
   
    

    return 'hello'

def terrestrial_parallax(model,target):
    
    
   

   return 'hello'
   
   
def space_parallax(target):
    
    return 'Hello'
    
def HJD_to_JD(t,target):    
    
    AU=149597870700
    c=299792458 
    Earth_position=slalib.sla_epv(t)
    Sun_position=-Earth_position[0]
    
    Sun_angles=slalib.sla_cc2s(Sun_position)
    
    Target_angles=[target.ra,target.dec]
    
    Time_correction=np.sqrt(Sun_position[0]**2+Sun_position[1]**2+Sun_position[2]**2)*AU/c*(np.sin(Sun_angles[1])*np.sin(Target_angles[1])+np.cos(Sun_angles[1])*np.cos(Sun_angles[1]*np.cos(Target_angles[0]-Sun_angles[0])))
    JD=t+Time_correction
    
    
    return JD