# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/ebachelet/.spyder2/.temp.py
"""
import MicroLFits

class Event(object):
    
        
    def __init__(self):
        
        self.kind = 'Microlensing'
        self.name='None'
        self.ra='None'
        self.dec='None'
        self.Teff=3000
        self.logg=4.5
        self.telescopes=[]
        self.fit_results=[]
        self.fit_covariance=[]
        self.fit_time=[]
        
        
    def fit(self, model, method):
        
        Methods=[0]
        if method not in Methods :
            
            print 'Wrong method request, has to be an integer'
        
        else :

            if self.kind=='Microlensing':
                
               
               
                
                available=['PSPL','FSPL'] 
            
                if model not in available :
            
                    print 'This is not a possible option for a microlensing fit yet'
                
                else :
                
                    if model=='PSPL':
                   
                        #print 'OK let go to a microlensing PSPL fit!' 
                        Fit=MicroLFits.ML_Fits(self,model,method)
                       

                        
    
                        self.fit_results=[Fit.model[0],Fit.method,Fit.fit_results]
                        self.fit_covariance=[Fit.model[0],Fit.method,Fit.fit_covariance]
                        self.fit_time=[Fit.model,Fit.fit_time]
                    if model=='FSPL':
                   
                        #print 'OK let go to a microlensing FSPL fit!' 
                        Fit=MicroLFits.ML_Fits(self,model,method)
                       

                        
    
                        self.fit_results=[Fit.model[0],Fit.method,Fit.fit_results]
                        self.fit_covariance=[Fit.model[0],Fit.method,Fit.fit_covariance]
                       
            else :
            
           
                
                print 'No possible fit yet for a non microlensing event, sorry :('
    
    def plot_lightcurve(self):
        import matplotlib.pyplot as plt
        for i in self.telescopes :
            time=i.lightcurve[:,0]
            if time[0]>2450000 :
               time=time-2450000
            plt.errorbar(time,i.lightcurve[:,1],yerr=i.lightcurve[:,2],linestyle='none')
        plt.gca().invert_yaxis()
        #plt.show()
        
    def plot_model(self,P):
        import matplotlib.pyplot as plt
        import numpy as np
        print P
        t=np.arange(min(self.telescopes[0].lightcurve[:,0]),max(self.telescopes[0].lightcurve[:,0]),0.01)

        to=P[0]
        if self.telescopes[0].lightcurve[0,0]>2450000 :
              t=t-2450000
              to=to-2450000
        uo=P[1]
        tE=P[2]
        fs=P[3]
        fb=P[4]*fs
        U=np.sqrt(uo**2+(t-to)**2/tE**2)
        A=(U**2+2)/(U*np.sqrt(U**2+4))
        
        plt.plot(t,27.4-2.5*np.log10(fs*A+fb))

    def telescopes_names(self):
        

        print [self.telescopes[i].name for i in xrange(len(self.telescopes))]
       
    def check(self):
                    
        if (type(self.name)=='None'):
                
            print 'ERROR : The event name ('+str(self.name)+') is not correct, it has to be a string'
            return
        
        #if (self.ra==0):
                
            #print 'ERROR : The event ra ('+str(self.ra)+') is not correct'
            #return
        
        #if (self.dec==0):
                
            #print 'ERROR : The event dec ('+str(self.dec)+') is not correct'
            #return
       
            

        print 'Everything is fine, this event can be treat' 

