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
import glob
import matplotlib.pyplot as plt

import numpy as np

import event
import telescopes
import microlmodels
# location = 'Space'
# model = 'PSPL'
second_order = [['None', 2457027.16667], ['None', 0], ['None', 0], 'None']


def main(command_line):
    #import pdb; pdb.set_trace()
    #events_names = np.loadtxt(command_line.input_directory+'OGLE_2015.txt',dtype='str')
    #doublons = np.loadtxt(command_line.input_directory+'multiples.list',dtype='str')
    #events_names = [os.path.split(x)[1] for x in glob.glob(command_line.input_directory + '/*.txt')]
    #print 'event_names = ', events_names

    #events_names=[event_name for event_name in os.listdir(command_line.input_directory) if 'Survey'  in event_name]
    events_names=[event_name for event_name in os.listdir(command_line.input_directory) if '.dat'  in event_name]
    #vents_names=['bul-'+str(i+1).zfill(2) for i in xrange(75)]
    
    #events_names=sorted(os.listdir(command_line.input_directory))
    #for i in doublons.ravel() :
     #  if i[0]=='K' :
      #      if i in events_names :
               
       #         index = np.where(events_names == i)[0]
        #        events_names = np.delete(events_names,index)

    #events_names2=[event_name for event_name in os.listdir(command_line.input_directory) if '.dat'  in event_name]
    # EEvents_names=[event_name for event_name in os.listdir(events_path) if '.phot' in event_name]
    events = []
    start = time.time()
    results = []
    errors = []
    #events_names=[i for i in events_names if '.dat' in i]
    #import pdb; pdb.set_trace()
    
    for event_name in events_names[0:]:
    #for j in xrange(10000):
        #j=j
        #event_name='bul-74'
        name='Lightcurve_'+str(5930)
        ##name = event_name[:-10]
        #name = 'OB150034'
        #name=event_name
        #name=event_name[:-10]
        #event_name='bul-74'
        #name='lc_0.0101_21.0_0.0_12.0_0.001'
        #name='lc_0.0001_81.0_0.0_13.0_0.011'
        current_event = event.Event()
        current_event.name = name
        current_event.ra = 269.39166666666665 
        current_event.dec = -29.22083333333333
        #event_telescopes = ['OGLE']
        event_telescopes = [i for i in events_names if name  in i]
        #event_telescopes = [i for i in events_names2 if 'dat'  in i]
        #import pdb; pdb.set_trace()
        #filters = ['I','J']    
        count=0
        #event_telescopes = [event_telescopes[1]]
        #second_order = [['Annual', 2457027], ['None', 0], ['None', 0], 'None']
        start=time.time()
        for event_telescope in event_telescopes:
            try :
               raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope, usecols=(0, 1, 2))
               
               lightcurve=np.array([raw_light_curve[:,0],raw_light_curve[:,1],raw_light_curve[:,2]]).T
               if lightcurve[0,0]>2450000 :
                   lightcurve[:,0] = lightcurve[:,0]-2450000
            except :
               lightcurve=np.array([raw_light_curve[0],raw_light_curve[1],raw_light_curve[2]]).T
            #import pdb; pdb.set_trace()

            #plt.scatter(raw_light_curve[:,0],raw_light_curve[:,1])
            #plt.gca().invert_yaxis()
            #lt.axis([min(raw_light_curve[:,0]),max(raw_light_curve[:,0]),max(raw_light_curve[:,1])+0.1,min(raw_light_curve[:,1])-0.5])
            #plt.show()
            #telescope = telescopes.Telescope(name=event_telescope[-10:-4], camera_filter='I', light_curve=lightcurve)
            telescope = telescopes.Telescope(name=event_telescope[2:-4], camera_filter=event_telescope[-5], light_curve=lightcurve)
            telescope.gamma=0.5
            if 'TG' in event_telescope :
                telescope.gamma = 0.22
                  
            else :
                pass
            current_event.telescopes.append(telescope)
            count+=1
            #try :
             #   raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope, usecols=(0, 1, 2))
              #  raw_light_curve=np.array([raw_light_curve[:,2],raw_light_curve[:,0],raw_light_curve[:,1]]).T
                #raw_light_curve = np.genfromtxt(command_line.input_directory + event_telescope, usecols=(0, 1, 2))
               # telescope = telescopes.Telescope(name=event_telescope[0]+event_telescope[-5], camera_filter=event_telescope[-4], light_curve=raw_light_curve)
                #telescope.name=k[-1][:-4]
                #telescope.name=k[1]
                #telescope.name=event_telescope[:4]
                # telescope.name=k[0]
                #telescope.filter = filters[count]            
                #telescope.find_gamma(5300.0, 4.5, command_line.claret)
                #current_event.telescopes.append(telescope)
                #count = count+1cc
            #except:
             #   pass
        #events.append(current_event)
        print 'Start;', current_event.name
        #import pdb; pdb.set_trace()
        # current_event.check_event()
        #telescopes_names = [i.name for i in current_event.telescopes]
       # if 'OI' in  telescopes_names:
            
            #current_event.find_survey('OI')
       # if :
            #import pdb; pdb.set_trace()
            #current_event.find_survey('KI')

        current_event.check_event()
        #current_event.find_survey('OGLE_I')
        
        #second_order = [['None', 2457027], ['None', 0], ['None', 0], 'None']
        #second_order = [['Annual', 2457027], ['None', 0], ['None', 0], 'None']
        #current_event.fit('FSPL', second_order,0)
        #current_event.fit('FSPL', second_order,0)
        #import pdb; pdb.set_trace()
        #second_order = [['Annual', 2457027.1], ['None', 0], ['None', 0], 'None']
        #current_event.fit(command_line.model, second_order,1)
        #current_event.telescopes.append(telescope)
        #if second_order[0][0]!='None' :
            #current_event.compute_parallax(second_order)
        Model = microlmodels.MLModels(current_event, command_line.model, second_order)
        
        current_event.fit(Model,'MCMC')
        import pdb; pdb.set_trace()

        #current_event.telescopes.append(telescope)
        #if second_order[0][0]!='None' :
            #current_event.compute_parallax(second_order)
       # Model = microlmodels.MLModels(current_event, command_line.model, second_order)
        
        #current_event.fit(Model,1)    
        #import pdb; pdb.set_trace()
    
        
        #Parameters,lightcurve_model,lightcurve_data = current_event.produce_outputs(0)
        #header = 'Lightcurve data for '+name+' \n\nColumn 1 : Timestamps in HJD \nColumn 2 : Magnitudes \nColumn 3 : Magnitudes error \nColumn 4 : Data provider'
        #lightcurve_data = lightcurve_data[lightcurve_data[:,0].argsort(),]
        #np.savetxt(name+'.data',lightcurve_data,header =  header,fmt = '%s')
        #header = 'Model lightcurve for '+name+' \n\nColumn 1 : Timestamps in HJD \nColumn 2 : Magnitudes \nColumn 3: Magnitudes error'
       
        #np.savetxt(name+'.model',lightcurve_model,header =  header,fmt = '%s')
        
       # np.savetxt(name+'.param',Parameters,fmt='%s')
        #current_event.produce_outputs(0)
        current_event.fits[0].produce_outputs()
        Results = []
        Errors = []
        
        Results.append(current_event.name)
        Errors.append(current_event.name)
        Results.append(current_event.fits[0].method)
        Errors.append(current_event.fits[0].method)
        
        #To,Uo,tE,Fs,Fb=map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]),zip(*np.percentile(current_event.fits[0].samples,[16,50,84],axis=0)))
        #A=[To,Uo,tE,Fs,Fb]
        #for i in xrange(5):
            
            #Results+=[A[i][0]]
            #Errors+=[A[i][1],A[i][2]]
        
        #Results.append(current_event.fits[0].fit_time)
        #header_model=['#Event', 'Method']+current_event.fits[0].model.model_dictionnary.keys()+['Chi2','Fit_time']
        #header_errors=['#Event', 'Method']+current_event.fits[0].model.model_dictionnary.keys()
        #current_event.produce_outputs(0) 
        #for i in current_event.fits[0].model.model_dictionnary.keys() :
           
         #           Results.append(current_event.fits[0].fit_results[current_event.fits[0].model.model_dictionnary[i]])
          #          Errors.append(current_event.fits[0].fit_errors[current_event.fits[0].model.model_dictionnary[i]])
        #Results.append(current_event.fits[0].fit_results[-1])
        #Results.append(current_event.fits[0].fit_time)
        #Results=np.array(Results)
        #Errors=np.array(Errors)
        #np.savetxt(os.path.join(command_line.output_directory,name+'.model'),np.hstack((header_model,Results)).reshape(2,Results.shape[0]),newline='\r\n',fmt='%s')
        #np.savetxt(os.path.join(command_line.output_directory,name+'.errors'),np.hstack((header_errors,Errors)).reshape(2,Errors.shape[0]),newline='\r\n',fmt='%s')
        #import pdb; pdb.set_trace()   
        #for ll in [0,1] :
         #       Results = []
          #      Errors = []
          #      current_event.produce_outputs(ll)  
          #      if ll==0 :
                    
          #          header_model=['#Event', 'Method']+current_event.fits[ll].model.model_dictionnary.keys()+['Chi2_earth','Time']
          #          name = name+'_earth'
          #      else :
          #          header_model=['#Event', 'Method']+current_event.fits[ll].model.model_dictionnary.keys()+['Chi2_earth','Chi2_swift','Time']
          #          name = name[:-6]+'_all'
                    
          #      Results.append(current_event.name)
          #      Errors.append(current_event.name)
          #      Results.append(current_event.fits[ll].method)
          #      Errors.append(current_event.fits[ll].method)
          #      for i in current_event.fits[ll].model.model_dictionnary.keys() :
           
          #          Results.append(current_event.fits[ll].fit_results[current_event.fits[ll].model.model_dictionnary[i]])
          #          Errors.append(current_event.fits[ll].fit_errors[current_event.fits[ll].model.model_dictionnary[i]])
        
                
          #      header_errors = header_model[:-2]
          #      if ll==1:
                    
          #          Results.append(current_event.fits[ll].fit_results[-2])
          #          header_errors = header_model[:-3]
          #      Results.append(current_event.fits[ll].fit_results[-1])
                    
          #      Results.append(current_event.fits[ll].fit_time)
                
                
          #      Results = np.array(Results)
          #      Errors = np.array(Errors)
          #      results.append(Results)
          #      errors.append(Errors)
          #      np.savetxt(os.path.join(command_line.output_directory,name+'.model'),np.hstack((header_model,Results)).reshape(2,Results.shape[0]),newline='\r\n',fmt='%s')
          #      np.savetxt(os.path.join(command_line.output_directory,name+'.errors'),np.hstack((header_errors,Errors)).reshape(2,Errors.shape[0]),newline='\r\n',fmt='%s')
        #import pdb; pdb.set_trace()  
        
        Results+=current_event.fits[0].fit_results
        Errors+=(current_event.fits[0].fit_covariance.diagonal()**0.5).tolist()
        Results.append(current_event.fits[0].fit_time) 
        
        results.append(Results)
        errors.append(Errors)
        print time.time()-start
    end = time.time()

    print end - start

    all_results = [('Fits.txt', results),
                   ('Fits_Error.txt', errors)]

    for file_name, values in all_results:
        np.savetxt(os.path.join(command_line.output_directory, file_name), np.array(values), fmt="%s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='FSPL')
    #parser.add_argument('-i', '--input_directory', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Artemis_2015/PSPL_2015/ProcessedData/2015/')
    #parser.add_argument('-i', '--input_directory', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Short_tE/Lightcurves/')
    parser.add_argument('-i', '--input_directory', default='/nethome/Desktop/Microlensing/OpenSourceProject/SimulationML/Lightcurves_Ground/Lightcurves/')
    #parser.add_argument('-i', '--input_directory', default='/home/ebachelet/Desktop/nethome/Desktop/Qatar/SuperComputer/OB13446/RawData/DATA/')
    #parser.add_argument('-i', '--input_directory', default='/home/ebachelet/Desktop/nethome/Desktop/RSTREET_PLANET/Early/')
    #parser.add_argument('-o', '--output_directory', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/OGLE_2000/')
    #parser.add_argument('-o', '--output_directory', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/MOA-2007-400/')
    parser.add_argument('-o', '--output_directory', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Developement/Fitter/FSPL/')
   # parser.add_argument('-o', '--output_directory', default='/home/ebachelet/Desktop/nethome/Desktop/RSTREET_PLANET/Fits_Early/')

    parser.add_argument('-c', '--claret', default='/home/ebachelet/Desktop/nethome/Desktop/Microlensing/OpenSourceProject/Claret2011/J_A+A_529_A75/')
    arguments = parser.parse_args()

    model = arguments.model

    main(arguments)
