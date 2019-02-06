# Welcome to pyLIMA tutorial! #

#We gonna see how to use your fancy fitting method and/or a different objective function
#instead of the standard pyLIMA fitting routines. We take the same example as example 1.

### First import the required libraries

import numpy as np
import matplotlib.pyplot as plt
import os, sys

from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA import microlmodels

### Create an event object. You can choose the name and RA,DEC in degrees :

your_event = event.Event()
your_event.name = 'your choice'
your_event.ra = 269.39166666666665 
your_event.dec = -29.22083333333333

## Now we need some observations. That's good, we obtain some data on two
### telescopes. Both are in I band and magnitude units :

data_1 = np.loadtxt('./Survey_1.dat')
telescope_1 = telescopes.Telescope(name='OGLE', camera_filter='I', light_curve_magnitude=data_1)

data_2 = np.loadtxt('./Followup_1.dat')
telescope_2 = telescopes.Telescope(name='LCOGT', camera_filter='I', light_curve_magnitude=data_2)

### Add the telescopes to your event :
your_event.telescopes.append(telescope_1)
your_event.telescopes.append(telescope_2)


### set gamma for each telescopes :

your_event.telescopes[0].gamma = 0.5
your_event.telescopes[1].gamma = 0.5

### Find the survey telescope :
your_event.find_survey('OGLE')

### Sanity check
your_event.check_event()

### Construct the model you want to fit. Let's go basic with a PSPL, without second_order effects :
model_1 = microlmodels.create_model('PSPL', your_event)

### Now we have to define your objective function. Here we take the simple chi^2, and fit in flux unit

def objective_function(fit_process_parameters, your_event, your_model):
    
        pyLIMA_parameters = your_model.compute_pyLIMA_parameters(fit_process_parameters)
        
        chichi = 0
        for telescope in your_event.telescopes:
            # Find the residuals of telescope observation regarding the parameters and model
            
            model = your_model.compute_the_microlensing_model(telescope, pyLIMA_parameters)
            flux= telescope.lightcurve_flux[:,1]
            errflux = telescope.lightcurve_flux[:,2]

            
            residus = (flux - model[0])/errflux 
            chichi += (residus ** 2).sum()
        
        return chichi
    
### Now we can use your fancy fitting routine. Let 's assume it is scipy.optimize.minimize

import scipy.optimize as so

### You need guess ....
your_guess=[79.963, -0.01, 9.6, 0.00027]
model_1.define_model_parameters()
result = so.minimize(objective_function, your_guess,args=(your_event,model_1))

print (result)



