'''
Welcome to pyLIMA (v2) tutorial 2!

This second tutorial will give you some basics about how to reconfigure your input parameters.
If you do not like the standard pyLIMA parameters, this is made for you.
We are going to fit the same light curves as in tutorial 1, but using different parametrization.
'''

### First import the required libraries as before.
import numpy as np
import matplotlib.pyplot as plt
import os, sys

from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA import microlmodels

### Create a new EVENT object and give it a name.
your_event = event.Event()
your_event.name = 'My event name'

### You now need to associate some data sets with this EVENT. 
### For this example, you will use simulated I-band data sets from two telescopes, OGLE and LCO.
### The data sets are pre-formatted: column 1 is the date, column 2 the magnitude and column 3 
### the uncertainty in the magnitude.
data_1 = np.loadtxt('./Survey_1.dat')
telescope_1 = telescopes.Telescope(name = 'OGLE', 
                                   camera_filter = 'I',
                                   light_curve = data_1.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

data_2 = np.loadtxt('./Followup_1.dat')
telescope_2 = telescopes.Telescope(name = 'LCO', 
                                   camera_filter = 'I',
                                   light_curve = data_2.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

### Append these two telescope data sets to your EVENT object.
your_event.telescopes.append(telescope_1)
your_event.telescopes.append(telescope_2)

### Define the survey telescope that you want to use to align all other data sets to.
### We recommend using the data set with the most measurements covering the gretest 
### time span of observations:
your_event.find_survey('OGLE')

### Run a quick sanity check on your input.
your_event.check_event()

### Set the microlensing limb-darkening coefficients (gamma) for each telescope:
your_event.telescopes[0].gamma = 0.5
your_event.telescopes[1].gamma = 0.5

### Fit an FSPL model to the data using the Trust Region Reflective (TRF) algorithm:
### We can make this faster by using the results from tutorial 1.
guess_parameters = [79.9, 0.008, 10.1, 0.023]

### Define the model and fit method (as in tutorial 1):
from pyLIMA.models import FSPL_model
fspl = FSPL_model.FSPLmodel(your_event)

### Import the TRF fitting algorithm
from pyLIMA.fits import TRF_fit
my_fit = TRF_fit.TRFfit(fspl)
my_fit.model_parameters_guess = guess_parameters
my_fit.fit()

### Let's see the plot. Zoom close to the peak again to see what is going on.
from pyLIMA.outputs import pyLIMA_plots
pyLIMA_plots.plot_lightcurves(fspl, my_fit.fit_results['best_model'])
plt.show()

### All right, looks fine, as before. 

### Now let's say you dislike the rho parameter and you want to change it to use log(rho) instead. 
### Let's see how to do this.

### We need to specify within pyLIMA how the new parameter depends on the old parameter.
### In essence, we need to define a transformation function within pyLIMA.
### For this particular transformation, i.e. from rho to log(rho), pyLIMA already provides the necessary functions to convert back and forth: 
from pyLIMA.fits import fancy_parameters
fancy_parameters.fancy_parameters_dictionnary = {'log_rho':'rho'}

### To see all available default options within pyLIMA, please look at the fancy_parameters module
### or type dir(fancy_parameters) in the Python prompt.

### Now we need to perform a new fit with the newly defined parameters.
### Use the same guess parameters as before but this time the last 
### parameter needs to be log(rho), as we previously defined it:
guess_parameters2 = [79.9, 0.008, 10.1, np.log10(0.023)]

### We need to tell the fit to use fancy_parameters, otherwise it will use the defaults.
my_fit2 = TRF_fit.TRFfit(fspl, fancy_parameters=True)
my_fit2.model_parameters_guess = guess_parameters2
my_fit2.fit()

### So this works great! 

### OK, try something more complicated now: define t_star = rho*tE and log_rho = log(rho).
### log_rho is already provided by pyLIMA, but t_star isn't. 
### So we need to tell pyLIMA what kind of change we want:
def t_star(x):
    return x.rho*x.tE

setattr(fancy_parameters, 't_star', t_star)

def tE(x):
    return x.t_star/10**(x.log_rho)

setattr(fancy_parameters, 'tE', tE)

### Update the parameter dictionary with the new definitions
fancy_parameters.fancy_parameters_dictionnary = {'log_rho':'rho', 't_star':'tE'}

### t_star = rho * tE so in our example that is 10.1 * 0.023 (see guess_parameters2 above):
guess_parameters3 = [79.9, 0.008, 10.1 * 0.023, np.log10(0.023)]

### Do the fit using the new parameter definitions:
my_fit3 = TRF_fit.TRFfit(fspl, fancy_parameters=True)
my_fit3.model_parameters_guess = guess_parameters3
my_fit3.fit()

### Let's look at the optimized parameters and the chi^2 of the fit:
my_fit3.fit_results['best_model']
my_fit3.fit_results['chi2']

### If you have forgotten the order of the parameters, do:
my_fit3.fit_parameters.keys()

### Note that the results now are displayed with our newly defined parameters.

### This concludes tutorial 2.
