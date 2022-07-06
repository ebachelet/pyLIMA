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


fancy_parameters.fancy_parameters_dictionnary = {'log_rho':'rho', 't_star':'tE'}

### t_star = rho * tE so in our example that is 10.1 * 0.023 (see guess_parameters2 above):
guess_parameters3 = [79.9, 0.008, 10.1 * 0.023, np.log10(0.023)]

my_fit3 = TRF_fit.TRFfit(fspl, fancy_parameters=True)
my_fit3.model_parameters_guess = guess_parameters3
my_fit3.fit()

# This means we change rho by log(rho) and tE by tstar in the fitting process.

### We need now to explain the mathematical transformation :
def logrho(x): return np.log10(x.rho)
def rho(x): return 10**x.logrho

def tstar(x): return x.rho*x.tE
def tE(x): return x.tstar/10**x.logrho

model_1.pyLIMA_to_fancy = {'logrho':pickle.loads(pickle.dumps(logrho)),'tstar':pickle.loads(pickle.dumps(tstar))}

### We also need to explain the inverse mathematical transformation :

model_1.fancy_to_pyLIMA = {'rho': pickle.loads(pickle.dumps(rho)),'tE': pickle.loads(pickle.dumps(tE))}

### Change tE boundaries to tstar boundaries (i.e [log10(rhomin)*tEmin, log10(rhomax)*tEmax]) :
model_1.parameters_boundaries[2] = [10**-5, 300 ]

### Change rho boundaries to logrho boundaries (i.e [log10(rhomin), log10(rhomax)]) :
model_1.parameters_boundaries[3] = [-5, -1]

### Give some guess for LM
model_1.parameters_guess = [79.93092292215124, 0.008144793661913143, 0.22, -1.6459136264565297]

### That's it, let's fit!
your_event.fit(model_1,'LM')

your_event.fits[-1].produce_outputs()
print('Chi2_LM :',your_event.fits[-1].outputs.fit_parameters.chichi)
print('tstar : ',your_event.fits[-1].outputs.fit_parameters.tstar)
print('Corresponding tE: ',your_event.fits[-1].outputs.fit_parameters.tstar/10**your_event.fits[-1].outputs.fit_parameters.logrho)

print('Log rho : ',your_event.fits[-1].outputs.fit_parameters.logrho)
print('Corresponding rho : ',10**your_event.fits[-1].outputs.fit_parameters.logrho)

plt.show()

### And what about the DE method? ### Let's try it!:
your_event.fit(model_1,'DE')


your_event.fits[-1].produce_outputs()

print('Chi2_DE :',your_event.fits[-1].outputs.fit_parameters.chichi)

print('tstar : ',your_event.fits[-1].outputs.fit_parameters.tstar)

print('Corresponding tE: ',your_event.fits[-1].outputs.fit_parameters.tstar/10**your_event.fits[-1].outputs.fit_parameters.logrho)



print('Log rho : ',your_event.fits[-1].outputs.fit_parameters.logrho)

print('Corresponding rho : ',10**your_event.fits[-1].outputs.fit_parameters.logrho)


plt.show()

# Bonus Track #
### Let's win some times by injecting some previous results

model_1.parameters_guess = [79.9, 0.008, 0.22849, -1.6459]

### Fit again, but using MCMC now. TAKE A WHILE....Wait until figures pop up.
your_event.fit(model_1,'MCMC',flux_estimation_MCMC='MCMC')
print('The fitting process is finished now, let produce some outputs....')

your_event.fits[-1].produce_outputs()

plt.show()
