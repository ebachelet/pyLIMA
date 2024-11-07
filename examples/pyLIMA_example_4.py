'''
Welcome to pyLIMA (v2) tutorial 4!

In this tutorial you will learn how to code your own objective function to be
optimised, instead of using the standard pyLIMA routines. For example, you might want
to use SIMPLEX chi^2 minimization, instead of LM.
In scipy.optimize the SIMPLEX method is called 'Nelder-Mead'.
We will use the same example light curves as in tutorial 1.
Please take some time to familiarize yourself with the pyLIMA documentation.
'''

import matplotlib.pyplot as plt
### Import the required libraries.
import numpy as np
import scipy.optimize as so
from pyLIMA.models import FSPL_model
from pyLIMA.outputs import pyLIMA_plots

from pyLIMA import event
from pyLIMA import telescopes

### Create a new EVENT object and give it a name.
your_event = event.Event()
your_event.name = 'My event name'

### You now need to associate some data sets with this EVENT. 
### For this example, you will use simulated I-band data sets from two telescopes,
# OGLE and LCO.
### The data sets are pre-formatted: column 1 is the date, column 2 the magnitude and
# column 3
### the uncertainty in the magnitude.
data_1 = np.loadtxt('./data/Survey_1.dat')
telescope_1 = telescopes.Telescope(name='OGLE',
                                   camera_filter='I',
                                   lightcurve=data_1.astype(float),
                                   lightcurve_names=['time', 'mag', 'err_mag'],
                                   lightcurve_units=['JD', 'mag', 'mag'])

data_2 = np.loadtxt('./data/Followup_1.dat')
telescope_2 = telescopes.Telescope(name='LCO',
                                   camera_filter='I',
                                   lightcurve=data_2.astype(float),
                                   lightcurve_names=['time', 'mag', 'err_mag'],
                                   lightcurve_units=['JD', 'mag', 'mag'])

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
your_event.telescopes[0].ld_gamma = 0.5
your_event.telescopes[1].ld_gamma = 0.5

### Next, construct the MODEL you want to fit and link it to the EVENT you prepared. 
### Let's go with a basic FSPL, without second order effects:

fspl = FSPL_model.FSPLmodel(your_event)


### Now we want to define the OBJECTIVE FUNCTION to use for the MODEL you prepared.
### Here we take a simple chi^2, and fit in flux units:
def chisq(fit_process_parameters, your_model):
    pyLIMA_parameters = your_model.compute_pyLIMA_parameters(fit_process_parameters)

    chichi = 0
    for telescope in your_model.event.telescopes:
        # Compute fit residuals
        model = your_model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[
            'photometry']
        flux = telescope.lightcurve['flux'].value
        errflux = telescope.lightcurve['err_flux'].value
        residus = (flux - model) / errflux
        chichi += (residus ** 2).sum()

    return chichi


### Now we can use your chisq OBJECTIVE FUNCTION for the fit.
### Let's assume it is scipy.optimize.minimize

### You need a reasonable starting guess ...
your_guess = [79.963, 0.01, 9.6, 0.04]

### This next command is done automatically in pyLIMA for the default optimizers but 
### since we defined our own OBJECTIVE FUNCTION here we need to call it explicitly
### to initialize the parameters in the model.
fspl.define_model_parameters()

### Now run the optimization using your chisq function:
result = so.minimize(chisq, your_guess, args=(fspl), method='Nelder-Mead')
print(result)

### Let's look at the optimized parameters and the chi^2 of the fit.
### In this particular case, the function we defined uses scipy.optimize, 
### where the optimized parameters are stored in result.x and the chi^2 in result.fun.
print("Optimized parameters:", result.x)
print("chi^2:", result.fun)

### In case you have forgotten, the order and names of the parameters can be obtained
# from:
fspl.model_dictionnary

### Finally, let's look at the plot of the fit. Import the pyLIMA plotting tools:

pyLIMA_plots.plot_lightcurves(fspl, result.x)
plt.show()

### This concludes tutorial 4.
