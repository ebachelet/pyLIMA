'''
Welcome to pyLIMA (v2) tutorial 1!

In this tutorial you will learn how pyLIMA works by fitting a simulated data set.
We will cover how to read in data files, call different fitting routines and how to
make plots.
Please take some time to familiarize yourself with the pyLIMA documentation.
'''
### Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from pyLIMA.fits import DE_fit
from pyLIMA.fits import LM_fit
from pyLIMA.fits import MCMC_fit
from pyLIMA.models import FSPL_model
from pyLIMA.models import PSPL_model
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

### Next, construct the MODEL you want to fit and link it to the EVENT you prepared. 
### Let's go with a basic PSPL, without second order effects:

pspl = PSPL_model.PSPLmodel(your_event)

### Let's try fitting the event with a simple Levenvberg_Marquardt (LM) algorithm.
### Define the FITTING ALGORITHM you want to use for the MODEL you prepared.
### For more information about the models and fitting algorithms available  
### please consult the pyLIMA documentation.

### Initialize the fit by declaring a simple FIT object using the MODEL you defined:
my_fit = LM_fit.LMfit(pspl)

### Before we run it, let's have a look at the initial fit parameters:
my_fit.fit_parameters

### Now fit the MODEL to the EVENT. This may take a few seconds.
my_fit.fit()

my_fit.fit_outputs()
### You can now recall the fit results on the screen by executing:
my_fit.fit_results

### You can now recall any entry in the output dictionary by using the appropriate key.
### For example, if you want to see the best fit results, you can access them like this:
my_fit.fit_results['best_model']

### If you don't remember which parameter each entry represents, you can always
# access the
### descriptions from fit_parameters.
my_fit.fit_parameters.keys()

### Let's see some plots. Import the pyLIMA plotting tools.

pyLIMA_plots.plot_lightcurves(pspl, my_fit.fit_results['best_model'])
plt.show()

### Let's try another fit with the differential evolution (DE) algorithm. 
### This will take longer... 

my_fit2 = DE_fit.DEfit(pspl,loss_function='chi2')
my_fit2.fit()

### Look at the results:
pyLIMA_plots.plot_lightcurves(pspl, my_fit2.fit_results['best_model'])
plt.show()

### You can use the Zoom-in function to look at the peak.
### There is strong evidence of finite source effects in this event, so let's try to
# fit this.
### You will need to import the FSPL MODEL to do this:

fspl = FSPL_model.FSPLmodel(your_event)

### You can still use the FITTING ALGORITHM that you imported previously. 
### Let's just use DE_fit for this:
my_fit3 = DE_fit.DEfit(fspl, loss_function='chi2')
my_fit3.fit()

### Let's see some plots. You can zoom close to the peak to see what is going on.
pyLIMA_plots.plot_lightcurves(fspl, my_fit3.fit_results['best_model'])
plt.show()

### There is evidently still some structure in the residuals. Could be some limb
# darkening going on!
### Let's try to fit for it.

### Set the microlensing limb-darkening coefficients (gamma) for each telescope:
your_event.telescopes[0].ld_gamma = 0.5
your_event.telescopes[1].ld_gamma = 0.5

### Fit again:
my_fit4 = DE_fit.DEfit(fspl, loss_function='chi2')
my_fit4.fit()

### And plot it. Then zoom at the peak again.
pyLIMA_plots.plot_lightcurves(fspl, my_fit4.fit_results['best_model'])
plt.show()

### You can use the results of a previous good fit as initial guesses 
### for the parameters in another fit:
guess_parameters = my_fit4.fit_results['best_model']
print(guess_parameters)

### These parameter guesses can now be used to start an MCMC run, for example.
### Using MCMC is recommended when you want to explore the posterior distribution of
# the parameters.
### Let's fit again using MCMC. This might take some time ...

my_fit5 = MCMC_fit.MCMCfit(fspl)
my_fit5.model_parameters_guess = guess_parameters
my_fit5.fit()

### Now your MCMC run is complete. Congratulations! 
### You can now plot the chains and explore how they evolve for each parameter.
### For example, to see how the chains for u0 evolve, do:
plt.plot(my_fit5.fit_results['MCMC_chains'][:, :, 1])
plt.show()

### The first part in the slice [:,:,1] represents the iteration number, the second
# the chain number
### and the last represents the parameter number (in addition to the likelihood at
# the end).
### The parameters are in the same order as in my_fit5.fit_parameters.keys()

### You can compare the MCMC distributions with the input values that were used to
# generate the light curve.
### For this, let's only consider the chains after the 1000th iteration (i.e. after
# burn-in).
### [:7] at the end is just so only the first 7 digits are printed.
MCMC_results = my_fit5.fit_results['MCMC_chains']
print('Parameters', ' Model', '   Fit', '     Errors')
print('-----------------------------------')
print('t_0:', '        79.9309 ', str(np.median(MCMC_results[1000:, :, 0]))[:7], '',
      str(np.std(MCMC_results[1000:, :, 0]))[:7])
print('u_0:', '        0.00826 ', str(np.median(MCMC_results[1000:, :, 1]))[:7], '',
      str(np.std(MCMC_results[1000:, :, 1]))[:7])
print('t_E:', '        10.1171 ', str(np.median(MCMC_results[1000:, :, 2]))[:7], '',
      str(np.std(MCMC_results[1000:, :, 2]))[:7])
print('rho:', '        0.02268 ', str(np.median(MCMC_results[1000:, :, 3]))[:7], '',
      str(np.std(MCMC_results[1000:, :, 3]))[:7])

### You can now plot the correlation between any two parameters.
### Import the relevant libraries:

### Now plot u0 against tE:
plt.hist2d(MCMC_results[1000:, :, 1].ravel(), MCMC_results[1000:, :, 2].ravel(),
           norm=LogNorm(), bins=50)
plt.xlabel('u0')
plt.ylabel('tE')
plt.show()

### You can consult the matplotlib.pyplot.hist2d documentation to see additional
# arguments.

### This concludes tutorial 1.
