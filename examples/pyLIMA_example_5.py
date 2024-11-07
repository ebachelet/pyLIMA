'''
Welcome to pyLIMA (v2) tutorial 5!

In this tutorial you will learn how to fit an actual planetary event using real data.
The event is OB150966 and the relevant publication is:
    https://ui.adsabs.harvard.edu/abs/2016ApJ...819...93S/

Please take some time to familiarize yourself with the pyLIMA documentation.
'''

import multiprocessing as mul

import matplotlib.pyplot as plt
### First import the required libraries
import numpy as np
from pyLIMA.fits import DE_fit
from pyLIMA.fits import TRF_fit
from pyLIMA.models import PSPL_model
from pyLIMA.models import USBL_model, pyLIMA_fancy_parameters
from pyLIMA.outputs import pyLIMA_plots

from pyLIMA import event
from pyLIMA import telescopes

### Create a new EVENT object and give it a name.
# Here RA and DEC matter (because the event has a strong parallax signal) !!!
your_event = event.Event(ra=268.75425, dec=-29.047111111111114)
your_event.name = 'OB150966'

### You now need to associate all data sets with this EVENT. 
### There are 11 sets of observations and we want to include all of them. 
### You could do this in a loop or load each of them individually as in this example.

### The data sets are already pre-formatted: 
###     column 1 is the date, column 2 the magnitude and column 3 
###     the uncertainty in the magnitude.
data_1 = np.loadtxt('./data/OGLE_OB150966.dat')
telescope_1 = telescopes.Telescope(name='OGLE',
                                   camera_filter='I',
                                   lightcurve=data_1.astype(float),
                                   lightcurve_names=['time', 'mag', 'err_mag'],
                                   lightcurve_units=['JD', 'mag', 'mag'])

data_2 = np.loadtxt('./data/MOA_OB150966.dat')
telescope_2 = telescopes.Telescope(name='MOA',
                                   camera_filter='I+R',
                                   lightcurve=data_2.astype(float),
                                   lightcurve_names=['time', 'mag', 'err_mag'],
                                   lightcurve_units=['JD', 'mag', 'mag'])

data_3 = np.loadtxt('./data/SPITZER_OB150966.dat')
telescope_3 = telescopes.Telescope(name='SPITZER',
                                   camera_filter='IRAC1',
                                   lightcurve=data_3.astype(float),
                                   lightcurve_names=['time', 'mag', 'err_mag'],
                                   lightcurve_units=['JD', 'mag', 'mag'])

### IMPORTANT: Tell the code that SPITZER is in space!
telescope_3.location = 'Space'
telescope_3.spacecraft_name = 'Spitzer'

data_4 = np.loadtxt('./data/DANISH_OB150966.dat')
telescope_4 = telescopes.Telescope(name='DANISH',
                                   camera_filter='Z+I',
                                   lightcurve=data_4.astype(float),
                                   lightcurve_names=['time', 'mag', 'err_mag'],
                                   lightcurve_units=['JD', 'mag', 'mag'])

data_5 = np.loadtxt('./data/LCO_CTIO_A_OB150966.dat')
telescope_5 = telescopes.Telescope(name='LCO_CTIO_A',
                                   camera_filter='I',
                                   lightcurve=data_5.astype(float),
                                   lightcurve_names=['time', 'mag', 'err_mag'],
                                   lightcurve_units=['JD', 'mag', 'mag'])

data_6 = np.loadtxt('./data/LCO_CTIO_B_OB150966.dat')
telescope_6 = telescopes.Telescope(name='LCO_CTIO_B',
                                   camera_filter='I',
                                   lightcurve=data_6.astype(float),
                                   lightcurve_names=['time', 'mag', 'err_mag'],
                                   lightcurve_units=['JD', 'mag', 'mag'])

data_7 = np.loadtxt('./data/LCO_CTIO_OB150966.dat')
telescope_7 = telescopes.Telescope(name='LCO_CTIO',
                                   camera_filter='I',
                                   lightcurve=data_7.astype(float),
                                   lightcurve_names=['time', 'mag', 'err_mag'],
                                   lightcurve_units=['JD', 'mag', 'mag'])

data_8 = np.loadtxt('./data/LCO_SAAO_OB150966.dat')
telescope_8 = telescopes.Telescope(name='LCO_SAAO',
                                   camera_filter='I',
                                   lightcurve=data_8.astype(float),
                                   lightcurve_names=['time', 'mag', 'err_mag'],
                                   lightcurve_units=['JD', 'mag', 'mag'])

data_9 = np.loadtxt('./data/LCO_SSO_A_OB150966.dat')
telescope_9 = telescopes.Telescope(name='LCO_SSO_A',
                                   camera_filter='I',
                                   lightcurve=data_9.astype(float),
                                   lightcurve_names=['time', 'mag', 'err_mag'],
                                   lightcurve_units=['JD', 'mag', 'mag'])

data_10 = np.loadtxt('./data/LCO_SSO_B_OB150966.dat')
telescope_10 = telescopes.Telescope(name='LCO_SSO_B',
                                    camera_filter='I',
                                    lightcurve=data_10.astype(float),
                                    lightcurve_names=['time', 'mag', 'err_mag'],
                                    lightcurve_units=['JD', 'mag', 'mag'])

data_11 = np.loadtxt('./data/LCO_SSO_OB150966.dat')
telescope_11 = telescopes.Telescope(name='LCO_SSO',
                                    camera_filter='I',
                                    lightcurve=data_11.astype(float),
                                    lightcurve_names=['time', 'mag', 'err_mag'],
                                    lightcurve_units=['JD', 'mag', 'mag'])

### Add the telescopes to your EVENT:
your_event.telescopes.append(telescope_1)
your_event.telescopes.append(telescope_2)
your_event.telescopes.append(telescope_3)
your_event.telescopes.append(telescope_4)
your_event.telescopes.append(telescope_5)
your_event.telescopes.append(telescope_6)
your_event.telescopes.append(telescope_7)
your_event.telescopes.append(telescope_8)
your_event.telescopes.append(telescope_9)
your_event.telescopes.append(telescope_10)
your_event.telescopes.append(telescope_11)

### Define the survey telescope that you want to use to align all other data sets to.
### We recommend using the data set with the most measurements covering the gretest 
### time span of observations:
your_event.find_survey('OGLE')

### Run a quick sanity check on your input.
your_event.check_event()

### You can now quickly browse some of the light curves to look for any obvious
# features.
### This should give you a hint as to which model you might want to explore first.

### Invert the y-axis of the plots so that light curves are displayed the correct way
# up.
plt.gca().invert_yaxis()

### Loop over the telescopes and select the ones you want to plot. Here we only display 
### the OGLE data:
for tel in your_event.telescopes:
    if tel.name == 'OGLE':
        tel.plot_data()

plt.show()

### The event is highly magnified but there seem to be no obvious strong 
### secondary features. You can try fitting it with a simple point-source, point-lens 
### (PSPL) model.

### Set up the PSPL MODEL you want to fit and link it to the EVENT. 

pspl = PSPL_model.PSPLmodel(your_event)

### Next you need to specify the fitting algorithm you want to use 
### e.g. [LM_fit, TRF_fit, DE_fit, MCMC_fit, etc]. Consult the documentation for 
### details on what each algorithm does. Let us try out a TRF fit and give it some 
### starting guess parameters for t0, u0 and tE. You can guess roughly what starting
# values
### you can try for these parameters by looking at the light curve again.

fit_1 = TRF_fit.TRFfit(pspl)
fit_1.model_parameters_guess = [2457205.5, 1.0, 100.0]

### Fit the model:
fit_1.fit()

### TRF doesn't explore the whole parameters space but it is good at narrowing in at
# a local miniumum.

### Let's plot it and look at the fit. Now all data will be aligned and displayed (
# with the exception of Spitzer, which is in space).
pyLIMA_plots.plot_lightcurves(pspl, fit_1.fit_results['best_model'])
plt.show()

### The fit looks reasonable, but Zoom closely around the peak and you will notice a
# secondary
### peak. The event is a binary! There are no clear caustic-crossing features in the
# light curve
### and the duration of the secondary peak is very short (less than a day), so this
# could be
### a planet. Since you have included Spitzer observations from space, you also now
# need to
### consider parallax in your model. All this implies that the next model we should
# allow
### for all these effects in our next model.

### Set up a new uniform-source, binary-lens (USBL) model and link it to the EVENT.
### For the USBL model, we will also need to specify four extra parameters: rho, s,
# q and alpha,
### as well as two more describing the parallax vector, piEN,piEE.
### In order, the USBL parameters to be fitted are (assuming we use fancy_parameters): 
### {'to': 0, 'uo': 1, 'log(tE)': 2, 'log(rho)': 3, 'log(s)': 4, 'log(q)': 5,
# 'alpha': 6} + [piEN,piEE]
### (note that are also secondary parameters to be optimized that allow for data
# offsets
### and blending)

# Use the default fancy parameters log(tE), log(rho), log(s), log(q)
fancy = pyLIMA_fancy_parameters.StandardFancyParameters()
usbl = USBL_model.USBLmodel(your_event, fancy_parameters=fancy,
                            parallax=['Full', 2457205.5])
### t0par = 2457265.5

### Note: When you fit for parallax (and/or orbital motion), you also need to provide a 
### reference time, t0par, from which to perform the computations. Good choices for
# t0par
### are times close to t0, or close to points of caustic entry.

### Specify the fitting algorithm. This time go for a differential evolution search of 
### the parameter space. 
fit_2 = DE_fit.DEfit(usbl, telescopes_fluxes_method='polyfit', DE_population_size=10,
                     max_iteration=10000, display_progress=True)

### You do not need to specify an initial position, but you do need to 
### provide allowed ranges for each parameter:
fit_2.fit_parameters['t0'][1] = [2457195.00, 2457215.00]  # t0 limits
fit_2.fit_parameters['u0'][1] = [0.001, 0.2]  # u0 limits
fit_2.fit_parameters['log_tE'][1] = [1.6, 2.0]  # logtE limits in days
fit_2.fit_parameters['log_rho'][1] = [-3.3, -1.3]  # logrho
fit_2.fit_parameters['log_separation'][1] = [0.0, 0.5]  # logs limits
fit_2.fit_parameters['log_mass_ratio'][1] = [-4.0, -1.3]  # logq limits
fit_2.fit_parameters['alpha'][1] = [-3.14, 3.14]  # alpha limits (in radians)
fit_2.fit_parameters['piEN'][1] = [-0.5, 0.5]
fit_2.fit_parameters['piEE'][1] = [-0.5, 0.5]

### Allow multiprocessing

pool = mul.Pool(processes=4)

### !!! WARNING !!!: By executing the next commands you will start a long 
### search of the parameter space. This is how you would do it in practice but since it 
### takes a long time, we recommend you skip this step by leaving perform_long_fit =
# False
### and using the precomputed optimized parameters given below.

perform_long_fit = False

### Fit the model:
if perform_long_fit is True:
    fit_2.fit(computational_pool=pool)

    # Save it
    np.save('results_USBL_DE_966.npy', fit_2.fit_results['DE_population'])

else:
    # Use the precomputed Differential Evolution (DE) results:
    fit_2.fit_results['DE_population'] = np.load('./data/results_USBL_DE_966.npy')
    fit_2.fit_results['best_model'] = fit_2.fit_results['DE_population'][346501][0:-1]
    # fit_2.fit_results['best_model'] = [2457205.21, 0.0109583755, 1.78218726,
    # -2.89415218, 0.0475121003, -3.79996021, 2.25499875, 0.0227712230, -0.227192561]

print('Best_model', fit_2.fit_results['best_model'])
# Plot the best fit model and the corresponding geometrical configuration
pyLIMA_plots.list_of_fake_telescopes = []
pyLIMA_plots.plot_lightcurves(usbl, fit_2.fit_results['best_model'])
pyLIMA_plots.plot_geometry(usbl, fit_2.fit_results['best_model'])
plt.show()

# This solution is close to the (+,+),wide solution reported in 
# https://ui.adsabs.harvard.edu/abs/2016ApJ...819...93S/
# (Table 1, Col 2 in the paper - with the units converted to our format):
# published_model_1 = [2457205.198, 0.0114, 1.76, -2.853, 0.0473, -3.78, 2.26,
# 0.0234, -0.238]

### This concludes tutorial 5.
