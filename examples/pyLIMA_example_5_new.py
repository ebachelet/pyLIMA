'''
Welcome to pyLIMA (v2) tutorial 5!

In this tutorial you will learn how to fit an actual planetary event using real data.
The event is OB150966 and the relevant publication is:
    https://ui.adsabs.harvard.edu/abs/2016ApJ...819...93S/

Please take some time to familiarize yourself with the pyLIMA documentation.
'''

### First import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import os, sys

from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.outputs import pyLIMA_plots

### Create a new EVENT object and give it a name.
your_event = event.Event()
your_event.name = 'OB150966'

#Here RA and DEC matter !!! 
your_event.ra = 268.75425
your_event.dec = -29.047111111111114

### You now need to associate all data sets with this EVENT. 
### There are 11 sets of observations and we need to load all of them.

### The data sets are already pre-formatted: 
###     column 1 is the date, column 2 the magnitude and column 3 
###     the uncertainty in the magnitude.
data_1 = np.loadtxt('./data/OGLE_OB150966.dat')
telescope_1 = telescopes.Telescope(name = 'OGLE', 
                                   camera_filter = 'I',
                                   light_curve = data_1.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

data_2 = np.loadtxt('./data/MOA_OB150966.dat')
telescope_2 = telescopes.Telescope(name = 'MOA', 
                                   camera_filter = 'I+R',
                                   light_curve = data_2.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

data_3 = np.loadtxt('./data/SPITZER_OB150966.dat')
telescope_3 = telescopes.Telescope(name = 'SPITZER', 
                                   camera_filter = 'IRAC1',
                                   light_curve = data_3.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

data_4 = np.loadtxt('./data/DANISH_OB150966.dat')
telescope_4 = telescopes.Telescope(name = 'DANISH', 
                                   camera_filter = 'Z+I',
                                   light_curve = data_4.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

data_5 = np.loadtxt('./data/LCO_CTIO_A_OB150966.dat')
telescope_5 = telescopes.Telescope(name = 'LCO_CTIO_A', 
                                   camera_filter = 'I',
                                   light_curve = data_5.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

data_6 = np.loadtxt('./data/LCO_CTIO_B_OB150966.dat')
telescope_6 = telescopes.Telescope(name = 'LCO_CTIO_B', 
                                   camera_filter = 'I',
                                   light_curve = data_6.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

data_7 = np.loadtxt('./data/LCO_CTIO_OB150966.dat')
telescope_7 = telescopes.Telescope(name = 'LCO_CTIO', 
                                   camera_filter = 'I',
                                   light_curve = data_7.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])


data_8 = np.loadtxt('./data/LCO_SAAO_OB150966.dat')
telescope_8 = telescopes.Telescope(name = 'LCO_SAAO', 
                                   camera_filter = 'I',
                                   light_curve = data_8.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

data_9 = np.loadtxt('./data/LCO_SSO_A_OB150966.dat')
telescope_9 = telescopes.Telescope(name = 'LCO_SSO_A', 
                                   camera_filter = 'I',
                                   light_curve = data_9.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

data_10 = np.loadtxt('./data/LCO_SSO_B_OB150966.dat')
telescope_10 = telescopes.Telescope(name = 'LCO_SSO_B', 
                                   camera_filter = 'I',
                                   light_curve = data_10.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

data_11 = np.loadtxt('./data/LCO_SSO_OB150966.dat')
telescope_11 = telescopes.Telescope(name = 'LCO_SSO', 
                                   camera_filter = 'I',
                                   light_curve = data_11.astype(float),
                                   light_curve_names = ['time','mag','err_mag'],
                                   light_curve_units = ['JD','mag','mag'])

### Add the telescopes to your event:
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

### Let's look at the data first to see what the light curve looks like.
## This should give us a hint as to which model we want to explore first.
for tel in your_event.telescopes:
    tel.plot_data()

plt.gca().invert_yaxis()
plt.show()

### Next, construct the MODEL you want to fit and link it to the EVENT you prepared. 
### Since this is obviously a binary, let's go with a USBL, without second order effects:
from pyLIMA.models import USBL_model
fspl = USBL_model.USBLmodel(your_event)

### Import the DE algorithm
from pyLIMA.fits import DE_fit

### Enable multithreading
import multiprocessing as mul
pool = mul.Pool(processes = 4)

### Perform the fit
my_fit = DE_fit.DEfit(fspl, telescopes_fluxes_method='polyfit', DE_population_size=10, max_iteration=10000, display_progress=True)
my_fit.fit(computational_pool = pool)

##############################


### looks to be some residuals on the event wings, maybe parallax.
### Lets try, you have to choose topar, the parallax time reference. Here we choose 2457850.
model_2 = microlmodels.create_model('PSPL', your_event,parallax=['Full',2457850])

# we can speed up computation, by adding guess from last fit :

#################### guess =   [to,uo,tE]                                + [piEN,piEE]  
model_2.parameters_guess = your_event.fits[0].fit_results[:3]+[0,0]

your_event.fit(model_2,'LM')

### Let's see some plots.
your_event.fits[-1].produce_outputs()
print ('Chi2_LM :',your_event.fits[-1].outputs.fit_parameters.chichi)
plt.show()


### That looks better! We can check with DE!
your_event.fit(model_2,'DE',DE_population_size=5)

### Let's see some plots.
your_event.fits[-1].produce_outputs()
print ('Chi2_LM :',your_event.fits[-1].outputs.fit_parameters.chichi)
plt.show()


### What about Space based parallax? Lets have a look to OB150966 :
### http://adsabs.harvard.edu/abs/2016ApJ...819...93S



your_event = event.Event()
your_event.name = 'OB150966'

#Here RA DEC matters !!
your_event.ra = 268.75425

your_event.dec = -29.047


## Now we need some observations. That's good, we obtain some data on two
### telescopes. Both are in I band and magnitude units :

data_1 = np.loadtxt('./data/OGLE_OB150966.dat')
telescope_1 = telescopes.Telescope(name='OGLE', camera_filter='I', light_curve_magnitude=data_1)

data_2 = np.loadtxt('./data/SPITZER_OB150966.dat')
telescope_2 = telescopes.Telescope(name='SPITZER', camera_filter='IRAC1', light_curve_magnitude=data_2)
telescope_2.location='Space'
telescope_2.spacecraft_name  = 'Spitzer'
### Add the telescopes to your event :
your_event.telescopes.append(telescope_1)
your_event.telescopes.append(telescope_2)



### Sanity check
your_event.check_event()

### Construct the model you want to fit. Let's go basic with a PSPL, without second_order effects :
model_1 = microlmodels.create_model('PSPL', your_event)


### Let's try with the simplest Levenvberg_Marquardt algorithm :
your_event.fit(model_1,'LM')


### Let's see some plots.
your_event.fits[-1].produce_outputs()
print ('Chi2_LM :',your_event.fits[-1].outputs.fit_parameters.chichi)
plt.show()

### Of course, not great at all! :

### Construct the model with parallax centered at 2457205:
model_2 = microlmodels.create_model('PSPL', your_event,parallax=['Full',2457205])
your_event.fit(model_2,'DE')


### Let's see some plots.
your_event.fits[-1].produce_outputs()
print ('Chi2_LM :',your_event.fits[-1].outputs.fit_parameters.chichi)
plt.show()

# Street et al. found piE =(0.0234, -0.238), close to your fit in principle.

### This concludes tutorial 5.

