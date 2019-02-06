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

#Here RA DEC matters !! 
your_event.ra = 266.25624999999997

your_event.dec = -22.261972222222223


## Now we need some observations. That's good, we obtain some data on two
### telescopes. Both are in I band and magnitude units :

data_1 = np.loadtxt('./Survey_parallax.dat')
telescope_1 = telescopes.Telescope(name='OGLE', camera_filter='I', light_curve_magnitude=data_1)

### Add the telescopes to your event :
your_event.telescopes.append(telescope_1)



### Sanity check
your_event.check_event()

### Construct the model you want to fit. Let's go basic with a PSPL, without second_order effects :
model_1 = microlmodels.create_model('PSPL', your_event)


### Let's try with the simplest Levenvberg_Marquardt algorithm :
your_event.fit(model_1,'LM')

### Let's see some plots.
your_event.fits[0].produce_outputs()
print('Chi2_LM :',your_event.fits[0].outputs.fit_parameters.chichi)
plt.show()


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

data_1 = np.loadtxt('./OGLE_OB150966.dat')
telescope_1 = telescopes.Telescope(name='OGLE', camera_filter='I', light_curve_magnitude=data_1)

data_2 = np.loadtxt('./SPITZER_OB150966.dat')
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

