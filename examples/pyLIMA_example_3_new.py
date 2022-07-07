'''
Welcome to pyLIMA (v2) tutorial 3!

In this tutorial you will learn how you can use pyLIMA to simulate a microlensing light curve.
We will cover how to call the pyLIMA microlensing simulator and generate a simple PSPL event.
We will also fit these events and see if we can recover the input parameters.
Please take some time to familiarize yourself with the pyLIMA documentation.
'''

### Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt
import os, sys

from pyLIMA import event
from pyLIMA import telescopes

### Import the simulator to be used for generating the simulated light curve
from pyLIMA.simulations import simulator

### Create a new EVENT object and give it a name.
### You can also set the coordinates for an event.
### In the context of the simulation, the coordinates will be used to check whether the
### target is observable from a specific observatory.
your_event = event.Event()
your_event.name = 'My simulated double source event'
your_event.ra = 270
your_event.dec = -30


### Create some telescope to observe the event from. The function we will use will create
### a generic telescope class (see pyLIMA documentation for details). We will need to create
### a new telescope for each observatory, telescope, filter combination. 
### For example, in this case, we will use SAAO_I, SAAO_V to simulate observations from SAAO 
### (South Africa) in the I and V bands and from CTIO (Chile) in the I ban. 
### Please refer to the documentation to see the full meaning of the input parameters.

### For example, to generate the SAAO_I data set:
### Name = SAAO_I,your_event, location = 'Earth', start_obs =2457585.5, end_obs = 2457615.5,
### sampling(hours) = 2, location='Earth', filter = 'I', uniform_sampling=False, altitude = 400 m, 
### longitude = 20.659279, latitude = -32.3959, bad_weather_percentage = 20%, 
### moon_windows_avoidance (degree)=15, minimum_alt=15)
SAAO_I = simulator.simulate_a_telescope('SAAO_I', your_event, 2457585.5, 2457615.5, 2, 'Earth','I',
                                        uniform_sampling=False, altitude=400, longitude = 20.659279, 
                                        latitude = -32.3959, bad_weather_percentage=20.0 / 100, 
                                        moon_windows_avoidance=20, minimum_alt=15)

SAAO_V = simulator.simulate_a_telescope('SAAO_V', your_event, 2457585.5, 2457615.5, 2, 'Earth','V',
                                        uniform_sampling=False, altitude=400, longitude = 20.659279, 
                                        latitude = -32.3959, bad_weather_percentage=20.0 / 100, 
                                        moon_windows_avoidance=20, minimum_alt=15)

CTIO_I = simulator.simulate_a_telescope('CTIO_I', your_event, 2457365.5,2457965.5, 4, 'Earth', 'I',
                                        uniform_sampling=False, altitude=1000, longitude = -109.285399, 
                                        latitude = -27.130, bad_weather_percentage=10.0 / 100, 
                                        moon_windows_avoidance=30, minimum_alt=30)

### Similar to tutorial 1, we need to associate these telescopes with the event we created:
your_event.telescopes.append(SAAO_I)
your_event.telescopes.append(SAAO_V)
your_event.telescopes.append(CTIO_I)

### Define which data set to align all data to:
your_event.find_survey('CTIO_I')

### Now construct the MODEL you want to deploy to construct the light curves and 
### link it to the EVENT you prepared.
### We will use the double-source point-lens (DSPL) model in this example.
from pyLIMA.models import DSPL_model
dspl = DSPL_model.DSPLmodel(your_event)

### Now that the MODEL is there, we need to set the relevant parameters.
### The parameters are drawn uniformly from the bounds defined but you can 
### also set them manually. Please consult the documentation for more 
### details on the parameters of the MODEL you want to use.
dspl_parameters = simulator.simulate_microlensing_model_parameters(dspl)
print (dspl_parameters)

### To see the order and names of the paramaters use:
dspl.model_dictionnary

### Now we need to define the magnitudes


### If you want to simulate a event from space, you can use :

#my_space_telescope = microlsimulator.simulate_a_telescope('Gaia',my_own_creation,  2457585.5, 2457615.5,2, 'Space','G',
#                                                          uniform_sampling=True, spacecraft_name='Gaia')
# Note that the spacecraft name shoudl match JPL horizon ephemeris, see microlparallax. If you include this 
# telescope in your analysis, you will need to give to the model parallax = ['Full,to_par] in order to have 
# correct simulation. 

#OK now we can choose the model we would like to simulate, here let's have a double source point lens one (DSPL). More details on models can be seen here [pyLIMA documentation](file/../../doc/build/html/pyLIMA.microlmodels.html)
#More details on parameters generation can be found here [pyLIMA documentation](file/../../doc/build/html/pyLIMA.microlsimulator.html)

### What model you want? Let's have DSPL!
my_own_model = microlsimulator.simulate_a_microlensing_model(my_own_creation, model='DSPL', parallax=['None', 0.0],
                                             xallarap=['None', 0.0],
                                             orbital_motion=['None', 0.0], source_spots='None')

# Find some model parameters. If you want specific parameters, you need to respet pyLIMA convention when you create your 
# parameters. For the DSPL example, my_own_parameters = [to, uo, delta_to, delta_uo, tE].
my_own_parameters = microlsimulator.simulate_microlensing_model_parameters(my_own_model)

# Which source magnitude? Which blending? 
# Same here, you can create your own flux parameters with the convention
# [ [magnitude_source_i, blending ratio_i]] for i in telescopes. In our case it looks : 
# [ [magnitude_source_survey, blending ratio_survey], [ magnitude_source_SAAO_I, blending ratio_SAAO_I],  
# [magnitude_source_SAAO_V, blending ratio_SAAO_V]], i.e [[18.5,0.3],[19.5,1.2],[20.2,1.6]] (example).

my_own_flux_parameters = microlsimulator.simulate_fluxes_parameters(my_own_creation.telescopes)
my_own_parameters += my_own_flux_parameters

#Now we need to transform these parameters into a parameter class object (this is a "technical" part but the interested reader can found the function here  [pyLIMA #documentation](file/../../doc/build/html/pyLIMA.microlmodels.html))

# Transform into pyLIMA standards
pyLIMA_parameters = my_own_model.compute_pyLIMA_parameters(my_own_parameters)

#Ok now we have the model we want to simulate, we then need to updates our telescopes observations!

# update the telescopes lightcurve in your event :
microlsimulator.simulate_lightcurve_flux(my_own_model, pyLIMA_parameters,  red_noise_apply='Yes')

#### Plot it!

for telescope in my_own_creation.telescopes:
    plt.errorbar(telescope.lightcurve_magnitude[:, 0]-2450000, telescope.lightcurve_magnitude[:, 1],
                 yerr=telescope.lightcurve_magnitude[:, 2], fmt='.',label=telescope.name)

    
# A list of commentary to explain parameters. Of couse, this is valable only for the DSPL models.
parameter_commentary = ['time of minimum impact parameter for source 1',
                        'minimum impact parameter for source 1',
                        'difference of time of minimum impact parameter between the two sources',
                        'difference of minimum impact parameters between the two sources',
                        'angular Einstein radius crossing time',
                        'flux ratio in I between source 1 and source 2',
                        'flux ratio in V between source 1 and source 2',
                        'source flux of source 1 for telescope survey',
                        'blending ratio of source 1 for telescope survey',
                        'source flux of source 1 for telescope SAAO_I',
                        'blending ratio of source 1 for telescope SAAO_I',
                        'source flux of source 1 for telescope SAAO_V',
                        'blending ratio of source 1 for telescope SAAO_V',
                        ]
for key in my_own_model.model_dictionnary.keys():
    indice = my_own_model.model_dictionnary[key]
    
    print (key, ' = ', my_own_parameters[indice], ' : ', parameter_commentary[indice] )

plt.gca().invert_yaxis()
plt.legend(numpoints=1)
plt.grid(True)
plt.show()

#Let's try to fit this now! You can go back to pyLIMA_example_1 for a more complete explanation if needed.
#Look in particular your DSPL fit parameters versus the model above.


from pyLIMA import microlmodels

model_1 = microlmodels.create_model('PSPL', my_own_creation)
my_own_creation.fit(model_1,'DE')

model_2 = microlmodels.create_model('DSPL', my_own_creation)
my_own_creation.fit(model_2,'DE')

my_own_creation.fits[0].produce_outputs()
my_own_creation.fits[1].produce_outputs()

print(my_own_creation.fits[0].model.model_type,'Chi2_LM :',my_own_creation.fits[0].outputs.fit_parameters.chichi)
print(my_own_creation.fits[1].model.model_type,'Chi2_LM :',my_own_creation.fits[1].outputs.fit_parameters.chichi)

plt.show()

### This concludes tutorial 3.

