'''
Welcome to pyLIMA (v2) tutorial 3!

In this tutorial you will learn how you can use pyLIMA to simulate a microlensing 
light curve. We will cover how to call the pyLIMA microlensing simulator and generate 
sample light curves.
We will also fit these light curves and see if we can recover the input parameters.
Please take some time to familiarize yourself with the pyLIMA documentation.
'''

### Import the required libraries.
import matplotlib.pyplot as plt
from pyLIMA.fits import DE_fit
from pyLIMA.models import PSPL_model
from pyLIMA.outputs import pyLIMA_plots
### Import the simulator to be used for generating the simulated light curve
from pyLIMA.simulations import simulator
from pyLIMA.toolbox import brightness_transformation

from pyLIMA import event

### Create a new EVENT object and give it a name.
### You can also set the coordinates for an event.
### In the context of the simulation, the coordinates will be used to check whether the
### target is observable from a specific observatory.
your_event = event.Event(ra=270, dec=-30)
your_event.name = 'My simulated event'

### Create some telescope(s) to observe the event from. The function we will use will
### create a generic telescope class (see pyLIMA documentation for details). We will 
### need to create a new telescope for each observatory, telescope, filter combination.
### Let us start simple and generate a single telescope first. We will also set 
### uniform_sampling=True, which will make sure the light curve generated will not 
### have any gaps due to the night/day cycle.
CTIO_I = simulator.simulate_a_telescope(name='CTIO_I', time_start=2457365.5,
                                        time_end=2457965.5, sampling=4,
                                        location='Earth', camera_filter='I',
                                        uniform_sampling=True, astrometry=False)

### Similar to tutorial 1, we need to associate this telescopee with the event we
### created:
your_event.telescopes.append(CTIO_I)

### Run a quick sanity check on your input.
your_event.check_event()

### Now construct the MODEL you want to deploy to construct the light curves and 
### link it to the EVENT you prepared.
### We will use a simple point-lens point-source (PSPL) model in this example.

pspl = PSPL_model.PSPLmodel(your_event)

### Now that the MODEL is there, we need to set the relevant parameters.
### The parameters are drawn uniformly from the bounds defined but you can 
### also set them manually. Please consult the documentation for more 
### details on the parameters of the MODEL you want to use. For the PSPL example,
### pspl_parameters = [to, uo, tE, flux_source, flux_blend]
pspl_parameters = simulator.simulate_microlensing_model_parameters(pspl)
print(pspl_parameters)

### Recall that to see the order and names of the paramaters you can always use:
pspl.model_dictionnary

### Transform the parameters into a pyLIMA class object. See the documentation for
### details.
pyLIMA_parameters_1 = pspl.compute_pyLIMA_parameters(pspl_parameters)

### Now we have defined the MODEL we want to simulate, we have defined the telescope
### details, so we just inject these into our simulator to produce a light curve:
simulator.simulate_lightcurve(pspl, pyLIMA_parameters_1)

#### Let's plot our simulated light curve using the pyLIMA plotter (recommended)!
pyLIMA_plots.plot_lightcurves(pspl, pspl_parameters)
plt.show()

### ... or you can just plot the results yourself any way you want to using matplotlib
plt.errorbar(CTIO_I.lightcurve['time'].value - 2450000,
             CTIO_I.lightcurve['mag'].value,
             yerr=CTIO_I.lightcurve['err_mag'].value,
             fmt='.', label=CTIO_I.name)

plt.gca().invert_yaxis()
plt.legend()
plt.show()

### OK, so now we want to simulate something more complicated. 
### Say, we have multiple telescopes around the world imaging the event in different
### bands and at different time intervals.
### In addition, we also want to simulate bad weather, avoid pointing too close to
### the moon, and also account for observing limitations due to the location of the
### target in the sky relative to the Sun. 
### (For a full list of the options available please consult the documentation!)

### Let's create a new event to observe:
your_event2 = event.Event(ra=264, dec=-28)
your_event2.name = 'My simulated event 2'

### We will simulate telescopes in South Africa (SAAO),  Chile (CTIO) and Australia 
### (SSO).
### For observing bands, we're simulate I-band for all sites, and also add a daily
### V-band observation from CTIO. 
### Each observing band counts as a seperate telescope, so we will need to
###create _four_ telescope objects:

SAAO_I = simulator.simulate_a_telescope(name='SAAO_I', time_start=2457575.5,
                                        time_end=2457625.5, sampling=2.5,
                                        location='Earth', camera_filter='I',
                                        uniform_sampling=False, altitude=400,
                                        longitude=20.659279,
                                        latitude=-32.3959,
                                        bad_weather_percentage=20.0 / 100,
                                        moon_windows_avoidance=20, minimum_alt=15,
                                        astrometry=False)

SSO_I = simulator.simulate_a_telescope('SSO_I', time_start=2457535.5,
                                       time_end=2457645.5, sampling=2.5,
                                       location='Earth', camera_filter='I',
                                       uniform_sampling=False, altitude=1165,
                                       longitude=149.0685,
                                       latitude=-31.2749,
                                       bad_weather_percentage=35.0 / 100,
                                       moon_windows_avoidance=20, minimum_alt=15,
                                       astrometry=False)

CTIO_I = simulator.simulate_a_telescope('CTIO_I', time_start=2457365.5,
                                        time_end=2457965.5, sampling=4.5,
                                        location='Earth', camera_filter='I',
                                        uniform_sampling=False, altitude=1000,
                                        longitude=-109.285399,
                                        latitude=-27.130,
                                        bad_weather_percentage=10.0 / 100,
                                        moon_windows_avoidance=30, minimum_alt=30,
                                        astrometry=False)

CTIO_V = simulator.simulate_a_telescope('CTIO_V', time_start=2457365.5,
                                        time_end=2457965.5, sampling=24.5,
                                        location='Earth', camera_filter='V',
                                        uniform_sampling=False, altitude=1000,
                                        longitude=-109.285399,
                                        latitude=-27.130,
                                        bad_weather_percentage=10.0 / 100,
                                        moon_windows_avoidance=30, minimum_alt=30,
                                        astrometry=False)

### The meaning of the parameters, in this example, for the SAAO_I data set are:
### Name = 'SAAO_I', location = 'Earth', start_obs =2457585.5, end_obs = 2457615.5,
### sampling(hours) = 2, location='Earth', filter = 'I', uniform_sampling=True,
### altitude = 400 m, longitude = 20.659279, latitude = -32.3959, 
### bad_weather_percentage = 20%, 
### moon_windows_avoidance (degrees)=20, minimum_alt=15, astrometry=False)

### Associate these telescopes with the event we created:
your_event2.telescopes.append(SAAO_I)
your_event2.telescopes.append(SSO_I)
your_event2.telescopes.append(CTIO_I)
your_event2.telescopes.append(CTIO_V)

### Run a quick sanity check on your input.
your_event2.check_event()

### Define which data set to align all data to (optional):
your_event2.find_survey('CTIO_I')

### Now construct the MODEL you want to deploy to construct the light curves and 
### link it to the EVENT you prepared.
### We will use the double-source point-lens (DSPL) model for this example.

dspl = PSPL_model.PSPLmodel(your_event2, double_source=['Static',2457500])

### Now that the MODEL is there, we need to set the relevant parameters.
### The parameters are drawn uniformly from the bounds defined but you can 
### also set them manually. Please consult the documentation for more 
### details on the parameters of the MODEL you want to use. For the DSPL example,
### dspl_parameters = [to, uo, tE, elta_to, delta_uo, q_fluxr_1, q_fluxr2, ...]
### where q_fluxr_* is the flux ratio in each observing band.
dspl_parameters = simulator.simulate_microlensing_model_parameters(dspl)
print(dspl_parameters)

### To see the order and names of the paramaters use:
print(dspl.model_dictionnary)

### pyLIMA has provided some random values for the fluxes drawn from uniform
### distributions. 
### These do not represent any physical system and are likely off for the
### telescope/filter combination that you as a user have defined, but they 
### can be used as placeholders for you to define your own values. We will 
### see how to do that later.
### For now, just use these temporary values for the simulation.

### Transform the parameters into pyLIMA standards:
pyLIMA_parameters = dspl.compute_pyLIMA_parameters(dspl_parameters)

### Now we have defined the MODEL we want to simulate, we have defined the telescopes
### and fluxes in each observing band, so we just inject these into our simulator to
### produce a light curve:
simulator.simulate_lightcurve(dspl, pyLIMA_parameters)

### Let's plot our simulated light curve!
### Plot with pyLIMA plotter (recommended):
pyLIMA_plots.list_of_fake_telescopes = []  # cleaning previous plots

pyLIMA_plots.plot_lightcurves(dspl, dspl_parameters)
plt.show()

### ... or plot it all manually if you prefer:
for telescope in your_event2.telescopes:
    plt.errorbar(telescope.lightcurve['time'].value - 2450000,
                 telescope.lightcurve['mag'].value,
                 yerr=telescope.lightcurve['err_mag'].value,
                 fmt='.', label=telescope.name)

plt.gca().invert_yaxis()
plt.legend()
plt.show()

### Say you want to define your own values to use, instead of having the pyLIMA
### simulators randomly guess.
### Here's how you can do that. Let's fix the DSPL parameters to some values where
### the binary source model produces two clear peaks, and then just adjust the flux 
### parameters.
dspl_parameters[0:7] = [2457760.216627234, 0.8605811108889658, 116.43231096591524, 143.4484970433387,
                        -0.6046788112617074,  0.15157064165919296,
                        0.18958495421162946]

### The order of the parameters is:
print(dspl.model_dictionnary)

### ... and we will replace all source and blend flux elements with our own values.
### We can assume the fluxes are calibrated. Set up the magnitude values you want:
magsource_CTIO_I = 17.32
magblend_CTIO_I = 20.89
magsource_SAAO_I = 17.32
magblend_SAAO_I = 20.89
magsource_SSO_I = 17.32
magblend_SSO_I = 20.89
magsource_CTIO_V = 19.18
magblend_CTIO_V = 21.22

### Now we need to convert these to fluxes. Set up an empty array to hold the values:
fluxes = []

### Import the magnitude to flux coversion function from pyLIMA and populate the array

for mag in [magsource_CTIO_I, magblend_CTIO_I, magsource_SAAO_I, magblend_SAAO_I,
            magsource_SSO_I, magblend_SSO_I, magsource_CTIO_V, magblend_CTIO_V]:
    flux = brightness_transformation.magnitude_to_flux(mag)
    fluxes.append(flux)

### Now we add these fluxes to the dspl_parameters we prepared earlier:
dspl_parameters[7:] = fluxes

### Transform the parameters into pyLIMA standards:
pyLIMA_parameters = dspl.compute_pyLIMA_parameters(dspl_parameters)

### Produce the lightcurve:
simulator.simulate_lightcurve(dspl, pyLIMA_parameters)

### Plot it:

pyLIMA_plots.plot_lightcurves(dspl, dspl_parameters)
plt.show()

### A short commentary to explain the DSPL parameters in this example:
print(dspl_parameters)
parameter_commentary = ['Time of minimum impact parameter for source 1',
                        'minimum impact parameter for source 1',
                        'angular Einstein radius crossing time',
                        'difference of time of minimum impact parameter between the '
                        'two sources',
                        'difference of minimum impact parameters between the two '
                        'sources',
                        'flux ratio in I between source 1 and source 2',
                        'flux ratio in V between source 1 and source 2',
                        'source flux of source 1 for telescope CTIO_I (survey '
                        'telescope)',
                        'blending ratio of source 1 for telescope CTIO_I (survey '
                        'telescope)',
                        'source flux of source 1 for telescope SAAO_I',
                        'blending ratio of source 1 for telescope SAAO_I',
                        'source flux of source 1 for telescope SSO_I',
                        'blending ratio of source 1 for telescope SSO_I',
                        'source flux of source 1 for telescope CTIO_V',
                        'blending ratio of source 1 for telescope CTIO_V',
                        ]

for key in dspl.model_dictionnary.keys():
    indice = dspl.model_dictionnary[key]
    print(key, ' = ', dspl_parameters[indice], ' : ', parameter_commentary[indice])

### Let's try to fit this now! (This can take a while!)
### You can check the first tutorial again for a detailed explanation if needed.

my_fit = DE_fit.DEfit(dspl, display_progress=True, loss_function='likelihood',strategy='best1bin')
my_fit.fit()

my_fit.fit_results['best_model']

### Compare your DSPL fit parameters with what you defined in the DSPL simulation above:
print(my_fit.fit_results['best_model'][:7] - dspl_parameters[0:7])

### Plot and constrast the optimized fit results and the simulated light curve:
pyLIMA_plots.plot_lightcurves(dspl, my_fit.fit_results['best_model'])
pyLIMA_plots.plot_lightcurves(dspl, dspl_parameters)
plt.show()
### This concludes tutorial 3.
