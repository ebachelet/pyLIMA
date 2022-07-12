'''
Welcome to pyLIMA (v2) tutorial 3b!

In this tutorial you will learn how to use the pyLIMA simulator to simulate 
observations from space. We will also learn how to add parallax to our models.
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
your_event.name = 'My simulated event'
your_event.ra = 270
your_event.dec = -30

### Create some telescope(s) to observe the event from. See tutorial 3 for more details.
### We will use CTIO_I (from Earth) and Gaia_G (from Space):
CTIO_I = simulator.simulate_a_telescope('CTIO_I', your_event, 2458365.5, 2458965.5, 4, 'Earth', 'I',
                                        uniform_sampling=False, altitude=1000, longitude = -109.285399, 
                                        latitude = -27.130, bad_weather_percentage=10.0 / 100, 
                                        moon_windows_avoidance=30, minimum_alt=30)

GAIA_G = simulator.simulate_a_telescope('GAIA_G', your_event, 2458365.5, 2458965.5, 168, 'Space', 'G',
                                        uniform_sampling=True, spacecraft_name='Gaia')

### Similar to tutorial 1, we need to associate this telescopee with the event we created:
your_event.telescopes.append(CTIO_I)
your_event.telescopes.append(GAIA_G)

### Run a quick sanity check on your input.
your_event.check_event()

### Now construct the MODEL you want to deploy:
### We will use a simple point-lens point-source (PSPL) model but we will also add parallax.
### This involves invoking the parralax= option when setting up our MODEL.
### Note that here we want to give a raference date to evalueate the parallax from, and this needs
### to be close to t0.
from pyLIMA.models import PSPL_model
pspl = PSPL_model.PSPLmodel(your_event, parallax=['Full', 2458565.5])
#pspl.define_pyLIMA_standard_parameters()

### Now that the MODEL is there, we need to set the relevant parameters.
### pspl_parameters = [to, uo, tE, flux_source, flux_blend]
pspl_parameters = simulator.simulate_microlensing_model_parameters(pspl)
print (pspl_parameters)

### Let's fix t0 to the value we set when we were preparing our MODEL (including parallax) above.
### This is the reference date (t0_\bar) we want to evaluate the parallax from:
pspl_parameters[0] = 2458565.25

### Transform the parameters into a pyLIMA class object. See the documentation for details.
pyLIMA_parameters_1 = pspl.compute_pyLIMA_parameters(pspl_parameters)

### Now we have defined the MODEL we want to simulate, we have defined the telescope details,
### so we just inject these into our simulator to produce a light curve:
simulator.simulate_lightcurve_flux(pspl, pyLIMA_parameters_1)

#### Let's plot our simulated light curve using the pyLIMA plotter:
from pyLIMA.outputs import pyLIMA_plots
pyLIMA_plots.plot_geometry(pspl, pspl_parameters)
plt.show()

### This concludes tutorial 3b.
