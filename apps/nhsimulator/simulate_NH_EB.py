
import numpy as np
import matplotlib.pyplot as plt
import os, sys


#lib_path = os.path.abspath(os.path.join('../'))
#sys.path.append(lib_path)

from pyLIMA import microlsimulator


my_own_creation = microlsimulator.simulate_a_microlensing_event(name ='A PSPL', 
                                                                ra=270, dec=-30)
my_own_creation2 = microlsimulator.simulate_a_microlensing_event(name ='A PSPL', 
                                                                ra=270, dec=-30)                                                                
                                                                
                                                                
my_survey = microlsimulator.simulate_a_telescope('survey',my_own_creation, 2456758.500000000,2461860.500000000,240, 'Earth','I',
                                                  uniform_sampling=True)
                                                  
nh = microlsimulator.simulate_a_telescope('NH',my_own_creation, 2456758.500000000,2461860.500000000,240, 'Space','I',
                                                  uniform_sampling=True)                                                 
nh2 = microlsimulator.simulate_a_telescope('NH',my_own_creation, 2456758.500000000,2461860.500000000,240, 'Space','I',
                                                  uniform_sampling=True)                                                 
                                                                                                       
                                                  
nh.spacecraft_name = 'NH'

data = np.loadtxt('NH_positions.txt')
data[:,3] *=  60*300000/150000000
nh.spacecraft_positions = data                                                  

my_own_creation.telescopes.append(my_survey)
my_own_creation.telescopes.append(nh)



my_own_creation.compute_parallax_all_telescopes(['Full', 2458000])

my_own_parameters = [2458000,0.01,150,0.005,0.005,1000,0.1,200,1.0]
#my_own_parameters = [2458000,0.01,30,0.5,0.5,1000,0.1,200,1.0]
microlensing_model = microlsimulator.simulate_a_microlensing_model(my_own_creation, model='PSPL', parallax=['Full', 2458000], xallarap=['None', 2458000])
pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(my_own_parameters)
microlsimulator.simulate_lightcurve_flux(microlensing_model, pyLIMA_parameters,  red_noise_apply='No')

microlensing_magnification1 = microlensing_model.model_magnification(my_own_creation.telescopes[0], pyLIMA_parameters)              
microlensing_magnification2 = microlensing_model.model_magnification(my_own_creation.telescopes[1], pyLIMA_parameters)                           



plt.plot(my_own_creation.telescopes[0].lightcurve_magnitude[:,0], microlensing_magnification1)
plt.plot(my_own_creation.telescopes[1].lightcurve_magnitude[:,0], microlensing_magnification2)
plt.show()

nh2.location='NewHorizon'
my_own_creation2.telescopes.append(nh2)
data = np.loadtxt('NH.txt')

nh2.spacecraft_positions = data             


microlensing_model2 = microlsimulator.simulate_a_microlensing_model(my_own_creation2, model='PSPL', parallax=['None', 2458010], xallarap=['None', 2458000])
microlensing_model3 = microlsimulator.simulate_a_microlensing_model(my_own_creation2, model='PSPL', parallax=['Full', 2458010], xallarap=['None', 2458000])
nh2.lightcurve_flux = nh.lightcurve_flux
nh2.lightcurve_magnitude = nh.lightcurve_magnitude



my_own_creation2.fit(microlensing_model2,'DE',DE_population_size=20)


my_own_creation2.fit(microlensing_model3,'DE',DE_population_size=20)


my_own_creation2.fits[0].produce_outputs()
my_own_creation2.fits[-1].produce_outputs()
plt.show()

import pdb;
pdb.set_trace()






import pdb;
pdb.set_trace()
