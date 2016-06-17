'''
Welcome to pyLIMA tutorial!
Let's learn how pyLIMA works by fitting an example.
Please help yourself with the pyLIMA documentation
'''

### First import the required libraries

import numpy as np

import matplotlib.pyplot as plt

import os, sys

lib_path = os.path.abspath(os.path.join('../'))

sys.path.append(lib_path)


from pyLIMA import event

from pyLIMA import telescopes

from pyLIMA import microlmodels


### Create an event object. You can choose the name and RA,DEC in degrees :

your_event = event.Event()

your_event.name = 'your choice'

your_event.ra = 269.39166666666665

your_event.dec = -29.22083333333333

### Now we need some observations. That's good, we obtain some data on two

### telescopes. Both are in I band and magnitude units :

data_1 = np.loadtxt('./Survey_1.dat')

telescope_1 = telescopes.Telescope(name='OGLE', camera_filter='I', light_curve_magnitude=data_1)

data_2 = np.loadtxt('./Followup_1.dat')

telescope_2 = telescopes.Telescope(name='LCOGT', camera_filter='I', light_curve_magnitude=data_2)

### Add the telescopes to your event :

your_event.telescopes.append(telescope_1)

your_event.telescopes.append(telescope_2)

### Find the survey telescope :

your_event.find_survey('OGLE')

### Sanity check

your_event.check_event()


### Construct the model you want to fit. Let's go basic with a PSPL, without second_order effects :

model_1 = microlmodels.MLModels(your_event, 'PSPL')

### Let's try with the simplest Levenvberg_Marquardt algorithm :

your_event.fit(model_1,'LM')

### Let's see some plots.

your_event.fits[0].produce_outputs()

print 'Chi2_LM :',your_event.fits[0].outputs.fit_parameters._chichi

plt.show()

### Let's try with differential evolution  algorithm. WAIT UNTIL THE FIGURE POP UP.

your_event.fit(model_1,'DE')

your_event.fits[1].produce_outputs()

print 'Chi2_DE :',your_event.fits[1].outputs.fit_parameters._chichi

plt.show()

### Let's go basic for FSPL :

model_2 = microlmodels.MLModels(your_event, 'FSPL')

your_event.fit(model_2,'LM')

### Let's see some plots. You can zoom close to the peak to see what is going on. CLOSE THE FIGURE TO CONTINUE.

your_event.fits[-1].produce_outputs()

print 'Chi2_LM :',your_event.fits[-1].outputs.fit_parameters._chichi

plt.show()

### set gamma for each telescopes :
your_event.telescopes[0].gamma = 0.5

your_event.telescopes[1].gamma = 0.5

### Fit again

your_event.fit(model_2,'LM')

your_event.fits[-1].produce_outputs()

print 'Chi2_LM :',your_event.fits[-1].outputs.fit_parameters._chichi

plt.show()

### Fit again

your_event.fit(model_2,'DE')

your_event.fits[-1].produce_outputs()

print 'Chi2_DE :',your_event.fits[-1].outputs.fit_parameters._chichi

plt.show()




print 'Parameters', ' Model','   Fit','     Errors'

print '-----------------------------------'

print 't_o', '        79.9309 ',str(your_event.fits[-1].outputs.fit_parameters.to)[:7],'',str(your_event.fits[-1].outputs.fit_errors.err_to)[:7]

print 'u_o', '        0.00826 ',str(your_event.fits[-1].outputs.fit_parameters.uo)[:7],'',str(your_event.fits[-1].outputs.fit_errors.err_uo)[:7]

print 't_E', '        10.1171 ',str(your_event.fits[-1].outputs.fit_parameters.tE)[:7],'',str(your_event.fits[-1].outputs.fit_errors.err_tE)[:7]

print 'rho', '        0.02268 ',str(your_event.fits[-1].outputs.fit_parameters.rho)[:7],'',str(your_event.fits[-1].outputs.fit_errors.err_rho)[:7]

print 'fs_OGLE', '    2915.76 ',str(your_event.fits[-1].outputs.fit_parameters.fs_OGLE)[:7],'',str(your_event.fits[-1].outputs.fit_errors.err_fs_OGLE)[:7]

print 'g_OGLE', '     0.07195 ',str(your_event.fits[-1].outputs.fit_parameters.g_OGLE)[:7],'',str(your_event.fits[-1].outputs.fit_errors.err_g_OGLE)[:7]

print 'fs_LCOGT', '   92936.7 ',str(your_event.fits[-1].outputs.fit_parameters.fs_LCOGT)[:7],'',str(your_event.fits[-1].outputs.fit_errors.err_fs_LCOGT)[:7]

print 'g_LCOGT', '    0.52460 ',str(your_event.fits[-1].outputs.fit_parameters.g_LCOGT)[:7],'',str(your_event.fits[-1].outputs.fit_errors.err_g_LCOGT)[:7]

### Fit again, but using MCMC now. TAKE A WHILE....Wait until figures pop up.
your_event.fit(model_2,'MCMC')
print 'The fitting process is finished now, let produce some outputs....'

your_event.fits[-1].produce_outputs()
plt.show()
