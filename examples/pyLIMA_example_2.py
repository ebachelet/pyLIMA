'''
Welcome to pyLIMA (v2) tutorial 2!

This second tutorial will give you some basics about how to reconfigure your input
parameters.
If you do not like the standard pyLIMA parameters, this is made for you. We will
demonstrate how to
replace them with parameters of your choice using the fancy_parameters module.
We are going to fit the same light curves as in tutorial 1, but using different
parametrization.
'''
### First import the required libraries as before.
import matplotlib.pyplot as plt
import numpy as np
from pyLIMA.fits import TRF_fit
from pyLIMA.models import FSPL_model
### Import fancy_parameters. This will allow us to change the definitions as required.
from pyLIMA.models import pyLIMA_fancy_parameters
from pyLIMA.outputs import pyLIMA_plots

from pyLIMA import event
from pyLIMA import telescopes

class MyFancyParameters(object):

    def __init__(self, fancy_parameters = {'tE': 'log_tE'},
                       fancy_boundaries = {'log_tE':(0,3)} ):

        self.fancy_parameters = fancy_parameters

        self.fancy_boundaries = fancy_boundaries


    def tE(self, fancy_params):

        return 10**fancy_params['log_tE']

    def log_tE(self, standard_params):

        return np.log10(standard_params['tE'])



### fancy_parameters already provides some commonly used options, for example:
pyLIMA_fancy_parameters.StandardFancyParameters

### Begin by create a new EVENT object and giving it a name, as in example 1.
your_event = event.Event()
your_event.name = 'My event name'

### Associate some data sets with this EVENT. 
### Again, you will use simulated I-band data sets from two telescopes, OGLE and LCO.
### The data sets are pre-formatted: column 1 is the date, column 2 the magnitude and
# column 3
### the uncertainty in the magnitude.

### Load up the data:
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

### If necessary, set the microlensing limb-darkening coefficients (gamma) for each
# telescope.
### We already saw in example 1 that setting limb darkening coefficients improves the
# fit for this event,
### so we set them again:
your_event.telescopes[0].ld_gamma = 0.5
your_event.telescopes[1].ld_gamma = 0.5

### Define the model and fit method (as in example 1) and let it know that you will
# be using alternative parameters.
### To do this, set the option fancy_parameters when you define the model. We will
# replace just one parameter, tE with log_tE.
### In essence, we need to define a transformation function within pyLIMA.
### For this particular transformation, i.e. from tE to log(tE), pyLIMA already
# provides the necessary functions to convert back and forth:
my_pars = MyFancyParameters()

fspl = FSPL_model.FSPLmodel(your_event, fancy_parameters=my_pars)

### We now want to fit this FSPL model to the data using the Trust Region Reflective
# (TRF) algorithm, but we have set it to
### use different parameters for the fit. Instead of tE, we have now set it to use
# log_tE.
### We can make this faster by using the results we obtained in example 1: [t0, u0,
# tE, rho] = [79.9, 0.008, 10.1, 0.023].
### Since the results in example 1 were given in the standard format, we need to
# adjust them so they match the new definition.
guess_parameters = [79.9, 0.008, np.log10(10.1), 0.023]

### Import the TRF fitting algorithm and fit

my_fit = TRF_fit.TRFfit(fspl)
my_fit.model_parameters_guess = guess_parameters
my_fit.fit()

### Let's see the plot. Zoom close to the peak again to see what is going on.


pyLIMA_plots.plot_lightcurves(fspl, my_fit.fit_results['best_model'])
plt.show()


### So this works as expected!

### OK, let's try something more complicated now: define t_star = rho*tE and use
# log_rho = log(rho).
### The log_rho definition is already provided by pyLIMA, but t_star isn't. 
### So we need to tell pyLIMA what kind of changes we want by defining them:

### Define the transformation from t_star --> t_E. This uses the default
# parameterisation.

class MyFancyParameters2(object):

    def __init__(self, fancy_parameters = {'rho': 'log_rho','tE':'t_star'},
                       fancy_boundaries = {'log_rho':(-5,0),'t_star':(0,1)} ):

        self.fancy_parameters = fancy_parameters

        self.fancy_boundaries = fancy_boundaries


    def rho(self, fancy_params):

        return 10**fancy_params['log_rho']

    def log_rho(self, standard_params):

        return np.log10(standard_params['rho'])
    
    def tE(self, fancy_params):

        return fancy_params['t_star']/10**fancy_params['log_rho']

    def t_star(self, standard_params):

        return standard_params['tE']*standard_params['rho']

### Update the fancy parameter dictionary with the new definitions
my_pars2 = MyFancyParameters2()
fspl2 = FSPL_model.FSPLmodel(your_event, fancy_parameters=my_pars2)

### Give it the guess parameters we obtained from example 1, formatted using the new
# definitions.
### t_star = rho * tE so in our example that is 10.1 * 0.023:
guess_parameters2 = [79.9, 0.008, 10.1 * 0.023, np.log10(0.023)]

### Perform the fit using the new parameter definitions:
my_fit2 = TRF_fit.TRFfit(fspl2)
my_fit2.model_parameters_guess = guess_parameters2
my_fit2.fit()

### To call all standard plotting options you can also use the fit_outputs module.
### If you want just the light curve, you can use plot_lightcurves as in example 1.
my_fit2.fit_outputs()
plt.show()

### Let's look at the optimized parameters and the chi^2 of the fit:
print("fit results: ", my_fit2.fit_results['best_model'])
print("chi2: ", my_fit2.fit_results['chi2'])

### If you have forgotten the order of the parameters, do:
my_fit2.fit_parameters.keys()

### Note that the results now are displayed with our newly defined parameters.

### This concludes tutorial 2.
