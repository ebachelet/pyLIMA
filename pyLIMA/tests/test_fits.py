import numpy as np
import pyLIMA.fits as pyfit
import pyLIMA.models as pymod

from pyLIMA import event
from pyLIMA import telescopes

np.random.seed(51)


def create_event():
    your_event = event.Event()

    data_1 = np.loadtxt('./examples/data/Survey_1.dat')
    telescope_1 = telescopes.Telescope(name='OGLE',
                                       camera_filter='I',
                                       light_curve=data_1.astype(float),
                                       light_curve_names=['time', 'mag', 'err_mag'],
                                       light_curve_units=['JD', 'mag', 'mag'])

    data_2 = np.loadtxt('./examples/data/Followup_1.dat')
    telescope_2 = telescopes.Telescope(name='LCO',
                                       camera_filter='I',
                                       light_curve=data_2.astype(float),
                                       light_curve_names=['time', 'mag', 'err_mag'],
                                       light_curve_units=['JD', 'mag', 'mag'])

    your_event.telescopes.append(telescope_1)
    your_event.telescopes.append(telescope_2)

    telescope_1.ld_gamma = 0.5
    telescope_2.ld_gamma = 0.5

    return your_event


def test_LM():
    eve = create_event()

    pspl = pymod.FSPLmodel(eve)

    my_fit = pyfit.TRFfit(pspl)
    my_fit.fit()

    values = [my_fit.fit_results[key] for key in my_fit.fit_results.keys()]

    assert np.allclose(values[0], np.array(
        [79.93092166436098, 0.008144359355309872, 10.110765454770114,
         0.022598878807753468, 2917.6397596937854, 208.25164854085628,
         92640.22522809621, 49189.89910261785]), atol=0,  rtol=0.001)

    assert np.allclose(values[1], 3851.0557824024704)
    assert np.allclose(values[3],
                       np.array([[6.49257550e-07, 1.40782955e-07, -1.97323807e-06,
                                  -3.70762247e-08, 9.85490647e-04, -9.76448768e-04,
                                  1.63793660e-03, 1.71403304e-04],
                                 [1.40782955e-07, 4.64977228e-08, -2.98946979e-06,
                                  -2.69722004e-09, 1.24031034e-03, -1.21483177e-03,
                                  3.43937685e-02, 3.55804762e-03],
                                 [-1.97323807e-06, -2.98946979e-06, 1.41395236e-03,
                                  -2.98528633e-06, -5.13112649e-01, 4.96004753e-01,
                                  -1.38713304e+01, -1.43501424e+00],
                                 [-3.70762247e-08, -2.69722004e-09, -2.98528633e-06,
                                  1.26526588e-08, 1.06192035e-03, -1.02463542e-03,
                                  3.37303951e-02, 3.48908060e-03],
                                 [9.85490647e-04, 1.24031034e-03, -5.13112649e-01,
                                  1.06192035e-03, 1.92433495e+02, -1.86739259e+02,
                                  5.10157189e+03, 5.27775375e+02],
                                 [-9.76448768e-04, -1.21483177e-03, 4.96004753e-01,
                                  -1.02463542e-03, -1.86739259e+02, 1.82314553e+02,
                                  -4.93919738e+03, -5.10979226e+02],
                                 [1.63793660e-03, 3.43937685e-02, -1.38713304e+01,
                                  3.37303951e-02, 5.10157189e+03, -4.93919738e+03,
                                  1.68845252e+05, 1.74649666e+04],
                                 [1.71403304e-04, 3.55804762e-03, -1.43501424e+00,
                                  3.48908060e-03, 5.27775375e+02, -5.10979226e+02,
                                  1.74649666e+04, 1.80653636e+03]]), atol=0, rtol=0.001)


def test_TRF():
    eve = create_event()

    pspl = pymod.FSPLmodel(eve)

    my_fit = pyfit.TRFfit(pspl)
    my_fit.fit()

    values = [my_fit.fit_results[key] for key in my_fit.fit_results.keys()]

    assert np.allclose(values[0], np.array(
        [79.93092166436098, 0.008144359355309872, 10.110765454770114,
         0.022598878807753468, 2917.6397596937854, 208.25164854085628,
         92640.22522809621, 49189.89910261785]), atol=0, rtol=0.001)

    assert np.allclose(values[1], 3851.0557824024704)

    assert np.allclose(values[3],
                       np.array([[6.49257550e-07, 1.40782955e-07, -1.97323807e-06,
                                  -3.70762247e-08, 9.85490647e-04, -9.76448768e-04,
                                  1.63793660e-03, 1.71403304e-04],
                                 [1.40782955e-07, 4.64977228e-08, -2.98946979e-06,
                                  -2.69722004e-09, 1.24031034e-03, -1.21483177e-03,
                                  3.43937685e-02, 3.55804762e-03],
                                 [-1.97323807e-06, -2.98946979e-06, 1.41395236e-03,
                                  -2.98528633e-06, -5.13112649e-01, 4.96004753e-01,
                                  -1.38713304e+01, -1.43501424e+00],
                                 [-3.70762247e-08, -2.69722004e-09, -2.98528633e-06,
                                  1.26526588e-08, 1.06192035e-03, -1.02463542e-03,
                                  3.37303951e-02, 3.48908060e-03],
                                 [9.85490647e-04, 1.24031034e-03, -5.13112649e-01,
                                  1.06192035e-03, 1.92433495e+02, -1.86739259e+02,
                                  5.10157189e+03, 5.27775375e+02],
                                 [-9.76448768e-04, -1.21483177e-03, 4.96004753e-01,
                                  -1.02463542e-03, -1.86739259e+02, 1.82314553e+02,
                                  -4.93919738e+03, -5.10979226e+02],
                                 [1.63793660e-03, 3.43937685e-02, -1.38713304e+01,
                                  3.37303951e-02, 5.10157189e+03, -4.93919738e+03,
                                  1.68845252e+05, 1.74649666e+04],
                                 [1.71403304e-04, 3.55804762e-03, -1.43501424e+00,
                                  3.48908060e-03, 5.27775375e+02, -5.10979226e+02,
                                  1.74649666e+04, 1.80653636e+03]]), atol=0, rtol=0.001)


def test_DE():
    eve = create_event()

    pspl = pymod.FSPLmodel(eve)

    my_fit = pyfit.DEfit(pspl, DE_population_size=1, max_iteration=10,
                         display_progress=False, strategy='best1bin')
    my_fit.fit()

    values = [my_fit.fit_results[key] for key in my_fit.fit_results.keys()]

    assert len(values) == 4

    assert len(values[0]) == 4

    assert values[3].shape == (55,9)

def test_MCMC():
    eve = create_event()

    pspl = pymod.FSPLmodel(eve)

    my_fit = pyfit.MCMCfit(pspl, MCMC_walkers=2, MCMC_links=10, )

    my_fit.model_parameters_guess = [79.93092166436098, 0.008144359355309872,
                                     10.110765454770114,
                                     0.022598878807753468, ]
    my_fit.fit()

    values = [my_fit.fit_results[key] for key in my_fit.fit_results.keys()]

    assert len(values) == 5

    assert len(values[0]) == 8

    assert values[2].shape == (10, 8, 5)

    assert values[3].shape == (10, 8, 9)


def test_objective_functions():

    eve = create_event()

    pspl = pymod.FSPLmodel(eve)

    my_fit = pyfit.TRFfit(pspl)
    my_fit.fit()

    de_fit = pyfit.TRFfit(pspl)
    mcmc_fit = pyfit.TRFfit(pspl)


    values = [my_fit.fit_results[key] for key in my_fit.fit_results.keys()]
    chi2_de = de_fit.model_chi2(values[0])[0]
    chi2_mcmc = mcmc_fit.model_chi2(values[0])[0]

    assert np.allclose(values[1], 3851.0557824024704)
    assert np.allclose(values[1], chi2_de)
    assert np.allclose(values[1], chi2_mcmc)