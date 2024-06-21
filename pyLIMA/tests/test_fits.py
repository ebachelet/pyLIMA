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
        [79.93092166158938, -0.008144355476847592, 10.110765704799517, 0.0225988791832888, 2917.639653172235,
         208.25175315434493, 92640.21776305692, 49189.94861271916]), atol=0,  rtol=0.001)

    assert np.allclose(values[1], 3851.0557824024704)
    assert np.allclose(values[3],
                       np.array([[ 6.49257596e-07, -1.40783055e-07, -1.97323951e-06,
                        -3.70762203e-08,  9.85491231e-04, -9.76449343e-04,
                         1.63795512e-03,  1.71405216e-04],
                       [-1.40783055e-07,  4.64977778e-08,  2.98947089e-06,
                         2.69722420e-09, -1.24031080e-03,  1.21483222e-03,
                        -3.43937791e-02, -3.55804863e-03],
                       [-1.97323951e-06,  2.98947089e-06,  1.41395249e-03,
                        -2.98528654e-06, -5.13112667e-01,  4.96004770e-01,
                        -1.38713305e+01, -1.43501420e+00],
                       [-3.70762203e-08,  2.69722420e-09, -2.98528654e-06,
                         1.26526584e-08,  1.06192035e-03, -1.02463542e-03,
                         3.37303934e-02,  3.48908032e-03],
                       [ 9.85491231e-04, -1.24031080e-03, -5.13112667e-01,
                         1.06192035e-03,  1.92433490e+02, -1.86739254e+02,
                         5.10157159e+03,  5.27775329e+02],
                       [-9.76449343e-04,  1.21483222e-03,  4.96004770e-01,
                        -1.02463542e-03, -1.86739254e+02,  1.82314547e+02,
                        -4.93919709e+03, -5.10979181e+02],
                       [ 1.63795512e-03, -3.43937791e-02, -1.38713305e+01,
                         3.37303934e-02,  5.10157159e+03, -4.93919709e+03,
                         1.68845236e+05,  1.74649644e+04],
                       [ 1.71405216e-04, -3.55804863e-03, -1.43501420e+00,
                         3.48908032e-03,  5.27775329e+02, -5.10979181e+02,
                         1.74649644e+04,  1.80653609e+03]]), atol=0, rtol=0.001)


def test_TRF():
    eve = create_event()

    pspl = pymod.FSPLmodel(eve)

    my_fit = pyfit.TRFfit(pspl)
    my_fit.fit()

    values = [my_fit.fit_results[key] for key in my_fit.fit_results.keys()]

    assert np.allclose(values[0], np.array([79.93092166158938, -0.008144355476847592, 10.110765704799517,
                                            0.0225988791832888, 2917.639653172235, 208.25175315434493,
                                            92640.21776305692, 49189.94861271916]), atol=0, rtol=0.001)

    assert np.allclose(values[1], 3851.0557824024704)

    assert np.allclose(values[3],
                       np.array([[ 6.49257596e-07, -1.40783055e-07, -1.97323951e-06,
                                    -3.70762203e-08,  9.85491231e-04, -9.76449343e-04,
                                     1.63795512e-03,  1.71405216e-04],
                                   [-1.40783055e-07,  4.64977778e-08,  2.98947089e-06,
                                     2.69722420e-09, -1.24031080e-03,  1.21483222e-03,
                                    -3.43937791e-02, -3.55804863e-03],
                                   [-1.97323951e-06,  2.98947089e-06,  1.41395249e-03,
                                    -2.98528654e-06, -5.13112667e-01,  4.96004770e-01,
                                    -1.38713305e+01, -1.43501420e+00],
                                   [-3.70762203e-08,  2.69722420e-09, -2.98528654e-06,
                                     1.26526584e-08,  1.06192035e-03, -1.02463542e-03,
                                     3.37303934e-02,  3.48908032e-03],
                                   [ 9.85491231e-04, -1.24031080e-03, -5.13112667e-01,
                                     1.06192035e-03,  1.92433490e+02, -1.86739254e+02,
                                     5.10157159e+03,  5.27775329e+02],
                                   [-9.76449343e-04,  1.21483222e-03,  4.96004770e-01,
                                    -1.02463542e-03, -1.86739254e+02,  1.82314547e+02,
                                    -4.93919709e+03, -5.10979181e+02],
                                   [ 1.63795512e-03, -3.43937791e-02, -1.38713305e+01,
                                     3.37303934e-02,  5.10157159e+03, -4.93919709e+03,
                                     1.68845236e+05,  1.74649644e+04],
                                   [ 1.71405216e-04, -3.55804863e-03, -1.43501420e+00,
                                     3.48908032e-03,  5.27775329e+02, -5.10979181e+02,
                                     1.74649644e+04,  1.80653609e+03]]), atol=0, rtol=0.001)


def test_DE():
    eve = create_event()

    pspl = pymod.FSPLmodel(eve)

    my_fit = pyfit.DEfit(pspl, DE_population_size=1, max_iteration=10,
                         display_progress=False, strategy='best1bin')
    my_fit.fit()

    values = [my_fit.fit_results[key] for key in my_fit.fit_results.keys()]
    assert len(values) == 4

    assert len(values[0]) == 4

    assert values[3].shape == (88, 9)

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

    de_fit = pyfit.DEfit(pspl, loss_function='chi2')
    mcmc_fit = pyfit.MCMCfit(pspl, loss_function='chi2')

    de_fit2 = pyfit.DEfit(pspl, loss_function='soft_l1')
    mcmc_fit2 = pyfit.MCMCfit(pspl, loss_function='soft_l1')

    de_fit3 = pyfit.DEfit(pspl, loss_function='likelihood')
    mcmc_fit3 = pyfit.MCMCfit(pspl, loss_function='likelihood')

    values = [my_fit.fit_results[key] for key in my_fit.fit_results.keys()]
    chi2_de = de_fit.model_chi2(values[0])[0]
    chi2_mcmc = mcmc_fit.model_chi2(values[0])[0]

    assert np.allclose(values[1], 3851.0557824024704)
    assert np.allclose(values[1], chi2_de)
    assert np.allclose(values[1], chi2_mcmc)
    assert np.allclose(values[1], de_fit.objective_function(np.array(values[0])))
    assert np.allclose(values[1], -mcmc_fit.objective_function(np.array(values[0])))

    assert np.allclose(my_fit.model_soft_l1(values[0])[0], 2688.1436988659825)
    assert np.allclose(de_fit.model_soft_l1(values[0])[0], 2688.1436988659825)
    assert np.allclose(mcmc_fit.model_soft_l1(values[0])[0], 2688.1436988659825)
    assert np.allclose(de_fit2.objective_function(np.array(values[0])),
                       2688.1436988659825)
    assert np.allclose(mcmc_fit2.objective_function(np.array(values[0])),
                       -2688.1436988659825)

    like, pym = de_fit3.model_likelihood(np.array(values[0]))

    #priors = [np.log(de_fit3.priors[key].pdf(pym[key])) for key in
    #          de_fit3.priors.keys()]

    assert np.allclose(de_fit3.objective_function(np.array(values[0])),
                       20467.16900026673)
    assert np.allclose(mcmc_fit3.objective_function(np.array(values[0])),
                       -20467.16900026673)






