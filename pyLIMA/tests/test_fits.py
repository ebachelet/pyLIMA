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
                                       lightcurve=data_1.astype(float),
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'])

    data_2 = np.loadtxt('./examples/data/Followup_1.dat')
    telescope_2 = telescopes.Telescope(name='LCO',
                                       camera_filter='I',
                                       lightcurve=data_2.astype(float),
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'])

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
        [79.93092166158918, 0.008144355475100886, 10.110765705232229, 0.022598879182430687,
         2917.6396530096204, 3125.8914063217517, 92640.21775628686, 141830.16639509634]), atol=0,  rtol=0.001)
    assert np.allclose(values[1], 3851.0557824024704)
    assert np.allclose(values[3],
                       np.array([[6.50143572e-07, 1.43401582e-07, -2.88106606e-06,
                                  -3.48199686e-08, 1.32141687e-03, 1.95156706e-05,
                                  1.36750059e-02, 1.42098775e-03],
                                 [1.43401582e-07, 4.91566822e-08, -3.57652040e-06,
                                  -1.11017601e-09, 1.46291460e-03, 3.18497790e-05,
                                  4.47699920e-02, 4.64960874e-03],
                                 [-2.88106606e-06, -3.57652040e-06, 1.47935327e-03,
                                  -3.20065516e-06, -5.39531281e-01, -1.76966964e-02,
                                  -1.58080974e+01, -1.64177757e+00],
                                 [-3.48199686e-08, -1.11017601e-09, -3.20065516e-06,
                                  1.33159844e-08, 1.14699123e-03, 3.93678840e-05,
                                  3.91789985e-02, 4.06883728e-03],
                                 [1.32141687e-03, 1.46291460e-03, -5.39531281e-01,
                                  1.14699123e-03, 2.03024354e+02, 5.93812779e+00,
                                  5.84490675e+03, 6.07036085e+02],
                                 [1.95156706e-05, 3.18497790e-05, -1.76966964e-02,
                                  3.93678840e-05, 5.93812779e+00, 1.27437842e+00,
                                  1.82726755e+02, 1.89771567e+01],
                                 [1.36750059e-02, 4.47699920e-02, -1.58080974e+01,
                                  3.91789985e-02, 5.84490675e+03, 1.82726755e+02,
                                  2.07384424e+05, 2.15371948e+04],
                                 [1.42098775e-03, 4.64960874e-03, -1.64177757e+00,
                                  4.06883728e-03, 6.07036085e+02, 1.89771567e+01,
                                  2.15371948e+04, 2.23667118e+03]]), atol=0, rtol=0.001)


def test_TRF():
    eve = create_event()

    pspl = pymod.FSPLmodel(eve)

    my_fit = pyfit.TRFfit(pspl)
    my_fit.fit()

    values = [my_fit.fit_results[key] for key in my_fit.fit_results.keys()]

    assert np.allclose(values[0], np.array(
        [79.93092166158918, 0.008144355475100886, 10.110765705232229, 0.022598879182430687,
         2917.6396530096204, 3125.8914063217517, 92640.21775628686, 141830.16639509634]), atol=0, rtol=0.001)

    assert np.allclose(values[1], 3851.0557824024704)

    assert np.allclose(values[3],
                        np.array([[ 6.50143572e-07,  1.43401582e-07, -2.88106606e-06,
                        -3.48199686e-08,  1.32141687e-03,  1.95156706e-05,
                        1.36750059e-02,  1.42098775e-03],
                        [ 1.43401582e-07,  4.91566822e-08, -3.57652040e-06,
                        -1.11017601e-09,  1.46291460e-03,  3.18497790e-05,
                        4.47699920e-02,  4.64960874e-03],
                        [-2.88106606e-06, -3.57652040e-06,  1.47935327e-03,
                        -3.20065516e-06, -5.39531281e-01, -1.76966964e-02,
                        -1.58080974e+01, -1.64177757e+00],
                        [-3.48199686e-08, -1.11017601e-09, -3.20065516e-06,
                        1.33159844e-08,  1.14699123e-03,  3.93678840e-05,
                        3.91789985e-02,  4.06883728e-03],
                        [ 1.32141687e-03,  1.46291460e-03, -5.39531281e-01,
                        1.14699123e-03,  2.03024354e+02,  5.93812779e+00,
                        5.84490675e+03,  6.07036085e+02],
                        [ 1.95156706e-05,  3.18497790e-05, -1.76966964e-02,
                        3.93678840e-05,  5.93812779e+00,  1.27437842e+00,
                        1.82726755e+02,  1.89771567e+01],
                        [ 1.36750059e-02,  4.47699920e-02, -1.58080974e+01,
                        3.91789985e-02,  5.84490675e+03,  1.82726755e+02,
                        2.07384424e+05,  2.15371948e+04],
                        [ 1.42098775e-03,  4.64960874e-03, -1.64177757e+00,
                        4.06883728e-03,  6.07036085e+02,  1.89771567e+01,
                        2.15371948e+04,  2.23667118e+03]]), atol=0, rtol=0.001)


def test_DE():
    eve = create_event()

    pspl = pymod.FSPLmodel(eve)

    my_fit = pyfit.DEfit(pspl, DE_population_size=1, max_iteration=10,
                         display_progress=False, strategy='best1bin')
    my_fit.fit()

    values = [my_fit.fit_results[key] for key in my_fit.fit_results.keys()]

    assert len(values) == 5

    assert len(values[0]) == 10

    assert values[3].shape == (88, 12)

def test_MCMC():

    eve = create_event()

    pspl = pymod.FSPLmodel(eve)

    my_fit = pyfit.MCMCfit(pspl, MCMC_walkers=2, MCMC_links=10, )

    my_fit.model_parameters_guess = [79.93092166436098, 0.008144359355309872,
                                     10.110765454770114,
                                     0.022598878807753468, ]
    my_fit.fit()
    values = [my_fit.fit_results[key] for key in my_fit.fit_results.keys()]

    assert len(values) == 6

    assert len(values[0]) == 8

    assert values[2].shape == (10, 8, 6)

    assert values[3].shape == (10, 8, 10)


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

    assert np.allclose(my_fit.model_soft_l1(values[0])[0], 2688.1437107410475)
    assert np.allclose(de_fit.model_soft_l1(values[0])[0], 2688.1437107410475)
    assert np.allclose(mcmc_fit.model_soft_l1(values[0])[0], 2688.1437107410475)
    assert np.allclose(de_fit2.objective_function(np.array(values[0])),
                       2688.1437107410475)
    assert np.allclose(mcmc_fit2.objective_function(np.array(values[0])),
                       -2688.1437107410475)

    like,priors, pym = de_fit3.model_likelihood(np.array(values[0]))

    #priors = [np.log(de_fit3.priors[key].pdf(pym[key])) for key in
    #          de_fit3.priors.keys()]

    assert np.allclose(de_fit3.objective_function(np.array(values[0])),
                       20478.7104956208)
    assert np.allclose(mcmc_fit3.objective_function(np.array(values[0])),
                       -20478.7104956208)






