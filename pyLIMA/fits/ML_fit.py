import sys
import numpy as np
from collections import OrderedDict, namedtuple
from bokeh.io import output_file, show

from pyLIMA.priors import parameters_boundaries
from pyLIMA.outputs import pyLIMA_plots
import pyLIMA.fits.objective_functions as objective_functions

from bokeh.layouts import gridplot, row
from bokeh.plotting import output_file, save

class FitException(Exception):
    pass


class MLfit(object):
    """
    ######## Fitter module ########

    This class contains the method to fit the event with the selected attributes.

    **WARNING**: All fits (and so results) are made using data in flux.

    Attributes :

        event : the event object on which you perform the fit on. More details on the event module.

        model : The microlensing model you want to fit. Has to be an object define in
                microlmodels module.
                More details on the microlmodels module.

        method : The fitting method you want to use for the fit.

        guess : The guess you can give to the fit or the guess return by the initial_guess function.

        fit_results : the fit parameters returned by method LM and DE.

        fit_covariance : the fit parameters covariance matrix returned by method LM and DE.

        fit_time : the time needed to fit.

        MCMC_chains : the MCMC chains returns by the MCMC method

        MCMC_probabilities : the objective function computed for each chains of the MCMC method

        fluxes_MCMC_method : a string describing how you want to estimate the model fluxes for the MCMC method.

        outputs : the standard pyLIMA outputs. More details in the microloutputs module.

    :param object event: the event object on which you perform the fit on. More details on the
                         event module.


    """

    def __init__(self, model, rescale_photometry=False, rescale_astrometry=False, telescopes_fluxes_method='fit'):
        """The fit class has to be intialized with an event object."""

        self.model = model
        self.rescale_photometry = rescale_photometry
        self.rescale_astrometry = rescale_astrometry
        self.telescopes_fluxes_method = telescopes_fluxes_method
        self.fit_parameters = []
        self.fit_results = {}
        self.priors = None

        self.model_parameters_guess = []
        self.rescale_photometry_parameters_guess = []
        self.rescale_astrometry_parameters_guess = []
        self.telescopes_fluxes_parameters_guess = []

        self.model_parameters_index = []
        self.rescale_photometry_parameters_index = []
        self.rescale_astrometry_parameters_index = []

        self.define_fit_parameters()

    def define_fit_parameters(self):

        fit_parameters_dictionnary_updated = self.model.model_dictionnary.copy()

        if self.telescopes_fluxes_method != 'fit':

            for telescope in self.model.event.telescopes:

                fit_parameters_dictionnary_updated.popitem()
                fit_parameters_dictionnary_updated.popitem()

        if self.rescale_photometry:

            for telescope in self.model.event.telescopes:

                if telescope.lightcurve_flux is not None:

                    fit_parameters_dictionnary_updated['logk_photometry_' + telescope.name] = \
                        len(fit_parameters_dictionnary_updated)

        if self.rescale_astrometry:

            for telescope in self.model.event.telescopes:

                if telescope.astrometry is not None:

                    fit_parameters_dictionnary_updated['logk_astrometry_ra_' + telescope.name] = \
                        len(fit_parameters_dictionnary_updated)

                    fit_parameters_dictionnary_updated['logk_astrometry_dec_' + telescope.name] = \
            len(fit_parameters_dictionnary_updated)

        self.fit_parameters = OrderedDict(
            sorted(fit_parameters_dictionnary_updated.items(), key=lambda x: x[1]))

        fit_parameters_boundaries = parameters_boundaries.parameters_boundaries(self.model.event, self.model.pyLIMA_standards_dictionnary)

        #t_0 limit fix
        mins_time = []
        maxs_time = []

        for telescope in self.model.event.telescopes:

            if telescope.lightcurve_flux is not None:

                mins_time.append(np.min(telescope.lightcurve_flux['time'].value))
                maxs_time.append(np.max(telescope.lightcurve_flux['time'].value))

            if telescope.astrometry is not None:

                mins_time.append(np.min(telescope.astrometry['time'].value))
                maxs_time.append(np.max(telescope.astrometry['time'].value))

        fit_parameters_boundaries[0] = [np.min(mins_time),np.max(maxs_time)]

        for ind, key in enumerate(self.fit_parameters.keys()):

            self.fit_parameters[key] = [ind, fit_parameters_boundaries[ind]]

        if len(self.model.fancy_to_pyLIMA_dictionnary) != 0:

            list_of_keys = [i for i in self.model.pyLIMA_standards_dictionnary]
            bounds = namedtuple('parameters', list_of_keys)

            for ind,key in enumerate(list_of_keys):

                setattr(bounds, key, np.array(fit_parameters_boundaries[ind]))

            for key in self.model.fancy_to_pyLIMA_dictionnary.keys():

                parameter = self.model.fancy_to_pyLIMA_dictionnary[key]
                index = np.where(parameter == np.array(list_of_keys))[0][0]

                if 'center' in key:

                    if key == 't_center':

                        new_bounds = fit_parameters_boundaries[0]

                    else:

                        new_bounds = parameters_boundaries.parameters_boundaries(self.model.event, {key:'0'})[0]

                else:

                    new_bounds = self.model.pyLIMA_to_fancy[key](bounds)

                new_bounds = np.sort(new_bounds)
                self.fit_parameters.pop(key)
                self.fit_parameters[key] = [index, new_bounds]

        self.fit_parameters = OrderedDict(
            sorted(self.fit_parameters.items(), key=lambda x: x[1]))

        self.model_parameters_index = [self.model.model_dictionnary[i] for i in self.model.model_dictionnary.keys() if
                                       i in self.fit_parameters.keys()]

        self.rescale_photometry_parameters_index = [self.fit_parameters[i][0] for i in self.fit_parameters.keys() if
                                                    'logk_photometry' in i]

        self.rescale_astrometry_parameters_index = [self.fit_parameters[i][0] for i in self.fit_parameters.keys() if
                                                    'logk_astrometry' in i]


    def objective_function(self):

       pass

    def get_priors(self, parameters):

        ln_likelihood = 0

        if self.priors is not None:

            for ind, prior_pdf in enumerate(self.priors):

                if prior_pdf is not None:

                    probability = prior_pdf.pdf(parameters[ind])

                    if probability > 0:

                        ln_likelihood += -np.log(probability)

                    else:

                        ln_likelihood = -np.inf

        return ln_likelihood



    def model_guess(self):
        """Try to estimate the microlensing parameters. Only use for PSPL and FSPL
           models. More details on microlguess module.

           :return guess_parameters: a list containing parameters guess related to the model.
           :rtype: list
        """
        import pyLIMA.priors.guess

        if len(self.model_parameters_guess) == 0:

            try:
                # Estimate the Paczynski parameters

                if self.model.model_type == 'PSPL':

                    guess_paczynski_parameters, f_source = pyLIMA.priors.guess.initial_guess_PSPL(self.model.event)

                if self.model.model_type == 'FSPL':

                    guess_paczynski_parameters, f_source = pyLIMA.priors.guess.initial_guess_FSPL(self.model.event)

                if self.model.model_type == 'FSPLee':

                    guess_paczynski_parameters, f_source = pyLIMA.priors.guess.initial_guess_FSPL(self.model.event)

                if self.model.model_type == 'FSPLarge':

                    guess_paczynski_parameters, f_source = pyLIMA.priors.guess.initial_guess_FSPL(self.model.event)

                if self.model.model_type == 'DSPL':

                    guess_paczynski_parameters, f_source = pyLIMA.priors.guess.initial_guess_DSPL(self.model.event)

                if 'theta_E' in self.fit_parameters.keys():

                    guess_paczynski_parameters = guess_paczynski_parameters + [1.0]

                if 'piEN' in self.fit_parameters.keys():
                    guess_paczynski_parameters = guess_paczynski_parameters + [0.0, 0.0]

                if 'XiEN' in self.fit_parameters.keys():
                    guess_paczynski_parameters = guess_paczynski_parameters + [0, 0]

                if 'dsdt' in self.fit_parameters.keys():
                    guess_paczynski_parameters = guess_paczynski_parameters + [0, 0]

                if 'spot_size' in self.fit_parameters.keys():
                    guess_paczynski_parameters = guess_paczynski_parameters + [0]

                self.model_parameters_guess = guess_paczynski_parameters

            except:

                raise FitException('Can not estimate guess, likely your model is too complex to automatic estimate. '
                                   'Please provide some in self.model_parameters_guess or run a DE fit.')
        else:


            self.model_parameters_guess = [float(i) for i in self.model_parameters_guess]

    def telescopes_fluxes_guess(self):

        if self.telescopes_fluxes_method == 'fit':

            if self.telescopes_fluxes_parameters_guess == []:

                telescopes_fluxes = self.find_fluxes(self.model_parameters_guess)


                self.telescopes_fluxes_parameters_guess = telescopes_fluxes

            self.telescopes_fluxes_parameters_guess = [float(i) for i in self.telescopes_fluxes_parameters_guess]

        else:

            self.telescopes_fluxes_parameters_guess = []

    def rescale_photometry_guess(self):

        if self.rescale_photometry:

            if self.rescale_photometry_parameters_guess == []:

                rescale_photometry_guess = []

                for telescope in self.model.event.telescopes:

                    if telescope.lightcurve_flux is not None:

                        rescale_photometry_guess.append(-4)

                self.rescale_photometry_parameters_guess = rescale_photometry_guess

            self.rescale_photometry_parameters_guess = [float(i) for i in self.rescale_photometry_parameters_guess]

        else:

            self.rescale_photometry_parameters_guess = []

    def rescale_astrometry_guess(self):

        if self.rescale_astrometry:

            if self.rescale_astrometry_parameters_guess == []:

                rescale_astrometry_guess = []

                for telescope in self.model.event.telescopes:

                    if telescope.astrometry is not None:

                        rescale_astrometry_guess.append(-4)
                        rescale_astrometry_guess.append(-4)

                self.rescale_astrometry_parameters_guess = rescale_astrometry_guess

            self.rescale_astrometry_parameters_guess = [float(i) for i in self.rescale_astrometry_parameters_guess]

        else:

            self.rescale_astrometry_parameters_guess = []
    def initial_guess(self):

        self.model_guess()
        self.telescopes_fluxes_guess()
        self.rescale_photometry_guess()
        self.rescale_astrometry_guess()

        fit_parameters_guess = self.model_parameters_guess+self.telescopes_fluxes_parameters_guess+\
                               self.rescale_photometry_parameters_guess+self.rescale_astrometry_parameters_guess
        fit_parameters_guess = [float(i) for i in fit_parameters_guess]

        if len(self.model.fancy_to_pyLIMA_dictionnary) != 0:

            list_of_keys = [i for i in self.fit_parameters.keys()]

            for key in self.model.fancy_to_pyLIMA_dictionnary.keys():

                try:

                    index = np.where(self.model.fancy_to_pyLIMA_dictionnary[key] == np.array(list_of_keys))[0][0]

                    parameter = self.model.fancy_to_pyLIMA_dictionnary[key]
                    value = fit_parameters_guess[index]
                    x = namedtuple('parameters', parameter)

                    setattr(x, parameter, value)
                    fit_parameters_guess[index] = self.model.pyLIMA_to_fancy[key](x)

                except:

                    pass


        if self.priors is not None:

            for ind,prior in enumerate(self.priors):

                probability = prior.pdf( fit_parameters_guess[ind])
                if probability<10**-10:

                    samples = prior.rvs(1000)

                    fit_parameters_guess[ind] = np.median(samples)

        for ind,param in enumerate(self.fit_parameters.keys()):

            if (fit_parameters_guess[ind] < self.fit_parameters[param][1][0]):

               fit_parameters_guess[ind] = self.fit_parameters[param][1][0]
               print( 'WARNING: Setting the '+param+' to the lower limit '+str(fit_parameters_guess[ind]))

            if (fit_parameters_guess[ind] > self.fit_parameters[param][1][1]):

                fit_parameters_guess[ind] = self.fit_parameters[param][1][1]
                print( 'WARNING: Setting the '+param+' to the upper limit '+str(fit_parameters_guess[ind]))

        print(sys._getframe().f_code.co_name, ' : Initial parameters guess SUCCESS')
        print('Using guess: ', fit_parameters_guess)
        return fit_parameters_guess


    def model_residuals(self, parameters):

        parameters = np.array(parameters)
        residus = {'photometry':[],'astrometry':[]}
        errors = residus.copy()

        if self.model.photometry:

            residuals_photometry, errors_flux = self.photometric_model_residuals(parameters)
            residus['photometry'] = residuals_photometry
            errors['photometry'] = errors_flux

        if self.model.astrometry:

            residuals_astrometry, errors_astrometry = self.astrometric_model_residuals(parameters)
            residus['astrometry'] = [np.concatenate((i[0],i[1])) for i in residuals_astrometry]
            errors['astrometry'] = [np.concatenate((i[0],i[1])) for i in errors_astrometry]

        return residus, errors

    def photometric_model_residuals(self, parameters):

        parameters = np.array(parameters)

        model_parameters = parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.rescale_photometry:

            rescaling_photometry_parameters = 10 ** (parameters[self.rescale_photometry_parameters_index])

        else:

            rescaling_photometry_parameters = None


        residus_photometry, errflux_photometry = objective_functions.all_telescope_photometric_residuals(
            self.model, pyLIMA_parameters,
            norm=False,
            rescaling_photometry_parameters=rescaling_photometry_parameters)

        return residus_photometry, errflux_photometry

    def astrometric_model_residuals(self, parameters):

        parameters = np.array(parameters)

        model_parameters = parameters[self.model_parameters_index]

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.rescale_astrometry:

            rescaling_astrometry_parameters = 10 ** (parameters[self.rescale_astrometry_parameters_index])

        else:

            rescaling_astrometry_parameters = None

        residus_ra, residus_dec, err_ra, err_dec = objective_functions.all_telescope_astrometric_residuals(
            self.model, pyLIMA_parameters,
            norm=False,
            rescaling_astrometry_parameters=rescaling_astrometry_parameters)


        return [residus_ra, residus_dec], [err_ra, err_dec]

    def model_chi2(self, parameters):

        parameters = np.array(parameters)

        residus, err = self.model_residuals(parameters)

        residuals = []
        errors = []

        for data_type in ['photometry','astrometry']:

            try:

                residuals.append(np.concatenate(residus[data_type])**2)
                errors.append(np.concatenate(err[data_type])**2)

            except:

                pass

        residuals = np.concatenate(residuals)
        errors = np.concatenate(errors)

        chi2 = np.sum(residuals / errors)

        return chi2

    def model_likelihood(self, parameters):

        parameters = np.array(parameters)

        residus, err = self.model_residuals(parameters)

        residuals = []
        errors = []

        for data_type in ['photometry', 'astrometry']:

            try:

                residuals.append(np.concatenate(residus[data_type]) ** 2)
                errors.append(np.concatenate(err[data_type]) ** 2)

            except:

                pass
        residuals = np.concatenate(residuals)
        errors = np.concatenate(errors)

        ln_likelihood = -0.5 * np.sum(residuals/errors + np.log(errors) + np.log(2 * np.pi))

        return ln_likelihood

    def chi2_photometry(self, parameters):

        parameters = np.array(parameters)

        residus, errors = self.photometric_model_residuals(parameters)
        residuals = np.concatenate(residus) ** 2
        errors = np.concatenate(errors) ** 2

        chi2 = np.sum(residuals/errors)

        return chi2

    def likelihood_photometry(self, parameters):

        parameters = np.array(parameters)

        residus, errors = self.photometric_model_residuals(parameters)

        residuals = np.concatenate(residus['photometry'])**2
        errors = np.concatenate(errors['photometry'])**2

        ln_likelihood = -0.5 * np.sum(residuals/errors + np.log(errors) + np.log(2 * np.pi))

        return ln_likelihood

    def chi2_astrometry(self, parameters):

        parameters = np.array(parameters)

        residus, errors = self.astrometric_model_residuals(parameters)

        residus = [np.concatenate((i[0], i[1])) for i in residus]
        errors = [np.concatenate((i[0],i[1])) for i in errors]

        residuals = np.concatenate(residus) ** 2
        errors = np.concatenate(errors) ** 2

        chi2 = np.sum(residuals/errors)

        return chi2

    def likelihood_photometry(self, parameters):

        parameters = np.array(parameters)

        residus, errors = self.astrometric_model_residuals(parameters)

        residus = [np.concatenate((i[0], i[1])) for i in residus]
        errors = [np.concatenate((i[0], i[1])) for i in errors]

        residuals = np.concatenate(residus) ** 2
        errors = np.concatenate(errors) ** 2

        ln_likelihood = -0.5 * np.sum(residuals/errors + np.log(errors) + np.log(2 * np.pi))

        return ln_likelihood


    def residuals_Jacobian(self, fit_process_parameters):

        photometric_jacobian = self.photometric_residuals_Jacobian(fit_process_parameters)

        #No Astrometry Yet

        return photometric_jacobian
    def photometric_residuals_Jacobian(self, fit_process_parameters):
        """Return the analytical Jacobian matrix, if requested by method LM.
        Available only for PSPL and FSPL without second_order.
        :param list fit_process_parameters: the model parameters ingested by the correpsonding
                                            fitting routine.
        :return: a numpy array which represents the jacobian matrix
        :rtype: array_like
        """

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)

        count = 0
        for telescope in self.model.event.telescopes:

            if count == 0:

                _jacobi = self.model.photometric_model_Jacobian(telescope, pyLIMA_parameters)/telescope.lightcurve_flux['err_flux'].value

            else:

                _jacobi = np.c_[_jacobi, self.model.photometric_model_Jacobian(telescope, pyLIMA_parameters)/telescope.lightcurve_flux['err_flux'].value]

            count += 1

        # The objective function is : (data-model)/errors

        _jacobi = -_jacobi
        jacobi = _jacobi[:-2]
        # Split the fs and g derivatives in several columns correpsonding to
        # each observatories
        start_index = 0
        dresdfs = _jacobi[-2]
        dresdg = _jacobi[-1]

        for telescope in self.model.event.telescopes:
            derivative_fs = np.zeros((len(dresdfs)))
            derivative_g = np.zeros((len(dresdg)))
            index = np.arange(start_index, start_index + len(telescope.lightcurve_flux))
            derivative_fs[index] = dresdfs[index]
            derivative_g[index] = dresdg[index]
            jacobi = np.r_[jacobi, np.array([derivative_fs, derivative_g])]

            start_index = index[-1] + 1

        return jacobi.T

    def find_fluxes(self, fit_process_parameters):
        """Find telescopes flux associated (fs,g) to the model. Used for initial_guess and LM
        method.

        :param list fit_process_parameters: the model parameters ingested by the correpsonding fitting
                                       routine.
        :param object model: a microlmodels which you want to compute the fs,g parameters.

        :return: a list of tuple with the (fs,g) telescopes flux parameters.
        :rtype: list
        """

        telescopes_fluxes = []
        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)

        for telescope in self.model.event.telescopes:

            if telescope.lightcurve_flux is not None:
                flux = telescope.lightcurve_flux['flux'].value

                ml_model = self.model.compute_the_microlensing_model(telescope, pyLIMA_parameters)

                f_source = ml_model['f_source']
                f_blend = ml_model['f_blend']

                # Prior here
                if (f_source <= self.fit_parameters['fsource_'+telescope.name][1][0]) | \
                   (f_source > self.fit_parameters['fsource_'+telescope.name][1][1]) |\
                   (f_blend <= self.fit_parameters['fblend_' + telescope.name][1][0]) |\
                   (f_blend > self.fit_parameters['fblend_' + telescope.name][1][1]) |\
                   (f_source + f_blend <= 0):

                    telescopes_fluxes.append(np.min(flux))
                    telescopes_fluxes.append(0.0)

                else:

                    telescopes_fluxes.append(f_source)
                    telescopes_fluxes.append(f_blend)


        return telescopes_fluxes

    def fit_outputs(self, bokeh_plot=None):

        matplotlib_lightcurves = None
        matplotlib_geometry = None
        matplotlib_astrometry = None
        matplotlib_distribution = None
        bokeh_figure = None

        if self.model.photometry:

            matplotlib_lightcurves, bokeh_lightcurves = pyLIMA_plots.plot_lightcurves(self.model, self.fit_results['best_model'], bokeh_plot=bokeh_plot)
            matplotlib_geometry, bokeh_geometry = pyLIMA_plots.plot_geometry(self.model, self.fit_results['best_model'], bokeh_plot=bokeh_plot)

        if self.model.astrometry:

            matplotlib_astrometry = pyLIMA_plots.plot_astrometry(self.model, self.fit_results['best_model'])

        parameters = [key for ind,key in enumerate(self.model.model_dictionnary.keys()) if ('fsource' not in key) and ('fblend' not in key) and ('gblend' not in key)]

        samples = self.samples_to_plot()
        samples_to_plot = samples[:, :len(parameters)]

        matplotlib_distribution, bokeh_distribution = pyLIMA_plots.plot_distribution(samples_to_plot, parameters_names=parameters,bokeh_plot=bokeh_plot)

        bokeh_figure = gridplot([[row(bokeh_lightcurves, gridplot([[bokeh_geometry]],toolbar_location='above'))]],toolbar_location=None)
        output_file(filename = self.model.event.name+'.html', title=self.model.event.name)
        save(bokeh_figure)

        return matplotlib_lightcurves, matplotlib_geometry, matplotlib_astrometry, matplotlib_distribution, bokeh_figure

