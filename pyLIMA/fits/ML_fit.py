import sys
from collections import OrderedDict, namedtuple
from multiprocessing import Manager

import numpy as np
import pyLIMA.fits.objective_functions as objective_functions
from bokeh.layouts import gridplot
from bokeh.plotting import output_file, save
from pyLIMA.outputs import pyLIMA_plots
from pyLIMA.priors import parameters_boundaries


class FitException(Exception):
    pass


class MLfit(object):
    """
    This class contains the method to fit the event with the selected attributes.

    **WARNING**: All fits (and so results) are made using data in flux in the
    lightcurves.

    Attributes
    ----------
    model : object, a microlensing model
    rescale_photometry : bool, turns on to rescale the photometric data
    rescale_astrometry : bool, turns on to rescale the astrometric data
    telescopes_fluxes_method : str, if not 'fit', then telescopes fluxes are
    estimated via np.polyfit
    loss_function : str, the loss_function used ('chi2','likelihood' or 'soft_l1')
    fit_parameters : dict, dictionnary containing the parameters name and boundaries
    fit_results : dict, dictionnary containing the fit results
    priors : list, a list of parameters priors (None by default)
    trials : list, a Manager().list() to collect all algorithm fit trials
    model_parameters_guess : list, a list containing the parameters guess
    rescale_photometry_parameters_guess : list, contains guess on rescaling photometry
    rescale_astrometry_parameters_guess : list, contains guess on rescaling astrometry
    telescopes_fluxes_parameters_guess : list, contains guess on telescopes fluxes
    model_parameters_index : list, indexes of models parameters
    rescale_photometry_parameters_index : list, indexes of photometry rescaling
    parameters
    rescale_astrometry_parameters_index : list, indexes of astrometry rescaling
    """

    def __init__(self, model, rescale_photometry=False, rescale_astrometry=False,
                 telescopes_fluxes_method='fit', loss_function='chi2'):
        """The fit class has to be intialized with an event object."""

        self.model = model
        self.rescale_photometry = rescale_photometry
        self.rescale_astrometry = rescale_astrometry
        self.telescopes_fluxes_method = telescopes_fluxes_method

        if rescale_astrometry | rescale_photometry:
            print(
                'Switching to likelihood objective function because of errorbars '
                'rescaling.')
            loss_function = 'likelihood'

        self.loss_function = loss_function

        self.fit_parameters = []
        self.priors_parameters = []
        self.fit_results = {}
        self.priors = None
        self.trials = Manager().list()  # to be recognize by all process during
        # parallelization

        self.model_parameters_guess = []
        self.rescale_photometry_parameters_guess = []
        self.rescale_astrometry_parameters_guess = []
        self.telescopes_fluxes_parameters_guess = []

        self.model_parameters_index = []
        self.rescale_photometry_parameters_index = []
        self.rescale_astrometry_parameters_index = []

        self.define_fit_parameters()
        self.define_priors_parameters()

    def define_parameters(self, include_telescopes_fluxes=True):
        """
        Define the parameters to fit and the indexes

        Parameters
        ----------
        include_telescopes_fluxes : bool, telescopes fluxes are part of the fit or not

        Returns
        -------
        fit_parameters : dict, a dictionnary with the parameters to fit and limits
        model_parameters_index : list, a list with the parameters indexes
        rescale_photometry_parameters_index : list, a list with indexes of photometry
        rescaling parameters
        rescale_astrometry_parameters_index : list, a list with indexes of astrometry
        rescaling parameters
        """
        standard_parameters_dictionnary = self.model.pyLIMA_standards_dictionnary.copy()
        standard_parameters_boundaries = \
            self.model.standard_parameters_boundaries.copy()

        fit_parameters_dictionnary_keys = []
        fit_parameters_indexes = []
        fit_parameters_boundaries = []

        thebounds = namedtuple('parameters',
                               [i for i in standard_parameters_dictionnary.keys()])

        for ind, key in enumerate(standard_parameters_dictionnary.keys()):
            setattr(thebounds, key, np.array(standard_parameters_boundaries[ind]))

        for ind, key in enumerate(standard_parameters_dictionnary.keys()):

            if (('fsource' in key) | ('fblend' in key) | ('gblend' in key)) & (
                    include_telescopes_fluxes is False):

                pass

            else:

                if key in self.model.fancy_to_pyLIMA.keys():

                    parameter = self.model.pyLIMA_to_fancy_dictionnary[key]

                    try:

                        new_bounds = np.sort(
                            self.model.pyLIMA_to_fancy[parameter](thebounds))

                    except TypeError:

                        new_bounds = standard_parameters_boundaries[ind]

                    thekey = parameter
                    theind = ind
                    theboundaries = new_bounds

                else:

                    thekey = key
                    theind = ind
                    theboundaries = standard_parameters_boundaries[ind]

                fit_parameters_dictionnary_keys.append(thekey)
                fit_parameters_indexes.append(theind)
                fit_parameters_boundaries.append(theboundaries)

        if self.rescale_photometry:

            for telescope in self.model.event.telescopes:

                if telescope.lightcurve_flux is not None:
                    thekey = 'logk_photometry_' + telescope.name
                    theind = len(fit_parameters_dictionnary_keys)
                    theboundaries = \
                        parameters_boundaries.parameters_boundaries(self.model.event,
                                                                    {thekey: 'dummy'})[
                            0]

                    fit_parameters_dictionnary_keys.append(thekey)
                    fit_parameters_indexes.append(theind)
                    fit_parameters_boundaries.append(theboundaries)

        if self.rescale_astrometry:

            for telescope in self.model.event.telescopes:

                if telescope.astrometry is not None:
                    thekey = 'logk_astrometry_ra' + telescope.name
                    theind = len(fit_parameters_dictionnary_keys)
                    theboundaries = \
                        parameters_boundaries.parameters_boundaries(self.model.event,
                                                                    {thekey: 'dummy'})[
                            0]

                    fit_parameters_dictionnary_keys.append(thekey)
                    fit_parameters_indexes.append(theind)
                    fit_parameters_boundaries.append(theboundaries)

                    thekey = 'logk_astrometry_dec' + telescope.name
                    theind = len(fit_parameters_dictionnary_keys)
                    theboundaries = \
                        parameters_boundaries.parameters_boundaries(self.model.event,
                                                                    {thekey: 'dummy'})[
                            0]

                    fit_parameters_dictionnary_keys.append(thekey)
                    fit_parameters_indexes.append(theind)
                    fit_parameters_boundaries.append(theboundaries)

        fit_parameters = {}

        for ind, key in enumerate(fit_parameters_dictionnary_keys):
            fit_parameters[key] = [fit_parameters_indexes[ind],
                                   fit_parameters_boundaries[ind]]

        fit_parameters = OrderedDict(
            sorted(fit_parameters.items(), key=lambda x: x[1]))

        # t_0 limit fix
        mins_time = []
        maxs_time = []

        for telescope in self.model.event.telescopes:

            if telescope.lightcurve_flux is not None:
                mins_time.append(np.min(telescope.lightcurve_flux['time'].value))
                maxs_time.append(np.max(telescope.lightcurve_flux['time'].value))

            if telescope.astrometry is not None:
                mins_time.append(np.min(telescope.astrometry['time'].value))
                maxs_time.append(np.max(telescope.astrometry['time'].value))

        if 't0' in fit_parameters.keys():
            fit_parameters['t0'][1] = (np.min(mins_time), np.max(maxs_time))

        if 't_center' in fit_parameters.keys():
            fit_parameters['t_center'][1] = (np.min(mins_time), np.max(maxs_time))

        model_parameters_index = [self.model.model_dictionnary[i] for i in
                                  self.model.model_dictionnary.keys() if
                                  i in fit_parameters.keys()]

        rescale_photometry_parameters_index = [fit_parameters[i][0] for i in
                                               fit_parameters.keys() if
                                               'logk_photometry' in i]

        rescale_astrometry_parameters_index = [fit_parameters[i][0] for i in
                                               fit_parameters.keys() if
                                               'logk_astrometry' in i]

        return fit_parameters, model_parameters_index, \
            rescale_photometry_parameters_index, rescale_astrometry_parameters_index

    def define_fit_parameters(self):
        """
        Define the parameters to fit

        Parameters
        ----------
        include_telescopes_fluxes : bool, telescopes fluxes are part of the fit or not

        Returns
        -------
        fit_parameters : dict, a dictionnary with the parameters to fit and limits
        model_parameters_index : list, a list with the parameters indexes
        rescale_photometry_parameters_index : list, a list with indexes of photometry
        rescaling parameters
        rescale_astrometry_parameters_index : list, a list with indexes of astrometry
        rescaling parameters
        """
        if self.telescopes_fluxes_method == 'fit':

            include_fluxes = True

        else:

            include_fluxes = False

        fit_parameters, model_parameters_index, rescale_photometry_parameters_index, \
            rescale_astrometry_parameters_index = self.define_parameters(
            include_telescopes_fluxes=include_fluxes)

        self.fit_parameters = fit_parameters
        self.model_parameters_index = model_parameters_index
        self.rescale_photometry_parameters_index = rescale_photometry_parameters_index
        self.rescale_astrometry_parameters_index = rescale_astrometry_parameters_index

    def define_priors_parameters(self):
        """
        Define the priors parameters to fit
        """
        include_fluxes = True

        fit_parameters, model_parameters_index, rescale_photometry_parameters_index, \
            rescale_astrometry_parameters_index = self.define_parameters(
            include_telescopes_fluxes=include_fluxes)

        self.priors_parameters = fit_parameters

    def standard_objective_function(self, fit_process_parameters):
        """
        Compute the objective function based on the model and fit_process_parameters

        Parameters
        ----------
        fit_process_parameters : list, list containing the fit parameters

        Returns
        -------

        objective : float, the value of the objective function
        """
        if self.loss_function == 'likelihood':
            likelihood, pyLIMA_parameters = self.model_likelihood(
                fit_process_parameters)
            objective = likelihood

        if self.loss_function == 'chi2':
            chi2, pyLIMA_parameters = self.model_chi2(fit_process_parameters)
            objective = chi2

        if self.loss_function == 'soft_l1':
            soft_l1, pyLIMA_parameters = self.model_soft_l1(fit_process_parameters)
            objective = soft_l1

        if self.telescopes_fluxes_method != 'fit':

            fluxes = []

            for tel in self.model.event.telescopes:

                if tel.lightcurve_flux is not None:
                    fluxes.append(getattr(pyLIMA_parameters, 'fsource_' + tel.name))
                    fluxes.append(getattr(pyLIMA_parameters, 'fblend_' + tel.name))

            self.trials.append(fit_process_parameters.tolist() + fluxes + [objective])

        else:

            self.trials.append(fit_process_parameters.tolist() + [objective])

        return objective

    def get_priors_probability(self, pyLIMA_parameters):
        """
        Transform the prior probability to ln space

        Parameters
        ----------
        pyLIMA_parameters : dict, a pyLIMA_parameters object

        Returns
        -------
        ln_likelihood : float, the value to add to the ln_likelihood from the priors
        """
        ln_likelihood = 0

        if self.priors is not None:

            for ind, prior_key in enumerate(self.priors.keys()):

                prior_pdf = self.priors[prior_key]

                if prior_pdf is not None:

                    probability = prior_pdf.pdf(getattr(pyLIMA_parameters, prior_key))

                    if probability > 0:

                        ln_likelihood += -np.log(probability)

                    else:

                        ln_likelihood = np.inf

        return ln_likelihood

    def model_guess(self):
        """
        Try to estimate the microlensing parameters.
        """
        import pyLIMA.priors.guess

        if len(self.model_parameters_guess) == 0:

            try:
                # Estimate the Paczynski parameters

                if self.model.model_type() == 'PSPL':
                    guess_paczynski_parameters, f_source = \
                        pyLIMA.priors.guess.initial_guess_PSPL(
                            self.model.event)

                if self.model.model_type() == 'FSPL':
                    guess_paczynski_parameters, f_source = \
                        pyLIMA.priors.guess.initial_guess_FSPL(
                            self.model.event)

                if self.model.model_type() == 'FSPLee':
                    guess_paczynski_parameters, f_source = \
                        pyLIMA.priors.guess.initial_guess_FSPL(
                            self.model.event)

                if self.model.model_type() == 'FSPLarge':
                    guess_paczynski_parameters, f_source = \
                        pyLIMA.priors.guess.initial_guess_FSPL(
                            self.model.event)

                if self.model.model_type() == 'DSPL':
                    guess_paczynski_parameters, f_source = \
                        pyLIMA.priors.guess.initial_guess_DSPL(
                            self.model.event)

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

            except ValueError:

                raise FitException(
                    'Can not estimate guess, likely your model is too complex to '
                    'automatic estimate. '
                    'Please provide some in self.model_parameters_guess or run a DE '
                    'fit.')
        else:

            self.model_parameters_guess = [float(i) for i in
                                           self.model_parameters_guess]

    def telescopes_fluxes_guess(self):
        """
        Estimate the telescopes fluxes guesses
        """
        if self.telescopes_fluxes_method == 'fit':

            if self.telescopes_fluxes_parameters_guess == []:
                telescopes_fluxes = self.model.find_telescopes_fluxes(
                    self.model_parameters_guess)
                telescopes_fluxes = self.check_telescopes_fluxes_limits(
                    telescopes_fluxes)

                self.telescopes_fluxes_parameters_guess = telescopes_fluxes

            self.telescopes_fluxes_parameters_guess = [float(i) for i in
                                                       self.telescopes_fluxes_parameters_guess]

        else:

            self.telescopes_fluxes_parameters_guess = []

    def rescale_photometry_guess(self):
        """
        Estimate the photometric rescaling guesses
        """
        if self.rescale_photometry:

            if self.rescale_photometry_parameters_guess == []:

                rescale_photometry_guess = []

                for telescope in self.model.event.telescopes:

                    if telescope.lightcurve_flux is not None:
                        rescale_photometry_guess.append(0)

                self.rescale_photometry_parameters_guess = rescale_photometry_guess

            self.rescale_photometry_parameters_guess = [float(i) for i in
                                                        self.rescale_photometry_parameters_guess]

        else:

            self.rescale_photometry_parameters_guess = []

    def rescale_astrometry_guess(self):
        """
        Estimate the astrometric rescaling guesses
        """
        if self.rescale_astrometry:

            if self.rescale_astrometry_parameters_guess == []:

                rescale_astrometry_guess = []

                for telescope in self.model.event.telescopes:

                    if telescope.astrometry is not None:
                        rescale_astrometry_guess.append(0)
                        rescale_astrometry_guess.append(0)

                self.rescale_astrometry_parameters_guess = rescale_astrometry_guess

            self.rescale_astrometry_parameters_guess = [float(i) for i in
                                                        self.rescale_astrometry_parameters_guess]

        else:

            self.rescale_astrometry_parameters_guess = []

    def initial_guess(self):
        """
        Estimate the fit guesses

        Returns
        -------
        fit_parameters_guess : list, a list of the parameters guess
        """
        self.model_guess()
        self.telescopes_fluxes_guess()
        self.rescale_photometry_guess()
        self.rescale_astrometry_guess()

        fit_parameters_guess = self.model_parameters_guess + \
                               self.telescopes_fluxes_parameters_guess + \
                               self.rescale_photometry_parameters_guess + \
                               self.rescale_astrometry_parameters_guess
        fit_parameters_guess = [float(i) for i in fit_parameters_guess]

        if len(self.model.fancy_to_pyLIMA_dictionnary) != 0:

            list_of_keys = [i for i in self.fit_parameters.keys()]

            for key in self.model.fancy_to_pyLIMA_dictionnary.keys():

                try:

                    index = np.where(
                        self.model.fancy_to_pyLIMA_dictionnary[key] == np.array(
                            list_of_keys))[0][0]

                    parameter = self.model.fancy_to_pyLIMA_dictionnary[key]
                    value = fit_parameters_guess[index]
                    x = namedtuple('parameters', parameter)

                    setattr(x, parameter, value)
                    fit_parameters_guess[index] = self.model.pyLIMA_to_fancy[key](x)

                except IndexError:

                    pass

        if self.priors is not None:

            for ind, prior_key in enumerate(self.fit_parameters.keys()):

                prior_pdf = self.priors[prior_key]

                probability = prior_pdf.pdf(fit_parameters_guess[ind])

                if probability < 10 ** -10:
                    samples = prior_pdf.rvs(1000)

                    fit_parameters_guess[ind] = np.median(samples)

        for ind, param in enumerate(self.fit_parameters.keys()):

            if (fit_parameters_guess[ind] < self.fit_parameters[param][1][0]):
                fit_parameters_guess[ind] = self.fit_parameters[param][1][0]
                print('WARNING: Setting the ' + param + ' to the lower limit ' + str(
                    fit_parameters_guess[ind]))

            if (fit_parameters_guess[ind] > self.fit_parameters[param][1][1]):
                fit_parameters_guess[ind] = self.fit_parameters[param][1][1]
                print('WARNING: Setting the ' + param + ' to the upper limit ' + str(
                    fit_parameters_guess[ind]))

        print(sys._getframe().f_code.co_name, ' : Initial parameters guess SUCCESS')
        print('Using guess: ', fit_parameters_guess)
        return fit_parameters_guess

    def model_residuals(self, pyLIMA_parameters, rescaling_photometry_parameters=None,
                        rescaling_astrometry_parameters=None):
        """
        Given a set of parameters, estimate the photometric and astrometric residuals

        Parameters
        ----------
        pyLIMA_parameters : dict, a pyLIMA_parameters object
        rescaling_photometry_parameters : bool, if the photometry is rescaled
        rescaling_astrometry_parameters : boold, if tje astrometry is rescaled

        Returns
        -------
        residus : array, an array containing the residuals , i.e. [res_photometry,
        res_astrometry]
        errors : array, an array containing the corresponding errors
        """
        if type(pyLIMA_parameters) is type:  # it is a pyLIMA_parameters object

            pass

        else:

            parameters = np.array(pyLIMA_parameters)

            model_parameters = parameters[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        residus = {'photometry': [], 'astrometry': []}
        errors = residus.copy()

        if self.model.photometry:
            residuals_photometry, errors_flux = self.photometric_model_residuals(
                pyLIMA_parameters,
                rescaling_photometry_parameters=rescaling_photometry_parameters)

            residus['photometry'] = residuals_photometry
            errors['photometry'] = errors_flux

        if self.model.astrometry:
            residuals_astrometry, errors_astrometry = self.astrometric_model_residuals(
                pyLIMA_parameters,
                rescaling_astrometry_parameters=rescaling_astrometry_parameters)

            residus['astrometry'] = [np.concatenate((i[0], i[1])) for i in
                                     residuals_astrometry]
            errors['astrometry'] = [np.concatenate((i[0], i[1])) for i in
                                    errors_astrometry]

        return residus, errors

    def photometric_model_residuals(self, pyLIMA_parameters,
                                    rescaling_photometry_parameters=None):
        """
        Given a set of parameters, estimate the photometric residuals

        Parameters
        ----------
        pyLIMA_parameters : dict, a pyLIMA_parameters object
        rescaling_photometry_parameters : bool, if the photometry is rescaled

        Returns
        -------
        residus_photometry : array, an array containing the photometry residuals ,
        i.e. [res_photometry_i] of telescope i
        errflux_photometry : array, an array containing the corresponding errors in flux
        """
        if type(pyLIMA_parameters) is type:  # it is a pyLIMA_parameters object

            pass

        else:

            parameters = np.array(pyLIMA_parameters)

            model_parameters = parameters[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        residus_photometry, errflux_photometry = \
            objective_functions.all_telescope_photometric_residuals(
                self.model, pyLIMA_parameters,
                norm=False,
                rescaling_photometry_parameters=rescaling_photometry_parameters)

        return residus_photometry, errflux_photometry

    def astrometric_model_residuals(self, pyLIMA_parameters,
                                    rescaling_astrometry_parameters=None):
        """
        Given a set of parameters, estimate the astrometric residuals

        Parameters
        ----------
        pyLIMA_parameters : dict, a pyLIMA_parameters object
        rescaling_astrometry_parameters : bool, if the astrometry is rescaled

        Returns
        -------
        residus_astrometry : array, an array containing the photometry residuals ,
        i.e. [res_ra_i,res_dec_i] of telescope i
        err_astrometry : array, an array containing the corresponding errors
        """
        if type(pyLIMA_parameters) is type:  # it is a pyLIMA_parameters object

            pass

        else:

            parameters = np.array(pyLIMA_parameters)

            model_parameters = parameters[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        residus_ra, residus_dec, err_ra, err_dec = \
            objective_functions.all_telescope_astrometric_residuals(
                self.model, pyLIMA_parameters,
                norm=False,
                rescaling_astrometry_parameters=rescaling_astrometry_parameters)

        return [residus_ra, residus_dec], [err_ra, err_dec]

    def model_chi2(self, parameters):
        """
        Given a set of parameters, estimate the chi^2, the sum of normalised residuals

        Parameters
        ----------
        parameters : , a pyLIMA_parameters object or an array of parameters

        Returns
        -------
        chi2 : float, the chi-square
        pyLIMA_parameters : dict, an updated pyLIMA_parameters object
        """
        if type(parameters) is type:  # it is a pyLIMA_parameters object

            pyLIMA_parameters = parameters

        else:

            params = np.array(parameters)

            model_parameters = params[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.rescale_photometry:

            rescaling_photometry_parameters = 10 ** (
                parameters[self.rescale_photometry_parameters_index])

        else:

            rescaling_photometry_parameters = None

        if self.rescale_astrometry:

            rescaling_astrometry_parameters = 10 ** (
                parameters[self.rescale_astrometry_parameters_index])

        else:

            rescaling_astrometry_parameters = None

        residus, err = self.model_residuals(pyLIMA_parameters,
                                            rescaling_photometry_parameters=rescaling_photometry_parameters,
                                            rescaling_astrometry_parameters=rescaling_astrometry_parameters)
        residuals = []
        errors = []

        for data_type in ['photometry', 'astrometry']:

            try:

                residuals.append(np.concatenate(residus[data_type]) ** 2)
                errors.append(np.concatenate(err[data_type]) ** 2)

            except ValueError:

                pass

        residuals = np.concatenate(residuals)
        errors = np.concatenate(errors)

        chi2 = np.sum(residuals / errors)

        return chi2, pyLIMA_parameters

    def model_likelihood(self, parameters):
        """
        Given a set of parameters, estimate the ln-likelihood, including priors if any

        Parameters
        ----------
        parameters : , a pyLIMA_parameters object or an array of parameters

        Returns
        -------
        ln_likelihood : float, the ln-likelihood
        pyLIMA_parameters : dict, an updated pyLIMA_parameters object
        """
        if type(parameters) is type:  # it is a pyLIMA_parameters object

            pyLIMA_parameters = parameters

        else:

            params = np.array(parameters)

            model_parameters = params[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.rescale_photometry:

            rescaling_photometry_parameters = 10 ** (
                parameters[self.rescale_photometry_parameters_index])

        else:

            rescaling_photometry_parameters = None

        if self.rescale_astrometry:

            rescaling_astrometry_parameters = 10 ** (
                parameters[self.rescale_astrometry_parameters_index])

        else:

            rescaling_astrometry_parameters = None

        residus, err = self.model_residuals(pyLIMA_parameters,
                                            rescaling_photometry_parameters=rescaling_photometry_parameters,
                                            rescaling_astrometry_parameters=rescaling_astrometry_parameters)

        residuals = []
        errors = []

        for data_type in ['photometry', 'astrometry']:

            try:

                residuals.append(np.concatenate(residus[data_type]) ** 2)
                errors.append(np.concatenate(err[data_type]) ** 2)

            except ValueError:

                pass

        residuals = np.concatenate(residuals)
        errors = np.concatenate(errors)

        ln_likelihood = np.sum(residuals / errors + np.log(errors) + np.log(2 * np.pi))

        prior = self.get_priors_probability(pyLIMA_parameters)

        ln_likelihood += prior

        ln_likelihood *= 0.5

        return ln_likelihood, pyLIMA_parameters

    def model_soft_l1(self, parameters):
        """
        Given a set of parameters, estimate the soft_l1 metric:
        soft_1 = 2 * np.sum(((1 + res**2 / errors**2) ** 0.5 - 1))

        Parameters
        ----------
        parameters : , a pyLIMA_parameters object or an array of parameters

        Returns
        -------
        soft_l1 : float, the soft_l1 metric
        pyLIMA_parameters : dict, an updated pyLIMA_parameters object
        """
        if type(parameters) is type:  # it is a pyLIMA_parameters object

            pyLIMA_parameters = parameters

        else:

            params = np.array(parameters)

            model_parameters = params[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        if self.rescale_photometry:

            rescaling_photometry_parameters = 10 ** (
                parameters[self.rescale_photometry_parameters_index])

        else:

            rescaling_photometry_parameters = None

        if self.rescale_astrometry:

            rescaling_astrometry_parameters = 10 ** (
                parameters[self.rescale_astrometry_parameters_index])

        else:

            rescaling_astrometry_parameters = None

        residus, err = self.model_residuals(pyLIMA_parameters,
                                            rescaling_photometry_parameters=rescaling_photometry_parameters,
                                            rescaling_astrometry_parameters=rescaling_astrometry_parameters)
        residuals = []
        errors = []

        for data_type in ['photometry', 'astrometry']:

            try:

                residuals.append(np.concatenate(residus[data_type]) ** 2)
                errors.append(np.concatenate(err[data_type]) ** 2)

            except ValueError:

                pass

        residuals = np.concatenate(residuals)
        errors = np.concatenate(errors)

        soft_l1 = 2 * np.sum(((1 + residuals / errors) ** 0.5 - 1))

        return soft_l1, pyLIMA_parameters

    def chi2_photometry(self, parameters):
        """
        Given a set of parameters, estimate the photometric chi^2, the sum of
        normalised residuals

        Parameters
        ----------
        parameters : , a pyLIMA_parameters object or an array of parameters

        Returns
        -------
        chi2 : float, the chi-square
        """
        residus, errors = self.photometric_model_residuals(parameters)
        residuals = np.concatenate(residus) ** 2
        errors = np.concatenate(errors) ** 2

        chi2 = np.sum(residuals / errors)

        return chi2

    def likelihood_photometry(self, parameters):
        """
        Given a set of parameters, estimate the photometric ln-likelihood

        Parameters
        ----------
        parameters : , a pyLIMA_parameters object or an array of parameters

        Returns
        -------
        ln_likeihood: float, the ln_likelihood
        """
        residus, errors = self.photometric_model_residuals(parameters)

        residuals = np.concatenate(residus) ** 2
        errors = np.concatenate(errors) ** 2

        ln_likelihood = 0.5 * np.sum(
            residuals / errors + np.log(errors) + np.log(2 * np.pi))

        return ln_likelihood

    def chi2_astrometry(self, parameters):
        """
        Given a set of parameters, estimate the astrometric chi^2, the sum of
        normalised residuals

        Parameters
        ----------
        parameters : , a pyLIMA_parameters object or an array of parameters

        Returns
        -------
        chi2 : float, the chi-square
        """
        residus, errors = self.astrometric_model_residuals(parameters)

        residus = [np.concatenate((i[0], i[1])) for i in residus]
        errors = [np.concatenate((i[0], i[1])) for i in errors]

        residuals = np.concatenate(residus) ** 2
        errors = np.concatenate(errors) ** 2

        chi2 = np.sum(residuals / errors)

        return chi2

    def likelihood_astrometry(self, parameters):
        """
        Given a set of parameters, estimate the astrometric ln-likelihood

        Parameters
        ----------
        parameters : , a pyLIMA_parameters object or an array of parameters

        Returns
        -------
        ln_likeihood: float, the ln_likelihood
        """
        residus, errors = self.astrometric_model_residuals(parameters)

        residus = [np.concatenate((i[0], i[1])) for i in residus]
        errors = [np.concatenate((i[0], i[1])) for i in errors]

        residuals = np.concatenate(residus) ** 2
        errors = np.concatenate(errors) ** 2

        ln_likelihood = 0.5 * np.sum(
            residuals / errors + np.log(errors) + np.log(2 * np.pi))

        return ln_likelihood

    def residuals_Jacobian(self, fit_process_parameters):
        """
        Given a set of parameters, estimate the Jacobian of residuals (no astrometry
        yet)

        Parameters
        ----------
        fit_process_parameters : , a pyLIMA_parameters object or an array of parameters

        Returns
        -------
        photometric_jacobian : array, an array containing the derivative of the
        residuals
        """
        photometric_jacobian = self.photometric_residuals_Jacobian(
            fit_process_parameters)

        # No Astrometry Yet

        return photometric_jacobian

    def photometric_residuals_Jacobian(self, fit_process_parameters):
        """
        Given a set of parameters, estimate the Jacobian of photometric residuals

        Parameters
        ----------
        fit_process_parameters : , a pyLIMA_parameters object or an array of parameters

        Returns
        -------
        jacobi : array, an array containing the derivative of the residuals
        """

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)

        count = 0
        for telescope in self.model.event.telescopes:

            if count == 0:

                _jacobi = self.model.photometric_model_Jacobian(telescope,
                                                                pyLIMA_parameters) / \
                          telescope.lightcurve_flux['err_flux'].value

            else:

                _jacobi = np.c_[
                    _jacobi, self.model.photometric_model_Jacobian(telescope,
                                                                   pyLIMA_parameters) /
                             telescope.lightcurve_flux['err_flux'].value]

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

    def check_telescopes_fluxes_limits(self, telescopes_fluxes):
        """
        Check,or set, the telescopes fluxes to the limits

        Parameters
        ----------
        telescopes_fluxes : dict, a dictionnary containing the telescopes fluxes values

        Returns
        -------
        new_fluxes : list, a list of the updated fluxes
        """
        new_fluxes = []

        for ind, key in enumerate(telescopes_fluxes._fields):

            flux = getattr(telescopes_fluxes, key)

            # Prior here
            if (flux <= self.fit_parameters[key][1][0]) | (
                    flux > self.fit_parameters[key][1][1]):

                if 'fsource' in key:

                    flux = np.max(self.fit_parameters[key][1])

                else:

                    flux = 0

            new_fluxes.append(flux)

        return new_fluxes

    def fit_outputs(self, bokeh_plot=None):
        """
        Produce the standard plots output

        Parameters
        ----------
        bokeh_plot : bool, to obtain a bokeh plot or not

        Returns
        -------
        matplotlib_lightcurves : matplotlib.fit, a matplotlib figure with the
        lightcurves
        matplotlib_geometry : matplotlib.fig, a matplotlib figure with the
        geometry
        matplotlib_astrometry : matotlib.fig, a matplotlib figure with the
        astrometry
        matplotlib_distribution : matplotlib.fig, a figure containing the parameters
        matplotlib_table : matplotlig.fig, a figure containing parameters table (not
        implemented yet)
        bokeh_figure : bokehh.fig, a bokeh.figure containing all the above
        """
        matplotlib_lightcurves = None
        matplotlib_geometry = None
        matplotlib_astrometry = None
        matplotlib_distribution = None
        bokeh_figure = None
        bokeh_lightcurves = None
        bokeh_astrometry = None

        if self.model.photometry:
            matplotlib_lightcurves, bokeh_lightcurves = pyLIMA_plots.plot_lightcurves(
                self.model, self.fit_results['best_model'], bokeh_plot=bokeh_plot)
            matplotlib_geometry, bokeh_geometry = pyLIMA_plots.plot_geometry(self.model,
                                                                             self.fit_results[
                                                                                 'best_model'],
                                                                             bokeh_plot=bokeh_plot)

        if self.model.astrometry:
            matplotlib_astrometry, bokeh_astrometry = pyLIMA_plots.plot_astrometry(
                self.model, self.fit_results['best_model'], bokeh_plot=bokeh_plot)

        parameters = [key for ind, key in enumerate(self.model.model_dictionnary.keys())
                      if ('fsource' not in key) and ('fblend' not in key) and (
                              'gblend' not in key)]

        samples = self.samples_to_plot()

        samples_to_plot = samples[:, :len(parameters)]

        matplotlib_distribution, bokeh_distribution = pyLIMA_plots.plot_distribution(
            samples_to_plot, parameters_names=parameters, bokeh_plot=bokeh_plot)
        # matplotlib_table = pyLIMA_plots.plot_parameters_table(samples,
        # parameters_names=[key for key in self.fit_parameters.keys()])
        matplotlib_table = None
        try:

            bokeh_figure = gridplot(
                [[bokeh_lightcurves, bokeh_geometry], [bokeh_astrometry, None]],
                toolbar_location='above')

        except ValueError:

            pass

        if bokeh_figure is not None:
            bokeh_plot_name = self.model.event.name.replace('-', '_').replace(' ', '_')

            output_file(filename='./' + bokeh_plot_name + '.html',
                        title=bokeh_plot_name)
            save(bokeh_figure)

        return matplotlib_lightcurves, matplotlib_geometry, matplotlib_astrometry, \
            matplotlib_distribution, \
            matplotlib_table, bokeh_figure
