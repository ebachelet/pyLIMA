import sys
from collections import OrderedDict
from multiprocessing import Manager

import numpy as np
import pyLIMA.fits.objective_functions as objective_functions
from pyLIMA.priors import parameters_boundaries
from pyLIMA.priors import parameters_priors

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
    trials_parameters : list, a Manager().list() to collect all algorithm fit trials_parameters
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
        self.extra_priors = None
        self.trials_parameters = Manager().list()  # to be recognize by all process during
        # parallelization
        self.trials_objective = Manager().list()
        self.trials_priors = Manager().list()

        self.model_parameters_guess = []
        self.rescale_photometry_parameters_guess = []
        self.rescale_astrometry_parameters_guess = []
        self.telescopes_fluxes_parameters_guess = []

        self.model_parameters_index = []
        self.rescale_photometry_parameters_index = []
        self.rescale_astrometry_parameters_index = []

        self.define_fit_parameters()
        self.define_priors_parameters()

        self.define_priors()


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

        model_parameters_dictionnary = self.model.model_dictionnary.copy()
        standard_parameters_boundaries = \
            self.model.standard_parameters_boundaries.copy()

        fit_parameters_dictionnary_keys = []
        fit_parameters_indexes = []
        fit_parameters_boundaries = []

        for ind, key in enumerate(model_parameters_dictionnary.keys()):

            if (('fsource' in key) | ('fblend' in key) | ('gblend' in key) | ('ftotal' in key)) & (
                    include_telescopes_fluxes is False):
                pass

            else:

                thekey = key
                theboundaries = standard_parameters_boundaries[ind]

                if self.model.fancy_parameters is not None:

                    if key in self.model.fancy_parameters.fancy_parameters.values():

                        theboundaries = self.model.fancy_parameters.fancy_boundaries[thekey]

                fit_parameters_dictionnary_keys.append(thekey)
                fit_parameters_indexes.append(ind)
                fit_parameters_boundaries.append(theboundaries)

        if self.rescale_photometry:

            for telescope in self.model.event.telescopes:

                if telescope.lightcurve is not None:
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
                                                    {thekey: 'dummy'})[0]

                    fit_parameters_dictionnary_keys.append(thekey)
                    fit_parameters_indexes.append(theind)
                    fit_parameters_boundaries.append(theboundaries)

                    thekey = 'logk_astrometry_dec' + telescope.name
                    theind = len(fit_parameters_dictionnary_keys)
                    theboundaries = \
                        parameters_boundaries.parameters_boundaries(self.model.event,
                                                    {thekey: 'dummy'})[0]

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

            if telescope.lightcurve is not None:
                mins_time.append(np.min(telescope.lightcurve['time'].value))
                maxs_time.append(np.max(telescope.lightcurve['time'].value))

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

    def define_priors(self):

        self.priors = parameters_priors.default_parameters_priors(
            self.priors_parameters)

    def fit_parameters_inside_limits(self, fit_process_parameters):

        for ind,key in enumerate(self.fit_parameters.keys()):

            if (fit_process_parameters[ind]<self.fit_parameters[key][1][0]) | (
                    fit_process_parameters[ind]>self.fit_parameters[key][1][1]):

                return np.inf

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
            likelihood, priors, pyLIMA_parameters = self.model_likelihood(
                fit_process_parameters)
            objective = likelihood

        if self.loss_function == 'chi2':
            chi2, pyLIMA_parameters = self.model_chi2(fit_process_parameters)
            objective = chi2
            priors = 0.0

        if self.loss_function == 'soft_l1':
            soft_l1, pyLIMA_parameters = self.model_soft_l1(fit_process_parameters)
            objective = soft_l1
            priors = 0.0

        if self.telescopes_fluxes_method != 'fit':

            fluxes = []

            for tel in self.model.event.telescopes:

                if tel.lightcurve is not None:

                        fluxes.append(pyLIMA_parameters['fsource_' + tel.name])

                        if self.model.blend_flux_parameter == 'gblend':
                            fluxes.append(pyLIMA_parameters['gblend_' + tel.name])

                        if self.model.blend_flux_parameter == 'fblend':
                            fluxes.append(pyLIMA_parameters['fblend_' + tel.name])

                        if self.model.blend_flux_parameter == 'ftotal':
                            fluxes.append(pyLIMA_parameters['ftotal_' + tel.name])

                        if self.model.blend_flux_parameter == 'noblend':
                            pass

            self.trials_parameters.append(fit_process_parameters.tolist() + fluxes+[
                objective,priors])

        else:

            self.trials_parameters.append(fit_process_parameters.tolist()+[
                objective,priors])

        self.trials_objective.append(objective)
        self.trials_priors.append(priors)

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

                    probability = prior_pdf.pdf(pyLIMA_parameters[prior_key])

                    if probability > 0:

                        ln_likelihood += np.log(probability)

                    else:

                        #ln_likelihood = -np.inf
                        ln_likelihood += -10**10
                        #return ln_likelihood

        if self.extra_priors is not None:

            for extra_prior in self.extra_priors:

                probability = extra_prior.pdf(pyLIMA_parameters)

                if probability > 0:

                    ln_likelihood += np.log(probability)

                else:

                    #ln_likelihood = -np.inf
                    ln_likelihood += -10 ** 10

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

                elif self.model.model_type() == 'FSPL':
                    guess_paczynski_parameters, f_source = \
                        pyLIMA.priors.guess.initial_guess_FSPL(
                            self.model.event)

                elif self.model.model_type() == 'FSPLee':
                    guess_paczynski_parameters, f_source = \
                        pyLIMA.priors.guess.initial_guess_FSPL(
                            self.model.event)

                elif self.model.model_type() == 'FSPLarge':
                    guess_paczynski_parameters, f_source = \
                        pyLIMA.priors.guess.initial_guess_FSPLarge(
                            self.model.event)

                elif self.model.model_type() == 'DSPL':
                    guess_paczynski_parameters, f_source = \
                        pyLIMA.priors.guess.initial_guess_DSPL(
                            self.model.event)
                
                else:
                    raise NotImplementedError(
                        "Guessing initial parameters for "
                        f"{self.model.model_type} is not yet supported. "
                        "This model requires manually setting initial parameters."
                    )

                if 'theta_E' in self.fit_parameters.keys():
                    guess_paczynski_parameters = guess_paczynski_parameters + [1.0]

                if 'piEN' in self.fit_parameters.keys():
                    guess_paczynski_parameters = guess_paczynski_parameters + [0.0, 0.0]

                if 'xiEN' in self.fit_parameters.keys():
                    guess_paczynski_parameters = guess_paczynski_parameters + [0, 0,
                                                                               1,0,0]

                if 'dsdt' in self.fit_parameters.keys():
                    guess_paczynski_parameters = guess_paczynski_parameters + [0, 0]

                if 'spot_size' in self.fit_parameters.keys():
                    guess_paczynski_parameters = guess_paczynski_parameters + [0]

                pyLIMA_parameters = self.model.compute_pyLIMA_parameters(
                    guess_paczynski_parameters, fancy_parameters=False)

                if self.model.fancy_parameters is not None:

                    self.model.pyLIMA_to_fancy_parameters(guess_paczynski_parameters)

                if self.model.fancy_parameters is not None:

                    pyLIMA_parameters = OrderedDict()

                    for standard_key, index in (
                            self.model.pyLIMA_standards_dictionnary.items()):
                        try:
                            pyLIMA_parameters[standard_key] = guess_paczynski_parameters[
                            index]

                        except IndexError:

                            pyLIMA_parameters[standard_key] = None

                    self.model.pyLIMA_to_fancy_parameters(pyLIMA_parameters)

                    new_guess_paczynski_parameters = []

                    for fancy_key, index in self.model.model_dictionnary.items():

                        try:
                            new_guess_paczynski_parameters.append(pyLIMA_parameters[
                                                                  fancy_key])
                        except IndexError:

                            pass
                else:

                    new_guess_paczynski_parameters = guess_paczynski_parameters


                pyLIMA_parameters = self.model.compute_pyLIMA_parameters(
                    new_guess_paczynski_parameters, fancy_parameters=False)

                final_guess_paczynski_parameters = []

                for ind, param in enumerate(list(self.fit_parameters.keys())[:len(
                        guess_paczynski_parameters)]):

                    param_value = pyLIMA_parameters[param]

                    if (param_value < self.fit_parameters[param][
                        1][0]) | (param_value >
                                  self.fit_parameters[param][1][1]):

                        new_value = (np.sign(self.fit_parameters[
                            param][1][1])-0.5)*np.abs(self.fit_parameters[
                            param][1][1])

                        final_guess_paczynski_parameters.append(new_value)

                    else:

                        final_guess_paczynski_parameters.append(param_value)

                self.model_parameters_guess = final_guess_paczynski_parameters

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

                    if telescope.lightcurve is not None:
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
        #fit_parameters_guess = self.model_parameters_guess.copy()

        #self.model_parameters_guess = fit_parameters_guess
        self.telescopes_fluxes_guess()
        self.rescale_photometry_guess()
        self.rescale_astrometry_guess()

        fit_parameters_guess = self.model_parameters_guess + \
                               self.telescopes_fluxes_parameters_guess + \
                               self.rescale_photometry_parameters_guess + \
                               self.rescale_astrometry_parameters_guess
        fit_parameters_guess = [float(i) for i in fit_parameters_guess]

        #if self.priors is not None:

        #    for ind, prior_key in enumerate(self.fit_parameters.keys()):

        #        prior_pdf = self.priors[prior_key]

        #        probability = prior_pdf.pdf(fit_parameters_guess[ind])

        #        if probability < 10 ** -10:
        #            samples = prior_pdf.rvs(1000)

        #           fit_parameters_guess[ind] = np.median(samples)
        for ind, param in enumerate(self.fit_parameters.keys()):

            if (fit_parameters_guess[ind] < self.fit_parameters[param][1][0]):
                #breakpoint()
                fit_parameters_guess = None
                print('WARNING: ' + param + ' is out of the parameters boundaries, '
                                             'abord fitting ')

                print(sys._getframe().f_code.co_name,
                      ' : Initial parameters guess FAIL')
                return fit_parameters_guess

            if (fit_parameters_guess[ind] > self.fit_parameters[param][1][1]):
                #breakpoint()
                fit_parameters_guess = None
                print('WARNING: ' + param + ' is out of the parameters boundaries, '
                                            'abord fitting ')
                print(sys._getframe().f_code.co_name,
                     ' : Initial parameters guess FAIL')
                return fit_parameters_guess

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

        # it is a pyLIMA_parameters object or not
        if (isinstance(pyLIMA_parameters,list) | isinstance(pyLIMA_parameters,np.ndarray)):

            parameters = np.array(pyLIMA_parameters)

            model_parameters = parameters[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        else:

            pass

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


            #breakpoint()
            residus['astrometry'] = [np.concatenate([residuals_astrometry[0][i], residuals_astrometry[
                1][i]]) for i in range(len(residuals_astrometry[0]))]

            errors['astrometry'] = [
                np.concatenate([errors_astrometry[0][i], errors_astrometry[
                    1][i]]) for i in range(len(errors_astrometry[0]))]

            #residus['astrometry'] = [np.concatenate((i[0], i[1])) for i in
            #                         residuals_astrometry]
            #errors['astrometry'] = [np.concatenate((i[0], i[1])) for i in
            #                        errors_astrometry]

            #residus['astrometry'] = np.ravel(residuals_astrometry)
            #errors['astrometry'] = np.ravel(errors_astrometry)

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

        # it is a pyLIMA_parameters object or not

        if (isinstance(pyLIMA_parameters, list) | isinstance(pyLIMA_parameters, np.ndarray)):

            parameters = np.array(pyLIMA_parameters)

            model_parameters = parameters[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        else:

            pass

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
        # it is a pyLIMA_parameters object or not
        if (isinstance(pyLIMA_parameters, list) | isinstance(pyLIMA_parameters, np.ndarray)):

            parameters = np.array(pyLIMA_parameters)

            model_parameters = parameters[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        else:

            pass

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
        # it is a pyLIMA_parameters object or not
        if (isinstance(parameters, list) | isinstance(parameters, np.ndarray)):

            parameters = np.array(parameters)

            model_parameters = parameters[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        else:

            pyLIMA_parameters = parameters

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
        # it is a pyLIMA_parameters object or not
        if (isinstance(parameters, list) | isinstance(parameters, np.ndarray)):

            parameters = np.array(parameters)

            model_parameters = parameters[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        else:

            pyLIMA_parameters = parameters

        if self.rescale_photometry:

            rescaling_photometry_parameters = 10 ** (
                parameters[self.rescale_photometry_parameters_index])
            #Not very nice...
            for ind,key in enumerate(self.fit_parameters.keys()):

                if key not in pyLIMA_parameters.keys():

                    pyLIMA_parameters[key] = parameters[ind]

        else:

            rescaling_photometry_parameters = None

        if self.rescale_astrometry:

            rescaling_astrometry_parameters = 10 ** (
                parameters[self.rescale_astrometry_parameters_index])

            for ind, key in enumerate(self.fit_parameters.keys()):

                if key not in pyLIMA_parameters.keys():
                    pyLIMA_parameters[key] = parameters[ind]
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

        ln_likelihood = 0.5*np.sum(residuals / errors + np.log(errors) +
                                np.log(2 * np.pi))

        prior = self.get_priors_probability(pyLIMA_parameters)

        ln_likelihood += -prior  # Default is -ln_likelihood

        return ln_likelihood, -prior, pyLIMA_parameters

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
        # it is a pyLIMA_parameters object or not

        if (isinstance(parameters, list) | isinstance(parameters, np.ndarray)):

            parameters = np.array(parameters)

            model_parameters = parameters[self.model_parameters_index]

            pyLIMA_parameters = self.model.compute_pyLIMA_parameters(model_parameters)

        else:

            pyLIMA_parameters = parameters

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
                          telescope.lightcurve['err_flux'].value

            else:

                _jacobi = np.c_[
                    _jacobi, self.model.photometric_model_Jacobian(telescope,
                                                                   pyLIMA_parameters) /
                             telescope.lightcurve['err_flux'].value]

            count += 1

        # The objective function is : (data-model)/errors

        _jacobi = -_jacobi
        # Split the fs and g derivatives in several columns correpsonding to
        # each observatories
        start_index = 0

        if self.model.blend_flux_parameter != 'noblend':
            jacobi = _jacobi[:-2]
            dresdfs = _jacobi[-2]
            dresdg = _jacobi[-1]

        else:
            jacobi = _jacobi[:-1]

            dresdfs = _jacobi[-1]
            dresdg = None

        for telescope in self.model.event.telescopes:
            derivative_fs = np.zeros((len(dresdfs)))
            index = np.arange(start_index, start_index + len(telescope.lightcurve))

            if self.model.blend_flux_parameter != 'noblend':
                derivative_g = np.zeros((len(dresdg)))

                derivative_fs[index] = dresdfs[index]
                derivative_g[index] = dresdg[index]
                jacobi = np.r_[jacobi, np.array([derivative_fs, derivative_g])]
            else:
                derivative_fs[index] = dresdfs[index]
                jacobi = np.r_[jacobi, [derivative_fs]]

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

        for ind, key in enumerate(telescopes_fluxes.keys()):

            flux = telescopes_fluxes[key]
            # Prior here
            if (flux <= self.fit_parameters[key][1][0]) | (
                    flux > self.fit_parameters[key][1][1]):

                flux = np.mean( self.fit_parameters[key][1])

            new_fluxes.append(flux)

        return new_fluxes

    def fit_outputs(self, bokeh_plot=None, bokeh_plot_name=None,
                    json_name='./pyLIMA_fit.json'):
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
        #Non standard call to libraries, but otherwise slow-down codes significantly
        from bokeh.layouts import gridplot
        from bokeh.plotting import output_file, save
        from pyLIMA.outputs import pyLIMA_plots, file_outputs

        file_outputs.json_output(self,json_name=json_name)

        matplotlib_lightcurves = None
        matplotlib_geometry = None
        matplotlib_astrometry = None
        matplotlib_distribution = None
        matplotlib_table = None
        bokeh_figure = None
        bokeh_lightcurves = None
        bokeh_astrometry = None
        bokeh_parameters = None

        pyLIMA_plots.update_matplotlib_colors(self.model.event)

        if self.model.photometry:
            matplotlib_lightcurves, bokeh_lightcurves = pyLIMA_plots.plot_lightcurves(
                self.model, self.fit_results['best_model'][self.model_parameters_index],
                bokeh_plot=bokeh_plot)
            matplotlib_geometry, bokeh_geometry = pyLIMA_plots.plot_geometry(self.model,
                                                                             self.fit_results[
                                                                                 'best_model'][self.model_parameters_index],
                                                                             bokeh_plot=bokeh_plot)

        if self.model.astrometry:
            matplotlib_astrometry, bokeh_astrometry = pyLIMA_plots.plot_astrometry(
                self.model, self.fit_results['best_model'][
                    self.model_parameters_index], bokeh_plot=bokeh_plot)

        parameters = [key for ind, key in enumerate(self.model.model_dictionnary.keys())
                      if ('fsource' not in key) and ('fblend' not in key) and (
                              'gblend' not in key) and ('ftotal' not in key)]
        try:
            chi2 = self.model_chi2(self.fit_results['best_model'])[0]
        except Exception as e:
            chi2 = None
            print (f"An error occurred while trying to get chi2 for the best model: {e}")
            pass
        
        samples = self.samples_to_plot()

        samples_to_plot = samples[:, :len(parameters)]

        try:
            matplotlib_distribution, bokeh_distribution = pyLIMA_plots.plot_distribution(
            samples_to_plot, parameters_names=parameters, bokeh_plot=bokeh_plot)
        except Exception as e:
            print(e)
            pass
        matplotlib_table, bokeh_parameters = pyLIMA_plots.plot_parameters_table(samples_to_plot,
                           parameters_names=parameters,
                           chi2 = chi2,
                           bokeh_plot=bokeh_plot)
        #matplotlib_table = None

        try:
            bokeh_figure = gridplot(
                [[bokeh_lightcurves, bokeh_geometry],
                 [bokeh_distribution, bokeh_parameters]],
#                 [bokeh_astrometry, None]],
                 toolbar_location='above')
        
        except Exception as e:

            print(f"An error occurred while creating the gridplot: {e}")
            
            pass
        
        if bokeh_plot is not None:

            if bokeh_plot_name is None:

                bokeh_plot_name = self.model.event.name.replace('-', '_').replace(' ', '_')
                bokeh_plot_name =  './' + bokeh_plot_name + '.html'

            try:
                output_file(filename= bokeh_plot_name,
                            title=self.model.event.name)
                save(bokeh_figure)
            except Exception as e:
                print(f"Warning: Failed to generate the HTML file due to {e}")

        return matplotlib_lightcurves, matplotlib_geometry, matplotlib_astrometry, \
            matplotlib_distribution, \
            matplotlib_table, bokeh_figure

    def print_fit_results(self):

        import pprint

        params = self.model.compute_pyLIMA_parameters(self.fit_results['best_model'])
        params[self.loss_function] = self.fit_results[self.loss_function]
        print('best model:')
        pprint.pprint(params)
