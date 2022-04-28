import sys
import numpy as np
from collections import OrderedDict


from pyLIMA.priors import parameters_boundaries

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

    def __init__(self, model, rescale_photometry=False):
        """The fit class has to be intialized with an event object."""

        self.model = model
        self.rescale_photometry = rescale_photometry
        self.fit_parameters = []

        self.model_parameters_guess = []
        self.rescale_photometry_parameters_guess = []
        self.telescopes_fluxes_parameters_guess = []

        self.model_parameters_index = []
        self.rescale_photometry_parameters_index = []

        self.model.define_model_parameters()
        self.define_fit_parameters()

    def define_fit_parameters(self):

        fit_parameters_dictionnary = self.model.paczynski_model_parameters()

        fit_parameters_dictionnary_updated = self.model.astrometric_model_parameters(fit_parameters_dictionnary)

        fit_parameters_dictionnary_updated = self.model.second_order_model_parameters(fit_parameters_dictionnary_updated)

        fit_parameters_dictionnary_updated = self.model.telescopes_fluxes_model_parameters(fit_parameters_dictionnary_updated)


        if self.rescale_photometry:

            for telescope in self.model.event.telescopes:

                if telescope.lightcurve_flux is not None:

                    fit_parameters_dictionnary_updated['logk_photometry_'+telescope.name] = \
                        len(fit_parameters_dictionnary_updated)

        self.fit_parameters = OrderedDict(
        sorted(fit_parameters_dictionnary_updated.items(), key=lambda x: x[1]))

        self.model_parameters_index = [self.model.model_dictionnary[i] for i in self.model.model_dictionnary.keys() if
                                 i in self.fit_parameters.keys()]
        self.rescale_photometry_parameters_index = [self.fit_parameters[i] for i in self.fit_parameters.keys() if 'logk_photometry' in i]

        fit_parameters_boundaries = parameters_boundaries.parameters_boundaries(self.model.event, self.fit_parameters)

        fit_parameters_boundaries = parameters_boundaries.parameters_boundaries(self.model.event, self.fit_parameters)

        for ind, key in enumerate(self.fit_parameters.keys()):

            self.fit_parameters[key] = [ind, fit_parameters_boundaries[ind]]

    def objective_function(self):

        likelihood_photometry = self.likelihood_photometry()
        likelihood_astrometry = self.likelihood_astrometry()

        return likelihood_photometry+likelihood_astrometry

    def covariance_matrix(self, extra_parameters=None):

        photometric_errors = np.hstack([i.lightcurve_flux['err_flux'].value for i in self.model.event.telescopes])

        errors = photometric_errors
        basic_covariance_matrix = np.zeros((len(errors),
                                            len(errors)))

        np.fill_diagonal(basic_covariance_matrix, errors**2)

        covariance = basic_covariance_matrix

        return covariance

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
                                   'Please provide some in model.parameters_guess or run a DE fit.')
        else:

            self.model_parameters_guess = [float(i) for i in self.model_parameters_guess ]

            pass

    def telescopes_fluxes_guess(self):

        if self.telescopes_fluxes_parameters_guess == []:

            telescopes_fluxes = self.find_fluxes(self.model_parameters_guess)

            self.telescopes_fluxes_parameters_guess = telescopes_fluxes

        self.telescopes_fluxes_parameters_guess = [float(i) for i in self.telescopes_fluxes_parameters_guess]

    def rescale_photometry_guess(self):

        if self.rescale_photometry:

            if self.rescale_photometry_parameters_guess == []:

                rescale_photometry_guess = []

                for telescope in self.model.event.telescopes:

                    if telescope.lightcurve_flux is not None:

                        rescale_photometry_guess.append(0.1)

                self.rescale_photometry_parameters_guess = rescale_photometry_guess

            self.rescale_photometry_parameters_guess = [float(i) for i in self.rescale_photometry_parameters_guess]

        else:

            self.rescale_photometry_parameters_guess = []


    def initial_guess(self):

        self.model_guess()
        self.telescopes_fluxes_guess()
        self.rescale_photometry_guess()

        fit_parameters_guess = self.model_parameters_guess+self.telescopes_fluxes_parameters_guess+self.rescale_photometry_parameters_guess
        fit_parameters_guess = [float(i) for i in fit_parameters_guess]

        print(sys._getframe().f_code.co_name, ' : Initial parameters guess SUCCESS')
        print('Using guess: ',fit_parameters_guess)
        return fit_parameters_guess

    def likelihood_astrometry(self):

        return 0
    def likelihood_photometry(self):

        import pyLIMA.fits.residuals
        return 0

    def produce_outputs(self):
        """ Produce the standard outputs for a fit.
        More details in microloutputs module.
        """

        outputs = microloutputs.fit_outputs(self)

        self.outputs = outputs

    def produce_fit_statistics(self):
        """ Produce the standard outputs for a fit.
        More details in microloutputs module.
        """

        stats_outputs = microloutputs.statistical_outputs(self)

        self.stats_outputs = stats_outputs

    def produce_pdf(self, output_directory='./'):
        """ ON CONSTRUCTION
        """
        microloutputs.pdf_output(self, output_directory)

    def produce_latex_table_results(self, output_directory='./'):
        """ ON CONSTRUCTION
        """
        microloutputs.latex_output(self, output_directory)


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
                f_blending = ml_model['f_blending']
                # Prior here
                if f_source < 0:

                    telescopes_fluxes.append(np.min(flux))
                    telescopes_fluxes.append(0.0)
                else:
                    telescopes_fluxes.append(f_source)
                    telescopes_fluxes.append(f_blending)
        return telescopes_fluxes

