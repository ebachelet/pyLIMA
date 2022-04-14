import sys

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

    def __init__(self, event, model):
        """The fit class has to be intialized with an event object."""

        self.event = event
        self.model = model


    def objective_function(self):

        likelihood_photometry = self.likelihood_photometry()
        likelihood_astrometry = self.likelihood_astrometry()

        return likelihood_astrometry+likelihood_astrometry

    def initial_guess(self):
        """Try to estimate the microlensing parameters. Only use for PSPL and FSPL
           models. More details on microlguess module.

           :return guess_parameters: a list containing parameters guess related to the model.
           :rtype: list
        """
        import pyLIMA.priors.guess
        if len(self.model.parameters_guess) == 0:

            # Estimate  the Paczynski parameters

            if self.model.model_type == 'PSPL':
                guess_paczynski_parameters, f_source = pyLIMA.priors.guess.initial_guess_PSPL(self.event)

            if self.model.model_type == 'FSPL':
                guess_paczynski_parameters, f_source = pyLIMA.priors.guess.initial_guess_FSPL(self.event)

            if self.model.model_type == 'FSPLee':
                guess_paczynski_parameters, f_source = pyLIMA.priors.guess.initial_guess_FSPL(self.event)

            if self.model.model_type == 'FSPLarge':
                guess_paczynski_parameters, f_source = pyLIMA.priors.guess.initial_guess_FSPL(self.event)

            if self.model.model_type == 'DSPL':
                guess_paczynski_parameters, f_source = pyLIMA.priors.guess.initial_guess_DSPL(self.event)

            if 'piEN' in self.model.model_dictionnary.keys():
                guess_paczynski_parameters = guess_paczynski_parameters + [0.0, 0.0]

            if 'XiEN' in self.model.model_dictionnary.keys():
                guess_paczynski_parameters = guess_paczynski_parameters + [0, 0]

            if 'dsdt' in self.model.model_dictionnary.keys():
                guess_paczynski_parameters = guess_paczynski_parameters + [0, 0]

            if 'spot_size' in self.model.model_dictionnary.keys():
                guess_paczynski_parameters = guess_paczynski_parameters + [0]

            # Estimate  the telescopes fluxes (flux_source + g_blending) parameters

            telescopes_fluxes = self.find_fluxes(guess_paczynski_parameters, self.model)

            # The survey fluxes are already known from microlguess
            telescopes_fluxes[0] = f_source
            telescopes_fluxes[1] = 0.0
        else:

            guess_paczynski_parameters = list(self.model.parameters_guess)

            telescopes_fluxes = self.find_fluxes(guess_paczynski_parameters)

        if len(guess_paczynski_parameters) != len(list(self.model.model_dictionnary.keys())):
            guess_paczynski_parameters += telescopes_fluxes

        print(sys._getframe().f_code.co_name, ' : Initial parameters guess SUCCESS')
        return guess_paczynski_parameters

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

        for telescope in self.event.telescopes:

            flux = telescope.lightcurve_flux['flux'].value

            ml_model, f_source, f_blending = self.model.compute_the_microlensing_model(telescope, pyLIMA_parameters)

            # Prior here
            if f_source < 0:

                telescopes_fluxes.append(np.min(flux))
                telescopes_fluxes.append(0.0)
            else:
                telescopes_fluxes.append(f_source)
                telescopes_fluxes.append(f_blending)
        return telescopes_fluxes

