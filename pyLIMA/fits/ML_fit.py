import sys
import numpy as np
from collections import OrderedDict, namedtuple

from pyLIMA.fits import fancy_parameters
from pyLIMA.priors import parameters_boundaries

### Standard fancy parameters functions



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

    def __init__(self, model, fancy_parameters=False, rescale_photometry=False, rescale_astrometry=False, telescopes_fluxes_method='fit'):
        """The fit class has to be intialized with an event object."""

        self.model = model
        self.fancy_parameters = fancy_parameters
        self.rescale_photometry = rescale_photometry
        self.rescale_astrometry = rescale_astrometry
        self.telescopes_fluxes_method = telescopes_fluxes_method
        self.fit_parameters = []
        self.fit_results = {}

        self.model_parameters_guess = []
        self.rescale_photometry_parameters_guess = []
        self.telescopes_fluxes_parameters_guess = []

        self.model_parameters_index = []
        self.rescale_photometry_parameters_index = []
        self.rescale_astrometry_parameters_index = []

        self.model.define_model_parameters()
        self.define_fancy_parameters()
        self.define_fit_parameters()

    def define_fancy_parameters(self):

        #self.model.model_dictionnary = {}
        #self.model.pyLIMA_standards_dictionnary = {}
        #self.model.fancy_to_pyLIMA_dictionnary = {}
        #self.model.pyLIMA_to_fancy = {}
        #self.model.fancy_to_pyLIMA = {}

        if self.fancy_parameters:

            import pickle

            try:

                fancy_parameters_dictionnary = fancy_parameters.fancy_parameters_dictionnary

            except:

                print('Loading the default fancy parameters!')
                fancy_parameters_dictionnary = fancy_parameters.standard_fancy_parameters

            for key in fancy_parameters_dictionnary.keys():

                parameter = fancy_parameters_dictionnary[key]

                if parameter in self.model.model_dictionnary.keys():

                    self.model.fancy_to_pyLIMA_dictionnary[key] = parameter

                    self.model.pyLIMA_to_fancy[key] = pickle.loads(pickle.dumps(eval('fancy_parameters.'+key)))
                    self.model.fancy_to_pyLIMA[parameter] = pickle.loads(pickle.dumps(eval('fancy_parameters.'+parameter)))

        self.model.define_model_parameters()

    def define_fit_parameters(self):

        fit_parameters_dictionnary = self.model.paczynski_model_parameters()

        fit_parameters_dictionnary_updated = self.model.astrometric_model_parameters(fit_parameters_dictionnary)

        fit_parameters_dictionnary_updated = self.model.second_order_model_parameters(
            fit_parameters_dictionnary_updated)

        if self.telescopes_fluxes_method == 'fit':

            fit_parameters_dictionnary_updated = self.model.telescopes_fluxes_model_parameters(
                fit_parameters_dictionnary_updated)

        if self.rescale_photometry:

            for telescope in self.model.event.telescopes:

                if telescope.lightcurve_flux is not None:

                    fit_parameters_dictionnary_updated['logk_photometry_' + telescope.name] = \
                        len(fit_parameters_dictionnary_updated)

        if self.rescale_astrometry:

            for telescope in self.model.event.telescopes:

                if telescope.astrometry is not None:

                    fit_parameters_dictionnary_updated['logk_astrometry_' + telescope.name] = \
                        len(fit_parameters_dictionnary_updated)

        self.fit_parameters = OrderedDict(
            sorted(fit_parameters_dictionnary_updated.items(), key=lambda x: x[1]))

        fit_parameters_boundaries = parameters_boundaries.parameters_boundaries(self.model.event, self.fit_parameters)

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

            list_of_keys = [i for i in self.fit_parameters.keys()]

            for key in self.model.fancy_to_pyLIMA_dictionnary.keys():
                new_bounds = []
                index = np.where(self.model.fancy_to_pyLIMA_dictionnary[key] == np.array(list_of_keys))[0][0]

                parameter = self.model.fancy_to_pyLIMA_dictionnary[key]
                bounds = self.fit_parameters[parameter][1]
                x = namedtuple('parameters', parameter)

                setattr(x, parameter, bounds[0])
                new_bounds.append(self.model.pyLIMA_to_fancy[key](x))
                setattr(x, parameter, bounds[1])
                new_bounds.append(self.model.pyLIMA_to_fancy[key](x))

                self.fit_parameters.pop(parameter)

                self.fit_parameters[key] = [index, new_bounds]

        self.fit_parameters = OrderedDict(
            sorted(self.fit_parameters.items(), key=lambda x: x[1]))

        self.model_parameters_index = [self.model.model_dictionnary[i] for i in self.model.model_dictionnary.keys() if
                                       i in self.fit_parameters.keys()]

        self.rescale_photometry_parameters_index = [self.fit_parameters[i][0] for i in self.fit_parameters.keys() if
                                                    'logk_photometry' in i]

        self.rescale_astrometry_parameters_index = [self.fit_parameters[i][0] for i in self.fit_parameters.keys() if
                                                    'logk_astrometry' in i]

    def fancy_parameters_to_pyLIMA_standard_parameters(self, fancy_parameters):
        """ Transform the fancy parameters to the pyLIMA standards. The output got all
        the necessary standard attributes, example to, uo, tE...


        :param object fancy_parameters: the fancy_parameters as namedtuple
        :return: the pyLIMA standards are added to the fancy parameters
        :rtype: object
        """
        # start_time = python_time.time()
        if len(self.fancy_to_pyLIMA) != 0:
            # import pdb;
            # pdb.set_trace()
            for key_parameter in self.fancy_to_pyLIMA.keys():
                setattr(fancy_parameters, key_parameter, self.fancy_to_pyLIMA[key_parameter](fancy_parameters))

        # print 'fancy to PYLIMA', python_time.time() - start_time
        return fancy_parameters

    def pyLIMA_standard_parameters_to_fancy_parameters(self, pyLIMA_parameters):
        """ Transform the  the pyLIMA standards parameters to the fancy parameters. The output got all
            the necessary fancy attributes.


        :param object pyLIMA_parameters: the  standard pyLIMA parameters as namedtuple
        :return: the fancy parameters are added to the fancy parameters
        :rtype: object
        """
        if len(self.pyLIMA_to_fancy) != 0:

            for key_parameter in self.pyLIMA_to_fancy.keys():
                setattr(pyLIMA_parameters, key_parameter, self.pyLIMA_to_fancy[key_parameter](pyLIMA_parameters))

        return pyLIMA_parameters

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
                if (f_source < 0) | (f_source+f_blending < 0) :

                    telescopes_fluxes.append(np.min(flux))
                    telescopes_fluxes.append(0.0)
                else:
                    telescopes_fluxes.append(f_source)
                    telescopes_fluxes.append(f_blending)
        return telescopes_fluxes


    def objective_function_Jacobian(self, fit_process_parameters):

        pyLIMA_parameters = self.model.compute_pyLIMA_parameters(fit_process_parameters)

        count = 0

        for telescope in self.event.telescopes:

            jack = self.model.magnification_Jacobian(telescope, pyLIMA_parameters)/telescope.lightcurve_flux['err_flux']

            if self.model.blend_flux_parameter == 'gblend':

                f_source = 2 * getattr(pyLIMA_parameters, 'fsource_' + telescope.name) / 2
                g_blending = 2 * getattr(pyLIMA_parameters, 'gblend_' + telescope.name) / 2

                jack[:,3] += g_blending/telescope.lightcurve_flux['err_flux']
                jack[:,4] *= f_source

            if count == 0:

                _jacobi = jack

            else:

                _jacobi = np.c_[_jacobi, jack]

            count += 1

        # The objective function is : (data-model)/errors
        _jacobi = -_jacobi
        jacobi = _jacobi[:-2]
        # Split the fs and g derivatives in several columns correpsonding to
        # each observatories
        start_index = 0
        dresdfs = _jacobi[-2]
        dresdg = _jacobi[-1]

        for telescope in self.event.telescopes:
            derivative_fs = np.zeros((len(dresdfs)))
            derivative_g = np.zeros((len(dresdg)))
            index = np.arange(start_index, start_index + len(telescope.lightcurve_flux['time'].value))
            derivative_fs[index] = dresdfs[index]
            derivative_g[index] = dresdg[index]
            jacobi = np.r_[jacobi, np.array([derivative_fs, derivative_g])]

            start_index = index[-1] + 1

        return jacobi.T