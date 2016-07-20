# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:38:14 2015

@author: ebachelet
"""
from __future__ import division
from datetime import datetime
from collections import OrderedDict
import collections
import copy
import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
from astropy.time import Time
from scipy.stats.distributions import t as student

import microltoolbox

plot_lightcurve_windows = 0.2
plot_residuals_windows = 0.2
max_plot_ticks = 3

def LM_outputs(fit):
    """Standard 'LM' and 'DE' outputs.

    :param object fit: a fit object. See the microlfits for more details.

    :return: a namedtuple containing the following attributes :

              fit_parameters : an namedtuple object containing all the fitted parameters

              fit_errors : an namedtuple object containing all the fitted parameters errors

              fit_correlation_matrix : a numpy array representing the fitted parameters
              correlation matrix

              figure_lightcurve : a two matplotlib figure showing the data and model and the
              correspoding residuals

    :rtype: object
    """

    results = LM_parameters_result(fit)
    covariance_matrix = fit.fit_covariance
    errors = LM_fit_errors(fit)
    correlation_matrix = cov2corr(covariance_matrix)
    figure_lightcurve = LM_plot_lightcurves(fit)
    # figure_parameters = LM_plot_parameters(fit)
    key_outputs = ['fit_parameters', 'fit_errors', 'fit_correlation_matrix', 'figure_lightcurve']
    outputs = collections.namedtuple('Fit_outputs', key_outputs)

    values_outputs = [results, errors, correlation_matrix, figure_lightcurve]

    count = 0
    for key in key_outputs:
        setattr(outputs, key, values_outputs[count])
        count += 1

    return outputs


def MCMC_outputs(fit):
    """Standard 'LM' and 'DE' outputs.

    :param object fit: a fit object. See the microlfits for more details.

    :return: a namedtuple containing the following attributes :

             MCMC_chains : a numpy array containing all the parameters chains + the corresponding
             objective function.

             MCMC_correlations : a numpy array representing the fitted parameters correlation
             matrix from the MCMC chains

             figure_lightcurve : a two matplotlib subplot showing the data and 35 models and the
             residuals corresponding to the best model.

             figure_distributions : a multiple matplotlib subplot representing the parameters
             distributions (2D slice + histogram)

    :rtype: object
    """

    chains = fit.MCMC_chains
    probabilities = fit.MCMC_probabilities

    CHAINS = chains[:, :, 0].ravel()
    for i in xrange(chains[0].shape[1] - 1):
        i += 1
        CHAINS = np.c_[CHAINS, chains[:, :, i].ravel()]

    BEST_PARAMETERS = CHAINS
    if chains[0].shape[1] != len(fit.model.model_dictionnary):
        fluxes = MCMC_compute_fs_g(fit, CHAINS)

        CHAINS = np.c_[CHAINS, fluxes]

    CHAINS = np.c_[CHAINS, probabilities.ravel()]
    BEST_PARAMETERS = np.c_[BEST_PARAMETERS, probabilities.ravel()]

    best_proba = np.argmax(CHAINS[:, -1])

    # cut to 6 sigma for plots
    index = np.where(CHAINS[:, -1] > CHAINS[best_proba, -1] - 36)[0]
    BEST = CHAINS[index]
    BEST = BEST[BEST[:, -1].argsort(),]

    BEST_PARAMETERS = BEST_PARAMETERS[index]
    BEST_PARAMETERS = BEST_PARAMETERS[BEST_PARAMETERS[:, -1].argsort(),]
    covariance_matrix = MCMC_covariance(CHAINS)
    correlation_matrix = cov2corr(covariance_matrix)

    figure_lightcurve = MCMC_plot_lightcurves(fit, BEST)
    figure_distributions = MCMC_plot_parameters_distribution(fit, BEST_PARAMETERS)

    key_outputs = ['MCMC_chains', 'MCMC_correlations', 'figure_lightcurve', 'figure_distributions']
    outputs = collections.namedtuple('Fit_outputs', key_outputs)

    values_outputs = [CHAINS, correlation_matrix, figure_lightcurve, figure_distributions]

    count = 0
    for key in key_outputs:
        setattr(outputs, key, values_outputs[count])
        count += 1

    return outputs


def MCMC_compute_fs_g(fit, mcmc_chains):
    """ Compute the corresponding source flux fs and blending factor g corresponding to each mcmc
    chain.

    :param fit: a fit object. See the microlfits for more details.
    :param mcmc_chains: a numpy array representing the mcmc chains.
    :return: a numpy array containing the corresponding fluxes parameters
    :rtype: array_type

    """
    Fluxes = np.zeros((len(mcmc_chains), 2 * len(fit.event.telescopes)))

    for i in xrange(len(mcmc_chains)):

        if Fluxes[i][0] == 0:
            index = np.where(mcmc_chains == mcmc_chains[i])[0]

            fluxes = fit.find_fluxes(mcmc_chains[i], fit.model)

            Fluxes[np.unique(index)] = fluxes

    return Fluxes


def MCMC_plot_parameters_distribution(fit, mcmc_best):
    """ Plot the fit parameters distributions.
    Only plot the best mcmc_chains are plotted.
    :param fit: a fit object. See the microlfits for more details.
    :param mcmc_best: a numpy array representing the best (<= 6 sigma) mcmc chains.
    :return: a multiple matplotlib subplot representing the parameters distributions (2D slice +
    histogram)
    :rtype: matplotlib_figure
    """

    mcmc_string_format = np.array([str(i) for i in mcmc_best.tolist()])
    mcmc_to_plot = np.unique(mcmc_string_format)
    mcmc_unique = np.array([json.loads(i) for i in mcmc_to_plot])

    mcmc_unique = mcmc_unique[mcmc_unique[:,-1].argsort(),]
    dimensions = mcmc_best.shape[1] - 1

    figure_distributions, axes2 = plt.subplots(dimensions, dimensions, sharex='col')

    count_i = 0

    for key_i in fit.model.model_dictionnary.keys()[: dimensions]:

        axes2[count_i, 0].set_ylabel(key_i, fontsize=int(100/dimensions))
        axes2[-1, count_i].set_xlabel(key_i, fontsize=int(100/dimensions))

        count_j = 0
        for key_j in fit.model.model_dictionnary.keys()[: dimensions]:

            axes2[count_i, count_j].ticklabel_format(useOffset=False, style='plain')
            axes2[count_i, count_j].ticklabel_format(useOffset=False, style='plain')


            if count_i == count_j:

                axes2[count_i, count_j].hist(mcmc_best[:, fit.model.model_dictionnary[key_i]], 100)
                axes2[count_i, count_j].xaxis.set_major_locator(MaxNLocator(max_plot_ticks))
                axes2[count_i, count_j].yaxis.set_major_locator(MaxNLocator(max_plot_ticks))
                axes2[count_i, count_j].tick_params(labelsize=int(75.0 / dimensions))

            else:

                if count_j < count_i:

                    axes2[count_i, count_j].scatter(
                        mcmc_unique[:, fit.model.model_dictionnary[key_j]],
                        mcmc_unique[:, fit.model.model_dictionnary[key_i]],
                        c=mcmc_unique[:, -1],
                        edgecolor='None')

                    axes2[count_i, count_j].set_xlim(
                        [min(mcmc_unique[:, fit.model.model_dictionnary[key_j]]),
                         max(mcmc_unique[:, fit.model.model_dictionnary[key_j]])])
                    axes2[count_i, count_j].xaxis.set_major_locator(MaxNLocator(max_plot_ticks))

                    axes2[count_i, count_j].set_ylim(
                        [min(mcmc_unique[:, fit.model.model_dictionnary[key_i]]),
                         max(mcmc_unique[:, fit.model.model_dictionnary[key_i]])])

                    axes2[count_i, count_j].tick_params(labelsize=int(75.0/dimensions))

                else:

                    axes2[count_i, count_j].axis('off')

                if count_j == 0 :

                    axes2[count_i, count_j].yaxis.set_major_locator(MaxNLocator(max_plot_ticks))
                else :

                    plt.setp(axes2[count_i, count_j].get_yticklabels(), visible=False)

            count_j += 1

        count_i += 1

    return figure_distributions


def MCMC_plot_lightcurves(fit, mcmc_best):
    """Plot 35 models from the mcmc_best sample. This is made to have  35 models equally spaced
    in term of objective funtion (~chichi)

    :param fit: a fit object. See the microlfits for more details.
    :param mcmc_best: a numpy array representing the best (<= 6 sigma) mcmc chains.
    :return: a two matplotlib subplot showing the data and 35 models and the residuals
    corresponding to the best model.
    :rtype: matplotlib_figure
    """
    figure_lightcurves, figure_axes = initialize_plot_lightcurve(fit)

    MCMC_plot_align_data(fit, mcmc_best[0], figure_axes[0])

    model_panel_chichi = np.linspace(max(mcmc_best[:, -1]), min(mcmc_best[:, -1]), 35).astype(int)
    color_normalization = matplotlib.colors.Normalize(vmin=np.min(mcmc_best[:, -1]),
                                                      vmax=np.max(mcmc_best[:, -1]))
    color_map = matplotlib.cm.jet

    scalar_couleur_map = matplotlib.cm.ScalarMappable(cmap=color_map, norm=color_normalization)
    scalar_couleur_map.set_array([])

    for model_chichi in model_panel_chichi:
        indice = np.searchsorted(mcmc_best[:, -1], model_chichi) - 1

        MCMC_plot_model(fit, mcmc_best[indice], mcmc_best[indice, -1], figure_axes[0],
                        scalar_couleur_map)

    cb = plt.colorbar(scalar_couleur_map, ax=figure_axes[0], orientation="horizontal")
    cb.locator = MaxNLocator(5)
    cb.update_ticks()
   # figure_axes[0].text(0.01, 0.97, 'provided by pyLIMA', style='italic', fontsize=10,
    #                    transform=figure_axes[0].transAxes)
    figure_axes[0].invert_yaxis()
    MCMC_plot_residuals(fit, mcmc_best[0], figure_axes[1])

    return figure_lightcurves


def MCMC_plot_model(fit, parameters, couleurs, figure_axes, scalar_couleur_map):
    """ Plot a  model to a given figure, with the color corresponding to the objective function
    of the model.

    :param fit: a fit object. See the microlfits for more details.
    :param parameters: the parameters [list] of the model you want to plot.
    :param couleurs: the values of the objective function for the model that match the color
    table scalar_couleur_map
    :param figure_axes: the axes where the plot are draw
    :param scalar_couleur_map: a matplotlib table that return a color given a scalar value (the
    objective function here)
    """
    min_time = min([min(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])
    max_time = max([max(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])

    time_of_model = np.arange(min_time, max_time + 100, 0.01)

    reference_telescope = copy.copy(fit.event.telescopes[0])
    reference_telescope.lightcurve_magnitude = np.array(
        [time_of_model, [0] * len(time_of_model), [0] * len(time_of_model)]).T
    reference_telescope.lightcurve_flux = reference_telescope.lightcurve_in_flux()
    reference_telescope.compute_parallax(fit.event, fit.model.parallax_model)

    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(parameters)
    flux_model = fit.model.compute_the_microlensing_model(reference_telescope, pyLIMA_parameters)[0]

    magnitude_model = microltoolbox.flux_to_magnitude(flux_model)

    figure_axes.plot(time_of_model, magnitude_model, color=scalar_couleur_map.to_rgba(couleurs),
                     alpha=0.5)


def MCMC_plot_align_data(fit, parameters, ax):
    """ Plot the data on the figure. Telescopes are aligned to the survey telescope (i.e number 0).

    :param fit: a fit object. See the microlfits for more details.
    :param parameters: the parameters [list] of the model you want to plot.
    :param ax: the matplotlib axes where you plot the data
    """
    reference_telescope = fit.event.telescopes[0].name
    fs_reference = parameters[fit.model.model_dictionnary['fs_' + reference_telescope]]
    g_reference = parameters[fit.model.model_dictionnary['g_' + reference_telescope]]

    for telescope in fit.event.telescopes:

        if telescope.name == reference_telescope:

            lightcurve = telescope.lightcurve_magnitude

        else:

            fs_telescope = parameters[fit.model.model_dictionnary['fs_' + telescope.name]]
            g_telescope = parameters[fit.model.model_dictionnary['g_' + telescope.name]]

            lightcurve = align_telescope_lightcurve(telescope.lightcurve_magnitude, fs_reference,
                                                    g_reference, fs_telescope, g_telescope)

        ax.errorbar(lightcurve[:, 0], lightcurve[:, 1], yerr=lightcurve[:, 2], fmt='.',
                    label=telescope.name)

    ax.legend(numpoints=1, fontsize=25)


def MCMC_plot_residuals(fit, parameters, ax):
    """Plot the data residual on the appropriate figure.

    :param fit: a fit object. See the microlfits for more details.
    :param parameters: the parameters [list] of the model you want to plot.
    :param ax: the matplotlib axes where you plot the data
    """

    for telescope in fit.event.telescopes:
        time = telescope.lightcurve_flux[:, 0]
        flux = telescope.lightcurve_flux[:, 1]
        error_flux = telescope.lightcurve_flux[:, 2]
        err_mag = microltoolbox.error_flux_to_error_magnitude(error_flux, flux)

        pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(parameters)
        flux_model = fit.model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[0]

        residuals = 2.5 * np.log10(flux_model / flux)
        ax.errorbar(time, residuals, yerr=err_mag, fmt='.')
    ax.set_ylim([-plot_residuals_windows, plot_residuals_windows])
    ax.invert_yaxis()

def LM_parameters_result(fit):
    """ Produce a namedtuple object containing the fitted parameters in the fit.fit_results.

    :param fit: a fit object. See the microlfits for more details.
    :param fit_parameters: a namedtuple object containing the fitted parameters.
    :rtype: object
    """

    fit_parameters = collections.namedtuple('Parameters', fit.model.model_dictionnary.keys())

    for parameter in fit.model.model_dictionnary.keys():
        setattr(fit_parameters, parameter, fit.fit_results[fit.model.model_dictionnary[parameter]])

    setattr(fit_parameters, 'chichi', fit.fit_results[-1])
    return fit_parameters


def MCMC_covariance(mcmc_chains):
    """ Estimate the covariance matrix from the mcmc_chains

    :param mcmc_chains: a numpy array representing the mcmc chains.
    :return : a numpy array representing the covariance matrix of your MCMC sampling.
    :rtype: array_like
    """
    esperances = []
    for i in xrange(mcmc_chains.shape[1] - 1):
        esperances.append(mcmc_chains[:, i] - np.median(mcmc_chains[:, i]))

    covariance_matrix = np.zeros((mcmc_chains.shape[1] - 1, mcmc_chains.shape[1] - 1))

    for i in xrange(mcmc_chains.shape[1] - 1):
        for j in np.arange(i, mcmc_chains.shape[1] - 1):
            covariance_matrix[i, j] = 1 / (len(mcmc_chains) - 1) * np.sum(
                esperances[i] * esperances[j])
            covariance_matrix[j, i] = 1 / (len(mcmc_chains) - 1) * np.sum(
                esperances[i] * esperances[j])

    return covariance_matrix


def LM_fit_errors(fit):
    """ Estimate the parameters errors from the fit.fit_covariance matrix.

    :param fit: a fit object. See the microlfits for more details.
    :return: a namedtuple object containing the square roots of parameters variance.
    :rtype: object
    """
    keys = ['err_' + parameter for parameter in fit.model.model_dictionnary.keys()]
    parameters_errors = collections.namedtuple('Errors_Parameters', keys)
    errors = fit.fit_covariance.diagonal() ** 0.5
    for i in fit.model.model_dictionnary.keys():
        setattr(parameters_errors, 'err_' + i, errors[fit.model.model_dictionnary[i]])

    return parameters_errors


def cov2corr(covariance_matrix):
    """Covariance matrix to correlation matrix.

    :param array_like covariance_matrix: a (square) numpy array representing the covariance matrix

    :return: a (square) numpy array representing the correlation matrix
    :rtype: array_like

    """

    d = np.sqrt(covariance_matrix.diagonal())
    correlation_matrix = ((covariance_matrix.T / d).T) / d

    return correlation_matrix


def LM_plot_lightcurves(fit):
    """Plot the aligned datasets and the best fit on the first subplot figure_axes[0] and residuals
    on the second subplot figure_axes[1].

    :param object fit: a fit object. See the microlfits for more details.
    :return: a figure representing data+model and residuals.
    :rtype: matplotlib_figure

    """
    figure, figure_axes = initialize_plot_lightcurve(fit)
    LM_plot_model(fit, figure_axes[0])
    LM_plot_align_data(fit, figure_axes[0])
    LM_plot_residuals(fit, figure_axes[1])

    return figure


def LM_plot_parameters(fit):
    """ NOT USED
    """
    figure, axes = initialize_plot_parameters()

    return figure


def initialize_plot_lightcurve(fit):
    """Initialize the lightcurve plot.

    :param object fit: a fit object. See the microlfits for more details.

    :return: a matplotlib figure  and the corresponding matplotlib axes
    :rtype: matplotlib_figure,matplotlib_axes

    """
    figure, figure_axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    figure_axes[0].grid()
    figure_axes[1].grid()
    # figure.suptitle(fit.event.name, fontsize=30)
    figure_axes[0].set_ylabel('Mag', fontsize=50)
    figure_axes[0].yaxis.set_major_locator(MaxNLocator(4))
    figure_axes[0].tick_params(labelsize=30)

    figure_axes[1].set_xlabel('Days', fontsize=50)
    figure_axes[1].xaxis.set_major_locator(MaxNLocator(8))
    figure_axes[1].yaxis.set_major_locator(MaxNLocator(4))

    figure_axes[1].set_ylabel('Residuals', fontsize=50)
    figure_axes[1].tick_params(labelsize=30)

    return figure, figure_axes


def initialize_plot_parameters(fit):
    """Initialize the parameters plot.

    :param object fit: a fit object. See the microlfits for more details.
    :return: a matplotlib figure  and the corresponding matplotlib axes.
    :rtype: matplotlib_figure,matplotlib_axes
    """
    dimension_y = np.floor(len(fit.fits_result) / 3)
    dimension_x = len(fit.fits_result) - 3 * dimension_y

    figure, figure_axes = plt.subplots(dimension_x, dimension_y)

    return figure, figure_axes


def LM_plot_model(fit, figure_axe):
    """Plot the microlensing model from the fit.

    :param object fit: a fit object. See the microlfits for more details.
    :param matplotlib_axes figure_axe: a matplotlib axes correpsonding to the figure.
    """

    min_time = min([min(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])
    max_time = max([max(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])

    time = np.linspace(min_time, max_time + 100, 30000)

    reference_telescope = copy.copy(fit.event.telescopes[0])
    reference_telescope.lightcurve_magnitude = np.array(
        [time, [0] * len(time), [0] * len(time)]).T
    reference_telescope.lightcurve_flux = reference_telescope.lightcurve_in_flux()
    reference_telescope.compute_parallax(fit.event, fit.model.parallax_model)

    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(fit.fit_results)
    flux_model = fit.model.compute_the_microlensing_model(reference_telescope, pyLIMA_parameters)[0]
    magnitude = microltoolbox.flux_to_magnitude(flux_model)

    figure_axe.plot(time, magnitude, '--k', lw=1)
    figure_axe.set_ylim(
        [min(magnitude) - plot_lightcurve_windows, max(magnitude) + plot_lightcurve_windows])
    figure_axe.invert_yaxis()
    # figure_axe.text(0.01, 0.97, 'provided by pyLIMA', style='italic', fontsize=10,
    # transform=figure_axe.transAxes)


def LM_plot_residuals(fit, figure_axe):
    """Plot the residuals from the fit.

    :param object fit: a fit object. See the microlfits for more details.
    :param matplotlib_axes figure_axe: a matplotlib axes correpsonding to the figure.
    """

    for telescope in fit.event.telescopes:
        time = telescope.lightcurve_flux[:, 0]
        flux = telescope.lightcurve_flux[:, 1]
        error_flux = telescope.lightcurve_flux[:, 2]
        err_mag = microltoolbox.error_flux_to_error_magnitude(error_flux, flux)

        pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(fit.fit_results)
        flux_model = fit.model.compute_the_microlensing_model(telescope, pyLIMA_parameters)[0]

        residuals = 2.5 * np.log10(flux_model / flux)
        figure_axe.errorbar(time, residuals, yerr=err_mag, fmt='.')
    figure_axe.set_ylim([-plot_residuals_windows, plot_residuals_windows])
    figure_axe.invert_yaxis()


def LM_plot_align_data(fit, figure_axe):
    """Plot the aligned data.

    :param object fit: a fit object. See the microlfits for more details.
    :param matplotlib_axes figure_axe: a matplotlib axes correpsonding to the figure.
    """
    reference_telescope = fit.event.telescopes[0].name
    fs_reference = fit.fit_results[fit.model.model_dictionnary['fs_' + reference_telescope]]
    g_reference = fit.fit_results[fit.model.model_dictionnary['g_' + reference_telescope]]

    for telescope in fit.event.telescopes:

        if telescope.name == reference_telescope:

            lightcurve = telescope.lightcurve_magnitude

        else:

            fs_telescope = fit.fit_results[fit.model.model_dictionnary['fs_' + telescope.name]]
            g_telescope = fit.fit_results[fit.model.model_dictionnary['g_' + telescope.name]]

            lightcurve = align_telescope_lightcurve(telescope.lightcurve_magnitude, fs_reference,
                                                    g_reference, fs_telescope, g_telescope)

        figure_axe.errorbar(lightcurve[:, 0], lightcurve[:, 1], yerr=lightcurve[:, 2], fmt='.',
                            label=telescope.name)

    figure_axe.legend(numpoints=1, fontsize=25)


def align_telescope_lightcurve(lightcurve_telescope_mag, fs_reference, g_reference, fs_telescope,
                               g_telescope):
    """Align data to the survey telescope (i.e telescope 0).

    :param array_like lightcurve_telescope_mag: the survey telescope in magnitude
    :param float fs_reference: the survey telescope reference source flux (i.e the fitted value)
    :param float g_reference: the survey telescope reference blending parameter (i.e the fitted
    value)
    :param float fs_telescope: the telescope source flux (i.e the fitted value)
    :param float g_reference: the telescope blending parameter (i.e the fitted value)

    :return: the aligned to survey lightcurve in magnitude
    :rtype: array_like
    """
    time = lightcurve_telescope_mag[:, 0]
    magnitude = lightcurve_telescope_mag[:, 1]
    err_mag = lightcurve_telescope_mag[:, 2]

    flux = microltoolbox.magnitude_to_flux(magnitude)

    flux_normalised = (flux - (fs_telescope * g_telescope)) / (
        fs_telescope) * fs_reference + fs_reference * g_reference

    magnitude_normalised = microltoolbox.flux_to_magnitude(flux_normalised)

    lightcurve_normalised = [time, magnitude_normalised, err_mag]

    lightcurve_mag_normalised = np.array(lightcurve_normalised).T

    return lightcurve_mag_normalised


### TO DO : some parts depreciated ####

def errors_on_fits(self, choice):
    if len(self.event.fits[choice].fit_covariance) == 0:

        print 'There is no way to produce errors without covariance at this stage'

    else:

        self.event.fits[choice].fit_errors = np.sqrt(
            self.event.fits[choice].fit_covariance.diagonal())


def find_observables(self):
    count = 0
    self.observables_dictionnary = {'to': 0, 'Ao': 1, 'tE': 2, 'Anow': 3, 'Ibaseline': 4,
                                    'Ipeak': 5, 'Inow': 6}
    self.observables_dictionnary = OrderedDict(
        sorted(self.observables_dictionnary.items(), key=lambda x: x[1]))
    for i in self.event.fits_results:
        observables = []
        parameters = i[3]
        to = parameters[self.event.fits_models[count][2].model_dictionnary['to']]
        uo = parameters[self.event.fits_models[count][2].model_dictionnary['uo']]
        tE = parameters[self.event.fits_models[count][2].model_dictionnary['tE']]

        t = Time(datetime.utcnow())
        # tnow=t.jd1+t.jd2
        tnow = 150
        Ao = microlmagnification.amplification(self.event.fits_models[count][2], np.array([to]),
                                               parameters, self.event.telescopes[0].gamma)[0][0]
        Anow = \
            microlmagnification.amplification(self.event.fits_models[count][2], np.array([tnow]),
                                              parameters, self.event.telescopes[0].gamma)[0][0]

        observables.append(to)
        observables.append(Ao)
        observables.append(tE)
        observables.append(Anow)

        Ibaseline = 27.4 - 2.5 * np.log10(
            parameters[self.event.fits_models[count][2].model_dictionnary[
                'fs_' + self.event.telescopes[0].name]] * (
                1 + parameters[self.event.fits_models[count][2].model_dictionnary[
                    'g_' + self.event.telescopes[0].name]]))

        Ipeak = 27.4 - 2.5 * np.log10(
            parameters[self.event.fits_models[count][2].model_dictionnary[
                'fs_' + self.event.telescopes[0].name]] * (
                Ao + parameters[self.event.fits_models[count][2].model_dictionnary[
                    'g_' + self.event.telescopes[0].name]]))

        Inow = 27.4 - 2.5 * np.log10(
            parameters[self.event.fits_models[count][2].model_dictionnary[
                'fs_' + self.event.telescopes[0].name]] * (
                Anow + parameters[self.event.fits_models[count][2].model_dictionnary[
                    'g_' + self.event.telescopes[0].name]]))

        observables.append(Ibaseline)
        observables.append(Ipeak)
        observables.append(Inow)

        self.observables.append([i[0], i[1], i[2], observables])


def find_observables_errors(self):
    for i in xrange(len(self.event.fits_results)):

        parameters = self.observables[i][2]
        parameters_errors = self.error_parameters[i][2]

        to = self.event.fits_results[i][2][0]
        uo = self.event.fits_results[i][2][1]
        tE = self.event.fits_results[i][2][2]

        Ao = parameters[1]
        err_Ao = parameters_errors[1] * 8 / (
            parameters[1] ** 2 * (parameters[1] ** 2 + 4) ** 1.5)
        Anow = parameters[3]
        jd1, jd2 = Time(datetime.datetime.utcnow())
        tnow = jd1 + jd2
        unow = np.sqrt(uo ** 2 + (tnow - to) ** 2 / tE ** 2)
        err_Anow = (uo * parameters_errors[1] * np.abs((tnow - to)) / tE ** 3 * (
            tE * parameters_errors[0] + np.abs((tnow - to)) * parameters_errors[2])) / unow
        observables_errors = []
        observables = []
        observables_errors.append(parameters_errors[0])
        observables_errors.append(err_Ao)
        observables_errors.append(parameters_errors[2])
        observables_errors.append(err_Anow)

        start = len(parameters) - 2 * len(self.event.telescopes) - 1
        for j in xrange(len(self.event.telescopes)):
            Ibaseline = 27.4 - 2.5 * np.log10(parameters[start] * (1 + parameters[start]))
            Ipeak = 27.4 - 2.5 * np.log10(parameters[start] * (Ao + parameters[start]))

            observables.append(Ibaseline)
            observables.append(Ipeak)

            start += 2

        self.observables.append([i[0], i[1], observables])


def errors_on_observables(self):
    for i in self.event.fits_covariance:
        self.error_parameters.append([i[0], i[1], np.sqrt(i[2].diagonal)])


def student_errors(self):
    alpha = 0.05
    ndata = len(self.event.telescopes[0].lightcurve_flux)
    npar = 5
    dof = ndata - npar
    tval = student.ppf(1 - alpha / 2, dof)

    lower = []
    upper = []

    for i in xrange(len(self.event.fits_covariance[0][2].diagonal())):
        sigma = self.event.fits_covariance[0][2].diagonal()[i] ** 0.5
        lower.append(self.event.fits_results[0][2][i] - sigma * tval)
        upper.append(self.event.fits_results[0][2][i] + sigma * tval)

    self.upper = upper
    self.lower = lower


def K2_C9_outputs(self):
    import matplotlib.pyplot as plt

    # first produce aligned lightcurve#

    time = []
    mag = []
    err_mag = []
    groups = []

    time = time + self.event.telescopes[0].lightcurve[:, 0].tolist()
    mag = mag + self.event.telescopes[0].lightcurve[:, 1].tolist()
    err_mag = err_mag + self.event.telescopes[0].lightcurve[:, 2].tolist()
    groups = groups + [self.event.telescopes[0].name] * len(self.event.telescopes[0].lightcurve)

    for i in self.event.telescopes[1:]:
        time = time + i.lightcurve[:, 0].tolist()
        Mag = i.lightcurve[:, 1]
        flux = 10 ** ((27.4 - Mag) / 2.5)
        err_flux = np.abs(-i.lightcurve[:, 2] * flux / (2.5) * np.log(10))
        flux_normalised = self.event.fits[0].fit_results[
                              self.event.fits[0].model.model_dictionnary[
                                  'fs_' + self.event.telescopes[0].name]] * ((
                                                                                 flux /
                                                                                 self.event.fits[
                                                                                     0].fit_results[
                                                                                     self.event.fits[
                                                                                         0].model.model_dictionnary[
                                                                                         'fs_' +
                                                                                         i.name]] -
                                                                                 self.event.fits[
                                                                                     0].fit_results[
                                                                                     self.event.fits[
                                                                                         0].model.model_dictionnary[
                                                                                         'g_'
                                                                                         +
                                                                                         i.name]]) +
                                                                             self.event.fits[
                                                                                 0].fit_results[
                                                                                 self.event.fits[
                                                                                     0].model.model_dictionnary[
                                                                                     'g_' +
                                                                                     self.event.telescopes[
                                                                                         0].name]])
        err_flux_norm = err_flux / flux * flux_normalised
        mag_norm = 27.4 - 2.5 * np.log10(flux_normalised)
        err_mag_norm = 2.5 * err_flux_norm / (flux_normalised * np.log(10))

        mag = mag + mag_norm.tolist()
        err_mag = err_mag + err_mag_norm.tolist()
        groups = groups + [i.name] * len(i.lightcurve)

    lightcurve_data = np.array([time, mag, err_mag, groups]).T

    # produce model lightcurve

    time = np.arange(min(self.event.telescopes[0].lightcurve[:, 0]), max(time) + 100, 0.01)
    ampli = microlmagnification.amplification(self.event.fits[0].model, time,
                                              self.event.fits[0].fit_results, 0.5)[0]
    flux = self.event.fits[0].fit_results[self.event.fits[0].model.model_dictionnary[
        'fs_' + self.event.telescopes[0].name]] * (
               ampli + self.event.fits[0].fit_results[
                   self.event.fits[0].model.model_dictionnary[
                       'g_' + self.event.telescopes[0].name]])
    mag = (27.4 - 2.5 * np.log10(flux)).tolist()
    err_mag = [0.001] * len(time)
    time = time.tolist()
    lightcurve_model = np.array([time, mag, err_mag]).T

    # produce parameters
    Parameters = []
    Names = []

    Uo = self.event.fits[0].fit_results[self.event.fits[0].model.model_dictionnary['uo']]
    Ao = (Uo ** 2 + 2) / (Uo * (Uo ** 2 + 4) ** 0.5)
    err_Ao = (8) / (Uo ** 2 * (Uo ** 2 + 4) ** 1.5) * \
             (self.event.fits[0].fit_covariance.diagonal() ** 0.5)[1]

    Parameters.append(Ao)
    Parameters.append(err_Ao)

    Names.append('PYLIMA.AO')
    Names.append('PYLIMA.SIG_AO')

    names = ['TE', 'TO', 'UO']
    Official = ['tE', 'to', 'uo']

    for i in xrange(len(Official)):
        index = self.event.fits[0].model.model_dictionnary[Official[i]]
        Parameters.append(self.event.fits[0].fit_results[index])
        Parameters.append((self.event.fits[0].fit_covariance.diagonal() ** 0.5)[index])

        Names.append('PYLIMA.' + names[i])
        Names.append('PYLIMA.SIG_' + names[i])
    Parameters = np.array([Names, Parameters]).T
    count = 0
    for i in self.event.telescopes:
        index = np.where(lightcurve_data[:, 3] == i.name)[0]
        colors = np.random.uniform(0, 10)
        plt.scatter(lightcurve_data[index, 0].astype(float),
                    lightcurve_data[index, 1].astype(float), c=(
                np.random.randint(0, float(len(self.event.telescopes))) / float(
                    len(self.event.telescopes)),
                np.random.randint(0, float(len(self.event.telescopes))) / float(
                    len(self.event.telescopes)),
                np.random.randint(0, float(len(self.event.telescopes))) / float(
                    len(self.event.telescopes)), 1), label=i.name, s=25)
        count += 1
    plt.legend(scatterpoints=1)
    plt.plot(lightcurve_model[:, 0], lightcurve_model[:, 1], 'g')
    plt.show()

    return Parameters, lightcurve_model, lightcurve_data
