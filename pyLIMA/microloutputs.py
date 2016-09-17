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
import microlmodels

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
    #Change matplotlib default colors

    n = len(fit.event.telescopes)
    color = plt.cm.jet(np.linspace(0.1, 0.99, n))  # This returns RGBA; convert:
    hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
                   tuple(color[:, 0:-1]))
    hexcolor[0] = '#000000'
    matplotlib.rcParams['axes.color_cycle'] = hexcolor

    results = LM_parameters_result(fit)
    covariance_matrix = fit.fit_covariance
    errors = LM_fit_errors(fit)
    correlation_matrix = cov2corr(covariance_matrix)
    figure_lightcurve = LM_plot_lightcurves(fit)
    figure_trajectory = plot_LM_ML_geometry(fit)
    key_outputs = ['fit_parameters', 'fit_errors', 'fit_correlation_matrix', 'figure_lightcurve', 'figure_geometry']
    outputs = collections.namedtuple('Fit_outputs', key_outputs)

    values_outputs = [results, errors, correlation_matrix, figure_lightcurve, figure_trajectory]

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

    # Change matplotlib default colors

    n = len(fit.event.telescopes)
    color = plt.cm.jet(np.linspace(0.01, 0.99, n))  # This returns RGBA; convert:
    hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
                   tuple(color[:, 0:-1]))
    matplotlib.rcParams['axes.color_cycle'] = hexcolor
    hexcolor[0] = '#000000'
    raw_chains = fit.MCMC_chains
    probabilities = fit.MCMC_probabilities

    mcmc_chains = raw_chains[:, :, 0].ravel()

    for i in xrange(raw_chains[0].shape[1] - 1):
        i += 1
        mcmc_chains = np.c_[mcmc_chains, raw_chains[:, :, i].ravel()]

    best_parameters = mcmc_chains

    if raw_chains[0].shape[1] != len(fit.model.model_dictionnary):
        fluxes = MCMC_compute_fs_g(fit, mcmc_chains)

        mcmc_chains = np.c_[mcmc_chains, fluxes]

    mcmc_chains = np.c_[mcmc_chains, probabilities.ravel()]
    best_parameters = np.c_[best_parameters, probabilities.ravel()]

    best_probability = np.argmax(mcmc_chains[:, -1])

    # cut to 6 sigma for plots
    index = np.where(mcmc_chains[:, -1] > mcmc_chains[best_probability, -1] - 36)[0]
    best_chains = mcmc_chains[index]
    best_chains = best_chains[best_chains[:, -1].argsort(),]

    best_parameters = best_parameters[index]
    best_parameters = best_parameters[best_parameters[:, -1].argsort(),]
    covariance_matrix = MCMC_covariance(mcmc_chains)
    correlation_matrix = cov2corr(covariance_matrix)

    figure_lightcurve = MCMC_plot_lightcurves(fit, best_chains)
    figure_distributions = MCMC_plot_parameters_distribution(fit, best_parameters)

    figure_geometry = plot_MCMC_ML_geometry(fit, best_chains)
    key_outputs = ['MCMC_chains', 'MCMC_correlations', 'figure_lightcurve', 'figure_distributions']
    outputs = collections.namedtuple('Fit_outputs', key_outputs)

    values_outputs = [mcmc_chains, correlation_matrix, figure_lightcurve, figure_distributions]

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
    fluxes_chains = np.zeros((len(mcmc_chains), 2 * len(fit.event.telescopes)))

    for i in xrange(len(mcmc_chains)):

        if fluxes_chains[i][0] == 0:
            index = np.where(mcmc_chains == mcmc_chains[i])[0]

            fluxes = fit.find_fluxes(mcmc_chains[i], fit.model)

            fluxes_chains[np.unique(index)] = fluxes

    return fluxes_chains


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

    mcmc_unique = mcmc_unique[mcmc_unique[:, -1].argsort(),]
    dimensions = mcmc_best.shape[1] - 1

    figure_distributions, axes2 = plt.subplots(dimensions, dimensions, sharex='col')

    count_i = 0

    for key_i in fit.model.model_dictionnary.keys()[: dimensions]:

        axes2[count_i, 0].set_ylabel(key_i, fontsize=int(100 / dimensions))
        axes2[-1, count_i].set_xlabel(key_i, fontsize=int(100 / dimensions))

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

                    axes2[count_i, count_j].tick_params(labelsize=int(75.0 / dimensions))

                else:

                    axes2[count_i, count_j].axis('off')

                if count_j == 0:

                    axes2[count_i, count_j].yaxis.set_major_locator(MaxNLocator(max_plot_ticks))
                else:

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
    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(mcmc_best[-1])
    MCMC_plot_align_data(fit, mcmc_best[0], figure_axes[0])

    model_panel_chichi = np.linspace(max(mcmc_best[:, -1]), min(mcmc_best[:, -1]), 35).astype(int)
    color_normalization = matplotlib.colors.Normalize(vmin=np.min(mcmc_best[:, -1]),
                                                      vmax=np.max(mcmc_best[:, -1]))
    color_map = matplotlib.cm.jet

    scalar_couleur_map = matplotlib.cm.ScalarMappable(cmap=color_map, norm=color_normalization)
    scalar_couleur_map.set_array([])

    min_time = min([min(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])
    max_time = max([max(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])

    time_of_model = np.linspace(min_time, max_time + 100, 30000)
    extra_time = np.linspace(pyLIMA_parameters.to - 2 * pyLIMA_parameters.tE,
                             pyLIMA_parameters.to + 2 * pyLIMA_parameters.tE, 30000)

    time_of_model = np.sort(np.append(time_of_model, extra_time))
    reference_telescope = copy.copy(fit.event.telescopes[0])
    reference_telescope = copy.copy(fit.event.telescopes[0])
    reference_telescope.lightcurve_magnitude = np.array(
        [time_of_model, [0] * len(time_of_model), [0] * len(time_of_model)]).T
    reference_telescope.lightcurve_flux = reference_telescope.lightcurve_in_flux()

    if fit.model.parallax_model[0] != 'None':
        reference_telescope.compute_parallax(fit.event, fit.model.parallax_model)

    for model_chichi in model_panel_chichi:
        indice = np.searchsorted(mcmc_best[:, -1], model_chichi) - 1

        MCMC_plot_model(fit, reference_telescope, mcmc_best[indice], mcmc_best[indice, -1], figure_axes[0],
                        scalar_couleur_map)

    colorbar = plt.colorbar(scalar_couleur_map, ax=figure_axes[0], orientation="horizontal")
    colorbar.locator = MaxNLocator(5)
    colorbar.formatter.set_useOffset(False)
    colorbar.update_ticks()

    figure_axes[0].text(0.01, 0.96, 'provided by pyLIMA', style='italic', fontsize=10,
                        transform=figure_axes[0].transAxes)
    figure_axes[0].invert_yaxis()
    MCMC_plot_residuals(fit, mcmc_best[0], figure_axes[1])

    return figure_lightcurves


def MCMC_plot_model(fit, reference_telescope, parameters, couleurs, figure_axes, scalar_couleur_map):
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

    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(parameters)
    flux_model = fit.model.compute_the_microlensing_model(reference_telescope, pyLIMA_parameters)[0]

    magnitude_model = microltoolbox.flux_to_magnitude(flux_model)

    figure_axes.plot(reference_telescope.lightcurve_magnitude[:, 0], magnitude_model,
                     color=scalar_couleur_map.to_rgba(couleurs),
                     alpha=0.5)


def MCMC_plot_align_data(fit, parameters, plot_axe):
    """ Plot the data on the figure. Telescopes are aligned to the survey telescope (i.e number 0).

    :param fit: a fit object. See the microlfits for more details.
    :param parameters: the parameters [list] of the model you want to plot.
    :param plot_axe: the matplotlib axes where you plot the data
    """
    reference_telescope = fit.event.telescopes[0].name
    fs_reference = parameters[fit.model.model_dictionnary['fs_' + reference_telescope]]
    g_reference = parameters[fit.model.model_dictionnary['g_' + reference_telescope]]

    for telescope in fit.event.telescopes:

        if telescope.name == reference_telescope:

            lightcurve_magnitude = telescope.lightcurve_magnitude

        else:

            fs_telescope = parameters[fit.model.model_dictionnary['fs_' + telescope.name]]
            g_telescope = parameters[fit.model.model_dictionnary['g_' + telescope.name]]

            lightcurve_magnitude = align_telescope_lightcurve(telescope.lightcurve_magnitude, fs_reference,
                                                              g_reference, fs_telescope, g_telescope)

        plot_axe.errorbar(lightcurve_magnitude[:, 0], lightcurve_magnitude[:, 1], yerr=lightcurve_magnitude[:, 2],
                          fmt='.',
                          label=telescope.name)

    plot_axe.legend(numpoints=1, fontsize=25)


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
    ax.xaxis.get_major_ticks()[0].draw = lambda *args: None
    ax.ticklabel_format(useOffset=False, style='plain')


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

    covariance_diagonal = np.sqrt(covariance_matrix.diagonal())
    correlation_matrix = ((covariance_matrix.T / covariance_diagonal).T) / covariance_diagonal

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
    fig_size=[10,10]
    figure, figure_axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},figsize=(fig_size[0], fig_size[1]))
    plt.subplots_adjust(top = 0.9, bottom = 0.15, left = 0.15, right = 0.9,  wspace = 0.2, hspace = 0.1 )
    figure_axes[0].grid()
    figure_axes[1].grid()
    #fig_size = plt.rcParams["figure.figsize"]
    figure.suptitle(fit.event.name,fontsize=30*fig_size[0]/len(fit.event.name))

    figure_axes[0].set_ylabel('Mag', fontsize=5*fig_size[1]*3/4.0)
    figure_axes[0].yaxis.set_major_locator(MaxNLocator(4))
    figure_axes[0].tick_params(axis='y', labelsize=2.5 * fig_size[1] * 3 / 4.0)

    figure_axes[1].set_xlabel('HJD', fontsize=5*fig_size[0]*3/4.0)
    figure_axes[1].xaxis.set_major_locator(MaxNLocator(6))
    figure_axes[1].yaxis.set_major_locator(MaxNLocator(4))
    figure_axes[1].xaxis.get_major_ticks()[0].draw = lambda *args: None
    figure_axes[1].ticklabel_format(useOffset=False, style='plain')
    figure_axes[1].set_ylabel('Residuals', fontsize=5*fig_size[1]*3/4.0)
    figure_axes[1].tick_params(axis='x', labelsize= 1.5*fig_size[0] * 3 / 4.0)
    figure_axes[1].tick_params(axis='y', labelsize=2.5 * fig_size[1] * 3 / 4.0)

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
    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(fit.fit_results)
    min_time = min([min(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])
    max_time = max([max(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])

    time = np.linspace(min_time, max_time + 100, 3000)
    extra_time = np.linspace(pyLIMA_parameters.to - 2 * pyLIMA_parameters.tE,
                             pyLIMA_parameters.to + 2 * pyLIMA_parameters.tE, 30000)
    time = np.sort(np.append(time, extra_time))
    reference_telescope = copy.copy(fit.event.telescopes[0])
    reference_telescope.lightcurve_magnitude = np.array(
        [time, [0] * len(time), [0] * len(time)]).T
    reference_telescope.lightcurve_flux = reference_telescope.lightcurve_in_flux()

    if fit.model.parallax_model[0] != 'None':
        reference_telescope.compute_parallax(fit.event, fit.model.parallax_model)

    flux_model = fit.model.compute_the_microlensing_model(reference_telescope, pyLIMA_parameters)[0]
    magnitude = microltoolbox.flux_to_magnitude(flux_model)

    figure_axe.plot(time, magnitude, '--k', label=fit.model.model_type, lw=1)
    figure_axe.set_ylim(
        [min(magnitude) - plot_lightcurve_windows, max(magnitude) + plot_lightcurve_windows])
    figure_axe.invert_yaxis()
    figure_axe.text(0.01, 0.96, 'provided by pyLIMA', style='italic', fontsize=10,
                    transform=figure_axe.transAxes)


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

    # xticks_labels = figure_axe.get_xticks()
    # figure_axe.set_xticklabels(xticks_labels, rotation=45)


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
    figure_axe.legend(numpoints=1, bbox_to_anchor=(0.01, 0.90), loc=2, borderaxespad=0.)


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


def plot_LM_ML_geometry(fit):
    """Plot the lensing geometry (i.e source trajectory) and the table of best parameters.
    :param object fit: a fit object. See the microlfits for more details.
    :param list best_parameters: a list containing the model you want to plot the trajectory
    """

    # Limits of the plot

    figure_trajectory_xlimit = 1.5
    figure_trajectory_ylimit = 1.5

    best_parameters = fit.fit_results
    fig_size = [15, 5]
    figure_trajectory = plt.figure(figsize=(fig_size[0], fig_size[1]))

    figure_axes = figure_trajectory.add_subplot(121, aspect=1)
    plt.subplots_adjust(top=0.8, bottom=0.1, wspace=0.5, hspace=0.2)
    einstein_ring = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--')
    figure_axes.add_artist(einstein_ring)

    min_time = min([min(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])
    max_time = max([max(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])

    time = np.linspace(min_time, max_time + 100, 3000)

    reference_telescope = copy.copy(fit.event.telescopes[0])
    reference_telescope.lightcurve_magnitude = np.array(
        [time, [0] * len(time), [0] * len(time)]).T

    reference_telescope.lightcurve_flux = np.array(
        [time, [0] * len(time), [0] * len(time)]).T

    if fit.model.parallax_model[0] != 'None':
        reference_telescope.compute_parallax(fit.event, fit.model.parallax_model)

    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(best_parameters)
    trajectory_x, trajectory_y = microlmodels.source_trajectory(reference_telescope, pyLIMA_parameters.to,
                                                                pyLIMA_parameters.uo, pyLIMA_parameters.tE,
                                                                pyLIMA_parameters)

    figure_axes.plot(trajectory_x, trajectory_y, 'b')

    # index_trajectory_limits = \
    #    np.where((np.abs(trajectory_x) < figure_trajectory_xlimit) & (np.abs(trajectory_y) < figure_trajectory_ylimit))[
    #        0]

    index_trajectory_limits = np.where((np.abs(time - pyLIMA_parameters.to) < 50))[0]

    if len(index_trajectory_limits) >= 3:
        midle = int(len(index_trajectory_limits) / 2)
        figure_axes.arrow(trajectory_x[index_trajectory_limits[midle]], trajectory_y[index_trajectory_limits[midle]],
                          trajectory_x[index_trajectory_limits[midle + 1]] - trajectory_x[
                              index_trajectory_limits[midle]],
                          trajectory_y[index_trajectory_limits[midle + 1]] - trajectory_y[
                              index_trajectory_limits[midle]],
                          head_width=0.1, head_length=0.2, color='b')

    if fit.model.model_type == 'DSPL':
        trajectory_x, trajectory_y = microlmodels.source_trajectory(reference_telescope,
                                                                    pyLIMA_parameters.to + pyLIMA_parameters.delta_to,
                                                                    pyLIMA_parameters.uo + pyLIMA_parameters.delta_uo,
                                                                    pyLIMA_parameters.tE,
                                                                    pyLIMA_parameters)
        figure_axes.plot(trajectory_x, trajectory_y, 'r')
        index_trajectory_limits = np.where((np.abs(time - pyLIMA_parameters.to) < 50))[0]

        if len(index_trajectory_limits) >= 3:
            midle = int(len(index_trajectory_limits) / 2)
            figure_axes.arrow(trajectory_x[index_trajectory_limits[midle]],
                          trajectory_y[index_trajectory_limits[midle]],
                          trajectory_x[index_trajectory_limits[midle + 1]] - trajectory_x[
                              index_trajectory_limits[midle]],
                          trajectory_y[index_trajectory_limits[midle + 1]] - trajectory_y[
                              index_trajectory_limits[midle]],
                          head_width=0.1, head_length=0.2, color='r')

    if 'BL' not in fit.model.model_type:
        figure_axes.scatter(0, 0, s=10, c='k')

    if ('PS' not in fit.model.model_type) & ('DS' not in fit.model.model_type):
        index_source = np.where((trajectory_x ** 2 + trajectory_y ** 2) ** 0.5 < max(1, pyLIMA_parameters.uo + 0.1))[0][
            0]
        source_disk = plt.Circle((trajectory_x[index_source], trajectory_y[index_source]), pyLIMA_parameters.rho,
                                 color='y')
        figure_axes.add_artist(source_disk)

    figure_axes.axis(
        [- figure_trajectory_xlimit, figure_trajectory_xlimit, - figure_trajectory_ylimit, figure_trajectory_ylimit])

    raw_labels = fit.model.model_dictionnary.keys() + ['Chi^2']
    column_labels = ['Parameters', 'Errors']

    table_val = [fit.fit_results, (fit.fit_covariance.diagonal() ** 0.5).tolist() + [0.0]]
    table_val = np.round(table_val, 5).tolist()
    table_val = np.array(table_val).T.tolist()

    table_colors = []
    raw_colors = []
    last_color = 'dodgerblue'
    for i in xrange(len(table_val)):
        table_colors.append([last_color, last_color])
        raw_colors.append(last_color)

        if last_color == 'dodgerblue':

            last_color = 'white'

        else:

            last_color = 'dodgerblue'

    table_axes = figure_trajectory.add_subplot(122, frameon=False)

    the_table = table_axes.table(cellText=table_val, cellColours=table_colors, rowColours=raw_colors,
                                 rowLabels=raw_labels,
                                 colLabels=column_labels, loc='center left')
    table_axes.get_yaxis().set_visible(False)
    table_axes.get_xaxis().set_visible(False)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(fig_size[0]*3/4.0)
    title = fit.model.event.name+' : '+fit.model.model_type
    figure_trajectory.suptitle(title, fontsize=30*fig_size[0]/len(title))
    return figure_trajectory


def plot_MCMC_ML_geometry(fit, best_chains):
    """Plot the lensing geometry (i.e source trajectory) and the table of best parameters.
    :param object fit: a fit object. See the microlfits for more details.
    :param list best_parameters: a list containing the model you want to plot the trajectory
    """

    # Limits of the plot

    figure_trajectory_xlimit = 1.5
    figure_trajectory_ylimit = 1.5

    best_parameters = best_chains[0]

    figure_trajectory = plt.figure()

    figure_axes = figure_trajectory.add_subplot(121, aspect=1)

    einstein_ring = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--')
    figure_axes.add_artist(einstein_ring)

    min_time = min([min(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])
    max_time = max([max(i.lightcurve_magnitude[:, 0]) for i in fit.event.telescopes])

    time = np.linspace(min_time, max_time + 100, 30000)

    reference_telescope = copy.copy(fit.event.telescopes[0])
    reference_telescope.lightcurve_magnitude = np.array(
        [time, [0] * len(time), [0] * len(time)]).T

    reference_telescope.lightcurve_flux = np.array(
        [time, [0] * len(time), [0] * len(time)]).T

    if fit.model.parallax_model[0] != 'None':
        reference_telescope.compute_parallax(fit.event, fit.model.parallax_model)

    pyLIMA_parameters = fit.model.compute_pyLIMA_parameters(best_parameters)
    trajectory_x, trajectory_y = microlmodels.source_trajectory(reference_telescope, pyLIMA_parameters.to,
                                                                pyLIMA_parameters.uo, pyLIMA_parameters.tE,
                                                                pyLIMA_parameters)

    figure_axes.plot(trajectory_x, trajectory_y, 'b')

    index_trajectory_limits = np.where((np.abs(time - pyLIMA_parameters.to) < 50))[0]
    if len(index_trajectory_limits) >= 3:
        midle = int(len(index_trajectory_limits)/2)
        figure_axes.arrow(trajectory_x[index_trajectory_limits[midle]], trajectory_y[index_trajectory_limits[midle]],
                          trajectory_x[index_trajectory_limits[midle+1]] - trajectory_x[index_trajectory_limits[midle]],
                          trajectory_y[index_trajectory_limits[midle+1]] - trajectory_y[index_trajectory_limits[midle]],
                          head_width=0.1, head_length=0.2, color='b')

    if fit.model.model_type == 'DSPL':

        trajectory_x, trajectory_y = microlmodels.source_trajectory(reference_telescope,
                                                                    pyLIMA_parameters.to + pyLIMA_parameters.delta_to,
                                                                    pyLIMA_parameters.uo + pyLIMA_parameters.delta_uo,
                                                                    pyLIMA_parameters.tE,
                                                                    pyLIMA_parameters)
        figure_axes.plot(trajectory_x, trajectory_y, 'r')
        index_trajectory_limits = np.where((np.abs(time - pyLIMA_parameters.to) < 50))[0]

        if len(index_trajectory_limits) >= 3:
            midle = int(len(index_trajectory_limits) / 2)
            figure_axes.arrow(trajectory_x[index_trajectory_limits[midle]],
                              trajectory_y[index_trajectory_limits[midle]],
                              trajectory_x[index_trajectory_limits[midle + 1]] - trajectory_x[
                                  index_trajectory_limits[midle]],
                              trajectory_y[index_trajectory_limits[midle + 1]] - trajectory_y[
                                  index_trajectory_limits[midle]],
                              head_width=0.1, head_length=0.2, color='r')

    if 'BL' not in fit.model.model_type:
        figure_axes.scatter(0, 0, s=10, c='k')
    if 'PS' not in fit.model.model_type:
        index_source = np.where((trajectory_x ** 2 + trajectory_y ** 2) ** 0.5 < max(1, pyLIMA_parameters.uo + 0.1))[0][
            0]
        source_disk = plt.Circle((trajectory_x[index_source], trajectory_y[index_source]), pyLIMA_parameters.rho,
                                 color='y')
        figure_axes.add_artist(source_disk)

    figure_axes.axis(
        [- figure_trajectory_xlimit, figure_trajectory_xlimit, - figure_trajectory_ylimit, figure_trajectory_ylimit])

    raw_labels = fit.model.model_dictionnary.keys()
    column_labels = ['Parameters 16%', 'Parameters 50%', 'Parameters 84%']
    table_val = []
    for i in xrange(len(fit.model.model_dictionnary.keys())):
        table_val.append([np.percentile(best_chains[:, i], 16), np.percentile(best_chains[:, i], 50),
                          np.percentile(best_chains[:, i], 84)])

    table_val = np.round(table_val, 5).tolist()

    table_colors = []
    raw_colors = []
    last_color = 'dodgerblue'
    for i in xrange(len(table_val)):

        table_colors.append([last_color, last_color, last_color])
        raw_colors.append(last_color)

        if last_color == 'dodgerblue':

            last_color = 'white'

        else:

            last_color = 'dodgerblue'

    table_axes = figure_trajectory.add_subplot(122, frameon=False)

    the_table = table_axes.table(cellText=table_val, cellColours=table_colors, rowColours=raw_colors,
                                 rowLabels=raw_labels,
                                 colLabels=column_labels, loc='center left')
    table_axes.get_yaxis().set_visible(False)
    table_axes.get_xaxis().set_visible(False)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(20)
    figure_trajectory.suptitle(fit.model.event.name + ' : ' + fit.model.model_type, fontsize=50)
    return figure_trajectory
