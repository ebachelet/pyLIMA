# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:48:29 2015

@author: ebachelet
"""
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import microlmagnification


class MLPlotter(object):
    def __init__(self, event):

        self.event = event
        self.filters_dictionnary = {'N': 'NIR', 'I': 'NIR', 'Red': 'NIR', 'V': 'V', 'J': 'IR',
                                    'H': 'IR', 'K': 'IR'}
        self.find_observed_bands()
        self.color_dictionnary = {'PSPL': 'k--', 'FSPL': 'r'}

        self.telescopes_reference = {}

        for i in self.observed_bands:

            for j in self.event.telescopes:

                if (self.filters_dictionnary[j.filter] == i) and i not in self.telescopes_reference:

                    self.telescopes_reference[i] = j.name

    def find_observed_bands(self):

        self.observed_bands = np.unique(
            [self.filters_dictionnary[i.filter] for i in self.event.telescopes])
        self.plots_dictionnary = {'NIR': 1}

        if 'V' in self.observed_bands:

            self.plots_dictionnary['V'] = len(self.plots_dictionnary) + 1

        if 'IR' in self.observed_bands:

            self.plots_dictionnary['IR'] = len(self.plots_dictionnary) + 1

    def initialize_plots(self, choice, observe):

        plot_model = self.event.fits_models[choice][2]

        for i in self.observed_bands:

            plot_location = self.plots_dictionnary[i]
            plt.subplot(1, len(self.observed_bands), plot_location)
            plt.title(i, fontsize=20)

            if observe == 'Mag':

                plt.gca().invert_yaxis()
            axes = plt.gca()
            limitmoins = self.event.fits_results[choice][3][
                             plot_model.model_dictionnary['to']] - 1.5 * \
                                                                   self.event.fits_results[choice][
                                                                       3][
                                                                       plot_model.model_dictionnary[
                                                                           'tE']]
            limitplus = self.event.fits_results[choice][3][
                            plot_model.model_dictionnary['to']] + 1.5 * \
                                                                  self.event.fits_results[choice][
                                                                      3][
                                                                      plot_model.model_dictionnary[
                                                                          'tE']]
            axes.set_xlim([limitmoins, limitplus])
        plt.suptitle(self.event.name, fontsize=30)

    def plot_lightcurves_mag(self, align):

        for i in self.event.telescopes:

            time = i.lightcurve[:, 0]
            if time[0] > 2450000:
                time = time - 2450000

            plot_location = self.plots_dictionnary[self.filters_dictionnary[i.filter]]
            plt.subplot(1, len(self.observed_bands), plot_location)

            if align == 'Yes':

                mag = 27.4 - 2.5 * np.log10((i.lightcurve_flux_aligned[:, 1]))
                err_mag = -2.5 * i.lightcurve_flux_aligned[:, 2] / (
                i.lightcurve_flux_aligned[:, 1] * np.log(10))

            else:

                mag = i.lightcurve[:, 1]
                err_mag = i.lightcurve[:, 2]

            plt.errorbar(time, mag, yerr=err_mag, linestyle='none', label=i.name)
            plt.legend(numpoints=1)

    def plot_lightcurves_flux(self, split):

        for i in self.event.telescopes:

            time = i.lightcurve_flux[:, 0]
            if time[0] > 2450000:
                time = time - 2450000

            plot_location = self.plots_dictionnary[self.filters_dictionnary[i.filter]]
            plt.subplot(1, len(self.observed_bands), plot_location)
            plt.errorbar(time, i.lightcurve_flux[:, 1], yerr=i.lightcurve_flux[:, 2],
                         linestyle='none', label=i.name)
            plt.legend(numpoints=1)

    def plot_model_mag(self, choice):

        plot_model = self.event.fits_models[choice][2]

        for i in self.observed_bands:

            plot_location = self.plots_dictionnary[i]
            plt.subplot(1, len(self.observed_bands), plot_location)
            limitmoins = self.event.fits_results[choice][3][
                             plot_model.model_dictionnary['to']] - 1.5 * \
                                                                   self.event.fits_results[choice][
                                                                       3][
                                                                       plot_model.model_dictionnary[
                                                                           'tE']]
            limitplus = self.event.fits_results[choice][3][
                            plot_model.model_dictionnary['to']] + 1.5 * \
                                                                  self.event.fits_results[choice][
                                                                      3][
                                                                      plot_model.model_dictionnary[
                                                                          'tE']]
            t = np.arange(limitmoins, limitplus, 0.01)

            parameters = self.event.fits_results[choice][3][:-1]
            if self.event.telescopes[0].lightcurve[0, 0] > 2450000:

                t = t - 2450000

            ampli = microlmagnification.amplification(plot_model, t, parameters,
                                                      self.event.telescopes[0].gamma)[0]
            fs = parameters[plot_model.model_dictionnary['fs_' + self.telescopes_reference[i]]]
            fb = parameters[plot_model.model_dictionnary['g_' + self.telescopes_reference[i]]] * fs

            plt.plot(t, 27.4 - 2.5 * np.log10(fs * ampli + fb),
                     self.color_dictionnary[self.event.fits_models[choice][2].paczynski_model],
                     lw=2, label=self.event.fits_models[choice][2].paczynski_model)
            plt.legend(numpoints=1)

    def plot_model_mag_uncertainties(self, choice):

        plot_model = self.event.fits_models[choice][2]

        for i in self.observed_bands:

            plot_location = self.plots_dictionnary[i]
            plt.subplot(1, len(self.observed_bands), plot_location)
            limitmoins = self.event.fits_results[choice][3][
                             plot_model.model_dictionnary['to']] - 1.5 * \
                                                                   self.event.fits_results[choice][
                                                                       3][
                                                                       plot_model.model_dictionnary[
                                                                           'tE']]
            limitplus = self.event.fits_results[choice][3][
                            plot_model.model_dictionnary['to']] + 1.5 * \
                                                                  self.event.fits_results[choice][
                                                                      3][
                                                                      plot_model.model_dictionnary[
                                                                          'tE']]
            t = np.arange(limitmoins, limitplus, 0.01)

            uncertain = [-1, 1]
            shadow = []
            for j in uncertain:

                parameters = self.event.fits_results[choice][3][:-1] + j * \
                                                                       self.event.fits_covariance[
                                                                           choice][
                                                                           3].diagonal() ** 0.5
                if self.event.telescopes[0].lightcurve[0, 0] > 2450000:

                    t = t - 2450000

                ampli = microlmagnification.amplification(plot_model, t, parameters,
                                                          self.event.telescopes[0].gamma)[0]

                fs = parameters[plot_model.model_dictionnary['fs_' + self.telescopes_reference[i]]]
                fb = parameters[
                         plot_model.model_dictionnary['g_' + self.telescopes_reference[i]]] * fs

                shadow.append(27.4 - 2.5 * np.log10(fs * ampli + fb))

            plt.fill_between(t, shadow[0], shadow[1], color='blue', alpha='0.2')

    def align_lightcurves(self, choice):

        plot_model = self.event.fits_models[choice][2]

        for i in self.event.telescopes:

            flux_align = (i.lightcurve_flux[:, 1] - self.event.fits_results[choice][3][
                plot_model.model_dictionnary[
                    'fs_' + i.name]] * self.event.fits_results[choice][3][
                              plot_model.model_dictionnary[
                                  'g_' + i.name]]) / self.event.fits_results[choice][3][
                             plot_model.model_dictionnary[
                                 'fs_' + i.name]] * self.event.fits_results[choice][3][
                             plot_model.model_dictionnary[
                                 'fs_' + self.telescopes_reference[
                                     self.filters_dictionnary[i.filter]]]] + \
                         self.event.fits_results[choice][3][plot_model.model_dictionnary[
                             'fs_' + self.telescopes_reference[
                                 self.filters_dictionnary[i.filter]]]] * \
                         self.event.fits_results[choice][3][plot_model.model_dictionnary[
                             'g_' + self.telescopes_reference[self.filters_dictionnary[i.filter]]]]

            err_flux_align = i.lightcurve_flux[:, 2] / i.lightcurve_flux[:, 1] * flux_align
            time = i.lightcurve_flux[:, 0]
            data = np.array([time, flux_align, err_flux_align]).T
            i.lightcurve_flux_aligned = data
