import matplotlib.pyplot as plt
import numpy as np
from pyLIMA.toolbox import fake_telescopes, plots
import pyLIMA.fits.objective_functions
from matplotlib.ticker import MaxNLocator

import cycler
import matplotlib


plot_lightcurve_windows = 0.2
plot_residuals_windows = 0.21
MAX_PLOT_TICKS = 2
MARKER_SYMBOLS = np.array([['o', '.', '*', 'v', '^', '<', '>', 's', 'p', 'd', 'x'] * 10])



def standard_light_curves_plot(microlensing_model, model_parameters):

    # Change matplotlib default colors
    n_telescopes = len(microlensing_model.event.telescopes)
    color = plt.cm.jet(np.linspace(0.01, 0.99, n_telescopes))  # This returns RGBA; convert:
    # hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
    #                tuple(color[:, 0:-1]))
    hexcolor = ['#' + format(int(i[0] * 255), 'x').zfill(2) + format(int(i[1] * 255), 'x').zfill(2) +
                format(int(i[2] * 255), 'x').zfill(2) for i in color]

    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler(color=hexcolor)

    figure, figure_axes = initialize_light_curves_plot(event_name=microlensing_model.event.name)
    plot_models(figure_axes[0], microlensing_model, model_parameters, plot_unit='Mag')
    plot_aligned_data(figure_axes[0], microlensing_model, model_parameters, plot_unit='Mag')
    plot_residuals(figure_axes[1], microlensing_model, model_parameters, plot_unit='Mag')
    figure_axes[0].invert_yaxis()
    figure_axes[1].invert_yaxis()
    legend = figure_axes[0].legend(shadow=True, fontsize='x-large',bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                                   mode="expand", borderaxespad=0, ncol=3)

def initialize_light_curves_plot(plot_unit='Mag', event_name='A microlensing event'):


    fig_size = [10, 10]
    figure, figure_axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                                       figsize=(fig_size[0], fig_size[1]), dpi=75)
    plt.subplots_adjust(top=0.84, bottom=0.15, left=0.20, right=0.99, wspace=0.2, hspace=0.1)
    figure_axes[0].grid()
    figure_axes[1].grid()
    figure.suptitle(event_name, fontsize=30 * fig_size[0] / len(event_name))

    figure_axes[0].set_ylabel(plot_unit, fontsize=5 * fig_size[1] * 3 / 4.0)
    figure_axes[0].yaxis.set_major_locator(MaxNLocator(4))
    figure_axes[0].tick_params(axis='y', labelsize=3.5 * fig_size[1] * 3 / 4.0)

    figure_axes[0].text(0.01, 0.96, 'provided by pyLIMA', style='italic', fontsize=10,
                        transform=figure_axes[0].transAxes)

    figure_axes[1].set_xlabel('JD', fontsize=5 * fig_size[0] * 3 / 4.0)
    figure_axes[1].xaxis.set_major_locator(MaxNLocator(3))
    figure_axes[1].yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))

    figure_axes[1].ticklabel_format(useOffset=False, style='plain')
    figure_axes[1].set_ylabel('Residuals', fontsize=5 * fig_size[1] * 2 / 4.0)
    figure_axes[1].tick_params(axis='x', labelsize=3.5 * fig_size[0] * 3 / 4.0)
    figure_axes[1].tick_params(axis='y', labelsize=3.5 * fig_size[1] * 3 / 4.0)

    return figure, figure_axes

def create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters):

    list_of_telescopes = []

    Earth = True

    for tel in microlensing_model.event.telescopes:

        if tel.location == 'Space':

            model_time = np.arange(np.min(tel.lightcurve_magnitude['time'].value),
                                     np.max(tel.lightcurve_magnitude['time'].value),
                                     0.01)
        else:

            model_time = np.arange(pyLIMA_parameters.to - 5 * pyLIMA_parameters.tE,
                                   pyLIMA_parameters.to + 5 * pyLIMA_parameters.tE,
                                   0.01)

        model_lightcurve = np.c_[model_time, [0] * len(model_time), [0] * len(model_time)]
        model_telescope = fake_telescopes.create_a_fake_telescope(model_lightcurve)

        model_telescope.name = tel.name
        model_telescope.filter = tel.filter
        model_telescope.location = tel.location

        if tel.location == 'Space':

            model_telescope.spacecraft_name = tel.spacecraft_name
            model_telescope.spacecraft_positions = tel.spacecraft_positions

            if microlensing_model.event.parallax_model != 'None':
                model_telescope.compute_parallax(microlensing_model.event.parallax_model)

            list_of_telescopes.append(model_telescope)

        if tel.location == 'Earth' and Earth:

            if microlensing_model.event.parallax_model != 'None':
                model_telescope.compute_parallax(microlensing_model.event.parallax_model)

            list_of_telescopes.append(model_telescope)
            Earth = False

    return list_of_telescopes


def plot_models(figure_axe, microlensing_model, model_parameters, plot_unit='Mag'):

    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    list_of_telescopes = create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters)
    telescopes_names = np.array([i.name for i in microlensing_model.event.telescopes])
    #plot models

    for tel in list_of_telescopes:

        model = microlensing_model.compute_the_microlensing_model(tel, pyLIMA_parameters)

        magnitude = pyLIMA.toolbox.brightness_transformation.ZERO_POINT-2.5*np.log10(model['flux'])

        name = tel.name

        index_color = np.where(name == telescopes_names)[0][0]
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][index_color]

        if tel.location == 'Earth':

            name = tel.location

        plots.plot_light_curve_magnitude(tel.lightcurve_magnitude['time'].value,
                                         magnitude, figure_axe=figure_axe, name=name, color=color)

def plot_aligned_data(figure_axe, microlensing_model, model_parameters, plot_unit='Mag'):

    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    #plot aligned data
    for ind, tel in enumerate(microlensing_model.event.telescopes):

        residus_in_mag = pyLIMA.fits.objective_functions.photometric_residuals_in_magnitude(tel, microlensing_model, pyLIMA_parameters)

        model_magnification = microlensing_model.model_magnification(tel, pyLIMA_parameters)
        f_source, f_blend = microlensing_model.derive_telescope_flux(tel, pyLIMA_parameters, model_magnification)

        if tel.location == 'Space':

            magnitude = pyLIMA.toolbox.brightness_transformation.ZERO_POINT - 2.5 * np.log10(
                f_source * model_magnification + f_blend)

        else:

            if ind == 0:
                ref_source = f_source
                ref_blend = f_blend

            magnitude = pyLIMA.toolbox.brightness_transformation.ZERO_POINT-2.5*np.log10(ref_source*model_magnification+ref_blend)

        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][ind]
        marker = str(MARKER_SYMBOLS[0][ind])

        plots.plot_light_curve_magnitude(tel.lightcurve_magnitude['time'].value,
                                         magnitude+residus_in_mag,
                                         tel.lightcurve_magnitude['err_mag'].value,
                                         figure_axe=figure_axe, color=color, marker=marker, name=tel.name)

def plot_residuals(figure_axe, microlensing_model, model_parameters, plot_unit='Mag'):

    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    #plot residuals

    for ind, tel in enumerate(microlensing_model.event.telescopes):

        residus_in_mag = pyLIMA.fits.objective_functions.photometric_residuals_in_magnitude(tel, microlensing_model, pyLIMA_parameters)

        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][ind]
        marker = str(MARKER_SYMBOLS[0][ind])

        plots.plot_light_curve_magnitude(tel.lightcurve_magnitude['time'].value,
                                         residus_in_mag,tel.lightcurve_magnitude['err_mag'].value,
                                         figure_axe=figure_axe, color=color, marker=marker, name=tel.name)

