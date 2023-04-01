import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import cycler
import matplotlib

from bokeh.io import output_file, show
from bokeh.layouts import gridplot, grid, layout, row
from bokeh.plotting import figure, curdoc
from bokeh.transform import linear_cmap, log_cmap
from bokeh.util.hex import hexbin
from bokeh.models import BasicTickFormatter
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import ColumnDataSource
from bokeh.models import Arrow, OpenHead
from bokeh.models import Circle

import pygtc


from pyLIMA.toolbox import fake_telescopes, plots
import pyLIMA.fits.objective_functions
from pyLIMA.parallax import parallax
from pyLIMA.astrometry import astrometric_positions




plot_lightcurve_windows = 0.2
plot_residuals_windows = 0.21
MAX_PLOT_TICKS = 2
MARKER_SYMBOLS = np.array([['o', '.', '*', 'v', '^', '<', '>', 's', 'p', 'd', 'x'] * 10])

list_of_fake_telescopes = []



def plot_geometry(microlensing_model, model_parameters, bokeh_plot=None):
    """Plot the lensing geometry (i.e source trajectory) and the table of best parameters.
    :param object fit: a fit object. See the microlfits for more details.
    :param list best_parameters: a list containing the model you want to plot the trajectory
    """

    list_of_fake_telescopes = []

    # Change matplotlib default colors
    n_telescopes = len(microlensing_model.event.telescopes)
    color = plt.cm.jet(np.linspace(0.01, 0.99, n_telescopes))  # This returns RGBA; convert:
    # hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
    #                tuple(color[:, 0:-1]))
    hexcolor = ['#' + format(int(i[0] * 255), 'x').zfill(2) + format(int(i[1] * 255), 'x').zfill(2) +
                format(int(i[2] * 255), 'x').zfill(2) for i in color]

    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler(color=hexcolor)
    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    faketelescopes = create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters)

    fig_size = [10, 10]
    figure_trajectory = plt.figure(figsize=(fig_size[0], fig_size[1]), dpi=75)

    figure_axes = figure_trajectory.add_subplot(111, aspect=1)
    plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)

    if bokeh_plot is not None:

        bokeh_geometry = figure(width=350, height=350, x_range=(-3, 3), y_range=(-3, 3),
                                x_axis_label='x [' + u'\u03B8\u2091'']', y_axis_label='y [' + u'\u03B8\u2091'']')

    else:

        bokeh_geometry =  None

    for telescope in faketelescopes:

        if telescope.lightcurve_flux is not None:
            platform = 'Earth'

            if telescope.location == 'Space':

                platform = telescope.name
                linestyle = '--'

            else:

                linestyle = '-'
                
            reference_telescope = telescope

            telescope_index = [i for i in range(len(microlensing_model.event.telescopes)) if
                               microlensing_model.event.telescopes[i].name == telescope.name][0]

            trajectory_x, trajectory_y, dseparation = microlensing_model.source_trajectory(telescope,pyLIMA_parameters,
                                                                                                         data_type='photometry')

            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index]
            figure_axes.plot(trajectory_x, trajectory_y,
                             c=color,
                             label=platform,linestyle=linestyle)

            bokeh_geometry.line(trajectory_x, trajectory_y,
                                color=color,
                                legend_label=platform)

            for index in [-1, 0, 1]:

                try:
                    index = np.argmin(np.abs(telescope.lightcurve_magnitude['time'].value -
                                             (pyLIMA_parameters.t0 + index * pyLIMA_parameters.tE)))
                    sign = np.sign(trajectory_x[index + 1] - trajectory_x[index])
                    derivative = (trajectory_y[index - 1] - trajectory_y[index + 1]) / (
                            trajectory_x[index - 1] - trajectory_x[index + 1])

                    figure_axes.annotate('', xy=(trajectory_x[index], trajectory_y[index]),
                                         xytext=(trajectory_x[index] - sign * 0.001,
                                                 trajectory_y[index] - sign * 0.001 * derivative),
                                         arrowprops=dict(arrowstyle="->", mutation_scale=35,
                                                         color=color))
                    oh = OpenHead(line_color=color, line_width=1)

                    bokeh_geometry.add_layout(Arrow(end=oh,
                                                    x_start=trajectory_x[index], y_start=trajectory_y[index],
                                                    x_end=trajectory_x[index] + sign * 0.001,
                                                    y_end=trajectory_y[index] + sign * 0.001 * derivative))


                except:

                    pass

            if microlensing_model.model_type == 'DSPL':

                _, _, trajectory_x, trajectory_y = microlensing_model.sources_trajectory(reference_telescope,
                                                                                               pyLIMA_parameters)

                figure_axes.plot(trajectory_x, trajectory_y,
                                 c=color, alpha=0.5)


                bokeh_geometry.line(trajectory_x, trajectory_y,
                                color=color, alpha=0.5)

        if 'BL' in microlensing_model.model_type:

            from pyLIMA.caustics import binary_caustics

            regime, caustics, cc = binary_caustics.find_2_lenses_caustics_and_critical_curves(
                pyLIMA_parameters.separation,
                pyLIMA_parameters.mass_ratio,
                resolution=5000)

            center_of_mass = pyLIMA_parameters.separation*pyLIMA_parameters.mass_ratio/(1+pyLIMA_parameters.mass_ratio)
            plt.scatter(-center_of_mass,0,s=10,c='k')
            plt.scatter(-center_of_mass+pyLIMA_parameters.separation, 0, s=10, c='k')

            for count, caustic in enumerate(caustics):

                try:
                    figure_axes.plot(caustic.real, caustic.imag, lw=3, c='r')
                    figure_axes.plot(cc[count].real, cc[count].imag, '--k')
                    bokeh_geometry.line(caustic.real, caustic.imag,
                                        color='red', line_width=3)
                    bokeh_geometry.line(cc[count].real, cc[count].imag, line_dash='dashed',
                                        color='black')

                except AttributeError:

                    pass

        else:

            figure_axes.scatter(0, 0, s=10, c='r')
            bokeh_geometry.scatter(0, 0, color='red')

            einstein_ring = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--')
            figure_axes.add_artist(einstein_ring)
            bokeh_geometry.circle(0, 0, radius=1, line_dash='dashed', line_color='black', fill_color=None)

    for telescope_index, telescope in enumerate(microlensing_model.event.telescopes):

        if telescope.lightcurve_flux is not None:

            trajectory_x, trajectory_y, separation = microlensing_model.source_trajectory(telescope,
                                                                                          pyLIMA_parameters,
                                                                                          data_type='photometry')

            if 'rho' in pyLIMA_parameters._fields:

                rho = pyLIMA_parameters.rho

            else:

                rho = 10 ** -5


            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index]

            patches = [plt.Circle((x, y), rho, color=color,
                                  alpha=0.2) for x, y in zip(trajectory_x, trajectory_y)]
            coll = matplotlib.collections.PatchCollection(patches, match_original=True)

            figure_axes.scatter(trajectory_x, trajectory_y,
                                c=color,
                                alpha=0.5, label=telescope.name, s=0.1)

            figure_axes.add_collection(coll)

            bokeh_geometry.circle(trajectory_x, trajectory_y, radius=rho,
                                  color=color,
                                  radius_dimension='max', fill_alpha=0.5)

        if microlensing_model.parallax_model[0] != 'None':

            piEN = pyLIMA_parameters.piEN
            piEE = pyLIMA_parameters.piEE

            EN_trajectory_angle = parallax.EN_trajectory_angle(piEN, piEE)

            plot_angle = -(EN_trajectory_angle)

            try:

                plot_angle += pyLIMA_parameters.alpha

            except:

                pass

            north = [0.1, 0]
            east = [0, 0.1]

            rota_mat = np.array([[np.cos(plot_angle), -np.sin(plot_angle)], [np.sin(plot_angle), np.cos(plot_angle)]])
            east = np.dot(rota_mat, east)
            north = np.dot(rota_mat, north)

            figure_axes.plot([0.8, 0.8 + east[0]], [0.8, 0.8 + east[1]], 'k',linestyle='--', transform=plt.gca().transAxes)
            bokeh_geometry.line([0.8, 0.8 + east[0]], [0.8, 0.8 + east[1]], line_dash='dashed',color='black')

            Ecoords = [0, 0.15]
            Ecoords = np.dot(rota_mat, Ecoords)
            figure_axes.text(0.8 + Ecoords[0], 0.8 + Ecoords[1], 'E', c='k', transform=plt.gca().transAxes,
                             size=25)

            figure_axes.plot([0.8, 0.8 + north[0]], [0.8, 0.8 + north[1]], 'k', transform=plt.gca().transAxes)
            bokeh_geometry.line([0.8, 0.8 + north[0]], [0.8, 0.8 + north[1]],  color='black')

            Ncoords = [0.15, 0.0]
            Ncoords = np.dot(rota_mat, Ncoords)
            figure_axes.text(0.8 + Ncoords[0], 0.8 + Ncoords[1], 'N', c='k', transform=plt.gca().transAxes, size=25)

    legend = figure_axes.legend(numpoints=1, loc='best', fancybox=True, framealpha=0.5)

    for handle in legend.legendHandles:

        try:

            handle.set_sizes([100])

        except:

            pass

    figure_axes.xaxis.set_major_locator(MaxNLocator(5))
    figure_axes.yaxis.set_major_locator(MaxNLocator(5))
    figure_axes.xaxis.get_major_ticks()[0].draw = lambda *args: None
    figure_axes.yaxis.get_major_ticks()[0].draw = lambda *args: None
    figure_axes.xaxis.get_major_ticks()[-1].draw = lambda *args: None
    figure_axes.yaxis.get_major_ticks()[-1].draw = lambda *args: None

    figure_axes.set_xlabel(r'$x(\theta_E)$', fontsize=25)
    figure_axes.set_ylabel(r'$y(\theta_E)$', fontsize=25)
    figure_axes.tick_params(axis='x', labelsize=15)
    figure_axes.tick_params(axis='y', labelsize=15)

    figure_axes.axis([-3, 3, -3, 3])
    title = microlensing_model.event.name + ' : ' + microlensing_model.model_type
    figure_trajectory.suptitle(title, fontsize=30 * fig_size[0] / len(title))

    return figure_trajectory, bokeh_geometry



def plot_astrometry(microlensing_model, model_parameters):

    list_of_fake_telescopes = []

    # Change matplotlib default colors
    n_telescopes = len(microlensing_model.event.telescopes)
    color = plt.cm.jet(np.linspace(0.01, 0.99, n_telescopes))  # This returns RGBA; convert:
    # hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
    #                tuple(color[:, 0:-1]))
    hexcolor = ['#' + format(int(i[0] * 255), 'x').zfill(2) + format(int(i[1] * 255), 'x').zfill(2) +
                format(int(i[2] * 255), 'x').zfill(2) for i in color]

    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler(color=hexcolor)

    mat_figure = plt.figure()
    mat_figure_ax = plt.axes()
    if len(model_parameters) != len(microlensing_model.model_dictionnary):

        telescopes_fluxes = microlensing_model.find_telescopes_fluxes(model_parameters)

        model_parameters = np.r_[model_parameters,telescopes_fluxes]

    plot_astrometric_models(mat_figure_ax, microlensing_model, model_parameters)
    plot_astrometric_data(mat_figure_ax, microlensing_model)

    legend = mat_figure_ax.legend(shadow=True, fontsize='x-large', bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                                   mode="expand", borderaxespad=0, ncol=3)


def plot_lightcurves(microlensing_model, model_parameters, bokeh_plot=None):

    list_of_fake_telescopes = []

    # Change matplotlib default colors
    n_telescopes = len(microlensing_model.event.telescopes)
    color = plt.cm.jet(np.linspace(0.01, 0.99, n_telescopes))  # This returns RGBA; convert:
    # hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
    #                tuple(color[:, 0:-1]))
    hexcolor = ['#' + format(int(i[0] * 255), 'x').zfill(2) + format(int(i[1] * 255), 'x').zfill(2) +
                format(int(i[2] * 255), 'x').zfill(2) for i in color]

    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler(color=hexcolor)

    mat_figure, mat_figure_axes = initialize_light_curves_plot(event_name=microlensing_model.event.name)

    if bokeh_plot is not None:

        bokeh_lightcurves = figure(width=800, height=600,toolbar_location=None,  y_axis_label='m [mag]')
        bokeh_residuals = figure(width=bokeh_lightcurves.width, height=200, x_range=bokeh_lightcurves.x_range,
                                 y_range=(0.18, -0.18), toolbar_location=None,
                                 x_axis_label='JD', y_axis_label=u'\u0394m [mag]')

        bokeh_lightcurves.xaxis.minor_tick_line_color = None
        bokeh_lightcurves.xaxis.major_tick_line_color = None
        bokeh_lightcurves.xaxis.major_label_text_font_size = '0pt'
        bokeh_lightcurves.y_range.flipped = True
        bokeh_lightcurves.xaxis.formatter = BasicTickFormatter(use_scientific=False)


        bokeh_residuals.xaxis.formatter = BasicTickFormatter(use_scientific=False)
        bokeh_residuals.xaxis.major_label_orientation = np.pi / 4
        bokeh_residuals.xaxis.minor_tick_line_color = None

    else:

        bokeh_lightcurves = None
        bokeh_residuals = None

    if len(model_parameters) != len(microlensing_model.model_dictionnary):

        telescopes_fluxes = microlensing_model.find_telescopes_fluxes(model_parameters)

        model_parameters = np.r_[model_parameters,telescopes_fluxes]

    plot_photometric_models(mat_figure_axes[0], microlensing_model, model_parameters, plot_unit='Mag',
                            bokeh_plot=bokeh_lightcurves)

    plot_aligned_data(mat_figure_axes[0], microlensing_model, model_parameters, plot_unit='Mag',
                      bokeh_plot=bokeh_lightcurves)

    plot_residuals(mat_figure_axes[1], microlensing_model, model_parameters, plot_unit='Mag',
                   bokeh_plot=bokeh_residuals)

    mat_figure_axes[0].invert_yaxis()
    mat_figure_axes[1].invert_yaxis()
    legend = mat_figure_axes[0].legend(shadow=True, fontsize='x-large', bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                                   mode="expand", borderaxespad=0, ncol=3)

    bokeh_lightcurves.legend.click_policy = "mute"
    legend = bokeh_lightcurves.legend[0]

    figure_bokeh = gridplot([[bokeh_lightcurves], [bokeh_residuals]],toolbar_location='above')

    return mat_figure, figure_bokeh

def initialize_light_curves_plot(plot_unit='Mag', event_name='A microlensing event'):


    fig_size = [10, 10]
    mat_figure, mat_figure_axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                                       figsize=(fig_size[0], fig_size[1]), dpi=75)
    plt.subplots_adjust(top=0.84, bottom=0.15, left=0.20, right=0.99, wspace=0.2, hspace=0.1)
    mat_figure_axes[0].grid()
    mat_figure_axes[1].grid()
    mat_figure.suptitle(event_name, fontsize=30 * fig_size[0] / len(event_name))

    mat_figure_axes[0].set_ylabel(plot_unit, fontsize=5 * fig_size[1] * 3 / 4.0)
    mat_figure_axes[0].yaxis.set_major_locator(MaxNLocator(4))
    mat_figure_axes[0].tick_params(axis='y', labelsize=3.5 * fig_size[1] * 3 / 4.0)

    mat_figure_axes[0].text(0.01, 0.96, 'provided by pyLIMA', style='italic', fontsize=10,
                        transform=mat_figure_axes[0].transAxes)

    mat_figure_axes[1].set_xlabel('JD', fontsize=5 * fig_size[0] * 3 / 4.0)
    mat_figure_axes[1].xaxis.set_major_locator(MaxNLocator(3))
    mat_figure_axes[1].yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))

    mat_figure_axes[1].ticklabel_format(useOffset=False, style='plain')
    mat_figure_axes[1].set_ylabel('Residuals', fontsize=5 * fig_size[1] * 2 / 4.0)
    mat_figure_axes[1].tick_params(axis='x', labelsize=3.5 * fig_size[0] * 3 / 4.0)
    mat_figure_axes[1].tick_params(axis='y', labelsize=3.5 * fig_size[1] * 3 / 4.0)

    return mat_figure, mat_figure_axes

def create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters):

    #recreate for bug fixing
    #list_of_fake_telescopes = []

    if len(list_of_fake_telescopes) == 0:

        Earth = True

        for tel in microlensing_model.event.telescopes:

            if tel.lightcurve_flux is not None:

                if tel.location == 'Space':

                    model_time = np.arange(np.min(tel.lightcurve_magnitude['time'].value),
                                              np.max(tel.lightcurve_magnitude['time'].value),
                                             0.01)


                    model_time = np.r_[model_time, tel.lightcurve_magnitude['time'].value]

                    model_time.sort()

                if Earth and tel.location == 'Earth':

                    model_time = np.arange(np.min((np.min(tel.lightcurve_magnitude['time'].value),pyLIMA_parameters.t0 - 5 * pyLIMA_parameters.tE)),
                                           np.max((np.max(tel.lightcurve_magnitude['time'].value),pyLIMA_parameters.t0 + 5 * pyLIMA_parameters.tE)),
                                           0.01)


                    for telescope in microlensing_model.event.telescopes:

                        if telescope.location =='Earth':

                            model_time = np.r_[model_time, telescope.lightcurve_magnitude['time'].value]

                            model_time.sort()

                model_lightcurve = np.c_[model_time, [0] * len(model_time), [0.1] * len(model_time)]
                model_telescope = fake_telescopes.create_a_fake_telescope(light_curve = model_lightcurve)

                model_telescope.name = tel.name
                model_telescope.filter = tel.filter
                model_telescope.location = tel.location
                model_telescope.ld_gamma = tel.ld_gamma
                model_telescope.ld_sigma = tel.ld_sigma
                model_telescope.ld_a1 = tel.ld_a1
                model_telescope.ld_a2 = tel.ld_a2


                model_telescope.location = tel.location
                model_telescope.spacecraft_name = tel.spacecraft_name

                if tel.location == 'Space':

                    model_telescope.spacecraft_name = tel.spacecraft_name
                    model_telescope.spacecraft_positions = tel.spacecraft_positions

                    if microlensing_model.parallax_model[0] != 'None':

                        import pyLIMA.parallax.parallax

                        parallax = pyLIMA.parallax.parallax.MLParallaxes(microlensing_model.event.ra,
                                                                         microlensing_model.event.dec,
                                                                         microlensing_model.parallax_model)

                        model_telescope.compute_parallax(parallax)

                    list_of_fake_telescopes.append(model_telescope)

                if tel.location == 'Earth' and Earth:

                    if microlensing_model.parallax_model[0] != 'None':

                        import pyLIMA.parallax.parallax

                        parallax = pyLIMA.parallax.parallax.MLParallaxes(microlensing_model.event.ra,
                                                                         microlensing_model.event.dec,
                                                                         microlensing_model.parallax_model)

                        model_telescope.compute_parallax(parallax)

                    list_of_fake_telescopes.append(model_telescope)
                    Earth = False

            if tel.astrometry is not None:

                if tel.location == 'Space':

                    model_time = np.arange(np.min(tel.astrometry['time'].value),
                                           np.max(tel.astrometry['time'].value),
                                           0.01)
                else:


                    model_time = np.arange(np.min((np.min(tel.astrometry['time'].value),pyLIMA_parameters.t0 - 5 * pyLIMA_parameters.tE)),
                                           np.max((np.max(tel.astrometry['time'].value),pyLIMA_parameters.t0 + 5 * pyLIMA_parameters.tE)),
                                           0.01)

                    for telescope in microlensing_model.event.telescopes:
                        model_time = np.r_[model_time, telescope.lightcurve_magnitude['time'].value]


                model_astrometry = np.c_[model_time, [0] * len(model_time), [0] * len(model_time),[0] * len(model_time), [0] * len(model_time)]
                model_telescope = fake_telescopes.create_a_fake_telescope(astrometry_curve = model_astrometry)

                model_telescope.name = tel.name
                model_telescope.filter = tel.filter
                model_telescope.location = tel.location
                model_telescope.gamma = tel.gamma
                model_telescope.pixel_scale = tel.pixel_scale

                if tel.location == 'Space':

                    model_telescope.spacecraft_name = tel.spacecraft_name
                    model_telescope.spacecraft_positions = tel.spacecraft_positions

                    if microlensing_model.parallax_model[0] != 'None':

                        import pyLIMA.parallax.parallax

                        parallax = pyLIMA.parallax.parallax.MLParallaxes(microlensing_model.event.ra,
                                                                         microlensing_model.event.dec,
                                                                         microlensing_model.parallax_model)

                        model_telescope.compute_parallax(parallax)

                    list_of_fake_telescopes.append(model_telescope)

                if tel.location == 'Earth':

                    if microlensing_model.parallax_model[0] != 'None':
                        import pyLIMA.parallax.parallax

                        parallax = pyLIMA.parallax.parallax.MLParallaxes(microlensing_model.event.ra,
                                                                         microlensing_model.event.dec,
                                                                         microlensing_model.parallax_model)

                        model_telescope.compute_parallax(parallax)

                    list_of_fake_telescopes.append(model_telescope)

    return list_of_fake_telescopes

def plot_photometric_models(figure_axe, microlensing_model, model_parameters, bokeh_plot = None, plot_unit='Mag'):

    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    list_of_telescopes = create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters)
    telescopes_names = np.array([i.name for i in microlensing_model.event.telescopes])

    #plot models
    index = 0

    for tel in list_of_telescopes:

        if tel.lightcurve_flux is not None:

            model = microlensing_model.compute_the_microlensing_model(tel, pyLIMA_parameters)

            magnitude = pyLIMA.toolbox.brightness_transformation.ZERO_POINT-2.5*np.log10(model['photometry'])
            f_source = model['f_source']
            f_blend = model['f_blend']

            if index == 0:

                ref_source = f_source
                ref_blend = f_blend
                index += 1

            delta_mag = -2.5 * np.log10(f_source + f_blend) + 2.5 * np.log10(ref_source + ref_blend)
            magnitude -= delta_mag

            name = tel.name

            index_color = np.where(name == telescopes_names)[0][0]
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][index_color]

            if tel.location == 'Earth':

                name = tel.location
                linestyle = '-'

            else:

                linestyle='--'

            plots.plot_light_curve_magnitude(tel.lightcurve_magnitude['time'].value,
                                             magnitude, figure_axe=figure_axe, name=name, color=color,linestyle=linestyle)

            if bokeh_plot is not None:

                bokeh_plot.line(tel.lightcurve_magnitude['time'].value, magnitude, legend_label=name, color=color)


def plot_aligned_data(figure_axe, microlensing_model, model_parameters, bokeh_plot=None, plot_unit='Mag'):

    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    #plot aligned data
    index = 0
    index_Earth = 0

    list_of_telescopes = create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters)

    ref_names = []
    ref_locations = []
    ref_magnification = []
    ref_fluxes = []

    for ref_tel in list_of_telescopes:

        model_magnification = microlensing_model.model_magnification(ref_tel, pyLIMA_parameters)
        f_source, f_blend = microlensing_model.derive_telescope_flux(ref_tel, pyLIMA_parameters, model_magnification)


        ref_names.append(ref_tel.name)
        ref_locations.append(ref_tel.location)
        ref_magnification.append(model_magnification)
        ref_fluxes.append([f_source,f_blend])

    for ind, tel in enumerate(microlensing_model.event.telescopes):

        if tel.lightcurve_flux is not None:

            if tel.location == 'Earth':

                ref_index = np.where(np.array(ref_locations) == 'Earth')[0][0]

            else:

                ref_index = np.where(np.array(ref_names) == tel.name)[0][0]

            residus_in_mag = pyLIMA.fits.objective_functions.photometric_residuals_in_magnitude(tel, microlensing_model,
                                                                                                pyLIMA_parameters)

            if ind == 0:

                reference_source = ref_fluxes[ind][0]
                reference_blend = ref_fluxes[ind][1]
                index += 1

            time_mask = [False for i in range(len(ref_magnification[ref_index]))]

            for time in tel.lightcurve_flux['time'].value:

                time_index = np.where(list_of_telescopes[ref_index].lightcurve_flux['time'].value == time)[0][0]
                time_mask[time_index] = True

            model_flux = ref_fluxes[ref_index][0]*ref_magnification[ref_index][time_mask]+ref_fluxes[ref_index][1]
            magnitude = pyLIMA.toolbox.brightness_transformation.ZERO_POINT - 2.5 * np.log10(model_flux)

            delta_mag = -2.5*np.log10( reference_source + reference_blend)+2.5*np.log10(ref_fluxes[ref_index][0] + ref_fluxes[ref_index][1])
            magnitude += delta_mag

            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][ind]
            marker = str(MARKER_SYMBOLS[0][ind])

            plots.plot_light_curve_magnitude(tel.lightcurve_magnitude['time'].value,
                                             magnitude+residus_in_mag,
                                             tel.lightcurve_magnitude['err_mag'].value,
                                             figure_axe=figure_axe, color=color, marker=marker, name=tel.name)

            if bokeh_plot is not None:

                bokeh_plot.scatter(tel.lightcurve_magnitude['time'].value,
                                   magnitude+residus_in_mag,
                                   color=color,
                                   size=5, legend_label=tel.name,
                                   muted_color=color,
                                   muted_alpha=0.2)

                err_xs = []
                err_ys = []

                for x, y, yerr in zip(tel.lightcurve_magnitude['time'].value, magnitude+residus_in_mag,
                                      tel.lightcurve_magnitude['err_mag'].value):
                    err_xs.append((x, x))
                    err_ys.append((y - yerr, y + yerr))

                bokeh_plot.multi_line(err_xs, err_ys, color=color,
                                      legend_label=tel.name,
                                      muted_color=color,
                                      muted_alpha=0.2)

def plot_residuals(figure_axe, microlensing_model, model_parameters, bokeh_plot=None, plot_unit='Mag'):

    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    #plot residuals

    for ind, tel in enumerate(microlensing_model.event.telescopes):

        if tel.lightcurve_flux is not None:

            residus_in_mag = pyLIMA.fits.objective_functions.photometric_residuals_in_magnitude(tel, microlensing_model, pyLIMA_parameters)

            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][ind]
            marker = str(MARKER_SYMBOLS[0][ind])

            plots.plot_light_curve_magnitude(tel.lightcurve_magnitude['time'].value,
                                             residus_in_mag,tel.lightcurve_magnitude['err_mag'].value,
                                             figure_axe=figure_axe, color=color, marker=marker, name=tel.name)

        if bokeh_plot is not None:

            bokeh_plot.scatter(tel.lightcurve_magnitude['time'].value,
                               residus_in_mag,
                               color=color,
                               size=5, legend_label=tel.name,
                               muted_color=color,
                               muted_alpha=0.2)

            err_xs = []
            err_ys = []

            for x, y, yerr in zip(tel.lightcurve_magnitude['time'].value,  residus_in_mag,
                                  tel.lightcurve_magnitude['err_mag'].value):
                err_xs.append((x, x))
                err_ys.append((y - yerr, y + yerr))

            bokeh_plot.multi_line(err_xs, err_ys, color=color,
                                  legend_label=tel.name,
                                  muted_color=color,
                                  muted_alpha=0.2)

    figure_axe.set_ylim([-0.1, 0.1])

def plot_astrometric_models(figure_axe, microlensing_model, model_parameters):

    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    list_of_telescopes = create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters)
    telescopes_names = np.array([i.name for i in microlensing_model.event.telescopes])

    #plot models
    Earth = True

    for tel in list_of_telescopes:

        if tel.astrometry is not None:

            model = microlensing_model.compute_the_microlensing_model(tel, pyLIMA_parameters)

            astrometric_model = model['astrometry']
            lens_E,lens_N = astrometric_positions.lens_astrometric_position(microlensing_model,
                                                                            tel,pyLIMA_parameters)
            name = tel.name

            index_color = np.where(name == telescopes_names)[0][0]
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][index_color]

            figure_axe.plot(astrometric_model[0], astrometric_model[1], c=color)

            if Earth is True:

                source_E, source_N = astrometric_positions.astrometric_position(tel, pyLIMA_parameters)

                figure_axe.plot(source_E, source_N, c='k',label='Source')
                figure_axe.plot(lens_E, lens_N, c='k', linestyle='--',label='Lens')

                for index in [-1, 0, 1]:

                        index_time = np.argmin(np.abs(tel.astrometry['time'].value -
                                                 (pyLIMA_parameters.t0 + index * pyLIMA_parameters.tE)))
                        sign = np.sign(source_E[index_time + 1] - source_E[index_time])
                        derivative = (source_N[index_time - 1] - source_N[index_time + 1]) / (
                                source_E[index_time - 1] - source_E[index_time + 1])

                        figure_axe.annotate('', xy=(source_E[index_time], source_N[index_time]),
                                             xytext=(source_E[index_time] - sign * 0.001,
                                                     source_N[index_time] - sign * 0.001 * derivative),
                                             arrowprops=dict(arrowstyle="->", mutation_scale=35,
                                                             color='k'))

                for index in [-1, 1]:

                        index_time = np.argmin(np.abs(tel.astrometry['time'].value -
                                                 (pyLIMA_parameters.t0 + index * pyLIMA_parameters.tE)))
                        sign = np.sign(lens_E[index_time + 1] - lens_E[index_time])
                        derivative = (lens_N[index_time - 1] - lens_N[index_time + 1]) / (
                                lens_E[index_time - 1] - lens_E[index_time + 1])

                        figure_axe.annotate('', xy=(lens_E[index_time], lens_N[index_time]),
                                            xytext=(lens_E[index_time] - sign * 0.001,
                                                    lens_N[index_time] - sign * 0.001 * derivative),
                                            arrowprops=dict(arrowstyle="->", mutation_scale=35,
                                                            color='k'))


                Earth = False



def plot_astrometric_data(figure_ax, microlensing_model):

    # plot data
    for ind, tel in enumerate(microlensing_model.event.telescopes):

        if tel.astrometry is not None:

            delta_ra = tel.astrometry['delta_ra'].value
            err_ra = tel.astrometry['err_delta_ra'].value

            delta_dec = tel.astrometry['delta_dec'].value
            err_dec = tel.astrometry['err_delta_dec'].value


            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][ind]
            marker = str(MARKER_SYMBOLS[0][ind])

            figure_ax.errorbar(delta_ra,delta_dec,xerr=err_ra,yerr=err_dec,fmt='.',ecolor=color,color=color,
                               label=tel.name )



def plot_distribution(samples,parameters_names=None):

    GTC = pygtc.plotGTC(chains=[samples], sigmaContourLevels=True, paramNames=parameters_names,
                        customLabelFont={'family': 'serif', 'size': 14},
                        customLegendFont={'family': 'serif', 'size': 14},
                        customTickFont={'family': 'serif', 'size': 7}, figureSize=7,nContourLevels=3)

    GTC.tight_layout(pad=0.2,w_pad=0.2,h_pad=0.2)