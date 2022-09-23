import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import cycler
import matplotlib
from bokeh.models import Arrow, OpenHead

from pyLIMA.toolbox import fake_telescopes, plots
import pyLIMA.fits.objective_functions
from pyLIMA.parallax import parallax
from pyLIMA.astrometry import astrometric_positions

plot_lightcurve_windows = 0.2
plot_residuals_windows = 0.21
MAX_PLOT_TICKS = 2
MARKER_SYMBOLS = np.array([['o', '.', '*', 'v', '^', '<', '>', 's', 'p', 'd', 'x'] * 10])

list_of_fake_telescopes = []


def plot_astrometry(microlensing_model, model_parameters):


    # Change matplotlib default colors
    n_telescopes = len(microlensing_model.event.telescopes)
    color = plt.cm.jet(np.linspace(0.01, 0.99, n_telescopes))  # This returns RGBA; convert:
    # hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
    #                tuple(color[:, 0:-1]))
    hexcolor = ['#' + format(int(i[0] * 255), 'x').zfill(2) + format(int(i[1] * 255), 'x').zfill(2) +
                format(int(i[2] * 255), 'x').zfill(2) for i in color]

    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler(color=hexcolor)

    figure = plt.figure()
    figure_ax = plt.axes()
    if len(model_parameters) != len(microlensing_model.model_dictionnary):

        telescopes_fluxes = microlensing_model.find_telescopes_fluxes(model_parameters)

        model_parameters = np.r_[model_parameters,telescopes_fluxes]

    plot_astrometric_models(figure_ax, microlensing_model, model_parameters)
    plot_astrometric_data(figure_ax, microlensing_model)

    legend = figure_ax.legend(shadow=True, fontsize='x-large', bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                                   mode="expand", borderaxespad=0, ncol=3)


def plot_lightcurves(microlensing_model, model_parameters):

    # Change matplotlib default colors
    n_telescopes = len(microlensing_model.event.telescopes)
    color = plt.cm.jet(np.linspace(0.01, 0.99, n_telescopes))  # This returns RGBA; convert:
    # hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255),
    #                tuple(color[:, 0:-1]))
    hexcolor = ['#' + format(int(i[0] * 255), 'x').zfill(2) + format(int(i[1] * 255), 'x').zfill(2) +
                format(int(i[2] * 255), 'x').zfill(2) for i in color]

    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler(color=hexcolor)

    figure, figure_axes = initialize_light_curves_plot(event_name=microlensing_model.event.name)

    if len(model_parameters) != len(microlensing_model.model_dictionnary):

        telescopes_fluxes = microlensing_model.find_telescopes_fluxes(model_parameters)

        model_parameters = np.r_[model_parameters,telescopes_fluxes]

    plot_photometric_models(figure_axes[0], microlensing_model, model_parameters, plot_unit='Mag')
    plot_aligned_data(figure_axes[0], microlensing_model, model_parameters, plot_unit='Mag')
    plot_residuals(figure_axes[1], microlensing_model, model_parameters, plot_unit='Mag')
    figure_axes[0].invert_yaxis()
    figure_axes[1].invert_yaxis()
    legend = figure_axes[0].legend(shadow=True, fontsize='x-large', bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
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

    if len(list_of_fake_telescopes) == 0:

        Earth = True

        for tel in microlensing_model.event.telescopes:

            if tel.lightcurve_flux is not None:

                if tel.location == 'Space':

                    model_time = np.arange(np.min(tel.lightcurve_magnitude['time'].value),
                                              np.max(tel.lightcurve_magnitude['time'].value),
                                             0.01)
                else:

                    model_time = np.arange(np.min((np.min(tel.lightcurve_magnitude['time'].value),pyLIMA_parameters.t0 - 5 * pyLIMA_parameters.tE)),
                                           np.max((np.max(tel.lightcurve_magnitude['time'].value),pyLIMA_parameters.t0 + 5 * pyLIMA_parameters.tE)),
                                           0.01)

                model_lightcurve = np.c_[model_time, [0] * len(model_time), [0] * len(model_time)]
                model_telescope = fake_telescopes.create_a_fake_telescope(light_curve = model_lightcurve)

                model_telescope.name = tel.name
                model_telescope.filter = tel.filter
                model_telescope.location = tel.location
                model_telescope.gamma = tel.gamma
                model_telescope.location = tel.location
                model_telescope.spacecraft_name = tel.spacecraft_name

                if tel.location == 'Space':

                    model_telescope.spacecraft_name = tel.spacecraft_name
                    model_telescope.spacecraft_positions = tel.spacecraft_positions

                    if microlensing_model.parallax_model != 'None':

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

                    if microlensing_model.parallax_model != 'None':
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

def plot_photometric_models(figure_axe, microlensing_model, model_parameters, plot_unit='Mag'):

    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    list_of_telescopes = create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters)
    telescopes_names = np.array([i.name for i in microlensing_model.event.telescopes])

    #plot models
    for tel in list_of_telescopes:

        if tel.lightcurve_flux is not None:

            model = microlensing_model.compute_the_microlensing_model(tel, pyLIMA_parameters)

            magnitude = pyLIMA.toolbox.brightness_transformation.ZERO_POINT-2.5*np.log10(model['photometry'])

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
    index = 0
    for ind, tel in enumerate(microlensing_model.event.telescopes):

        if tel.lightcurve_flux is not None:

            residus_in_mag = pyLIMA.fits.objective_functions.photometric_residuals_in_magnitude(tel, microlensing_model, pyLIMA_parameters)

            model_magnification = microlensing_model.model_magnification(tel, pyLIMA_parameters)
            f_source, f_blend = microlensing_model.derive_telescope_flux(tel, pyLIMA_parameters, model_magnification)

            if tel.location == 'Space':

                magnitude = pyLIMA.toolbox.brightness_transformation.ZERO_POINT - 2.5 * np.log10(
                    f_source * model_magnification + f_blend)

            else:

                if index == 0:

                    ref_source = f_source
                    ref_blend = f_blend
                    index += 1
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

        if tel.lightcurve_flux is not None:

            residus_in_mag = pyLIMA.fits.objective_functions.photometric_residuals_in_magnitude(tel, microlensing_model, pyLIMA_parameters)

            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][ind]
            marker = str(MARKER_SYMBOLS[0][ind])

            plots.plot_light_curve_magnitude(tel.lightcurve_magnitude['time'].value,
                                             residus_in_mag,tel.lightcurve_magnitude['err_mag'].value,
                                             figure_axe=figure_axe, color=color, marker=marker, name=tel.name)

    figure_axe.set_ylim([-0.1, 0.1])


def plot_geometry(microlensing_model, model_parameters):
    """Plot the lensing geometry (i.e source trajectory) and the table of best parameters.
    :param object fit: a fit object. See the microlfits for more details.
    :param list best_parameters: a list containing the model you want to plot the trajectory
    """
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

    for telescope in faketelescopes:

        if telescope.lightcurve_flux is not None:
            platform = 'Earth'

            if telescope.location == 'Space':

                platform = telescope.name

            reference_telescope = telescope

            telescope_index = [i for i in range(len(microlensing_model.event.telescopes)) if
                               microlensing_model.event.telescopes[i].name == telescope.name][0]

            trajectory_x, trajectory_y, dseparation = microlensing_model.source_trajectory(telescope,pyLIMA_parameters,
                                                                                                         data_type='photometry')


            figure_axes.plot(trajectory_x, trajectory_y,
                             c=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],
                             label=platform)

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
                                                         color=plt.rcParams["axes.prop_cycle"].by_key()["color"][
                                                             telescope_index]))

                except:

                    pass

            if microlensing_model.model_type == 'DSPL':

                _, _, trajectory_x, trajectory_y = microlensing_model.sources_trajectory(reference_telescope,
                                                                                               pyLIMA_parameters)

                figure_axes.plot(trajectory_x, trajectory_y,
                                 c=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index], alpha=0.5)

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

                except AttributeError:

                    pass

        else:

            figure_axes.scatter(0, 0, s=10, c='r')

            einstein_ring = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--')
            figure_axes.add_artist(einstein_ring)

    for telescope_index, telescope in enumerate(microlensing_model.event.telescopes):

        if telescope.lightcurve_flux is not None:

            trajectory_x, trajectory_y, separation = microlensing_model.source_trajectory(telescope,
                                                                                          pyLIMA_parameters,
                                                                                          data_type='photometry')

            if 'rho' in pyLIMA_parameters._fields:

                rho = pyLIMA_parameters.rho

            else:

                rho = 10 ** -3

            patches = [plt.Circle((x, y), rho, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],
                                  alpha=0.2) for x, y in zip(trajectory_x, trajectory_y)]
            coll = matplotlib.collections.PatchCollection(patches, match_original=True)

            figure_axes.scatter(trajectory_x, trajectory_y,
                                c=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],
                                alpha=0.5, label=telescope.name, s=0.1)

            figure_axes.add_collection(coll)

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

            figure_axes.plot([0.8, 0.8 + east[0]], [0.8, 0.8 + east[1]], 'k', transform=plt.gca().transAxes)
            Ecoords = [0, 0.15]
            Ecoords = np.dot(rota_mat, Ecoords)
            figure_axes.text(0.8 + Ecoords[0], 0.8 + Ecoords[1], 'E', c='k', transform=plt.gca().transAxes, size=25)

            figure_axes.plot([0.8, 0.8 + north[0]], [0.8, 0.8 + north[1]], 'k', transform=plt.gca().transAxes)
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

    return figure_trajectory


def bokeh_plot_geometry(microlensing_model, model_parameters):
    """Plot the lensing geometry (i.e source trajectory) and the table of best parameters.
    :param object fit: a fit object. See the microlfits for more details.
    :param list best_parameters: a list containing the model you want to plot the trajectory
    """

    bokeh_geometry = plt.figure(width=350, height=350, x_range=(-3, 3), y_range=(-3, 3), toolbar_location=None,
                                x_axis_label='x [' + u'\u03B8\u2091'']', y_axis_label='y [' + u'\u03B8\u2091'']'
                                )

    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    fake_telescopes = create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters)


    for telescope in fake_telescopes:

        platform = 'Earth'

        if telescope.location == 'Space':
            platform = telescope.name

        reference_telescope = telescope

        telescope_index = [i for i in range(len(microlensing_model.event.telescopes)) if
                           microlensing_model.event.telescopes[i].name == telescope.name][0]

        trajectory_x, trajectory_y, dseparation = microlensing_model.source_trajectory(telescope,
                                                                                       pyLIMA_parameters,
                                                                                       data_type='photometry')
        bokeh_geometry.line(trajectory_x, trajectory_y,
                            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],
                            legend=platform)

        for index in [-1, 0, 1]:

            try:
                index = np.argmin(np.abs(telescope.lightcurve_magnitude['time'].value -
                                         (pyLIMA_parameters.t0 + index * pyLIMA_parameters.tE)))
                sign = np.sign(trajectory_x[index + 1] - trajectory_x[index])
                derivative = (trajectory_y[index - 1] - trajectory_y[index + 1]) / (
                        trajectory_x[index - 1] - trajectory_x[index + 1])


                bokeh_geometry.add_layout(Arrow(end=OpenHead(line_color=
                                                             plt.rcParams["axes.prop_cycle"].by_key()["color"][
                                                                 telescope_index]),
                                                x_start=trajectory_x[index], y_start=trajectory_y[index],
                                                x_end=trajectory_x[index] + sign * 0.001,
                                                y_end=trajectory_y[index] + sign * 0.001 * derivative
                                                ))
            except:
                pass

        if microlensing_model.model_type == 'DSPL':
            _, _, trajectory_x, trajectory_y = microlensing_model.sources_trajectory(reference_telescope,
                                                                                     pyLIMA_parameters)

            bokeh_geometry.line(trajectory_x, trajectory_y,
                                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index], alpha=0.5,
                                legend=platform)

    if 'BL' in microlensing_model.model_type:

        from pyLIMA.caustics import binary_caustics

        regime, caustics, cc = binary_caustics.find_2_lenses_caustics_and_critical_curves(
            pyLIMA_parameters.separation,
            pyLIMA_parameters.mass_ratio,
            resolution=5000)

        for count, caustic in enumerate(caustics):

            try:

                bokeh_geometry.line(caustic.real, caustic.imag,
                                    color='red', line_width=3)
                bokeh_geometry.line(cc[count].real, cc[count].imag, line_dash='dashed',
                                    color='black')
            except AttributeError:

                pass

    else:

        bokeh_geometry.scatter(0, 0, color='red')

        einstein_ring = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--')

        bokeh_geometry.circle(0, 0, radius=1, line_dash='dashed', line_color='black', fill_color=None)

    for telescope_index, telescope in enumerate(microlensing_model.event.telescopes):

        trajectory_x, trajectory_y, separation = microlensing_model.source_trajectory(telescope,
                                                                                      pyLIMA_parameters)

        if 'rho' in pyLIMA_parameters._fields:

            rho = pyLIMA_parameters.rho

        else:

            rho = 10 ** -3


        bokeh_geometry.circle(trajectory_x, trajectory_y, radius=rho,
                              color=plt.rcParams["axes.prop_cycle"].by_key()["color"][telescope_index],
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

    return bokeh_geometry


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