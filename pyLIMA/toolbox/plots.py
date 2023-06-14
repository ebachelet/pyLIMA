import matplotlib.pyplot as plt


def plot_light_curve_magnitude(time, mag, mag_error=None, figure_axe=None, color=None,
                               linestyle='-', marker=None, name=None):
    """
    Plot a lightcurve in magnitude

    Parameters
    ----------
    time : array, the time to plot
    mag : array, the magnitude to plot
    mag_error : array, the magnitude error
    figure_axe : matplotlib.axe, an axe to plot
    color : str, a color string
    linestyle : str, the matplotlib linestyle desired
    marker : str, the matplotlib marker
    name : str, the points name
    """
    if figure_axe:

        pass

    else:

        figure, figure_axe = plt.subplots()

    if mag_error is None:

        figure_axe.plot(time, mag, c=color, label=name, linestyle=linestyle)

    else:

        figure_axe.errorbar(time, mag, mag_error, color=color, marker=marker,
                            label=name, linestyle='')


def plot_light_curve_flux(time, flux, flux_error=None, figure_axe=None):
    """
    Plot a lightcurve in flux

    Parameters
    ----------
    time : array, the time to plot
    flux : array, the flux to plot
    flux_error : array, the flux error
    figure_axe : matplotlib.axe, an axe to plot
    """

    if figure_axe:

        pass

    else:

        figure, figure_axe = plt.subplots()

    if flux_error is None:

        figure_axe.plot(time, flux)


    else:

        figure_axe.errorbar(time, flux, flux_error, fmt='.')
