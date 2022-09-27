import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_light_curve_magnitude(time, mag, mag_error=None, figure_axe=None, color=None, marker=None, name=None):

    if figure_axe:

        pass

    else:

        figure, figure_axe = plt.subplots()

    if mag_error is None:

        figure_axe.plot(time, mag, c=color, label=name)

    else:

        figure_axe.errorbar(time, mag, mag_error, color=color, marker=marker, label=name,  linestyle='')


def plot_light_curve_flux(time, flux, flux_error=None, figure_axe=None):

    if figure_axe:

        pass

    else:

        figure, figure_axe = plt.subplots()

    if flux_error is None:

        figure_axe.plot(time, flux)


    else:

        figure_axe.errorbar(time,flux, flux_error, fmt='.')