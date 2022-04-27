import numpy as np

def to_boundaries(event):

    minimum_observing_time_telescopes = []
    maximum_observing_time_telescopes = []

    for telescope in event.telescopes:

        try:

            minimum_observing_time_telescopes.append(min(telescope.lightcurve_flux['time'].value))
            maximum_observing_time_telescopes.append(max(telescope.lightcurve_flux['time'].value))

        except:

            minimum_observing_time_telescopes.append(min(telescope.astrometry['time'].value))
            maximum_observing_time_telescopes.append(max(telescope.astrometry['time'].value))

    to_boundaries = (min(minimum_observing_time_telescopes), max(maximum_observing_time_telescopes))

    return to_boundaries

def delta_to_boundaries():

    return (-150, 150)

def delta_uo_boundaries():

    return (-1.0,1.0)

def uo_boundaries():

    return (0.0,1.0)

def tE_boundaries():

    return (0.1,500)

def rho_boundaries():

    return (5 * 10 ** -5, 0.05)

def q_flux_boundaries():

    return (0.001,1.0)

def logs_boundaries():

    return (-1.0,1.0)

def logq_boundaries():

    return (-6.0,0.0)

def alpha_boundaries():

    return (-np.pi,np.pi)

def piEN_boundaries():

    return (-2.0,2.0)

def piEE_boundaries():

    return (-2.0,2.0)

def dsdt_boundaries():

    return (-10.0,10.0)

def dalphadt_boundaries():

    return (-10.0,10.0)

def theta_E_boundaries():

    return (0.0,10.0)

def fsource_boundaries(flux):

    return (0,np.max(flux))

def fblend_boundaries(flux):

    return (-np.max(flux),np.max(flux))

def gblend_boundaries():

    return (-1.0,1000)


def rescale_photometry_boundaries():

    return (0.0,10.0)

def parameters_boundaries(event, model_dictionnary):
    """ This function define the parameters boundaries for a specific model.

       :param object model: a microlmodels object.

       :return: parameters_boundaries, a list of tuple containing parameters limits
       :rtype: list
    """

    bounds = []
    telescopes_names = [i.name for i in event.telescopes]

    for key in model_dictionnary.keys():

        try:

            function_name = key + '_boundaries()'

            if ('fsource_' in key):

                telescope_ind = np.where(key.split('fsource_')[1]==np.array(telescopes_names))[0][0]
                flux = event.telescopes[telescope_ind].lightcurve_flux['flux'].value
                function_name = key.split('_')[0] + '_boundaries(flux)'

            if ('fblend_' in key):

                telescope_ind = np.where(key.split('fblend_')[1]==np.array(telescopes_names))[0][0]
                flux = event.telescopes[telescope_ind].lightcurve_flux['flux'].value
                function_name = key.split('_')[0] + '_boundaries(flux)'

            if 'to' in key:

                function_name = key + '_boundaries(event)'

            if 'k_photometry' in key:

                function_name = 'rescale_photometry_boundaries()'

            bounds.append(eval(function_name))

        except:

            pass
    return bounds

def telescopes_fluxes_boundaries(event):

    fluxes_boundaries = []

    for telescope in event.telescopes:

        if telescope.lightcurve_flux is not None:

            fluxes_boundaries.append((0,np.max(telescope.lightcurve_flux['flux'].value)))

            if model.blend_flux_parameter == 'fblend':

                fluxes_boundaries.append((-fluxes_boundaries[-1][1], fluxes_boundaries[-1][1]))

            if model.blend_flux_parameter == 'gblend':

                fluxes_boundaries.append((-1, 1000))

    return fluxes_boundaries

def rescaling_photometry_boundaries(model):

    rescaling_boundaries = []

    for telescope in model.event.telescopes:

        if telescope.lightcurve_flux is not None:

            rescaling_boundaries.append((0,10))

    return rescaling_boundaries