import numpy as np


def t0_boundaries():
    # minimum_observing_time_telescopes = []
    # maximum_observing_time_telescopes = []

    # for telescope in event.telescopes:

    #    try:

    #        minimum_observing_time_telescopes.append(min(telescope.lightcurve[
    #        'time'].value))
    #        maximum_observing_time_telescopes.append(max(telescope.lightcurve[
    #        'time'].value))

    #    except:

    #        minimum_observing_time_telescopes.append(min(telescope.astrometry[
    #        'time'].value))
    #        maximum_observing_time_telescopes.append(max(telescope.astrometry[
    #        'time'].value))

    to_boundaries = (2400000, 2500000)

    return to_boundaries


def delta_t0_boundaries():
    return (-150, 150)


def delta_u0_boundaries():
    return (-1.0, 1.0)


def u0_boundaries():
    return (-1.0, 1.0)


def tE_boundaries():
    return (0.1, 500)


def rho_boundaries():
    return (5 * 10 ** -5, 0.05)


def q_flux_boundaries():
    return (0.0, 2)


def separation_boundaries():
    return (0.1, 10.0)


def mass_ratio_boundaries():
    return (10 ** (-6.0), 1.0)


def alpha_boundaries():
    return (0.0, 2 * np.pi)


def piEN_boundaries():
    return (-0.5, 0.5)


def piEE_boundaries():
    return (-0.5, 0.5)


def v_para_boundaries():
    return (-10.0, 10.0)


def v_perp_boundaries():
    return (-10.0, 10.0)


def v_radial_boundaries():
    return (-10.0, 10.0)


def theta_E_boundaries():
    return (0.0, 10.0)


def r_s_boundaries():
    return (-10, 10)


def a_s_boundaries():
    return (0.500001, 10)


def rE_boundaries():
    return (0.1, 100.0)  # AU


def fsource_boundaries(flux):
    return (0.0, np.max(flux))


def fblend_boundaries(flux):
    return (-np.max(flux), np.max(flux))

def ftotal_boundaries(flux):
    return (0, np.max(flux))
def gblend_boundaries():
    return (-1.0, 1000)


def rescale_photometry_boundaries():
    return (-5, 5.)


def rescale_astrometry_boundaries():
    return (-5, 5)


def pi_source_boundaries():
    return (0.01, 10)  # mas


def mu_source_N_boundaries():
    return (-20, 20)  # pixel/yr


def mu_source_E_boundaries():
    return (-20, 20)  # pixel/yr


def position_pixel_boundaries():
    return (-4096, 4096)  # pix


def position_ra_boundaries():
    return (0, 360)  # (0,360) degree


def position_dec_boundaries():
    return (-90, 90)  # (-90,90) degree


def t_center_boundaries():
    return t0_boundaries()


def u_center_boundaries():
    return (-1, 1)

def xi_para_boundaries():
    return (-0.25, 0.25)
def xi_perp_boundaries():
    return (-0.25, 0.25)
def xi_angular_velocity_boundaries():
    return (0.0, 2*np.pi) #/d
def xi_phase_boundaries():
    return (0.0, 2*np.pi)
def xi_inclination_boundaries():
    return (-np.pi/2, np.pi/2)
def xi_mass_ratio_boundaries():
    return (0.0,1)


def parameters_boundaries(event, model_dictionnary):
    """
    Function to find initial DSPL guess

    Parameters
    ----------
    event : object, an event object
    model_dictionnary : dict, a dictionnary containing the parameetrs

    Returns
    -------
    bounds : list, [[bound_min,bound_max]_i] for all i parameters
    """
    bounds = []
    telescopes_names = [i.name for i in event.telescopes]

    for key in model_dictionnary.keys():

        arguments = []

        try:

            function_name = key + '_boundaries'
            if ('rho_' in key):
                function_name = 'rho_boundaries'

            if ('q_flux' in key):
                function_name = 'q_flux_boundaries'

            if ('fsource_' in key):
                telescope_ind = \
                    np.where(key.split('fsource_')[1] == np.array(telescopes_names))[0][
                        0]
                flux = event.telescopes[telescope_ind].lightcurve['flux'].value
                arguments = [flux]
                function_name = key.split('_')[0] + '_boundaries'

            if ('fblend_' in key):
                telescope_ind = \
                    np.where(key.split('fblend_')[1] == np.array(telescopes_names))[0][
                        0]
                flux = event.telescopes[telescope_ind].lightcurve['flux'].value
                arguments = [flux]

                function_name = key.split('_')[0] + '_boundaries'

            if ('ftotal_' in key):
                telescope_ind = \
                    np.where(key.split('ftotal_')[1] == np.array(telescopes_names))[0][
                        0]
                flux = event.telescopes[telescope_ind].lightcurve['flux'].value
                arguments = [flux]

                function_name = key.split('_')[0] + '_boundaries'

            if ('gblend_' in key):
                function_name = key.split('_')[0] + '_boundaries'

            if 'logk_photometry' in key:
                function_name = 'rescale_photometry_boundaries'

            if 'logk_astrometry' in key:
                function_name = 'rescale_astrometry_boundaries'

            if 'position_source' in key:

                try:

                    telescope_ind = np.where(
                        key.split('position_source_N_')[1] == np.array(
                            telescopes_names))[0][0]

                except IndexError:

                    telescope_ind = np.where(
                        key.split('position_source_E_')[1] == np.array(
                            telescopes_names))[0][0]

                if event.telescopes[telescope_ind].astrometry['ra'].unit == 'deg':

                    if 'position_source_E' in key:

                        function_name = 'position_ra_boundaries'

                    else:

                        function_name = 'position_dec_boundaries'


                else:

                    function_name = 'position_pixel_boundaries'

            if 'position_blend' in key:

                try:

                    telescope_ind = np.where(
                        key.split('position_blend_N_')[1] == np.array(
                            telescopes_names))[0][0]

                except IndexError:

                    telescope_ind = np.where(
                        key.split('position_blend_E_')[1] == np.array(
                            telescopes_names))[0][0]

                if event.telescopes[telescope_ind].astrometry['ra'].unit == 'deg':

                    if 'position_source_E' in key:

                        function_name = 'position_ra_boundaries'

                    else:

                        function_name = 'position_dec_boundaries'

                else:

                    function_name = 'position_pixel_boundaries'

            bounds.append(eval(function_name)(*arguments))

        except AttributeError:

            breakpoint()
            pass

    return bounds
