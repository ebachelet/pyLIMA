import numpy as np

def parameters_boundaries(model):
    """ This function define the parameters boundaries for a specific model.

       :param object model: a microlmodels object.

       :return: parameters_boundaries, a list of tuple containing parameters limits
       :rtype: list
    """

    minimum_observing_time_telescopes = []
    maximum_observing_time_telescopes = []

    for telescope in model.event.telescopes:

        try:

            minimum_observing_time_telescopes.append(min(telescope.lightcurve_flux['time'].value))
            maximum_observing_time_telescopes.append(max(telescope.lightcurve_flux['time'].value))

        except:

            minimum_observing_time_telescopes.append(min(telescope.astrometry['time'].value))
            maximum_observing_time_telescopes.append(max(telescope.astrometry['time'].value))

    to_boundaries = (min(minimum_observing_time_telescopes), max(maximum_observing_time_telescopes))
    delta_to_boundaries = (-150, 150)
    delta_uo_boundaries = (-1.0, 1.0)
    uo_boundaries = (0.0, 1.0)
    tE_boundaries = (0.1, 500)
    rho_boundaries = (5 * 10 ** -5, 0.05)
    q_flux_boundaries = (0.001, 1.0)

    logs_boundaries = (-1.0, 1.0)
    logq_boundaries = (-6.0, 0.0)
    alpha_boundaries = (-np.pi, np.pi)

    piEN_boundaries = (-2.0, 2.0)
    piEE_boundaries = (-2.0, 2.0)
    XiEN_boundaries = (-2.0, 2.0)
    XiEE_boundaries = (-2.0, 2.0)

    dsdt_boundaries = (-10, 10)
    dalphadt_boundaries = (-10, 10)
    v_boundaries = (-2, 2)
    mass_boundaries = [10 ** -1, 10]
    rE_boundaries = [10 ** -1, 100]
    theta_E_boundaries = (0.0,10.0)

    v_boundaries = (-2, 2)

    ra_xal_boundaries = [0, 360]
    dec_xal_boundaries = [-90, 90]
    period_xal_boundaries = [0.001, 1000]
    ecc_xal_boundaries = [0, 1]
    t_peri_xal_boundaries = to_boundaries
    period_variable = (0.001, 1000)
    phase_variable = (-np.pi, np.pi)
    amplitude_variable = (0.0, 3.0)
    octave_variable = (10 ** -10, 1)
    q_boundaries = (-2, 2)

    # Paczynski models boundaries
    if model.model_type == 'PSPL':
        parameters_boundaries = [to_boundaries, uo_boundaries, tE_boundaries]

    if model.model_type == 'FSPL':
        parameters_boundaries = [to_boundaries, uo_boundaries, tE_boundaries, rho_boundaries]

    if model.model_type == 'DSPL':
        parameters_boundaries = [to_boundaries, uo_boundaries, delta_to_boundaries,
                                 delta_uo_boundaries, tE_boundaries]
        filters = [telescope.filter for telescope in model.event.telescopes]

        unique_filters = np.unique(filters)

        parameters_boundaries += [q_flux_boundaries] * len(unique_filters)

    if model.model_type == 'DFSPL':
        parameters_boundaries = [to_boundaries, uo_boundaries, delta_to_boundaries,
                                 delta_uo_boundaries, tE_boundaries, rho_boundaries, rho_boundaries]
        filters = [telescope.filter for telescope in model.event.telescopes]

        unique_filters = np.unique(filters)

        parameters_boundaries += [q_flux_boundaries] * len(unique_filters)
    if model.model_type == 'PSBL':
        parameters_boundaries = [to_boundaries, uo_boundaries, tE_boundaries, logs_boundaries,
                                 logq_boundaries, alpha_boundaries]

    if (model.model_type == 'USBL') or (model.model_type == 'FSBL'):
        parameters_boundaries = [to_boundaries, uo_boundaries, tE_boundaries, rho_boundaries, logs_boundaries,
                                 logq_boundaries, alpha_boundaries]
        # fluxes = [(0,np.max(telescope.lightcurve_flux[:,1])) for telescope in model.event.telescopes]
        # blend = [(0,100) for telescope in model.event.telescopes]

        # for ind,telo in enumerate(model.event.telescopes):
        # parameters_boundaries+=[fluxes[ind], blend[ind]]


    if model.astrometry:

        parameters_boundaries.append(theta_E_boundaries)

    if model.model_type == 'VariablePL':
        parameters_boundaries = [to_boundaries, uo_boundaries, tE_boundaries, rho_boundaries, period_variable]

        filters = [telescope.filter for telescope in model.event.telescopes]

        unique_filters = np.unique(filters)
        for i in range(model.number_of_harmonics):
            for j in unique_filters:
                parameters_boundaries += [amplitude_variable]
                parameters_boundaries += [phase_variable]
                parameters_boundaries += [octave_variable]

    # Second order boundaries
    if model.parallax_model[0] != 'None':
        parameters_boundaries.append(piEN_boundaries)
        parameters_boundaries.append(piEE_boundaries)

    if model.xallarap_model[0] != 'None':
        parameters_boundaries.append(XiEN_boundaries)
        parameters_boundaries.append(XiEE_boundaries)
        parameters_boundaries.append(ra_xal_boundaries)
        parameters_boundaries.append(dec_xal_boundaries)
        parameters_boundaries.append(period_xal_boundaries)

        if model.xallarap_model[0] != 'Circular':
            parameters_boundaries.append(ecc_xal_boundaries)
            parameters_boundaries.append(t_peri_xal_boundaries)

    if model.orbital_motion_model[0] == '2D':
        parameters_boundaries.append(dsdt_boundaries)
        parameters_boundaries.append(dalphadt_boundaries)

    if model.orbital_motion_model[0] == 'Circular':
        parameters_boundaries.append(dsdt_boundaries)
        parameters_boundaries.append(dsdt_boundaries)
        parameters_boundaries.append(dsdt_boundaries)

    if model.orbital_motion_model[0] == 'Keplerian':
        parameters_boundaries.append(logs_boundaries)
        parameters_boundaries.append(v_boundaries)
        parameters_boundaries.append(v_boundaries)
        parameters_boundaries.append(v_boundaries)
        parameters_boundaries.append(mass_boundaries)
        parameters_boundaries.append(rE_boundaries)

    # if source_spots

    return parameters_boundaries


def telescopes_fluxes_boundaries(model):

    fluxes_boundaries = []

    for telescope in model.event.telescopes:

        fluxes_boundaries.append((0,np.max(telescope.lightcurve_flux['flux'].value)))

        if model.blend_flux_parameter == 'fb':

            fluxes_boundaries.append((-fluxes_boundaries[-1][1], fluxes_boundaries[-1][1]))

        if model.blend_flux_parameter == 'g':

            fluxes_boundaries.append((-1, 1000))

    return fluxes_boundaries

def rescaling_boundaries(model):

    rescaling_boundaries = []

    for telescope in model.event.telescopes:

        rescaling_boundaries.append((0,10))

    return rescaling_boundaries