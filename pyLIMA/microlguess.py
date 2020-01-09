# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:00:51 2016

@author: ebachelet
"""

import numpy as np
import scipy.signal as ss

from pyLIMA import microltoolbox


def initial_guess_PSPL(event):
    """Function to find initial PSPL guess for Levenberg-Marquardt solver (method=='LM').
       This assumes no blending.

    :param object event: the event object on which you perform the fit on. More details on the
                            event module.

    :return: the PSPL guess for this event. A list of parameters associated to the PSPL model + the source flux of
    :return: the PSPL guess for this event. A list of parameters associated to the PSPL model + the source flux of
                the survey telescope.
    :rtype: list,float
    """

    # to estimation
    to_estimations = []
    maximum_flux_estimations = []
    errors_magnitude = []

    for telescope in event.telescopes:
        # Lot of process here, if one fails, just skip
        lightcurve_magnitude = telescope.lightcurve_magnitude
        mean_error_magnitude = np.mean(lightcurve_magnitude[:, 2])
        try:

            # only the best photometry
            good_photometry_indexes = np.where((lightcurve_magnitude[:, 2] <
                                                max(0.1, mean_error_magnitude)))[0]
            lightcurve_bis = lightcurve_magnitude[good_photometry_indexes]

            lightcurve_bis = lightcurve_bis[lightcurve_bis[:, 0].argsort(), :]

            mag = lightcurve_bis[:, 1]
            flux = microltoolbox.magnitude_to_flux(mag)

            # clean the lightcurve using Savitzky-Golay filter on 3 points, degree 1.
            mag_clean = ss.savgol_filter(mag, 3, 1)
            time = lightcurve_bis[:, 0]
            flux_clean = microltoolbox.flux_to_magnitude(mag_clean)
            errmag = lightcurve_bis[:, 2]

            flux_source = min(flux_clean)
            good_points = np.where(flux_clean > flux_source)[0]

            while (np.std(time[good_points]) > 5) | (len(good_points) > 100):

                indexes = \
                    np.where((flux_clean[good_points] > np.median(flux_clean[good_points])) & (
                        errmag[good_points] <= max(0.1, 2.0 * np.mean(errmag[good_points]))))[0]

                if len(indexes) < 1:

                    break

                else:

                    good_points = good_points[indexes]

                    # gravity = (
                    #   np.median(time[good_points]), np.median(flux_clean[good_points]),
                    #    np.mean(errmag[good_points]))

                    # distances = np.sqrt((time[good_points] - gravity[0]) ** 2 / gravity[0] ** 2)

            to = np.median(time[good_points])
            max_flux = max(flux[good_points])
            to_estimations.append(to)
            maximum_flux_estimations.append(max_flux)
            errors_magnitude.append(np.mean(lightcurve_bis[good_points, 2]))

        except:

            time = lightcurve_magnitude[:, 0]
            flux = microltoolbox.magnitude_to_flux(lightcurve_magnitude[:, 1])
            to = np.median(time)
            max_flux = max(flux)
            to_estimations.append(to)
            maximum_flux_estimations.append(max_flux)

            errors_magnitude.append(mean_error_magnitude)

    to_guess = sum(np.array(to_estimations) / np.array(errors_magnitude) ** 2) / sum(
        1 / np.array(errors_magnitude) ** 2)
    survey = event.telescopes[0]
    lightcurve = survey.lightcurve_magnitude
    lightcurve = lightcurve[lightcurve[:, 0].argsort(), :]

    ## fs, uo, tE estimations only one the survey telescope


    time = lightcurve[:, 0]
    flux = microltoolbox.magnitude_to_flux(lightcurve[:, 1])
    errflux = microltoolbox.error_magnitude_to_error_flux(lightcurve[:, 2], flux)

    # fs estimation, no blend

    baseline_flux_0 = np.min(flux)
    baseline_flux = np.median(flux)

    while np.abs(baseline_flux_0 - baseline_flux) > 0.01 * baseline_flux:

        baseline_flux_0 = baseline_flux
        indexes = np.where((flux < baseline_flux))[0].tolist() + np.where(
            np.abs(flux - baseline_flux) < np.abs(errflux))[0].tolist()
        baseline_flux = np.median(flux[indexes])

        if len(indexes) < 100:
            baseline_flux = np.median(flux[flux.argsort()[:100]])
            break

    fs_guess = baseline_flux

    # uo estimation
    max_flux = maximum_flux_estimations[0]
    Amax = max_flux / fs_guess
    if (Amax < 1.0) | np.isnan(Amax):
        Amax = 1.1
    uo_guess = np.sqrt(-2 + 2 * np.sqrt(1 - 1 / (1 - Amax ** 2)))

    # tE estimations
    tE_guesses = []

    # Method 1 : flux(t_demi_amplification) = 0.5 * fs_guess * (Amax + 1)

    half_magnification = 0.5 * (Amax + 1)

    flux_demi_amplification = fs_guess * half_magnification

    index_plus = np.where((time > to_guess) & (flux < flux_demi_amplification))[0]
    index_moins = np.where((time < to_guess) & (flux < flux_demi_amplification))[0]

    if len(index_plus) != 0:

        if len(index_moins) != 0:
            t_demi_amplification = (time[index_plus[0]] - time[index_moins[-1]])
            tE_demi_amplification = t_demi_amplification / (
                2 * np.sqrt(-2 + 2 * np.sqrt(1 + 1 / (half_magnification ** 2 - 1)) - uo_guess ** 2))

            tE_guesses.append(tE_demi_amplification)

        else:
            t_demi_amplification = time[index_plus[0]] - to_guess
            tE_demi_amplification = t_demi_amplification / np.sqrt(
                -2 + 2 * np.sqrt(1 + 1 / (half_magnification ** 2 - 1)) - uo_guess ** 2)

            tE_guesses.append(tE_demi_amplification)
    else:

        if len(index_moins) != 0:
            t_demi_amplification = to_guess - time[index_moins[-1]]
            tE_demi_amplification = t_demi_amplification / np.sqrt(
                -2 + 2 * np.sqrt(1 + 1 / (half_magnification ** 2 - 1)) - uo_guess ** 2)

            tE_guesses.append(tE_demi_amplification)

    # Method 2 : flux(t_E) = fs_guess * (uo^+3)/[(uo^2+1)^0.5*(uo^2+5)^0.5]

    amplification_tE = (uo_guess ** 2 + 3) / ((uo_guess ** 2 + 1) ** 0.5 * np.sqrt(uo_guess ** 2 + 5))
    flux_tE = fs_guess * amplification_tE

    index_tE_plus = np.where((flux < flux_tE) & (time > to))[0]
    index_tE_moins = np.where((flux < flux_tE) & (time < to))[0]

    if len(index_tE_moins) != 0:
        index_tE_moins = index_tE_moins[-1]
        tE_moins = to_guess - time[index_tE_moins]

        tE_guesses.append(tE_moins)

    if len(index_tE_plus) != 0:
        index_tE_plus = index_tE_plus[0]
        tE_plus = time[index_tE_plus] - to_guess

        tE_guesses.append(tE_plus)

    # Method 3 : the first points before/after to_guess that reach the baseline. Very rough
    # approximation ot tE.

    index_tE_baseline_plus = np.where((time > to) & (np.abs(flux - fs_guess) < np.abs(errflux)))[0]
    index_tE_baseline_moins = np.where((time < to) & (np.abs(flux - fs_guess) < np.abs(errflux)))[0]

    if len(index_tE_baseline_plus) != 0:
        tEPlus = time[index_tE_baseline_plus[0]] - to_guess

        tE_guesses.append(tEPlus)

    if len(index_tE_baseline_moins) != 0:
        tEMoins = to_guess - time[index_tE_baseline_moins[-1]]

        tE_guesses.append(tEMoins)

    tE_guess = np.median(tE_guesses)

    # safety reason, unlikely
    if (tE_guess < 0.1) | np.isnan(tE_guess):
        tE_guess = 20.0

    # [to,uo,tE], fsource

    return [to_guess, uo_guess, tE_guess], fs_guess


def initial_guess_FSPL(event):
    """Function to find initial FSPL guess for Levenberg-Marquardt solver (method=='LM').
       This assumes no blending.

    :param object event: the event object on which you perform the fit on. More details on the
                            event module.

    :return: the FSPL guess for this event. A list of parameters associated to the FSPL model + the source flux of
                the survey telescope.
    :rtype: list,float
    """
    PSPL_guess, fs_guess = initial_guess_PSPL(event)
    # Dummy guess
    rho_guess = 0.05

    FSPL_guess = PSPL_guess + [rho_guess]

    # [to,uo,tE,rho], fsource
    return FSPL_guess, fs_guess


def initial_guess_DSPL(event):
    """Function to find initial DSPL guess for Levenberg-Marquardt solver (method=='LM').
       This assumes no blending.

       :param object event: the event object on which you perform the fit on. More details on the
                            event module.

       :return: the DSPL guess for this event. A list of parameters associated to the DSPL model + the source flux of
                the survey telescope.
       :rtype: list,float
    """
    PSPL_guess, fs_guess = initial_guess_PSPL(event)

    filters = [telescope.filter for telescope in event.telescopes]

    unique_filters = np.unique(filters)

    # Dummy guess
    delta_to_guess = 5  # days
    delta_uo_guess = 0.01
    q_flux_guess = 0.5

    DSPL_guess = PSPL_guess[:2] + [delta_to_guess] + [delta_uo_guess] + \
                 [PSPL_guess[2]] + [q_flux_guess] * len(unique_filters)

    # [to1,uo1,delta_to,uo2,tE,q_F_i], fsource
    return DSPL_guess, fs_guess


def differential_evolution_parameters_boundaries(model):
    """ This function define the parameters boundaries for a specific model.

       :param object model: a microlmodels object.

       :return: parameters_boundaries, a list of tuple containing parameters limits
       :rtype: list
    """

    minimum_observing_time_telescopes = [min(telescope.lightcurve_flux[:, 0]) - 0 for telescope in
                                         model.event.telescopes]
    maximum_observing_time_telescopes = [max(telescope.lightcurve_flux[:, 0]) + 0 for telescope in
                                         model.event.telescopes]

    to_boundaries = (min(minimum_observing_time_telescopes), max(maximum_observing_time_telescopes))
    delta_to_boundaries = (-150, 150)
    delta_uo_boundaries = (-1.0, 1.0)
    uo_boundaries = (0.0, 1.0)
    tE_boundaries = (1.0, 500)
    rho_boundaries = (5 * 10 ** -5, 0.05)
    q_flux_boundaries = (0.001, 1.0)

    logs_boundaries = (-1.0, 1.0)
    logq_boundaries = (-6.0, 0.0)
    alpha_boundaries = (-np.pi, np.pi)

    piEN_boundaries = (-2.0, 2.0)
    piEE_boundaries = (-2.0, 2.0)
    XiEN_boundaries = (-2.0, 2.0)
    XiEE_boundaries = (-2.0, 2.0)

    dsdt_boundaries = (-10,10)
    dalphadt_boundaries = (-10,10)
    v_boundaries = (-2,2)
    mass_boundaries = [10**-1,10]
    rE_boundaries = [10**-1,100]

    v_boundaries = (-2,2)
    
    ra_xal_boundaries = [0,360]
    dec_xal_boundaries = [-90,90]
    period_xal_boundaries = [0.001,1000]
    ecc_xal_boundaries = [0,1]
    t_peri_xal_boundaries = to_boundaries

    # model_xallarap_boundaries = {'None': [], 'True': [(-2.0, 2.0), (-2.0, 2.0)]}

    # model_orbital_motion_boundaries = {'None': [], '2D': [], '3D': []}

    # model_source_spots_boundaries = {'None': []}

    period_variable = (0.001,1000)
    phase_variable = (-np.pi, np.pi)
    amplitude_variable = (0.0, 3.0)
    octave_variable = (10**-10,1)
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
                                 delta_uo_boundaries, tE_boundaries,rho_boundaries,rho_boundaries]
        filters = [telescope.filter for telescope in model.event.telescopes]

        unique_filters = np.unique(filters)

        parameters_boundaries += [q_flux_boundaries] * len(unique_filters)
    if model.model_type == 'PSBL':
        parameters_boundaries = [to_boundaries, uo_boundaries, tE_boundaries, logs_boundaries,
                                 logq_boundaries, alpha_boundaries]

    if (model.model_type == 'USBL') or (model.model_type == 'FSBL'):
        parameters_boundaries = [to_boundaries, uo_boundaries, tE_boundaries, rho_boundaries, logs_boundaries,
                                 logq_boundaries, alpha_boundaries]
        #fluxes = [(0,np.max(telescope.lightcurve_flux[:,1])) for telescope in model.event.telescopes]
        #blend = [(0,100) for telescope in model.event.telescopes]

        #for ind,telo in enumerate(model.event.telescopes):
             #parameters_boundaries+=[fluxes[ind], blend[ind]]

   
    if model.model_type == 'VariablePL':
        parameters_boundaries = [to_boundaries, uo_boundaries, tE_boundaries,rho_boundaries, period_variable]

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


def MCMC_parameters_initialization(parameter_key, parameters_dictionnary, parameters):
    """Generate a random parameter for the MCMC initialization.

        :param str parameter_key: the parameter on which we apply the function
        :param dict parameters_dictionnary: the dictionnary of parameters keys associared to the parameters input
        :param list parameters: a list of float which indicate the model parameters

        :return: a list containing the trial(s) associated to the parameter_key string
        :rtype: list of float
     """
    if ('to' in parameter_key) :
        epsilon = np.random.uniform(-0.01, 0.01)
        to_parameters_trial = parameters[parameters_dictionnary[parameter_key]] + epsilon

        return [to_parameters_trial]

    if 'fs' in parameter_key:
        epsilon = np.random.uniform(0.99, 1.00)

        fs_trial = parameters[parameters_dictionnary[parameter_key]] * epsilon
        g_trial = (1 + parameters[parameters_dictionnary[parameter_key] + 1]) / epsilon - 1

        return [fs_trial, g_trial]
        # return
    if ('g_' in parameter_key) or ('fb_' in parameter_key):
        return

    # if 'pi' in parameter_key:

    #    epsilon = np.random.uniform(0.9, 1.1)
    #    sign = np.random.choice([-1,1])

    #    pi_trial = sign*parameters[parameters_dictionnary[parameter_key]] * epsilon

    #    return [pi_trial]

    if 'rho' in parameter_key:
        epsilon = np.random.uniform(0.99, 1.01)
        rho_parameters_trial = parameters[parameters_dictionnary[parameter_key]] * epsilon
        return [rho_parameters_trial]

    if 'logs' in parameter_key:
        epsilon = np.random.uniform(-0.05, 0.05)

        logs_parameters_trial = parameters[parameters_dictionnary[parameter_key]] + epsilon
        return [logs_parameters_trial]
    if 'logq' in parameter_key:
        epsilon = np.random.uniform(-0.05, 0.05)

        logq_parameters_trial = parameters[parameters_dictionnary[parameter_key]] + epsilon
        return [logq_parameters_trial]
    epsilon = np.random.uniform(0.99, 1.01)
    all_other_parameter_trial = parameters[parameters_dictionnary[parameter_key]] * epsilon

    return [all_other_parameter_trial]
