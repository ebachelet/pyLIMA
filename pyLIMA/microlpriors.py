# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:49:26 2016

@author: ebachelet
"""
import numpy as np
from pyLIMA import microlcaustics


def priors_on_models(pyLIMA_parameters, model, binary_regime = None):

    keys = pyLIMA_parameters._fields
    for index,bounds in enumerate(model.parameters_boundaries):

        parameter = getattr(pyLIMA_parameters, keys[index])

        if (parameter < bounds[0]) or (parameter>bounds[1]):

            return np.inf

    if binary_regime:

        regime = microlcaustics.find_2_lenses_caustic_regime(10**pyLIMA_parameters.logs, 10**pyLIMA_parameters.logq)

        if regime != binary_regime:

            return np.inf

    if model.orbital_motion_model[0] == 'Keplerian':
        prior = kinematic_energy_prior(pyLIMA_parameters)

        if prior !=0 :
            return prior
    return 0

def microlensing_flux_priors(size_dataset, f_source, g_blending):
    # Little prior here, need to be chaneged

    if (f_source < 0) | (g_blending < -1.0):

        prior_flux_impossible = np.inf
        return prior_flux_impossible

    else:

        prior_flux_impossible = 0.0

    prior_flux_negative_blending = np.log(size_dataset) * 1 / (1 + g_blending)

    prior = prior_flux_impossible + prior_flux_negative_blending
    return prior


def microlensing_parameters_limits_priors(parameters, limits):
    for i in range(len(limits)):

        if (parameters[i] > limits[i][1]) | (parameters[i] < limits[i][0]):

            return np.inf

        else:

            pass
    return 42.0


def kinematic_energy_prior(pyLIMA_parameters):

    v_para = pyLIMA_parameters.v_para
    v_perp = pyLIMA_parameters.v_perp
    v_radial = pyLIMA_parameters.v_radial
    separation_0 = 10 ** pyLIMA_parameters.logs
    separation_z = 10 ** pyLIMA_parameters.logs_z
    mass_lens = pyLIMA_parameters.mass_lens
    rE = pyLIMA_parameters.rE





    r_0 = np.array([separation_0, 0, separation_z]) * rE
    v_0 = r_0[0] *np.array([v_para, v_perp, v_radial])


    eps = np.sum(( v_0) ** 2) / 2 - 4 * np.pi ** 2 * mass_lens / np.sum(r_0 ** 2) ** 0.5

    if eps>0:

        prior = np.inf
    else:
        prior = 0
    return prior