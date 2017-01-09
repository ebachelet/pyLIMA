# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:49:26 2016

@author: ebachelet
"""
import numpy as np


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
    for i in xrange(len(limits)):

        if (parameters[i] > limits[i][1]) | (parameters[i] < limits[i][0]):

            return np.inf

        else:

            pass
    return 42.0
