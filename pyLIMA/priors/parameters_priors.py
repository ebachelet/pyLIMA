import scipy.stats as ss
import numpy as np

def parameters_priors(fit_parameters):
    """ This function define the parameters boundaries for a specific model.

       :param object model: a microlmodels object.

       :return: parameters_boundaries, a list of tuple containing parameters limits
       :rtype: list
    """

    priors = []

    for key in fit_parameters.keys():

            bounds = fit_parameters[key][1]
            priors.append(ss.uniform(loc=bounds[0], scale=bounds[1]-bounds[0]))

    return priors