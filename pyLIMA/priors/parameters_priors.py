import numpy as np


### Define manual distribution objects are much faster than scipy.pdf....
class UniformDistribution(object):

    def __init__(self, bound_min, bound_max):

        self.bound_min = bound_min
        self.bound_max = bound_max
        self.probability = 1 / (bound_max - bound_min)

    def pdf(self, x):

        if (x > self.bound_min) & (x < self.bound_max):

            return self.probability

        else:

            return 0

    def rvs(self, size):

        sample = np.random.uniform(0, 1, size)
        samples = self.bound_min+sample*(self.bound_max-self.bound_min)

        return samples

class NormalDistribution(object):

    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def pdf(self, x):
        probability = 1 / self.sigma / np.sqrt(2 * np.pi) * \
                      np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)

        return probability

    def rvs(self, size):

        samples = np.random.normal(self.mean, self.sigma, size)

        return samples

def default_parameters_priors(fit_parameters):
    """
    Function to return default priors on parameters (i.e. uniform)

    Parameters
    ----------
    fit_parameters : dict, a dictionnary containing the parameters bounds

    Returns
    -------
    priors : dict, {'i':prior_i} for all i parameters
    """

    priors = {}

    for key in fit_parameters.keys():
        bounds = fit_parameters[key][1]
        # priors[key] = ss.uniform(loc=bounds[0], scale=bounds[1]-bounds[0])
        priors[key] = UniformDistribution(bounds[0], bounds[1])

    return priors
