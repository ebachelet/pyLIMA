import numpy as np
from emcee import autocorr


def Gelman_Rubin(chain):
    W = np.mean(np.var(chain, axis=0), axis=0)
    theta_B = np.mean(chain, axis=0)
    theta_BB = np.mean(theta_B, axis=0)
    B = chain.shape[0] / (chain.shape[1] - 1) * np.sum((theta_B - theta_BB) ** 2,
                                                       axis=0)
    var_theta = (1 - 1 / chain.shape[1]) * W + 1 / chain.shape[1] * B
    GR = (var_theta / W) ** 0.5
    return GR


def split_R(chain):
    """https://arxiv.org/pdf/1903.08008.pdf"""

    W = np.mean(np.var(chain, axis=0), axis=0)
    theta_B = np.mean(chain, axis=0)
    theta_BB = np.mean(theta_B, axis=0)
    B = chain.shape[0] / (chain.shape[1] - 1) * np.sum((theta_B - theta_BB) ** 2,
                                                       axis=0)
    var_theta = (1 - 1 / chain.shape[1]) * W + 1 / chain.shape[1] * B
    GR = (var_theta / W) ** 0.5
    return GR


def autocorrelation_time(chain):
    tau = autocorr.integrated_time(chain)

    return tau
