import numpy as np
import scipy.stats as ss


def normal_Kolmogorov_Smirnov(sample):
    """The moon illumination expressed as a percentage.

                :param astropy sun: the sun ephemeris
                :param astropy moon: the moon ephemeris

                :return: a numpy array like indicated the moon illumination.

                :rtype: array_like

    """

    mu, sigma = ss.norm.fit(sample)
    # use mu sigma for anomaly, 0,1 for rescaling???
    KS_stat, KS_pvalue = ss.kstest(sample, 'norm', args=(0, 1))

    # the sample is likely Gaussian-like if KS_stat (~ maximum distance between
    # sample and theoritical distribution) -> 0
    # the null hypothesis can not be rejected ( i.e the distribution of sample come
    # from a Gaussian) if KS_pvalue -> 1

    KS_judgement = 0

    if KS_pvalue > 0.01:
        KS_judgement = 1

    if KS_pvalue > 0.05:
        KS_judgement = 2

    return KS_stat, KS_pvalue, KS_judgement


def normal_Anderson_Darling(sample):
    """Compute a Anderson-Darling tests on the sample versus a normal distribution
    with mu = 0, sigma = 1

            :param array_like sample: the sample you want to check the "Gaussianity"
            :returns: the Anderson-Darling statistic, the Anderson-Darling critical
            values associated to the significance
            level of 15 % and the Anderson-Darling judgement
            :rtype: float, array_like, array_like
    """

    AD_stat, AD_critical_values, AD_significance_levels = ss.anderson(sample)

    # the sample is likely Gaussian-like if AD_stat (~ maximum distance between
    # sample and theoritical distribution) -> 0
    # the null hypothesis can not be rejected ( i.e the distribution of sample come
    # from a Gaussian) if AD_pvalue -> 1

    AD_judgement = 0

    if AD_stat < 2 * AD_critical_values[-1]:
        AD_judgement = 1

    if AD_stat < AD_critical_values[-1]:
        AD_judgement = 2
    return AD_stat, AD_critical_values[-1], AD_judgement


def normal_Shapiro_Wilk(sample):
    """Compute a Shapiro-Wilk tests on the sample versus a normal distribution with
    mu = 0, sigma = 1

            :param array_like sample: the sample you want to check the "Gaussianity"
            :returns: the Shapiro-Wilk statistic and its related p_value
            :rtype: float, float
    """

    SW_stat, SW_pvalue = ss.shapiro(sample)

    # the null hypothesis can not be rejected ( i.e the distribution of sample come
    # from a Gaussian) if SW_stat -> 1
    # the null hypothesis can not be rejected ( i.e the distribution of sample come
    # from a Gaussian) if SW_pvalue -> 1

    # Judegement made on the STATISTIC because 'W tests statistic is accurate but the
    # p-value may not be" (see scipy doc)
    SW_judgement = 0

    if SW_pvalue > 0.01:
        SW_judgement = 1

    if SW_pvalue > 0.05:
        SW_judgement = 2

    return SW_stat, SW_pvalue, SW_judgement


### Statistics fit quality metrics

def normalized_chi2(chi2, n_data, n_parameters):
    """Compute the chi^2/dof

            :param float chi2: the chi^2
            :param int n_data: the number of data_points
            :param int n_parameters: the number of model parameters

            :returns: the chi^2/dof and the chi2dof_judgement
            :rtype: float
    """

    chi2_sur_dof = chi2 / (n_data - n_parameters)

    chi2dof_judgement = 0
    if chi2_sur_dof < 2:
        chi2dof_judgement = 2

    return chi2_sur_dof, chi2dof_judgement


def Bayesian_Information_Criterion(chi2, n_data, n_parameters):
    """Compute the BIC statistic.

            :param float chi2: the chi^2
            :param int n_data: the number of data_points
            :param int n_parameters: the number of model parameters

            :returns: the chi^2/dof
            :rtype: float
    """
    BIC = chi2 + n_parameters * np.log(n_data)

    return BIC


def Akaike_Information_Criterion(chi2, n_parameters):
    """Compute the BIC statistic.

            :param float chi2: the chi^2
            :param int n_parameters: the number of model parameters

            :returns: the chi^2/dof
            :rtype: float
    """
    AIC = chi2 + 2 * n_parameters

    return AIC
