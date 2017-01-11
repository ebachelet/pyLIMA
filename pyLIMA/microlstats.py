import scipy.stats
import numpy as np


### Tests on residuals

def normal_Kolmogorov_Smirnov(sample):
    """ Compute a Kolmorogov-Smirnov test on the sample versus a normal distribution with mu = 0, sigma = 1

        :param array_like sample: the sample you want to check the "Gaussianity"
        :returns: the Kologorov-Smirnov statistic, its related p_value and the  Kolmorogov-Smirnov judgement
        :rtype: float, float
    """

    KS_stat, KS_pvalue = scipy.stats.kstest(sample, 'norm')

    # the sample is likely Gaussian-like if KS_stat (~ maximum distance between sample and theoritical distribution) -> 0
    # the null hypothesis can not be rejected ( i.e the distribution of sample come from a Gaussian) if KS_pvalue -> 1

    KS_judgement = 0

    if KS_pvalue > 0.1:

        KS_judgement = 1

    if KS_pvalue > 0.5:

        KS_judgement = 2

    return KS_stat, KS_pvalue, KS_judgement

def normal_Anderson_Darling(sample):
    """ Compute a Anderson-Darling test on the sample versus a normal distribution with mu = 0, sigma = 1

        :param array_like sample: the sample you want to check the "Gaussianity"
        :returns: the Anderson-Darling statistic, the Anderson-Darling critical values associated to the significance
        level of 15 % and the Anderson-Darling judgement
        :rtype: float, array_like, array_like
    """

    AD_stat, AD_critical_values, AD_significance_level = scipy.stats.anderson(sample, 'norm')

    # the null hypothesis ( i.e the distribution of sample come from a Gaussian) can be rejected for each
    # AD_significance_level if the AD_stat is GREATER than the corresponding AD_critical_values.
    # Example :
    # AD_stat = 0.44, AD_critical_values = [0.536,0.61,0.732,0.854,1.016], AD_significance_level(%) = [15,10,5,2.5,1]
    # In this case, the null hypothesis is not rejected (i.e the sample can come from a Gaussian)

    AD_critical_value = AD_critical_values[0]

    AD_judgement = 0

    if AD_stat < AD_critical_value:
        AD_judgement = 2

    return  AD_stat, AD_critical_value, AD_judgement

def normal_Shapiro_Wilk(sample):
    """ Compute a Shapiro-Wilk test on the sample versus a normal distribution with mu = 0, sigma = 1

        :param array_like sample: the sample you want to check the "Gaussianity"
        :returns: the Shapiro-Wilk statistic and its related p_value
        :rtype: float, float
    """

    SW_stat, SW_pvalue = scipy.stats.shapiro(sample)

    # the null hypothesis can not be rejected ( i.e the distribution of sample come from a Gaussian) if SW_stat -> 1
    # the null hypothesis can not be rejected ( i.e the distribution of sample come from a Gaussian) if SW_pvalue -> 1

    SW_judgement = 0

    if SW_pvalue > 0.1:
        SW_judgement = 1

    if SW_pvalue > 0.5:
        SW_judgement = 2


    return  SW_stat, SW_pvalue, SW_judgement


### Statistics fit quality metrics

def normalized_chi2(chi2, n_data, n_parameters) :
    """ Compute the chi^2/dof

        :param float chi2: the chi^2
        :param int n_data: the number of data_points
        :param int n_parameters: the number of model parameters

        :returns: the chi^2/dof and the chi2dof_judgement
        :rtype: float
    """

    chi2_sur_dof = chi2/(n_data-n_parameters)

    chi2dof_judgement = 0
    if chi2_sur_dof < 2 :
        chi2dof_judgement = 2

    return chi2_sur_dof,chi2dof_judgement

def Bayesian_Information_Criterion(chi2, n_data, n_parameters):
    """ Compute the BIC statistic.

        :param float chi2: the chi^2
        :param int n_data: the number of data_points
        :param int n_parameters: the number of model parameters

        :returns: the chi^2/dof
        :rtype: float
    """
    BIC = chi2 + n_parameters*np.log(n_data)

    return BIC

def Akaike_Information_Criterion(chi2, n_parameters):
    """ Compute the BIC statistic.

        :param float chi2: the chi^2
        :param int n_parameters: the number of model parameters

        :returns: the chi^2/dof
        :rtype: float
    """
    AIC = chi2 + 2*n_parameters

    return AIC


