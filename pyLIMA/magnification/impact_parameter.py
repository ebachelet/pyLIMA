def impact_parameter(tau, beta):
    """
    The impact parameter U(t).
    See http://adsabs.harvard.edu/abs/1986ApJ...304....1P
    http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    Parameters
    ----------
    tau : array, (t-t0)/tE
    beta : array, [u0]*len(t)

    Returns
    -------
    impact_param : array, u(t)
    """

    impact_param = (tau ** 2 + beta ** 2) ** 0.5  # u(t)

    return impact_param
