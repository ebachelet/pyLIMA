def magnification_PSPL(tau, beta, return_impact_parameter=False):
    """
    The Paczynski Point Source Point Lens magnification and the impact parameter U(t).
    See http://adsabs.harvard.edu/abs/1986ApJ...304....1P

    Parameters
    ----------
    tau : array, (t-t0)/tE
    beta : array, [u0]*len(t)
    return_impact_parameter : bool, if the impact parameter is needed or not

    Returns
    -------
    magnification_PSPL : array, A(t) for PSPL
    impact_parameter : array, u(t)
    """

    # For notations, check for example :
    # http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    import pyLIMA.magnification.impact_parameter

    impact_parameter = pyLIMA.magnification.impact_parameter.impact_parameter(tau,
                                                                              beta)  #
    # u(t)
    impact_parameter_square = impact_parameter ** 2  # u(t)^2

    magnification_pspl = (impact_parameter_square + 2) / (
            impact_parameter * (impact_parameter_square + 4) ** 0.5)

    if return_impact_parameter:

        # return both
        return magnification_pspl, impact_parameter

    else:

        # return magnification
        return magnification_pspl
