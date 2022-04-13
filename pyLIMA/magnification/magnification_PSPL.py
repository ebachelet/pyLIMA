def magnification_PSPL(tau, uo, return_impact_parameter=False):
    """
    The Paczynski Point Source Point Lens magnification and the impact parameter U(t).
    "Gravitational microlensing by the galactic halo",Paczynski, B. 1986
    http://adsabs.harvard.edu/abs/1986ApJ...304....1P

    :param array_like tau: the tau define for example in
                           http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :param array_like uo: the uo define for example in
                         http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    :param boolean return_impact_parameter: if the impact parameter is needed or not

    :return: the PSPL magnification A_PSPL(t) and the impact parameter U(t)
    :rtype: tuple, tuple of two array_like
    """
    # For notations, check for example : http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    import pyLIMA.magnification.impact_parameter
    
    impact_parameter = pyLIMA.magnification.impact_parameter.impact_parameter(tau, uo)  # u(t)
    impact_parameter_square = impact_parameter ** 2  # u(t)^2

    magnification_pspl = (impact_parameter_square + 2) / (impact_parameter * (impact_parameter_square + 4) ** 0.5)

    if return_impact_parameter:
        
        # return both
        return magnification_pspl, impact_parameter
    
    else:
        
        # return magnification
        return magnification_pspl
