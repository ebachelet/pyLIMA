def impact_parameter(tau, uo):
    """
    The impact parameter U(t).
    "Gravitational microlensing by the galactic halo",Paczynski, B. 1986
    http://adsabs.harvard.edu/abs/1986ApJ...304....1P

    :param array_like tau: the tau define for example in
                               http://adsabs.harvard.edu/abs/2015ApJ...804...20C
    :param array_like uo: the uo define for example in
                              http://adsabs.harvard.edu/abs/2015ApJ...804...20C

    :return: the impact parameter U(t)
    :rtype: array_like
    """
    impact_param = (tau ** 2 + uo ** 2) ** 0.5  # u(t)

    return impact_param
