from pyLIMA.orbitalmotion import orbital_motion_2D, orbital_motion_3D


def orbital_motion_shifts(orbital_motion_model, time, pyLIMA_parameters):
    """
    Compute the orbital motion shifts depending on the model.
    See https://ui.adsabs.harvard.edu/abs/2011ApJ...738...87S/abstract

    Parameters
    ----------
    orbital_motion_model : list, [str,float] the modeel type (2D, Circular or
    Keplerian) and t0,om
    time : array, containing the time to treat
    pyLIMA_parameters : a pyLIMA_parameters object

    Returns
    -------
    dseparation : array, it contains the binary separation variation
    dalpha : array, containts the variation of the lens trajectory angle du to the
    motion of the lens
    """
    if orbital_motion_model[0] == '2D':

        ds_dt = pyLIMA_parameters['v_para'] * pyLIMA_parameters['separation']
        dseparation = orbital_motion_2D.orbital_motion_2D_separation_shift(time,
                                                                           orbital_motion_model[
                                                                               1],
                                                                           ds_dt)

        dalpha_dt = pyLIMA_parameters['v_perp']
        dalpha = orbital_motion_2D.orbital_motion_2D_trajectory_shift(time,
                                                                      orbital_motion_model[
                                                                          1], dalpha_dt)

    else:

        dseparation, dalpha = orbital_motion_3D.orbital_motion_keplerian(time,
                                                                         pyLIMA_parameters,
                                                                         orbital_motion_model)

    return dseparation, dalpha
