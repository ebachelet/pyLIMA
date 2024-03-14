def orbital_motion_2D_trajectory_shift(time, t0_om, dalpha_dt):
    """
    Compute the trajectory curvature induced by the linear orbital motion of the lens.

    Parameters
    ----------
    time : array, containing the time to treat
    t0_om : float, the time of reference of the orbital motion
    dalpha_dt :  float, the linear rate of lens rotation

    Returns
    -------
    dalpha : array, containts the variation of the lens trajectory angle du to the
    motion of the lens
    """
    dalpha = dalpha_dt * (time - t0_om) / 365.25

    return dalpha


def orbital_motion_2D_separation_shift(time, t0_om, ds_dt):
    """
    Compute the separation change induced by the linear orbital motion of the lens.

    Parameters
    ----------
    time : array, containing the time to treat
    t0_om : float, the time of reference of the orbital motion
    ds_dt :  float, the linear rate of lens separation

    Returns
    -------
    dseparation : array, containts the variation of the lens separation du to the
    motion of the lens
    """
    dseparation = ds_dt * (time - t0_om) / 365.25

    return dseparation
