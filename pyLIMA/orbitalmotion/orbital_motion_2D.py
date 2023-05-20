def orbital_motion_2D_trajectory_shift(time,to_om, dalpha_dt):
    """ Compute the trajectory curvature induced by the orbital motion of the lens.

    :param float to_om: the reference time for the orbital motion
    :param array_like time: the time array to compute the trajectory shift
    :param float dalpha_dt: the angle change rate, in radian/day

    :return: dalpha, the angle shift
    :rtype: array_like
    """

    dalpha = dalpha_dt * (time - to_om)/365.25

    return -dalpha


def orbital_motion_2D_separation_shift(time,to_om, ds_dt):
    """ Compute the binary separation change induced by the orbital motion of the lens.

    :param float to_om: the reference time for the orbital motion
    :param array_like time: the time array to compute the trajectory shift
    :param float ds_dt: the binary separation change rate, in einstein_ring_unit/day

    :return: dseparation, the binary separation shift
    :rtype: array_like
    """
    dseparation = ds_dt * (time - to_om)/365.25

    return dseparation