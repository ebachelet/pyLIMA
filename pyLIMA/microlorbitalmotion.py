import numpy as np


def orbital_motion_2D_trajectory_shift(to_om, time, dalpha_dt):
    """ Compute the trajectory curvature induced by the orbital motion of the lens.

    :param float to_om: the reference time for the orbital motion
    :param array_like time: the time array to compute the trajectory shift
    :param float dalpha_dt: the angle change rate, in radian/yr

    :return: dalpha, the angle shift
    :rtype: array_like
    """

    dalpha = dalpha_dt * (time - to_om)

    return dalpha


def orbital_motion_2D_separation_shift(to_om, time, ds_dt):
    """ Compute the binary separation change induced by the orbital motion of the lens.

    :param float to_om: the reference time for the orbital motion
    :param array_like time: the time array to compute the trajectory shift
    :param float ds_dt: the binary separation change rate, in einstein_ring_unit/yr

    :return: dseparation, the binary separation shift
    :rtype: array_like
    """
    dseparation = ds_dt * (time - to_om)

    return dseparation


def orbital_motion_3D_separation_shift(time, period, ds_dt):
    """ Compute the binary separation change induced by the orbital motion of the lens.

    :param float to_om: the reference time for the orbital motion
    :param array_like time: the time array to compute the trajectory shift
    :param float ds_dt: the binary separation change rate, in einstein_ring_unit/yr

    :return: dseparation, the binary separation shift
    :rtype: array_like
    """
    dseparation = ds_dt * (time - to_om)

    return dseparation