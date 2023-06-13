from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time


def Earth_ephemerides(time_to_treat):
    """
    Find the Earth positions and speeds

    Parameters
    ----------
    time_to_treat : array, array of time to treat

    Returns
    -------
    Earth_position_speed : list, [positions,speed]
    """
    time_jd_reference = Time(time_to_treat, format='jd')
    Earth_position_speed = get_body_barycentric_posvel('Earth', time_jd_reference)

    return Earth_position_speed
