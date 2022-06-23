from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel


def Earth_ephemerides(time_to_treat):

    time_jd_reference = Time(time_to_treat, format='jd')
    Earth_position_speed = get_body_barycentric_posvel('Earth', time_jd_reference)

    return Earth_position_speed