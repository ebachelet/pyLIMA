import numpy as np

def PSPL_shifts_no_blend(source_x, source_y, theta_E):
    """ See https: // arxiv.org / pdf / 1705.01767.pdf
    """
    shifts = (source_x,source_y)/(source_x**2+source_y**2+2)*theta_E

    return shifts


def PSPL_shifts_with_blend(source_x, source_y, theta_E, g_blend):
    """ See https: // arxiv.org / pdf / 1705.01767.pdf
    """
    u_square = source_x**2+source_y**2
    factor = (4+u_square)**0.5

    delta_s = (u_square**0.5-g_blend*u_square*factor)/(2+u_square+g_blend*u_square**0.5*factor)
    shifts = delta_s+g_blend/(1+g_blend)*u_square**0.5
    shifts *= theta_E

    return shifts
