import numpy as np


def PSPL_shifts_no_blend(source_x, source_y, theta_E):
    """
    The PSPL astreomtric shifts without blend, see https: // arxiv.org / pdf /
    1705.01767.pdf

    Parameters
    ----------
    source_x : array, the positions of the source in x
    source_y: array, the positions of the source in y
    theta_E : float, the angular Einsteing ring radius in mas

    Returns
    -------
    shifts : array,the astrometric shifts, in x and y
    """
    shifts = (source_x, source_y) / (source_x ** 2 + source_y ** 2 + 2) * theta_E

    return shifts


def PSPL_shifts_with_blend(source_x, source_y, theta_E, g_blend):
    """
    The PSPL astreomtric shifts with blend, see https: // arxiv.org / pdf /
    1705.01767.pdf

    Parameters
    ----------
    source_x : array, the positions of the source in x
    source_y: array, the positions of the source in y
    theta_E : float, the angular Einsteing ring radius in mas
    g_Blend : float, the blend_ratio, i.e. f_blend/f_source

    Returns
    -------
    shifts : array,the astrometric shifts, in x and y
    """
    u_square = source_x ** 2 + source_y ** 2
    factor = (4 + u_square) ** 0.5

    delta_s = (u_square ** 0.5 - g_blend * u_square * factor) / (
            2 + u_square + g_blend * u_square ** 0.5 * factor)
    shifts = delta_s + g_blend / (1 + g_blend) * u_square ** 0.5
    shifts *= theta_E

    alpha = np.arctan2(source_y, source_x)
    shifts = np.array([shifts * np.cos(alpha), shifts * np.sin(alpha)])

    return shifts
