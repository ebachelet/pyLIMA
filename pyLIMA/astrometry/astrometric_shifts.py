import numpy as np

def PSPL_shifts(source_x, source_y, theta_E):

    shifts = (source_x,source_y)/(source_x**2+source_y**2+2)*theta_E

    return shifts
