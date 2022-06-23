import numpy as np

def xy_shifts_to_NE_shifts(xy_shifts,piEN,piEE):

    centroid_x = xy_shifts[0]
    centroid_y = xy_shifts[1]

    angle = np.arctan2(piEE,piEN)

    Delta_dec = centroid_x * np.cos(angle) - np.sin(angle) * centroid_y
    Delta_ra = centroid_x * np.sin(angle) + np.cos(angle) * centroid_y

    return Delta_ra, Delta_dec

def astrometric_position(time,time_ref,earth_vector,parallax,ref_N,ref_E,mu_N,mu_E):


    projected =earth_vector*parallax
    position_N = mu_N / 365.25 * (time -time_ref) + ref_N+projected[0]
    position_E = mu_E / 365.25 * (time -time_ref) + ref_E+projected[1]

    return position_N,position_E

def source_position(telescope, pyLIMA_parameters, shifts = None):

    time = telescope.astrometry['time'].value
    time_ref = pyLIMA_parameters.t0
    earth_vector = telescope.Earth_positions
    parallax = pyLIMA_parameters.parallax_source
    ref_N = getattr(pyLIMA_parameters, 'position_source_N_'+telescope.name)
    ref_E = getattr(pyLIMA_parameters, 'position_source_E_'+telescope.name)
    mu_N = pyLIMA_parameters.mu_source_N
    mu_E = pyLIMA_parameters.mu_source_E

    position_N, position_E = astrometric_position(time, time_ref, earth_vector, parallax, ref_N, ref_E, mu_N, mu_E)

    if telescope.astrometry['delta_ra'].unit == 'mas':

        position_dec = shifts[1] + position_N * telescope.pixel_scale
        position_ra = shifts[0] + position_E * telescope.pixel_scale

    else:

        position_dec = shifts[1] / telescope.pixel_scale + position_N
        position_ra = shifts[0] / telescope.pixel_scale + position_E


    return position_ra, position_dec