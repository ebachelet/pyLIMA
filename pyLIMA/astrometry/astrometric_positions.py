import numpy as np


def xy_shifts_to_NE_shifts(xy_shifts, piEN, piEE):
    """
    Transform the x,y positions into the North, East reference,
    i.e. rotate by phi = np.arctan2(piEE,piEN)

    Parameters
    ----------
    xy_shifts : array, the x and y positions
    piEN : float, the North component of the parallax vector
    piEE : float, a list of [string,float] indicating the xallarap model

    Returns
    -------
    Detlta_ra : array, the astrometric shift in North
    Detlta_dec : array, the astrometric shift in East
    """

    centroid_x = xy_shifts[0]
    centroid_y = xy_shifts[1]

    angle = np.arctan2(piEE, piEN)

    Delta_dec = centroid_x * np.cos(angle) - np.sin(angle) * centroid_y
    Delta_ra = centroid_x * np.sin(angle) + np.cos(angle) * centroid_y

    return Delta_ra, Delta_dec


def astrometric_positions_of_the_source(telescope, pyLIMA_parameters, time_ref=None):
    """
    The source astrometric positions without astrometric shifts

    Parameters
    ----------
    telescope : object, the telescope object containing astrometric data
    pyLIMA_parameters : dict, the dictionnary containing the model parameters
    time_ref : float, time of reference chosen, default is t0

    Returns
    -------
    position_ra : array, the astrometric position in North, degree or pixels
    position_dec : array, the astrometric position in East, degree or pixels
    """
    time = telescope.astrometry['time'].value

    if time_ref is None:
        time_ref = pyLIMA_parameters['t0']

    ref_N = pyLIMA_parameters['position_source_N_' + telescope.name]
    ref_E = pyLIMA_parameters['position_source_E_' + telescope.name]
    mu_N = pyLIMA_parameters['mu_source_N']
    mu_E = pyLIMA_parameters['mu_source_E']

    earth_vector = telescope.Earth_positions_projected['astrometry']
    parallax_source = pyLIMA_parameters['pi_source']
    Earth_projected = earth_vector * parallax_source  # mas

    if telescope.astrometry['ra'].unit == 'deg':

        position_N = mu_N / 365.25 * (time - time_ref) / 3600 / 1000 + ref_N  # deg
        position_E = mu_E / 365.25 * (time - time_ref) / 3600 / 1000 + ref_E

        position_dec = position_N - Earth_projected[0] / 3600 / 1000
        position_ra = position_E - Earth_projected[1] / 3600 / 1000

    else:

        position_N = mu_N / 365.25 * (time - time_ref) + ref_N  # pix
        position_E = mu_E / 365.25 * (time - time_ref) + ref_E

        position_dec = position_N - Earth_projected[0] / telescope.pixel_scale
        position_ra = position_E - Earth_projected[1] / telescope.pixel_scale

    return position_ra, position_dec


def source_astrometric_positions(telescope, pyLIMA_parameters, shifts=None,
                                 time_ref=None):
    """
    The source astrometric positions with or without astrometric shifts


    Parameters
    ----------
    telescope : object, the telescope object containing astrometric data
    pyLIMA_parameters : dict, the dictionnary containing the model parameters
    shifts : array, North and East astrometric shifts in mas or pixel
    time_ref : float, time of reference chosen, default is t0

    Returns
    -------
    position_ra : array, the astrometric position in North, degree or pixels
    position_dec : array, the astrometric position in East, degree or pixels
    """
    if shifts is None:
        shifts = np.zeros(2)

    position_E, position_N = astrometric_positions_of_the_source(telescope,
                                                                 pyLIMA_parameters,
                                                                 time_ref)

    if telescope.astrometry['ra'].unit == 'deg':

        position_ra = shifts[0] / 3600. / 1000 + position_E
        position_dec = shifts[1] / 3600. / 1000 + position_N

    else:

        position_ra = shifts[0] / telescope.pixel_scale + position_E
        position_dec = shifts[1] / telescope.pixel_scale + position_N

    return position_ra, position_dec


def lens_astrometric_positions(model, telescope, pyLIMA_parameters, time_ref=None):
    """
    The lens astrometric positions

    Parameters
    ----------
    model : object, the microlensing model object to compute the source trajectory
     relative to lens
    telescope : object, the telescope object containing astrometric data
    pyLIMA_parameters : dict, the dictionnary containing the model parameters
    time_ref : float, time of reference chosen, default is t0

    Returns
    -------
    position_ra : array, the astrometric position in North, degree or pixels
    position_dec : array, the astrometric position in East, degree or pixels
    """
    source_EN = source_astrometric_positions(telescope, pyLIMA_parameters, shifts=None,
                                             time_ref=time_ref)

    (source1_trajectory_x, source1_trajectory_y,
    source2_trajectory_x, source2_trajectory_y,
    dseparation, dalpha) = model.sources_trajectory(
        telescope, pyLIMA_parameters,
        data_type='astrometry')

    source_relative_to_lens_EN = xy_shifts_to_NE_shifts(
        (source1_trajectory_x, source1_trajectory_y)
        , pyLIMA_parameters['piEN'], pyLIMA_parameters['piEE'])

    source_relative_to_lens_EN = (np.array(source_relative_to_lens_EN) *
                                  pyLIMA_parameters['theta_E']) #mas

    if telescope.astrometry['ra'].unit == 'deg':

        lens_EN = source_EN - source_relative_to_lens_EN / 3600. / 1000.

    else:

        lens_EN = source_EN - source_relative_to_lens_EN / telescope.pixel_scale

    return lens_EN
