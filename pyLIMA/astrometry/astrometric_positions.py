import numpy as np


def xy_shifts_to_NE_shifts(xy_shifts, piEN, piEE):
    """
    Transform the x,y positions into the North, East reference,
    i.e. rotate by phi = np.arctan2(piEE,piEN)

    :param array xy_shifts: the x and y positions
    :param float piEN : the North component of the parallax vector
    :param float piEE: a list of [string,float] indicating the xallarap model

    :return: Delta_ra, Delta_dec position in North and East
    :rtype: tuple, tuple of two array_like
    """
    centroid_x = xy_shifts[0]
    centroid_y = xy_shifts[1]

    angle = np.arctan2(piEE, piEN)

    Delta_dec = centroid_x * np.cos(angle) - np.sin(angle) * centroid_y
    Delta_ra = centroid_x * np.sin(angle) + np.cos(angle) * centroid_y

    return Delta_ra, Delta_dec


def astrometric_positions_of_the_source(telescope, pyLIMA_parameters, time_ref=None):
    """ The source astrometric positions without astrometric shifts

        :param object telescope: the telescope object containing astrometric data
        :param dictionnary pyLIMA_parameters : the dictionnary containing the model
        parameters
        :param float time_ref: time of reference chosen, default is t0

        :return: position_ra, position_dec astrometric position in North and East, units
        depend of the data units (pixels or degrees)
        :rtype: tuple, tuple of two array_like
    """

    time = telescope.astrometry['time'].value

    if time_ref is None:
        time_ref = pyLIMA_parameters.t0

    ref_N = getattr(pyLIMA_parameters, 'position_source_N_' + telescope.name)
    ref_E = getattr(pyLIMA_parameters, 'position_source_E_' + telescope.name)
    mu_N = pyLIMA_parameters.mu_source_N
    mu_E = pyLIMA_parameters.mu_source_E

    earth_vector = telescope.Earth_positions_projected['astrometry']
    parallax_source = pyLIMA_parameters.parallax_source
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
    """ The source astrometric positions with or without astrometric shifts

        :param object telescope: the telescope object containing astrometric data
        :param dictionnary pyLIMA_parameters : the dictionnary containing the model
        parameters
        :param array shifts : arrays of North and East astrometric shifts in mas or
        pixel
        :param float time_ref: time of reference chosen, default is t0

        :return: position_ra, position_dec, astrometric position of the source in North
        and East, units depend of the data units (pixels or degrees)
        :rtype: tuple, tuple of two array_like
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
    """ The lens astrometric positions

        :param object model: the microlensing model object to compute the source
        trajectory relative to lens
        :param object telescope: the telescope object containing astrometric data
        :param dictionnary pyLIMA_parameters : the dictionnary containing the model
        parameters
        :param float time_ref: time of reference chosen, default is t0

        :return: position_ra, position_dec astrometric position of the lens in North
        and East, units depend of the data units (pixels or degrees)
        :rtype: tuple, tuple of two array_like
    """
    source_EN = source_astrometric_positions(telescope, pyLIMA_parameters, shifts=None,
                                             time_ref=time_ref)

    source_relative_to_lens = model.source_trajectory(telescope, pyLIMA_parameters,
                                                      data_type='astrometry')

    source_relative_to_lens_EN = xy_shifts_to_NE_shifts(
        (source_relative_to_lens[0], source_relative_to_lens[1])
        , pyLIMA_parameters.piEN, pyLIMA_parameters.piEE)

    lens_EN = np.array(source_relative_to_lens_EN) * pyLIMA_parameters.theta_E

    if telescope.astrometry['ra'].unit == 'deg':

        lens_EN = source_EN - lens_EN / 3600. / 1000.

    else:

        lens_EN = source_EN - lens_EN / telescope.pixel_scale

    return lens_EN
