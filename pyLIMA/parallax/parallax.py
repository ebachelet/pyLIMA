import numpy as np
from astropy import constants as astronomical_constants
from scipy import interpolate
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, spherical_to_cartesian

from pyLIMA.parallax import astropy_ephemerides

AU = astronomical_constants.au.value
SPEED_OF_LIGHT = astronomical_constants.c.value
EARTH_RADIUS = astronomical_constants.R_earth.value


def EN_trajectory_angle(piEN, piEE):
    """Find the angle between the North vector and the lens trajectory (at t0par). See Gould2004, RESOLUTION OF THE MACHO-LMC-5 PUZZLE: THE JERK-PARALLAX MICROLENS DEGENERACY

    :param float piEN: the North parallax component
    :param float piEE: the East parallax component

    :return: the angle in radians
    :rtype: float
    """

    angle = np.arctan2(piEE, piEN)

    return angle


def compute_parallax_curvature(piE, delta_positions):
    """ Compute the curvature induce by the parallax of from
    deltas_positions of a telescope.

    :param array_like piE: the microlensing parallax vector. Have a look :
                           http://adsabs.harvard.edu/abs/2004ApJ...606..319G
    :param array_like delta_positions: the delta_positions of the telescope. More details in microlparallax module.
    :return: delta_tau and delta_u, the shift introduce by parallax
    :rtype: array_like,array_like
    """

    delta_tau = np.dot(piE, delta_positions)
    delta_beta = np.cross(piE, delta_positions.T)

    return delta_tau, delta_beta

def parallax_combination(telescope, parallax_model, North_vector, East_vector):
        """ Compute, and set, the deltas_positions attributes of the telescope object
        inside the list of telescopes. deltas_positions is the offset between the position of the
        observatory at the time t, and the
        center of the Earth at the date to_par. More details on each parallax functions.

            :param object telescope:  a telescope object on which you want to set the deltas_positions
            due to parallax.

        """

        for data_type in ['astrometry', 'photometry']:

            delta_North = 0
            delta_East = 0

            if data_type == 'photometry':

                data = telescope.lightcurve_flux
            else:

                data = telescope.astrometry

            if data is not None:

                time = data['time'].value
                earth_positions = telescope.Earth_positions[data_type]
                Earth_projected_North = np.dot(earth_positions, North_vector)
                Earth_projected_East = np.dot(earth_positions, East_vector)

                telescope.Earth_positions_projected[data_type] = np.array([Earth_projected_North, Earth_projected_East])

                earth_speeds = telescope.Earth_positions[data_type]
                Earth_projected_North = np.dot(earth_speeds, North_vector)
                Earth_projected_East = np.dot(earth_speeds, East_vector)

                telescope.Earth_speeds_projected[data_type] = np.array(
                    [Earth_projected_North, Earth_projected_East])

                if (parallax_model[0] == 'Annual') | (parallax_model[0] == 'Full'):

                        annual_positions = annual_parallax(time, earth_positions, parallax_model[1])

                        delta_North += np.dot(annual_positions, North_vector)
                        delta_East += np.dot(annual_positions, East_vector)

                if (parallax_model[0] == 'Terrestrial') | (parallax_model[0] == 'Full') \
                        | (telescope.location == 'Space'):

                    telescope_positions = telescope.telescope_positions[data_type]
                    delta_North += np.dot(telescope_positions, North_vector)
                    delta_East += np.dot(telescope_positions, East_vector)

                deltas_position = np.array([delta_North, delta_East])

                telescope.deltas_positions[data_type] = deltas_position


def Earth_ephemerides(time_to_treat):
        """Compute the position shift due to the Earth movement. Please have a look on :
        "Resolution of the MACHO-LMC-5 Puzzle: The Jerk-Parallax Microlens Degeneracy"
        Gould, Andrew 2004. http://adsabs.harvard.edu/abs/2004ApJ...606..319G

        :param  time_to_treat: a numpy array containing the time where you want to compute this
        effect.
        :return: the shift induce by the Earth motion around the Sun
        :rtype: array_like

        **WARNING** : this is a geocentric point of view.
                      slalib use MJD time definition, which is MJD = JD-2400000.5
        """
        with solar_system_ephemeris.set('builtin'):

            Earth_ephemeris = astropy_ephemerides.Earth_ephemerides(time_to_treat)
            Earth_positions = Earth_ephemeris[0].xyz.value.T
            Earth_speeds = Earth_ephemeris[1].xyz.value.T

            return Earth_positions, Earth_speeds

def Earth_telescope_sidereal_times(time_to_treat, sidereal_type='mean'):
        """ Compute the position shift due to the distance of the obervatories from the Earth
        center.
        Please have a look on :
        "Parallax effects in binary microlensing events"
        Hardy, S.J and Walker, M.A. 1995. http://adsabs.harvard.edu/abs/1995MNRAS.276L..79H

        :param  time_to_treat: a numpy array containing the time where you want to compute this
        effect.
        :param altitude: the altitude of the telescope in meter
        :param longitude: the longitude of the telescope in degree
        :param latitude: the latitude of the telescope in degree
        :return: the shift induce by the distance of the telescope to the Earth center.
        :rtype: array_like

        **WARNING** : slalib use MJD time definition, which is MJD = JD-2400000.5
        """

        times = Time(time_to_treat, format='jd')
        sideral_times = times.sidereal_time(sidereal_type,'greenwich').value/24*2*np.pi

        return sideral_times

def space_ephemerides(telescope, time_to_treat, data_type='photometry'):
        """ Compute the position shift due to the distance of the obervatories from the Earth
        center.
        Please have a look on :
        "Parallax effects in binary microlensing events"
        Hardy, S.J and Walker, M.A. 1995. http://adsabs.harvard.edu/abs/1995MNRAS.276L..79H

        :param  time_to_treat: a numpy array containing the time where you want to compute this
        effect.
        :param satellite_name: the name of the observatory. Have to be recognize by JPL HORIZON.
        :return: the shift induce by the distance of the telescope to the Earth center.
        :rtype: array_like
        **WARNING** : slalib use MJD time definition, which is MJD = JD-2400000.5
        """

        satellite_name = telescope.spacecraft_name

        if len(telescope.spacecraft_positions[data_type]) != 0:

            spacecraft_positions = telescope.spacecraft_positions[data_type]

        else:

            # call JPL!
            from pyLIMA.parallax import JPL_ephemerides
            spacecraft_positions = JPL_ephemerides.horizons_API(satellite_name, time_to_treat, observatory='Geocentric')[1]

        satellite_positions = np.array(spacecraft_positions)
        dates = satellite_positions[:, 0].astype(float)
        ra = satellite_positions[:, 1].astype(float)
        dec = satellite_positions[:, 2].astype(float)
        distances = satellite_positions[:, 3].astype(float)

        x, y, z = spherical_to_cartesian(r=distances, lat=dec * np.pi / 180, lon=ra * np.pi / 180)

        interpolated_x = interpolate.interp1d(dates, x)
        interpolated_y = interpolate.interp1d(dates, y)
        interpolated_z = interpolate.interp1d(dates, z)

        x_value = interpolated_x(time_to_treat)
        y_value = interpolated_y(time_to_treat)
        z_value = interpolated_z(time_to_treat)

        satellite_positions = -np.c_[x_value, y_value, z_value]

        return satellite_positions, spacecraft_positions

def annual_parallax(time_to_treat, earth_positions, t0_par):

        """Compute the position shift due to the Earth movement. Please have a look on :
        "Resolution of the MACHO-LMC-5 Puzzle: The Jerk-Parallax Microlens Degeneracy"
        Gould, Andrew 2004. http://adsabs.harvard.edu/abs/2004ApJ...606..319G

        :param  time_to_treat: a numpy array containing the time where you want to compute this
        effect.
        :return: the shift induce by the Earth motion around the Sun
        :rtype: array_like

        **WARNING** : this is a geocentric point of view.
                      slalib use MJD time definition, which is MJD = JD-2400000.5
        """

        Earth_position_time_reference = Earth_ephemerides(t0_par)
        Sun_position_time_reference = -Earth_position_time_reference[0]
        Sun_speed_time_reference = -Earth_position_time_reference[1]

        Sun_position = -earth_positions
        delta_Sun = Sun_position- np.c_[time_to_treat - t0_par] * Sun_speed_time_reference - Sun_position_time_reference

        return delta_Sun

def terrestrial_parallax(sidereal_times, altitude, longitude, latitude):
    """ Compute the position shift due to the distance of the obervatories from the Earth
    center.
    Please have a look on :
    "Parallax effects in binary microlensing events"
    Hardy, S.J and Walker, M.A. 1995. http://adsabs.harvard.edu/abs/1995MNRAS.276L..79H

    :param  time_to_treat: a numpy array containing the time where you want to compute this
    effect.
    :param altitude: the altitude of the telescope in meter
    :param longitude: the longitude of the telescope in degree
    :param latitude: the latitude of the telescope in degree
    :return: the shift induce by the distance of the telescope to the Earth center.
    :rtype: array_like

    **WARNING** : slalib use MJD time definition, which is MJD = JD-2400000.5
    """

    radius = (EARTH_RADIUS + altitude) / AU
    Longitude = longitude * np.pi / 180.0
    Latitude = latitude * np.pi / 180.0

    telescope_longitudes = Longitude - sidereal_times

    x,y,z = spherical_to_cartesian(r=radius, lat=Latitude, lon=telescope_longitudes)
    #breakpoint()
    delta_telescope = -np.c_[x.value,y.value,z.value]

    return delta_telescope

