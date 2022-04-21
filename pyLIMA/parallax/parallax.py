import numpy as np
from astropy import constants as astronomical_constants
from scipy import interpolate

from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation,spherical_to_cartesian, cartesian_to_spherical
from astropy.coordinates import get_body, get_body_barycentric_posvel

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


class MLParallaxes(object):
    """
    ######## Parallax module ########


    This module compute the parallax shifts due to different parallax effects.

    Attributes :

    event : the event object on which you perform the fit on. More details on the event module.

    parallax_model : The parallax effect you want to fit. Have to be a list containing the
    parallax model name
    and the reference time to_par (in JD unit). Example : ['Annual',2457000.0]

    AU : the astronomical unit,  as defined by astropy (in meter)

    speed_of_light : the speed light c,  as defined by astropy (in meter/second)

    Earth_radius : the Earth equatorial radius,  as defined by astropy (in meter)

    target_angles_in_the_sky : a list containing [RA,DEC] of the target in radians unit.

   :param event: the event object on which you perform the fit on. More details on the event module.
   :param parallax_model: The parallax effect you want to fit. Have to be a list containing the
   parallax model name
        and the to_par value. Example : ['Annual',2457000.0]

    """

    def __init__(self, event_ra, event_dec, parallax_model):
        """Initialization of the attributes described above."""

        self.parallax_model = parallax_model[0]
        self.to_par = parallax_model[1]
        self.AU = astronomical_constants.au.value
        self.speed_of_light = astronomical_constants.c.value
        self.Earth_radius = astronomical_constants.R_earth.value

        self.target_angles_in_the_sky = [event_ra * np.pi / 180, event_dec * np.pi / 180]
        self.North_East_vectors_target()

    def North_East_vectors_target(self):
        """This function define the North and East vectors projected on the sky plane
        perpendicular to the line
        of sight (i.e the line define by RA,DEC of the event).
        """
        target_angles_in_the_sky = self.target_angles_in_the_sky
        Target = np.array(
            [np.cos(target_angles_in_the_sky[1]) * np.cos(target_angles_in_the_sky[0]),
             np.cos(target_angles_in_the_sky[1]) * np.sin(target_angles_in_the_sky[0]),
             np.sin(target_angles_in_the_sky[1])])

        self.East = np.array(
            [-np.sin(target_angles_in_the_sky[0]), np.cos(target_angles_in_the_sky[0]), 0.0])
        self.North = np.cross(Target, self.East)


    def parallax_combination(self, telescope):
        """ Compute, and set, the deltas_positions attributes of the telescope object
       inside the list of telescopes. deltas_positions is the offset between the position of the
       observatory at the time t, and the
       center of the Earth at the date to_par. More details on each parallax functions.

       :param object telescope:  a telescope object on which you want to set the deltas_positions
       due to parallax.

       """

        location = telescope.location

        time = telescope.lightcurve_flux['time'].value
        delta_North = np.array([])
        delta_East = np.array([])

        if location == 'NewHorizon':
            delta_North, delta_East = self.lonely_satellite(time, telescope)

        if location == 'Earth':

            if (self.parallax_model == 'Annual'):
                telescope_positions = self.annual_parallax(time)

                delta_North = np.append(delta_North, telescope_positions[0])
                delta_East = np.append(delta_East, telescope_positions[1])

            if (self.parallax_model == 'Terrestrial'):
                altitude = telescope.altitude
                longitude = telescope.longitude
                latitude = telescope.latitude

                telescope_positions = -self.terrestrial_parallax(time, altitude, longitude, latitude)

                delta_North = np.append(delta_North, telescope_positions[0])
                delta_East = np.append(delta_East, telescope_positions[1])

            if (self.parallax_model == 'Full'):
                telescope_positions = self.annual_parallax(time)

                delta_North = np.append(delta_North, telescope_positions[0])
                delta_East = np.append(delta_East, telescope_positions[1])

                altitude = telescope.altitude
                longitude = telescope.longitude
                latitude = telescope.latitude

                telescope_positions = -self.terrestrial_parallax(time, altitude, longitude, latitude)

                delta_North += telescope_positions[0]
                delta_East += telescope_positions[1]

        if location == 'Space':
            telescope_positions = self.annual_parallax(time)
            delta_North = np.append(delta_North, telescope_positions[0])
            delta_East = np.append(delta_East, telescope_positions[1])
            name = telescope.spacecraft_name

            telescope_positions = -self.space_parallax(time, name, telescope)
            # import pdb;
            # pdb.set_trace()
            # delta_North = np.append(delta_North, telescope_positions[0])
            # delta_East = np.append(delta_East, telescope_positions[1])

            delta_North += telescope_positions[0]
            delta_East += telescope_positions[1]

        deltas_position = np.array([delta_North, delta_East])

        # set the attributes deltas_positions for the telescope object
        telescope.deltas_positions = deltas_position

    def annual_parallax(self, time_to_treat):
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
            time_jd_reference = Time(self.to_par, format='jd')
            Earth_position_time_reference = get_body_barycentric_posvel('Earth', time_jd_reference)
            Sun_position_time_reference = -Earth_position_time_reference[0]
            Sun_speed_time_reference = -Earth_position_time_reference[1]

            time_jd = Time(time_to_treat, format='jd')
            Earth_position = get_body_barycentric_posvel('Earth', time_jd)
            Sun_position = -Earth_position[0]

            delta_Sun = Sun_position.xyz.value.T - np.c_[
                time_to_treat - self.to_par] * Sun_speed_time_reference.xyz.value \
                        - Sun_position_time_reference.xyz.value

            delta_Sun_projected = np.array(
                [np.dot(delta_Sun, self.North), np.dot(delta_Sun, self.East)])

            return delta_Sun_projected

    def terrestrial_parallax(self, time_to_treat, altitude, longitude, latitude):
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

        radius = (self.Earth_radius + altitude) / self.AU
        Longitude = longitude * np.pi / 180.0
        Latitude = latitude * np.pi / 180.0

        #delta_telescope = []
        #for time in time_to_treat:
        #    time_mjd = time - 2400000.5
        #    sideral_time = slalib.sla_gmst(time_mjd)
        #    telescope_longitude = - Longitude - self.target_angles_in_the_sky[
        #        0] + sideral_time

        #    delta_telescope.append(radius * slalib.sla_dcs2c(telescope_longitude, Latitude))
        #    import pdb;
        #    pdb.set_trace()
        times = Time(time_to_treat, format='jd')
        sideral_times = times.sidereal_time('apparent','greenwich').value/24*2*np.pi
        telescope_longitudes = - Longitude - self.target_angles_in_the_sky[0] + sideral_times

        x,y,z = spherical_to_cartesian(radius, Latitude, telescope_longitudes)
        delta_telescope = np.c_[x.value,y.value,z.value]
        delta_telescope_projected = np.array(
            [np.dot(delta_telescope, self.North), np.dot(delta_telescope, self.East)])
        return delta_telescope_projected

    def space_parallax(self, time_to_treat, satellite_name, telescope):
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

        tstart = min(time_to_treat) - 1
        tend = max(time_to_treat) + 1
        if len(telescope.spacecraft_positions) != 0:
            # allow to pass the user to give his own ephemeris
            satellite_positions = np.array(telescope.spacecraft_positions)
        else:
            # call JPL!
            import pyLIMA.parallax.JPL_ephemerides
            satellite_positions = pyLIMA.parallax.JPL_ephemerides.produce_horizons_ephem(satellite_name, tstart, tend, observatory='Geocentric',
                                                         step_size='1440m', verbose=False)[1]
            telescope.spacecraft_positions = np.array(satellite_positions).astype(float)
        satellite_positions = np.array(satellite_positions)
        dates = satellite_positions[:, 0].astype(float)
        ra = satellite_positions[:, 1].astype(float)
        dec = satellite_positions[:, 2].astype(float)
        distances = satellite_positions[:, 3].astype(float)

        interpolated_ra = interpolate.interp1d(dates, ra)
        interpolated_dec = interpolate.interp1d(dates, dec)
        interpolated_distance = interpolate.interp1d(dates, distances)

        ra_interpolated = interpolated_ra(time_to_treat)
        dec_interpolated = interpolated_dec(time_to_treat)
        distance_interpolated = interpolated_distance(time_to_treat)

        #delta_satellite = []
        #for index_time in range(len(time_to_treat)):
        #    delta_satellite.append(distance_interpolated[index_time] * slalib.sla_dcs2c(
        #        ra_interpolated[index_time] * np.pi / 180,
        #        dec_interpolated[index_time] * np.pi / 180))

        x, y, z = spherical_to_cartesian(distance_interpolated,  dec_interpolated* np.pi / 180,
                                         ra_interpolated * np.pi / 180)
        delta_satellite = np.c_[x.value, y.value, z.value]
        #delta_satellite = np.array(delta_satellite)
        delta_satellite_projected = np.array(
            [np.dot(delta_satellite, self.North), np.dot(delta_satellite, self.East)])

        return delta_satellite_projected

    def lonely_satellite(self, time_to_treat, telescope):
        """
        """

        satellite_positions = np.array(telescope.spacecraft_positions)

        dates = satellite_positions[:, 0].astype(float)
        X = satellite_positions[:, 1].astype(float)
        Y = satellite_positions[:, 2].astype(float)
        Z = satellite_positions[:, 3].astype(float)

        interpolated_X = interpolate.interp1d(dates, X)
        interpolated_Y = interpolate.interp1d(dates, Y)
        interpolated_Z = interpolate.interp1d(dates, Z)

        spacecraft_position_time_reference = np.array(
            [interpolated_X(self.to_par), interpolated_Y(self.to_par), interpolated_Z(self.to_par)])
        spacecraft_position_time_reference1 = np.array(
            [interpolated_X(self.to_par - 1), interpolated_Y(self.to_par - 1), interpolated_Z(self.to_par - 1)])
        spacecraft_position_time_reference2 = np.array(
            [interpolated_X(self.to_par + 1), interpolated_Y(self.to_par + 1), interpolated_Z(self.to_par + 1)])
        spacecraft_speed_time_reference = (
                                                      spacecraft_position_time_reference2 - spacecraft_position_time_reference1) / 2
        delta_spacecraft = []

        for time in time_to_treat:
            sat_position = np.array([interpolated_X(time), interpolated_Y(time), interpolated_Z(time)])

            delta_satellite = sat_position - (
                        time - self.to_par) * spacecraft_speed_time_reference - spacecraft_position_time_reference

            delta_spacecraft.append(delta_satellite.tolist())

        delta_Sat = np.array(delta_spacecraft)

        delta_spacecraft_projected = np.array(
            [np.dot(delta_Sat, self.North), np.dot(delta_Sat, self.East)])

        return delta_spacecraft_projected
