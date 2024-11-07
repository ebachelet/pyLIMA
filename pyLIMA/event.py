import sys

import numpy as np


class EventException(Exception):
    pass


class Event(object):
    """
    This class contains all information relative to a microlensing event, including
    position in the sky, telescopes that observed etc...

    Attributes
    ----------

    kind : str, the type of event (i.e. Microlensing)
    name : str, the event name
    ra : float, the right ascension in degree
    dec : float, the declination in degree
    North : array, the North vector projected in the plane of sky
    East : array, the East vector projected in the plane of sky
    telescopes : list, a list of telescope object
    survey : str, the survey associated to the event, to align plot to

    """

    def __init__(self, ra=266.416792, dec=-29.007806):

        self.kind = 'Microlensing'
        self.name = 'Sagittarius A*'
        self.ra = ra
        self.dec = dec
        self.North = []
        self.East = []
        self.telescopes = []
        self.survey = None

        self.North_East_vectors()

    def telescopes_names(self):
        """
        Print the telescopes names
        """
        print([self.telescopes[i].name for i in range(len(self.telescopes))])

    def check_event(self):
        """
        Check some basics informations.
        """
        for telescope in self.telescopes:

            if (telescope.lightcurve is None) & (
                    telescope.astrometry is None):
                print(
                    'WARNING : The telescope ' + telescope.name + ' is empty (no '
                                                                  'lightcurves or '
                                                                  'astrometry'
                                                                  ', it is useless, '
                                                                  'so I deleted it)')

                self.telescopes.remove(telescope)

        if not isinstance(self.name, str):
            raise EventException('ERROR : The event name (' + str(
                self.name) + ') is not correct, it has to be a string')

        if (self.ra > 360) or (self.ra < 0):
            raise EventException('ERROR : The event ra (' + str(
                self.ra) + ') is not correct, it has to be a float between 0 and 360 '
                           'degrees')

        if (self.dec > 90) or (self.dec < -90):
            raise EventException('ERROR : The event dec (' + str(
                self.dec) + ') is not correct, it has to be between -90 and 90 degrees')

        if len(self.telescopes) == 0:
            raise EventException(
                'There is no telescope associated to your event, no fit possible!')

        else:

            for telescope in self.telescopes:

                if len(telescope.lightcurve) == 0:
                    print(
                        'ERROR : There is no associated lightcurve in magnitude or '
                        'flux with ' \
                        'this telescopes : ' \
                        + telescope.name + ', add one with telescope.lightcurve = '
                                           'your_data')
                    raise EventException(
                        'There is no lightcurve associated to the  telescope ' + str(
                            telescope.name) + ', no fit possible!')

        print(sys._getframe().f_code.co_name, ' : Everything looks fine...')

    def find_survey(self, choice=None):
        """
        Find the survey telescope and place it first in the telescopes list

        Parameters
        ----------
        choice : str, the survey name
        """
        self.survey = choice or self.telescopes[0].name

        names = [telescope.name for telescope in self.telescopes]
        if any(self.survey in name for name in names):

            index = np.where(self.survey == np.array(names))[0]
            sorting = np.arange(0, len(self.telescopes))
            sorting = np.delete(sorting, index)
            sorting = np.insert(sorting, 0, index)
            self.telescopes = [self.telescopes[i] for i in sorting]

        else:

            print('ERROR : There is no telescope names containing ' + self.survey)
            return

    def compute_parallax_all_telescopes(self, parallax_model):
        """
        Launch the parallax computation for all telescopes

        Parameters
        ----------
        parallax_model : list, [str,float] the parallax model
        """

        for telescope in self.telescopes:
            telescope.compute_parallax(parallax_model, self.North, self.East)

    def total_number_of_data_points(self):
        """
        Return the total number of photometric observations

        Returns
        -------
        n_data : float, the total number of data points
        """
        n_data = 0.0

        for telescope in self.telescopes:
            n_data = n_data + telescope.n_data('flux')

        return n_data

    def North_East_vectors(self):
        """
        Compute the North,East vectors in the sky
        """
        target_angles_in_the_sky = [self.ra * np.pi / 180, self.dec * np.pi / 180]
        Target = np.array(
            [np.cos(target_angles_in_the_sky[1]) * np.cos(target_angles_in_the_sky[0]),
             np.cos(target_angles_in_the_sky[1]) * np.sin(target_angles_in_the_sky[0]),
             np.sin(target_angles_in_the_sky[1])])

        self.East = np.array(
            [-np.sin(target_angles_in_the_sky[0]), np.cos(target_angles_in_the_sky[0]),
             0.0])
        self.North = np.cross(Target, self.East)
