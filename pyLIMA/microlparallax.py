# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:49:44 2015

@author: ebachelet
"""
from __future__ import division
import telnetlib

import numpy as np
from astropy import constants as const
from scipy import interpolate
from astropy.time import Time
from pyslalib import slalib


def horizons_obscodes(observatory):
    """Map LCOGT sitecodes to those needed for JPL HORIZONS.
    McDonald Observatory (ELP/V37) is used as a default if the look-up fails"""

    JPL_HORIZONS_ID = {'ALMA': '-7',
                       'VLA': '-5',
                       'GBT': '-9',
                       'MAUNAKEA': '-80',
                       'OVRO': '-81',
                       'ELP': 'V37',
                       'FTN': 'F65',
                       'FTS': 'E10',
                       'SQA': 'G51',
                       'W85': 'W85',
                       'W86': 'W86',
                       'W87': 'W87',
                       'CPT': 'K92',
                       'K91': 'K91',
                       'K92': 'K92',
                       'K93': 'K93',
                       'Geocentric': '500',
		       'Kepler':'-227'
                       }

    # Check if we were passed the JPL site code directly
    if (observatory in JPL_HORIZONS_ID.values()):
        OBSERVATORY_ID = observatory
    else:
        # Lookup observatory name in map, use ELP's code as a default if not found
        OBSERVATORY_ID = JPL_HORIZONS_ID.get(observatory, 'V37')

    return OBSERVATORY_ID


def optcallback(socket, command, option):
    cnum = ord(command)
    onum = ord(option)
    if cnum == telnetlib.WILL:  # and onum == ECHO:
        socket.write(telnetlib.IAC + telnetlib.DONT + onum)
    if cnum == telnetlib.DO and onum == telnetlib.TTYPE:
        socket.write(telnetlib.IAC + telnetlib.WONT + telnetlib.TTYPE)


class MLParallaxes(object):
    def __init__(self, event, model):
        """ Initialization of the attributes described above. """
        self.AU = const.au.value
        self.speed_of_light = const.c.value
        self.Earth_radius = const.R_earth.value
        self.event = event
        self.model = model[0]
        self.topar = model[1]
        self.delta_tau = []
        self.delta_u = []
        self.target_angles = [self.event.ra * np.pi / 180, self.event.dec * np.pi / 180]

    def N_E_vectors_target(self):

        target_angles = self.target_angles
        Target = np.array([np.cos(target_angles[1]) * np.cos(target_angles[0]),
                           np.cos(target_angles[1]) * np.sin(target_angles[0]),
                           np.sin(target_angles[1])])

        self.East = np.array([-np.sin(target_angles[0]), np.cos(target_angles[0]), 0.0])
        self.North = np.cross(Target, self.East)

    def HJD_to_JD(self, t):

        AU = self.AU
        light_speed = self.speed_of_light

        time_correction = []
        # DTT=[]
        t = t

        for i in t:

            count = 0
            jd = np.copy(i)

            while count < 3:

                Earth_position = slalib.sla_epv(jd)
                Sun_position = -Earth_position[0]

                Sun_angles = slalib.sla_dcc2s(Sun_position)
                target_angles = self.target_angles

                t_correction = np.sqrt(Sun_position[0] ** 2 + Sun_position[1] ** 2 + Sun_position[
                    2] ** 2) * AU / light_speed * (
                               np.sin(Sun_angles[1]) * np.sin(target_angles[1]) + np.cos(
                                   Sun_angles[1]) * np.cos(target_angles[1]) * np.cos(
                                   target_angles[0] - Sun_angles[0])) / (3600 * 24.0)
                count = count + 1

        # DTT.append(slalib.sla_dtt(jd)/(3600*24))
        time_correction.append(t_correction)

        JD = t + np.array(time_correction)

        return JD

    def parallax_combination(self, telescopes):
       
	
        for i in telescopes:

            self.N_E_vectors_target()
            delta_position_North = np.array([])
            delta_position_East = np.array([])

            kind = i.location
            #t = self.HJD_to_JD(i.lightcurve_flux[:,0])
            #t = i.lightcurve_flux[:, 0]
	    t = i.lightcurve_flux[:,0]
            delta_North = np.array([])
            delta_East = np.array([])

            if kind == 'Earth':

                if (self.model == 'Annual'):

                    positions = self.annual_parallax(t)
                    # import pdb; pdb.set_trace()
                    delta_North = np.append(delta_North, positions[0])
                    delta_East = np.append(delta_East, positions[1])

                if (self.model == 'Terrestrial'):

                    altitude = i.altitude
                    longitude = i.longitude
                    latitude = i.latitude

                    positions = self.terrestrial_parallax(t, altitude, longitude, latitude)
                    delta_North = np.append(delta_North, positions[0])
                    delta_East = np.append(delta_East, positions[1])

                if (self.model == 'Full'):

                    positions = self.annual_parallax(t)
                    delta_North = np.append(delta_North, positions[0])
                    delta_East = np.append(delta_East, positions[1])

                    altitude = i.altitude
                    longitude = i.longitude
                    latitude = i.latitude
		    import pdb; pdb.set_trace()
                    positions = self.terrestrial_parallax(t, altitude, longitude, latitude)
                    delta_North = np.append(delta_North, positions[0])
                    delta_East = np.append(delta_East, positions[1])


            else:
                positions = self.annual_parallax(t)
                delta_North = np.append(delta_North, positions[0])
                delta_East = np.append(delta_East, positions[1])
                name = i.name
                try:
                    positions = self.space_parallax(t, name)
                except:
                    print 'TIME OUT CONNECTION TO JPL'
                    import pdb;
                    pdb.set_trace()

                    positions = self.space_parallax(t, name)
		            
		delta_North = delta_North + positions[0]
                delta_East = delta_East + positions[1]
	    
            delta_position_North = np.append(delta_position_North, delta_North)
            delta_position_East = np.append(delta_position_East, delta_East)

            deltas_position = np.array([delta_position_North, delta_position_East])
            i.deltas_positions = deltas_position

    def annual_parallax(self, t):
        
        #topar=self.HJD_to_JD(np.array([self.topar]))-2400000.5
        #topar = self.topar - 2400000.5
	topar = self.topar - 2400000.5   
	Earth_position_ref = slalib.sla_epv(topar)
        Sun_position_ref = -Earth_position_ref[0]
        Sun_speed_ref = -Earth_position_ref[1]
        delta_Sun = []

        for i in t:

            tt = i - 2400000.5

            Earth_position = slalib.sla_epv(tt)
            Sun_position = -Earth_position[0]
            delta_sun = Sun_position - (tt - topar) * Sun_speed_ref - Sun_position_ref
            

            delta_Sun.append(delta_sun.tolist())

        delta_Sun = np.array(delta_Sun)
        delta_Sun_proj = np.array([np.dot(delta_Sun, self.North), np.dot(delta_Sun, self.East)])

        return delta_Sun_proj

    def terrestrial_parallax(self, t, altitude, longitude, latitude):

        radius = (self.Earth_radius + altitude)/self.AU
        Longitude = longitude * np.pi / 180.0
        Latitude = latitude * np.pi / 180.0
	
        delta_telescope = []
        for i in t:

            tt = i - 2400000.5
            sideral_time = slalib.sla_gmst(tt)
            telescope_longitude = - Longitude - self.target_angles[0]*np.pi/180 + sideral_time/24.0*np.pi
	    delta_telescope.append(radius*slalib.sla_dcs2c(telescope_longitude,Latitude))            


        delta_telescope = np.array(delta_telescope)
	delta_telescope_projected = np.array([np.dot(delta_telescope, self.North), np.dot(delta_telescope, self.East)])
        return delta_telescope_projected

    def space_parallax(self, t, name):
        # tstart = self.HJD_to_JD(np.array([t[0]]))
        # tend = self.HJD_to_JD(np.array([t[-1]]))
        #import pdb; pdb.set_trace()

        tstart = t[0] - 1
        tend = t[-1] + 1
        
        positions = self.produce_horizons_ephem(name, tstart, tend, observatory='Geocentric',step_size='60m', verbose=False)[1]
        #positions = np.loadtxt('SWIFT.dat')
        #positions = positions[:, :-1]
	positions = np.array(positions)
        dates = positions[:, 0].astype(float)
        ra = positions[:, 1].astype(float)
        dec = positions[:, 2].astype(float)
        distances = positions[:, 3].astype(float)
	
        interpol_ra = interpolate.interp1d(dates, ra)
        interpol_dec = interpolate.interp1d(dates, dec)
        interpol_dist = interpolate.interp1d(dates, distances)
        # times=self.HJD_to_JD(t)

        ra_interpolated = interpol_ra(t)
        dec_interpolated = interpol_dec(t)
        distance_interpolated = interpol_dist(t)

        
	delta_satellite = []
        for i in xrange(len(t)):

            tt = i
            #delta_North.append(distance_interpolated[tt] * (np.sin(dec_interpolated[tt]*np.pi/180) * np.cos(
            #    self.target_angles[1]) - np.cos(dec_interpolated[tt]*np.pi/180) * np.sin(
            #    self.target_angles[1]) * np.cos(ra_interpolated[tt]*np.pi/180-self.target_angles[0])))
            #delta_East.append(distance_interpolated[tt] * np.cos(dec_interpolated[tt]*np.pi/180) * np.sin(
            #    ra_interpolated[tt]*np.pi/180-self.target_angles[0]))
	    
       	    
	    delta_satellite.append(distance_interpolated[tt]*slalib.sla_dcs2c(ra_interpolated[tt]*np.pi/180,dec_interpolated[tt]*np.pi/180))
	
	delta_satellite=np.array(delta_satellite)
	delta_satellite_projected = np.array([np.dot(delta_satellite, self.North), np.dot(delta_satellite, self.East)])


        return delta_satellite_projected

    def parallax_outputs(self, PiE):

        piE = np.array(PiE)
        delta_tau = np.dot(piE, self.delta_position)
        delta_u = np.cross(piE, self.delta_position.T)

        return delta_tau, delta_u

    def produce_horizons_ephem(self, body, start_time, end_time, observatory='ELP', step_size='60m',
                               verbose=False):
        """
        Write by Tim Lister.
        example interactive session:
        telnet://horizons.jpl.nasa.gov:6775
        606 # = Titan
        e  # for ephemeris
        o  # for observables
        -7 # for ALMA
        y  # confirm
        2011-Apr-23 00:00  #  UT
        2011-Apr-23 01:00  #  UT
        10m #  interval
        y  # default output
        1,3,4,9,19,20,23 # RA/DEC and rates (Rarcsec/hour), Az & El, Vis. mag, Helio. range (r),
        Earth range (delta), Elong.
        space  # to get to next prompt
        q   # quit
        """
	body = str(-227)
        # Lookup observatory name
        OBSERVATORY_ID = horizons_obscodes(observatory)
        if (verbose):
            print "Observatory ID= ", OBSERVATORY_ID
	#import pdb; pdb.set_trace()
        tstart = Time(start_time,format='jd')

        tstart = 'JD' + str(tstart.isot)
	tstart = tstart[:12]+' '+tstart[13:-7]
	tstart ='JD' + str(start_time)


        if (verbose):
            print "tstart = ", tstart
        # tstop = end_time.strftime('%Y-%m-%d %H:%M')
        #tstop = Time(end_time,format='jd')

        tstop = 'JD' + str(end_time)
	#tstop = tstop[:12]+' '+tstop[13:-7]
        timeout = 5  # seconds
        t = telnetlib.Telnet('horizons.jpl.nasa.gov', 6775)
        t.set_option_negotiation_callback(optcallback)
        data = t.read_until('Horizons> ')
        if (verbose):
            print "data = ", data
            #        print "hex string = %s\n\n" % binascii.hexlify(data)
        while (data.find('Horizons>') < 0):
            t.write('\n')
            data = t.read_until('Horizons> ')
            if (verbose):
                print "data = ", data
        t.write(body + '\n')
        data = t.read_until('Select ... [E]phemeris, [F]tp, [M]ail, [R]edisplay, ?, <cr>: ',
                            timeout)
        if len(data) == 0:
            print 'No connection to JPL, sorry :('
            flag = 'No connection to JPL'
            return flag

        if (verbose):
            print "data = ", data
        if (data.find('phemeris') < 0):
            if (data.find('EXACT') >= 0):
                t.write('\n')
                data = t.read_until('Select ... [E]phemeris, [F]tp, [M]ail, [R]edisplay, ?, <cr>: ',
                                    timeout)
                if (verbose):
                    print data
                useID = ''
            else:
                # then we have a conflict in the name.
                # e.g. Titan vs. Titania, or Mars vs. Mars Barycenter
                # Try to resolve by forcing an exact match.
                lines = data.split('\n')
                if (verbose):
                    print "Multiple entries found, using exact match"
                    print "nlines = %d" % (len(lines))
                firstline = -1
                lastvalidline = -1
                l = 0
                useID = -1
                for line in lines:
                    if (verbose):
                        print line
                    if (line.find('-----') >= 0):
                        if (firstline == -1):
                            firstline = l + 1
                    else:
                        tokens = line.split()
                        if (firstline >= 0 and lastvalidline == -1):
                            if (len(tokens) < 2):
                                lastvalidline = l - 1
                            elif (tokens[1] == body and len(tokens) < 3):
                                # The <3 is necessary to filter out entries for a planet's
                                # barycenter
                                useID = int(tokens[0])
                                useBody = tokens[1]
                                if (verbose):
                                    print "Use instead the id = %s = %d" % (tokens[0], useID)
                    l = l + 1
                if (useID == -1):
                    # Try again with only the first letter capitalized, Probably not necessary
                    body = str.upper(body[0]) + str.lower(body[1:])
                    #          print "Try the exact match search again with body = ", body
                    firstline = -1
                    lastvalidline = -1
                    l = 0
                    for line in lines:
                        if (verbose):
                            print line
                        if (line.find('-----') >= 0):
                            if (firstline == -1):
                                firstline = l + 1
                        elif (firstline > 0):
                            if (verbose):
                                print "Splitting this line = %s" % (line)
                            tokens = line.split()
                            if (verbose):
                                print "length=%d,  %d tokens found" % (len(line), len(tokens))
                            if (firstline >= 0 and lastvalidline == -1):
                                if (len(tokens) < 2):
                                    # this is the final (i.e. blank) line in the list
                                    lastvalidline = l - 1
                                elif (tokens[1] == body):
                                    #                  print "%s %s is equal to %s." % (tokens[
                                    # 0],tokens[1],body)
                                    useID = int(tokens[0])
                                    useBody = tokens[1]
                                    if (len(tokens) < 3):
                                        if (verbose):
                                            print "Use instead the id = %s = %d" % (
                                            tokens[0], useID)
                                    elif (len(tokens[2].split()) < 1):
                                        if (verbose):
                                            print "Use instead the id = ", tokens[0]
                                else:
                                    if (verbose):
                                        print "%s %s is not equal to %s." % (
                                        tokens[0], tokens[1], body)
                        l = l + 1
                if (verbose):
                    print "line with first possible source = ", firstline
                    print "line with last possible source = ", lastvalidline
                    print "first possible source = ", (lines[firstline].split())[1]
                    print "last possible source = ", (lines[lastvalidline].split())[1]
                    print "Writing ", useID
                t.write(str(useID) + '\n')
                data = t.read_until('Select ... [E]phemeris, [F]tp, [M]ail, [R]edisplay, ?, <cr>: ')
                if (verbose):
                    print data
        else:
            useID = ''
        t.write('e\n')
        data = t.read_until('Observe, Elements, Vectors  [o,e,v,?] : ')
        if (verbose):
            print data
        t.write('o\n')
        data = t.read_until('Coordinate center [ <id>,coord,geo  ] : ')
        if (verbose):
            print data
        t.write('%s\n' % OBSERVATORY_ID)
        data = t.read_until('[ y/n ] --> ')
        pointer = data.find('----------------')
        ending = data[pointer:]
        lines = ending.split('\n')
        try:
            if (verbose):
                print "Parsing line = %s" % (lines)
            tokens = lines[1].split()
        except:
            print "Telescope code unrecognized by JPL."
            return ([], [], [])

        if (verbose):
            print data
        obsname = ''
        for i in range(4, len(tokens)):
            obsname += tokens[i]
            if (i < len(tokens) + 1):
                obsname += ' '
        print "Confirmed Observatory name = ", obsname
        if (useID != ''):
            print "Confirmed Target ID = %d = %s" % (useID, useBody)
        t.write('y\n')
        data = t.read_until('] : ', 1)
        if (verbose):
            print data

        t.write(tstart + '\n')
        data = t.read_until('] : ', 1)
        if (verbose):
            print data
        t.write(tstop + '\n')
        data = t.read_until(' ? ] : ', timeout)
        if (verbose):
            print data
        t.write(step_size + '\n')
        data = t.read_until(', ?] : ', timeout)
        if (verbose):
            print data
        if (1 == 1):
            # t.write('n\n1,3,4,9,19,20,23,\nJ2000\n\n\nMIN\nDEG\nYES\n\n\nYES\n\n\n\n\n\n\n\n')
            t.write('n\n1,20,\nJ2000\n\n\JD\nMIN\nDEG\nYES\n\n\nYES\n\n\n\n\n\n\n\n')
        else:
            t.write('y\n')  # accept default output?
            data = t.read_until(', ?] : ')  # ,timeout)
            if (verbose):
                print data
            t.write('1,3\n')

        t.read_until('$$SOE', timeout)
        data = t.read_until('$$EOE', timeout)
        if (verbose):
            print data

        t.close()
        lines = data.split('\n')
        horemp = []
        for hor_line in lines:
            if (verbose):
                print "hor_line = ", hor_line
                print len(hor_line.split())
            data_line = True
            # print hor_line

	   
            if (len(hor_line.split()) == 4):

                (time, raDegrees, decDegrees, light_dist) = hor_line.split()

            elif (len(hor_line.split()) == 0 or len(hor_line.split()) == 1):
                data_line = False
            else:
                data_line = False
                print "Wrong number of fields (", len(hor_line.split()), ")"
                print "hor_line=", hor_line
            if (data_line == True):

                horemp_line = [time, raDegrees, decDegrees, light_dist]
                if (verbose):
                    print horemp_line
                horemp.append(horemp_line)

                # Construct ephem_info
        ephem_info = {'obj_id': body,
                      'emp_sitecode': OBSERVATORY_ID,
                      'emp_timesys': '(UT)',
                      'emp_rateunits': '"/min'
                      }
        flag = 'Succes connection to JPL'
        return flag, horemp
