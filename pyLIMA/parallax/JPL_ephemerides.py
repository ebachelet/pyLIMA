import telnetlib



TIMEOUT_JPL = 120  # seconds. The time you allow telnetlib to discuss with JPL, see space_parallax.
JPL_TYPICAL_REQUEST_TIME_PER_LINE = 0.002  # seconds.


def optcallback(socket, command, option):
    """Write by Tim Lister, thanks :)
    """
    cnum = ord(command)
    onum = ord(option)
    if cnum == telnetlib.WILL:  # and onum == ECHO:
        socket.write(telnetlib.IAC + telnetlib.DONT + onum)
    if cnum == telnetlib.DO and onum == telnetlib.TTYPE:
        socket.write(telnetlib.IAC + telnetlib.WONT + telnetlib.TTYPE)

        ### Uncomment the following if the spacecraft dataset is huge! and also the global variables
        # at the begining of the module
        # width = struct.pack('H', MAX_WINDOW_WIDTH)
        # height = struct.pack('H', MAX_WINDOW_HEIGHT)
        # socket.send(telnetlib.IAC + telnetlib.SB + telnetlib.NAWS + width + height + telnetlib.IAC + telnetlib.SE)


def horizons_obscodes(observatory):
    """Transform observatory names to JPL horizon codes.
    Write by Tim Lister, thanks :)

    :param str observatory: the satellite name you would like to obtain ephemeris. As to be in the dictionnary
           JPL_HORIZONS_ID (exact name matching!).

    :return: the JPL ID of your satellite.
    :rtype: int
    """

    JPL_HORIZONS_ID = {
        'Geocentric': '500',
        'Kepler': '-227',
        'Spitzer': '-79',
        'HST': '-48',
        'Gaia': '-139479',
        'New Horizons': '-98'
    }

    # Check if we were passed the JPL site code directly
    if (observatory in list(JPL_HORIZONS_ID.values())):
        OBSERVATORY_ID = observatory
    else:
        # Lookup observatory name in map, use ELP's code as a default if not found
        OBSERVATORY_ID = JPL_HORIZONS_ID.get(observatory, 'V37')

    return OBSERVATORY_ID


def produce_horizons_ephem(body, start_time, end_time, observatory='ELP', step_size='60m',
                           verbose=False):
    """
    Write by Tim Lister. Thanks for sharing :) Produce RA,DEC and distance from the Geocentric Center.

    """
    # Lookup observatory name
    OBSERVATORY_ID = horizons_obscodes(observatory)
    body = horizons_obscodes(body)
    if (verbose):
        print("Observatory ID= ", OBSERVATORY_ID)

    tstart = 'JD' + str(start_time)

    if (verbose):
        print("tstart = ", tstart)

    tstop = 'JD' + str(end_time)
    # timeout = TIMEOUT_JPL
    expected_number_of_lines = (end_time - start_time) * 24
    timeout = max(JPL_TYPICAL_REQUEST_TIME_PER_LINE * expected_number_of_lines, 5)
    t = telnetlib.Telnet('horizons.jpl.nasa.gov', 6775)
    t.set_option_negotiation_callback(optcallback)
    data = t.read_until('Horizons> '.encode('utf-8'))
    if (verbose):
        print("data = ", data)
        #        print "hex string = %s\n\n" % binascii.hexlify(data)
    while (data.find('Horizons>'.encode('utf-8')) < 0):
        t.write('\n'.encode('utf-8'))
        data = t.read_until('Horizons> '.encode('utf-8'))
        if (verbose):
            print("data = ", data)
    t.write((body + '\n').encode('utf-8'))
    data = t.read_until('Select ... [E]phemeris, [F]tp, [M]ail, [R]edisplay, ?, <cr>: '.encode('utf-8'),
                        timeout)
    if len(data) == 0:
        print('No connection to JPL, sorry :(')
        flag = 'No connection to JPL'
        return flag

    if (verbose):
        print("data = ", data)
    if (data.find('phemeris'.encode('utf-8')) < 0):
        if (data.find('EXACT'.encode('utf-8')) >= 0):
            t.write('\n'.encode('utf-8'))
            data = t.read_until('Select ... [E]phemeris, [F]tp, [M]ail, [R]edisplay, ?, <cr>: '.encode('utf-8'),
                                timeout)
            if (verbose):
                print(data)
            useID = ''
        else:
            # then we have a conflict in the name.
            # e.g. Titan vs. Titania, or Mars vs. Mars Barycenter
            # Try to resolve by forcing an exact match.
            lines = data.split('\n')
            if (verbose):
                print("Multiple entries found, using exact match")
                print("nlines = %d" % (len(lines)))
            firstline = -1
            lastvalidline = -1
            l = 0
            useID = -1
            for line in lines:
                if (verbose):
                    print(line)
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
                                print("Use instead the id = %s = %d" % (tokens[0], useID))
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
                        print(line)
                    if (line.find('-----') >= 0):
                        if (firstline == -1):
                            firstline = l + 1
                    elif (firstline > 0):
                        if (verbose):
                            print("Splitting this line = %s" % (line))
                        tokens = line.split()
                        if (verbose):
                            print("length=%d,  %d tokens found" % (len(line), len(tokens)))
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
                                        print("Use instead the id = %s = %d" % (
                                            tokens[0], useID))
                                elif (len(tokens[2].split()) < 1):
                                    if (verbose):
                                        print("Use instead the id = ", tokens[0])
                            else:
                                if (verbose):
                                    print("%s %s is not equal to %s." % (
                                        tokens[0], tokens[1], body))
                    l = l + 1
            if (verbose):
                print("line with first possible source = ", firstline)
                print("line with last possible source = ", lastvalidline)
                print("first possible source = ", (lines[firstline].split())[1])
                print("last possible source = ", (lines[lastvalidline].split())[1])
                print("Writing ", useID)
            t.write((str(useID) + '\n').encode('utf-8'))
            data = t.read_until('Select ... [E]phemeris, [F]tp, [M]ail, [R]edisplay, ?, <cr>: '.encode('utf-8'))
            if (verbose):
                print(data)
    else:
        useID = ''

    t.write('e\n'.encode('utf-8'))
    data = t.read_until('Observe, Elements, Vectors  [o,e,v,?] : '.encode('utf-8'))
    if (verbose):
        print(data)
    t.write('o\n'.encode('utf-8'))
    data = t.read_until('Coordinate center [ <id>,coord,geo  ] : '.encode('utf-8'))
    if (verbose):
        print(data)
    t.write(('%s\n' % OBSERVATORY_ID).encode('utf-8'))
    data = t.read_until('[ y/n ] --> '.encode('utf-8'))
    pointer = data.find('----------------'.encode('utf-8'))
    ending = data[pointer:]
    lines = ending.split('\n'.encode('utf-8'))
    try:
        if (verbose):
            print("Parsing line = %s" % (lines))
        tokens = lines[1].split()
    except:
        print("Telescope code unrecognized by JPL.")
        return ([], [], [])

    if (verbose):
        print(data)
    obsname = ''
    for i in range(4, len(tokens)):
        obsname += (tokens[i]).decode('utf-8')
        if (i < len(tokens) + 1):
            obsname += ' '
    print("Confirmed Observatory name = ", obsname)
    if (useID != ''):
        print("Confirmed Target ID = %d = %s" % (useID, useBody))
    t.write('y\n'.encode('utf-8'))
    data = t.read_until('] : '.encode('utf-8'), 1)
    if (verbose):
        print(data)

    t.write((tstart + '\n').encode('utf-8'))
    data = t.read_until('] : '.encode('utf-8'), 1)
    if (verbose):
        print(data)
    t.write((tstop + '\n').encode('utf-8'))
    data = t.read_until(' ? ] : '.encode('utf-8'), timeout)
    if (verbose):
        print(data)
    t.write((step_size + '\n').encode('utf-8'))
    data = t.read_until(', ?] : '.encode('utf-8'), timeout)
    if (verbose):
        print(data)
    if (1 == 1):
        # t.write('n\n1,3,4,9,19,20,23,\nJ2000\n\n\nMIN\nDEG\nYES\n\n\nYES\n\n\n\n\n\n\n\n')
        t.write('n\n1,20,\nJ2000\n\n\JD\nMIN\nDEG\nYES\n\n\nYES\n\n\n\n\n\n\n100000\n\n'.encode('utf-8'))
    else:
        t.write('y\n'.encode('utf-8'))  # accept default output?
        data = t.read_until(', ?] : '.encode('utf-8'))  # ,timeout)
        if (verbose):
            print(data)
        t.write('1,3\n'.encode('utf-8'))

    t.read_until('$$SOE'.encode('utf-8'), timeout)
    data = t.read_until('$$EOE'.encode('utf-8'), timeout)
    if (verbose):
        print(data)

    t.close()
    lines = data.split('\n'.encode('utf-8'))
    horemp = []
    for hor_line in lines:
        if (verbose):
            print("hor_line = ", hor_line)
            print(len(hor_line.split()))
        data_line = True
        # print hor_line

        if (len(hor_line.split()) == 4):

            (time, raDegrees, decDegrees, light_dist) = hor_line.split()

        elif (len(hor_line.split()) == 0 or len(hor_line.split()) == 1):
            data_line = False
        else:
            data_line = False
            print("Wrong number of fields (", len(hor_line.split()), ")")
            print("hor_line=", hor_line)

        if (data_line == True):

            horemp_line = [time, raDegrees, decDegrees, light_dist]
            if (verbose):
                print(horemp_line)
            horemp.append(horemp_line)

            # Construct ephem_info
    ephem_info = {'obj_id': body,
                  'emp_sitecode': OBSERVATORY_ID,
                  'emp_timesys': '(UT)',
                  'emp_rateunits': '"/min'
                  }
    flag = 'Succes connection to JPL'
    print('Successfully ephemeris from JPL!')

    return flag, horemp



def horizons_API(body, start_time, end_time, observatory='ELP', step_size='60m'):
    # Lookup observatory name
    OBSERVATORY_ID = horizons_obscodes(observatory)
    body = horizons_obscodes(body)

    tstart = 'JD' + str(start_time)

    tstop = 'JD' + str(end_time)

    import sys
    import requests
    f = open(sys.argv[1])
    request = 'https: // ssd.jpl.nasa.gov / api / horizons.api?format = text & COMMAND = '499' & OBJ_DATA = 'YES' & MAKE_EPHEM = 'YES' & EPHEM_TYPE = 'OBSERVER' & CENTER = '500' & START_TIME = 'JD2457000' & STOP_TIME = 'JD2457200' & STEP_SIZE = '1%20d' & QUANTITIES = '1,3,6''
    print(r.text)
    f.close()