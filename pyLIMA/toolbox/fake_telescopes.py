from pyLIMA import telescopes


def create_a_fake_telescope(light_curve, astrometry_curve=None, name='A Fake Telescope'):

    telescope = telescopes.Telescope(name=name, light_curve=light_curve,  light_curve_names = ['time', 'mag', 'err_mag'],
                                     light_curve_units=['JD', 'mag', 'mag'], astrometry=astrometry_curve, )

    return telescope