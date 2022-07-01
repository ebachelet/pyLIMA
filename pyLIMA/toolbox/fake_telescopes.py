from pyLIMA import telescopes


def create_a_fake_telescope(light_curve=None, astrometry_curve=None, name='A Fake Telescope'):

    telescope = telescopes.Telescope(name=name, light_curve=light_curve,  light_curve_names = ['time', 'mag', 'err_mag'],
                                     light_curve_units=['JD', 'mag', 'mag'], astrometry=astrometry_curve,
                                     astrometry_names = ['time', 'delta_ra', 'err_delta_ra','delta_ec','err_delta_dec'],
                                     astrometry_units=['JD', 'pix', 'pix','pix','pix'] )

    return telescope