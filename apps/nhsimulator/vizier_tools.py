# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:35:31 2018

@author: rstreet
"""

from sys import argv
from astroquery.vizier import Vizier
from astropy import wcs, coordinates, units, visualization

def search_vizier_for_sources(ra, dec, radius, catalog, row_limit=-1):
    """Function to perform online query of the 2MASS catalogue and return
    a catalogue of known objects within the field of view
    
    Inputs:
        :param str ra: RA J2000 in sexigesimal format
        :param str dec: Dec J2000 in sexigesimal format
        :param float radius: Search radius in arcmin
        :param str catalog: Catalog to search.  Options include:
                                    ['2MASS', 'VPHAS+']
    """
    
    supported_catalogs = { '2MASS': ['2MASS', 
                                     ['_RAJ2000', '_DEJ2000', 'Jmag', 'e_Jmag', \
                                    'Hmag', 'e_Hmag','Kmag', 'e_Kmag'],
                                    {'Jmag':'<20'}],
                           'VPHAS+': ['II/341', 
                                      ['_RAJ2000', '_DEJ2000', 'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'clean'],
                                    {}]
                           }

    (cat_id,cat_col_list,cat_filters) = supported_catalogs[catalog]
    
    v = Vizier(columns=cat_col_list,\
                column_filters=cat_filters)

    v.ROW_LIMIT = row_limit
    c = coordinates.SkyCoord(ra+' '+dec, frame='icrs', unit=(units.hourangle, units.deg))
    r = radius * units.arcminute
    
    catalog_list = Vizier.find_catalogs(cat_id)
    
    result=v.query_region(c,radius=r,catalog=[cat_id])
    
    if len(result) == 1:
        result = result[0]
        
    return result

if __name__ == '__main__':
    
    if len(argv) == 1:
        
        ra = raw_input('Please enter search centroid RA in sexigesimal format: ')
        dec = raw_input('Please enter search centroid Dec in sexigesimal format: ')
        radius = raw_input('Please enter search radius in arcmin: ')
        catalog = raw_input('Please enter the ID of the catalog to search [2MASS, VPHAS+]: ')
        
    else:

        ra = argv[1]
        dec = argv[2]
        radius = argv[3]
        catalog = argv[4]
    
    radius = float(radius)
    
    qs = search_vizier_for_sources(ra, dec, radius, catalog)
    
    print qs