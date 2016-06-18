# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:39:32 2015

@author: ebachelet
"""
from pyLIMA import microllimb_darkening 

DATA = """ 0.00  3500. -5.0  2.0  0.6278 Kp L  ATLAS
 0.50  3500. -5.0  2.0  0.6161 Kp L  ATLAS
 1.00  3500. -5.0  2.0  0.5889 Kp L  ATLAS
 1.50  3500. -5.0  2.0  0.5627 Kp L  ATLAS
 2.00  3500. -5.0  2.0  0.5512 Kp L  ATLAS
 2.50  3500. -5.0  2.0  0.5526 Kp L  ATLAS
 3.00  3500. -5.0  2.0  0.5629 Kp L  ATLAS
 0.00  3750. -5.0  2.0  0.6270 Kp L  ATLAS
 0.50  3750. -5.0  2.0  0.6190 Kp L  ATLAS
 1.00  3750. -5.0  2.0  0.6019 Kp L  ATLAS
 4.50 32000.  1.0  2.0  0.1512 z' F  PHOENIX
 5.00 32000.  1.0  2.0  0.1419 z' F  PHOENIX
 4.50 33000.  1.0  2.0  0.1486 z' F  PHOENIX
 5.00 33000.  1.0  2.0  0.1397 z' F  PHOENIX
 4.50 34000.  1.0  2.0  0.1450 z' F  PHOENIX
 5.00 34000.  1.0  2.0  0.1369 z' F  PHOENIX
 4.50 35000.  1.0  2.0  0.1407 z' F  PHOENIX
 5.00 35000.  1.0  2.0  0.1334 z' F  PHOENIX
 5.00 37500.  1.0  2.0  0.1226 z' F  PHOENIX
 5.00 40000.  1.0  2.0  0.1150 z' F  PHOENIX
"""


def test_reading_file():
    data = list(microllimb_darkening.read_claret_data(DATA, camera_filter='all'))

    assert len(data) == 20


def test_read_row_object():
    data = list(microllimb_darkening.read_claret_data(DATA, camera_filter='all'))

    first_row_object = data[0]

    assert first_row_object.log_g == 0.0
    assert first_row_object.Teff == 3500.0
    assert first_row_object.metallicity == -5.0
    assert first_row_object.microturbulent_velocity == 2.0
    assert first_row_object.linear_limb_darkening == 0.6278
    assert first_row_object.filter == 'Kp'
    assert first_row_object.method == 'L'
    assert first_row_object.model == 'ATLAS'


def test_filter_by_filter():
    data = list(microllimb_darkening.read_claret_data(DATA, camera_filter="z'"))

    assert len(data) == 10
    assert data[0].filter == "z'"
