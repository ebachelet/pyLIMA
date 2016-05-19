# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:46:17 2015

@author: ebachelet
"""


import microlfits


def test_mlfits():
    
    fit = list(read_claret_data(DATA, camera_filter='all'))

    assert len(data) == 20