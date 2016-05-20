# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:08:11 2016

@author: ebachelet
"""
import mock

import microlmodels

def test_define_parameters():
    
    event = mock.MagicMock()
    Model = microlmodels.MLModels(event,model='FSPL', second_order=[['None', 0.0], ['None', 0.0], ['None', 0.0], 'None'])

   