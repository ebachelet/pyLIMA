import os.path

import numpy as np

from pyLIMA import microlpriors


def test_microlensing_flux_priors_negative_source_flux():


	priors = microlpriors.microlensing_flux_priors(100,-1.0,0)

	assert priors == np.inf

	
def test_microlensing_flux_priors_too_much_negative_blending_flux():


	priors = microlpriors.microlensing_flux_priors(100,10.0,-50.0)

	assert priors == np.inf	
