import copy
import time as python_time

import numpy as np
from pyLIMA.fits.ML_fit import MLfit
from pyLIMA.models import generate_model
from tqdm import tqdm

from pyLIMA import event


class BOOTSTRAPfit(MLfit):
    """Under Construction"""

    def __init__(self, model, bootstrap_fitter='TRF', telescopes_fluxes_method='fit'):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, telescopes_fluxes_method=telescopes_fluxes_method)

    def generate_new_model(self):

        # create a new event

        new_event = event.Event(ra=self.model.event.ra, dec=self.model.event.dec)

        for tel in self.model.event.telescopes:

            if tel.lightcurve is not None:

                bootstrap_indexes_photometry = np.random.randint(0,
                                                                 len(tel.lightcurve),
                                                                 len(tel.lightcurve))

            else:

                bootstrap_indexes_photometry = None

            if tel.astrometry is not None:

                bootstrap_indexes_astrometry = np.random.randint(0, len(tel.astrometry),
                                                                 len(tel.astrometry))

            else:

                bootstrap_indexes_astrometry = None

            new_telescope = copy.deepcopy(tel)
            new_telescope.trim_data(photometry_mask=bootstrap_indexes_photometry,
                                    astrometry_mask=bootstrap_indexes_astrometry)

            new_event.telescopes.append(new_telescope)

        new_model = generate_model.create_model(self.model.model_type(), new_event,
                                                parallax=self.model.parallax_model,
                                                xallarap=self.model.xallarap_model,
                                                orbital_motion=self.model.orbital_motion_model,
                                                origin=self.model.origin,
                                                blend_flux_parameter=self.model.blend_flux_parameter,
                                                fancy_parameters=self.model.fancy_to_pyLIMA_dictionnary)

        return new_model

    def new_step(self, popi, popo):

        from pyLIMA.fits import TRF_fit
        np.random.seed(popi)
        updated_model = self.generate_new_model()

        trf = TRF_fit.TRFfit(updated_model)
        trf.model_parameters_guess = self.model_parameters_guess

        for key in self.fit_parameters.keys():
            trf.fit_parameters[key][1] = self.fit_parameters[key][1]

        trf.fit()

        return trf.fit_results['best_model']

    def fit(self, number_of_samples=100, computational_pool=None):

        start_time = python_time.time()

        samples = []

        if computational_pool is not None:

            number_of_loop = 1

        else:

            number_of_loop = number_of_samples

        for step in tqdm(range(number_of_loop)):

            if computational_pool is not None:

                iterable = [(i, i) for i in range(number_of_samples)]

                new_step = computational_pool.starmap(self.new_step, iterable)
                for samp in new_step:
                    samples.append(samp)
            else:

                new_step = self.new_step(step, step)

                samples.append(new_step)

        computation_time = python_time.time() - start_time

        samples = np.array(samples)

        self.fit_results = {'samples': samples, 'fit_time': computation_time}
