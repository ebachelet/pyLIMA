from tqdm import tqdm
import numpy as np

from pyLIMA.fits.ML_fit import MLfit
from pyLIMA.models import generate_model
from pyLIMA import event
from pyLIMA import telescopes

class BOOTSTRAPfit(MLfit):

    def __init__(self, model, bootstrap_fitter = 'TRF', fancy_parameters=False,telescopes_fluxes_method='fit'):
        """The fit class has to be intialized with an event object."""

        super().__init__(model, fancy_parameters=fancy_parameters, telescopes_fluxes_method=telescopes_fluxes_method)

    def generate_new_model(self):

        #create a new event

        new_event = event.Event()
        new_event.ra = self.model.event.ra
        new_event.dec = self.model.event.dec


        for tel in self.model.event.telescopes:



                if tel.lightcurve_flux is not None:


                    bootstrap_indexes = np.random.randint(0,len(tel.lightcurve_flux),len(tel.lightcurve_flux))

                    lightcurve =  np.c_[[tel.lightcurve_flux[key].value for key in tel.lightcurve_flux.columns]].T[bootstrap_indexes]
                    light_curve_names =  [tel.lightcurve_flux[key].info.name for key in tel.lightcurve_flux.columns]
                    light_curve_units =  [tel.lightcurve_flux[key].info.unit for key in tel.lightcurve_flux.columns]

                else:

                    lightcurve = None
                    light_curve_names = None
                    light_curve_units = None


                if tel.astrometry is not None:

                    bootstrap_indexes = np.random.randint(0,len(tel.astrometry),len(tel.astrometry))

                    astrometry = np.c_[[tel.astrometry[key].value for key in tel.astrometry.columns]].T[
                        bootstrap_indexes]
                    astrometry_names =  [tel.astrometry[key].info.name for key in tel.astrometry.columns]
                    astrometry_units =  [tel.astrometry[key].info.unit for key in tel.astrometry.columns]

                else:

                    astrometry = None
                    astrometry_names = None
                    astrometry_units = None

                name = tel.name
                camera_filter = tel.filter
                location = tel.location
                gamma = tel.gamma
                spacecraft_name = tel.spacecraft_name
                spacecraft_positions = tel.spacecraft_positions

                new_telescope = telescopes.Telescope(name=name, camera_filter=camera_filter, light_curve=lightcurve,
                    light_curve_names=light_curve_names, light_curve_units=light_curve_units, clean_the_light_curve=False,
                    location=location, spacecraft_name=spacecraft_name,
                    astrometry=astrometry, astrometry_names=astrometry_names, astrometry_units=astrometry_units)


                new_telescope.altitude = tel.altitude
                new_telescope.longitude = tel.longitude
                new_telescope.latitude = tel.latitude

                new_telescope.spacecraft_positions = tel.spacecraft_positions


                new_event.telescopes.append(new_telescope)

        new_model = generate_model.create_model(self.model.model_type,new_event, parallax=self.model.parallax_model,
                                                xallarap=self.model.xallarap_model,
                                                orbital_motion=self.model.orbital_motion_model,
                                                blend_flux_parameter=self.model.blend_flux_parameter)

        return new_model



    def new_step(self,popi,popo):

        from pyLIMA.fits import TRF_fit
        np.random.seed(popi)
        updated_model = self.generate_new_model()
        trf = TRF_fit.TRFfit(updated_model)
        trf.model_parameters_guess = self.model_parameters_guess
        trf.fit()

        return trf.fit_results['best_model']

    def fit(self,number_of_samples=100, computational_pool = None ):

        #original_datasets = {}

        #for telescope in self.model.event.telescopes:

        #    try:

        #        photometry = telescope.lightcurve_flux

        #    except:

        #        photometry = []

        #    try:

        #        astrometry = telescope.astrometry

        #    except:

        #        astrometry = []


        #    original_datasets['telescope.name']['photometry'] = photometry
        #    original_datasets['telescope.name']['astrometry'] = astrometry


        samples = []

        if computational_pool is not None:

            number_of_loop = 1

        else:

            number_of_loop = number_of_samples

        for step in tqdm(range(number_of_loop)):

            iterable = [(i,i) for i in range(number_of_samples)]
            if computational_pool is not None:

                new_step = computational_pool.starmap(self.new_step,iterable)
                for samp in new_step:

                    samples.append(samp)
            else:
                new_step = self.new_step()

                samples.append(new_step)
           

        return np.array(samples)