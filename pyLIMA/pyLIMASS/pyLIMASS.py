import astropy.units as u
import numpy as np
import pkg_resources
import scipy.interpolate as si
import scipy.optimize as so
import sklearn.mixture as skmix
import speclite
import speclite.filters
from astropy.table import QTable

ISOCHRONES_HEADER = ['Fe', 'logAge', 'logMass', 'logL', 'logTe', 'logg', 'mbolmag',
                     'Umag', 'Bmag', 'Vmag', 'Rmag', 'Imag', 'Jmag', 'Hmag', 'Kmag',
                     'umag', 'gmag', 'rmag', 'imag', 'zmag', 'Gmag', 'G_BPmag',
                     'G_RPmag', 'F062mag', 'F087mag', 'F106mag', 'F129mag', 'F158mag',
                     'F184mag', 'F146mag', 'F213mag']

resource_path = '/'.join(('data', 'Roman_Filters.dat'))
template = pkg_resources.resource_filename('pyLIMA', resource_path)

ROMAN_FILTERS_RESPONSE = np.loadtxt(template)

ROMAN_FILTERS_RESPONSE = QTable(ROMAN_FILTERS_RESPONSE,
                                names=['wavelength'] + ['F062mag', 'F087mag', 'F106mag',
                                                        'F129mag', 'F158mag', 'F184mag',
                                                        'F146mag', 'F213mag'])

filter_names = [i for i in ROMAN_FILTERS_RESPONSE.columns.keys()][1:]

for key in filter_names:
    filt = speclite.filters.FilterResponse(wavelength=
                                           ROMAN_FILTERS_RESPONSE[
                                               'wavelength'] * 10 ** 4 * u.AA, response=
                                           ROMAN_FILTERS_RESPONSE[key],
                                           meta=dict(group_name='ROMAN', band_name=key))
class SourceLensProbabilities(object):

    def __init__(self, observables={}, catalog='Isochrones', stellar_lens=False,
                 t0=None, ra=None, dec=None):

        self.catalog = catalog
        self.stellar_lens = stellar_lens

        self.t0 = t0
        self.ra = ra
        self.dec = dec

        self.default_observables = {'log10(M_s)': None,
                                    'log10(D_s)': None,
                                    'log10(Teff_s)': None,
                                    'Fe_s': None,
                                    'logg_s': None,
                                    'log10(L_s)': None,
                                    'mu_sN': None,
                                    'mu_sE': None,
                                    'log10(Av_s)': None,
                                    'log10(theta_s)': None,
                                    'log10(R_s)': None,
                                    'log10(M_l)': None,
                                    'log10(epsilon_D_l)': None,
                                    'log10(D_l)': None,
                                    'log10(Teff_l)': None,
                                    'Fe_l': None,
                                    'logg_l': None,
                                    'log10(L_l)': None,
                                    'mu_lN': None,
                                    'mu_lE': None,
                                    'log10(epsilon_Av_l)': None,
                                    'log10(Av_l)': None,
                                    'log10(theta_l)': None,
                                    'log10(R_l)': None,
                                    'log10(t_E)': None,
                                    'log10(rho_s)': None,
                                    'log10(theta_E)': None,
                                    'pi_EN': None,
                                    'pi_EE': None,
                                    'log10(pi_E)': None,
                                    'mu_relN': None,
                                    'mu_relE': None,
                                    'log10(mu_rel)': None,
                                    'mu_rel_hel_N': None,
                                    'mu_rel_hel_E': None,
                                    'log10(mu_rel_hel)': None,
                                    'mags_source': None,
                                    'mags_lens': None,
                                    'mags_baseline': None,
                                    }

        self.observables = self.default_observables.copy()

        for key in observables.keys():
            self.observables[key] = observables[key]

        if self.observables['mags_lens'] is not None:
            print('Switching stellar_lens to True since lens magnitudes are provided')
            self.stellar_lens = True

        self.log10_Mass_bounds = [-1, 2]
        self.log10_Distance_bounds = [-2, 2]
        self.Fe_bounds = [-2, 0.5]
        self.log10_Teff_bounds = [3, 4]
        self.logg_bounds = [0, 6]
        self.muN_bounds = [-20, 20]
        self.muE_bounds = [-20, 20]
        self.log10_Av_bounds = [-2, 1]
        self.log10_epsilon = [-2, 0]

        self.bounds = [self.log10_Mass_bounds, self.log10_Distance_bounds,
                       self.log10_Teff_bounds, self.Fe_bounds, self.logg_bounds,
                       self.muN_bounds, self.muE_bounds, self.log10_Av_bounds,
                       self.log10_Mass_bounds, self.log10_epsilon,
                       self.log10_Teff_bounds, self.Fe_bounds, self.logg_bounds,
                       self.muN_bounds, self.muE_bounds, self.log10_epsilon]

        self.load_isochrones()

        self.build_filters()

        self.Earth_speeds_at_t0()

    def modify_observables(self, observables={}, stellar_lens=False, t0=None, ra=None,
                           dec=None):

        self.stellar_lens = stellar_lens

        self.t0 = t0
        self.ra = ra
        self.dec = dec

        self.observables = self.default_observables.copy()
        for key in observables.keys():
            self.observables[key] = observables[key]

        if self.observables['mags_lens'] is not None:
            print('Switching stellar_lens to True since lens magnitudes are provided')
            self.stellar_lens = True

        self.build_filters()

        self.Earth_speeds_at_t0()

    def Earth_speeds_at_t0(self):

        if self.t0 is not None:
            target_angles_in_the_sky = [self.ra * np.pi / 180, self.dec * np.pi / 180]
            Target = np.array(
                [np.cos(target_angles_in_the_sky[1]) * np.cos(
                    target_angles_in_the_sky[0]),
                 np.cos(target_angles_in_the_sky[1]) * np.sin(
                     target_angles_in_the_sky[0]),
                 np.sin(target_angles_in_the_sky[1])])

            self.East = np.array(
                [-np.sin(target_angles_in_the_sky[0]),
                 np.cos(target_angles_in_the_sky[0]),
                 0.0])
            self.North = np.cross(Target, self.East)

            from pyLIMA.parallax import astropy_ephemerides

            Earth_pos_speed = astropy_ephemerides.Earth_ephemerides(self.t0)

            Earth_speed = Earth_pos_speed[1].xyz.value * 365.25

            self.Earth_speed_projected_at_t0 = np.array(
                [np.dot(Earth_speed, self.North), np.dot(Earth_speed, self.East)])

    def Wang_absorption_law(self, Av, lamb):
        # https://iopscience.iop.org/article/10.3847/1538-4357/ab1c61/pdf
        alambda = np.zeros(len(lamb))
        mask = lamb < 1
        Y = 1 / lamb - 1.82

        alambda[mask] = Av * (
                    1 + 0.7499 * Y[mask] - 0.1086 * Y[mask] ** 2 - 0.08909 * Y[
                mask] ** 3 + 0.02905 * Y[mask] ** 4 + 0.01069 * Y[
                        mask] ** 5 + 0.001707 * Y[mask] ** 6 - 0.001002 * Y[mask] ** 7)

        alambda[~mask] = Av * 0.3722 * lamb[~mask] ** -2.070
        mask = alambda < 0
        alambda[mask] = 0

        return alambda

    def build_filters(self):

        filters = {}
        all_filters = {}

        twomass_filters = speclite.filters.load_filters('twomass-*')
        bessel_filters = speclite.filters.load_filters('bessell-*')
        sdss_filters = speclite.filters.load_filters('sdss2010-*')
        gaia_filters = speclite.filters.load_filters('gaiadr2-*')

        for key in self.observables.keys():

            table = np.c_[self.isochrones['logTe'].value, self.isochrones['Fe'].value,
            self.isochrones['logg'].value]

            if 'mags' in key:

                for filter_name in ISOCHRONES_HEADER[7:]:

                    if 'F' in filter_name:
                        filt = speclite.filters.load_filters('ROMAN-' + filter_name)[0]

                    if ('Jmag' in filter_name):
                        filt = twomass_filters[0]

                    if ('Hmag' in filter_name):
                        filt = twomass_filters[1]

                    if ('Kmag' in filter_name):
                        filt = twomass_filters[2]

                    if ('Umag' in filter_name):
                        filt = bessel_filters[0]

                    if ('Bmag' in filter_name):
                        filt = bessel_filters[1]

                    if ('Vmag' in filter_name):
                        filt = bessel_filters[2]

                    if ('Rmag' in filter_name):
                        filt = bessel_filters[3]

                    if ('Imag' in filter_name):
                        filt = bessel_filters[4]

                    if ('umag' in filter_name):
                        filt = sdss_filters[0]

                    if ('gmag' in filter_name):
                        filt = sdss_filters[1]

                    if ('rmag' in filter_name):
                        filt = sdss_filters[2]

                    if ('imag' in filter_name):
                        filt = sdss_filters[3]

                    if ('zmag' in filter_name):
                        filt = sdss_filters[4]

                    if ('Gmag' in filter_name):
                        filt = gaia_filters[1]

                    if ('G_BPmag' in filter_name):
                        filt = gaia_filters[0]

                    if ('G_RPmag' in filter_name):
                        filt = gaia_filters[2]

                    if self.observables[key] is not None:

                        if filter_name in self.observables[key].keys():
                            absolute_mag_interpolator = si.LinearNDInterpolator(table,
                                                                                self.isochrones[
                                                                                    filter_name].value)
                            absolute_mag_nearest = si.NearestNDInterpolator(table,
                                                                            self.isochrones[
                                                                                filter_name].value)

                            absorption = np.sum(
                                filt.response * self.Wang_absorption_law(1,
                                                                         filt.wavelength / 10000)) / np.sum(
                                filt.response)

                            filters[filter_name] = {'filter': filt,
                                                    'AB_correction': float(0),
                                                    'M0_interpolator': [
                                                        absolute_mag_interpolator,
                                                        absolute_mag_nearest],
                                                    'absorption': absorption}
                # ab_corr = Spyctres.derive_AB_correction([filt])

                # all_filters[filter_name] = {'filter':filt,'AB_correction': float(
                # 0),'M0_interpolator':[absolute_mag_interpolator,absolute_mag_nearest]}

        self.filters = filters
        # self.all_filters = all_filters

    def load_isochrones(self):

        resource_path = '/'.join(('data', 'Bressan_Isochrones.dat'))
        template = pkg_resources.resource_filename('pyLIMA', resource_path)

        ISO = np.loadtxt(template, dtype=str)[1:].astype(float)
        ISO[:, 2] = np.log10(ISO[:, 2])

        ISO = QTable(ISO, names=ISOCHRONES_HEADER)
        # Needs tunable isochrones cuts
        mask = (ISO['logMass'].value < 2) & (ISO['logg'].value < 6) & (
                    ISO['logg'].value > 0) & (ISO['logAge'].value > 9) & (
                           ISO['Vmag'].value < 20)
        ISO = ISO[mask]

        self.isoLogL = [si.LinearNDInterpolator(
            np.c_[ISO['logTe'].value, ISO['Fe'].value, ISO['logg'].value], ISO['logL']),
                        si.NearestNDInterpolator(np.c_[ISO['logTe'].value, ISO[
                            'Fe'].value, ISO['logg'].value], ISO['logL'])]
        self.isochrones = ISO

        #######Interpolator with 1/dist_isochrones**2 seems efficient and robust...

    def local_isochrones(self, params):

        M, Teff, Fe, logg = params

        local_iso = self.isochrones
        dist = (local_iso['logMass'].value - np.log10(M)) ** 2 + (
                    local_iso['Fe'].value - Fe) ** 2 + (
                           local_iso['logTe'].value - np.log10(Teff)) ** 2 + (
                           local_iso['logg'].value - logg) ** 2

        local_isochrones_dist = dist

        return local_iso, local_isochrones_dist

    def reconstruct_mags_from_spectrum(self, params):

        pass

    def reconstruct_mags_from_isochrones(self, params, observation, local_isochrones,
                                         local_isochrones_dist):

        D, Av, Teff, Fe, logg = params

        lumi = float(self.isoLogL[0](np.log10(Teff), Fe, logg))
        if np.isnan(lumi):
            # lumi = float(self.isoLogL[1](np.log10(Teff),Fe,logg))
            lumi = -999

        mu = 5 * np.log10(D * 1000) - 5

        mags = []
        for obs in observation.keys():

            filt = self.filters[obs]
            vega_mag = float(filt['M0_interpolator'][0](np.log10(Teff), Fe, logg))
            # vega_mag = np.nan
            if np.isnan(vega_mag):
                vega_mag = float(filt['M0_interpolator'][1](np.log10(Teff), Fe, logg))
                # vega_mag = float(filt['M0_interpolator'][1](np.log10(Teff),Fe,logg))
                # vega_mag = -999

            absorption = filt['absorption'] * Av
            # breakpoint()
            vega_mag += mu + absorption

            mags.append(float(vega_mag))
        # breakpoint()
        return mags, lumi

    def generate_observables(self, parameters):

        (log10_M_s, log10_D_s, log10_Teff_s, Fe_s, logg_s, mu_sN, mu_sE,
         log10_Av_s) = parameters[:8]

        (log10_M_l, log10_epsilon_D_l, log10_Teff_l, Fe_l, logg_l, mu_lN, mu_lE,
         log10_epsilon_Av_l) = parameters[8:]

        Ms = 10 ** log10_M_s
        Ml = 10 ** log10_M_l

        Ds = 10 ** log10_D_s
        Dl = 10 ** log10_epsilon_D_l * Ds

        Avs = 10 ** log10_Av_s
        Avl = 10 ** log10_epsilon_Av_l * Avs

        pirel = 1 / Dl - 1 / Ds

        theta_E = (8.144 * Ml * pirel) ** 0.5
        pi_E = pirel / theta_E

        mu_rel_vector = np.array((mu_lN, mu_lE)) - (mu_sN, mu_sE)

        mu_rel = np.sqrt(np.sum(mu_rel_vector ** 2))

        pi_EN, pi_EE = mu_rel_vector * pi_E / mu_rel

        t_E = theta_E / mu_rel * 365.25

        Rs = 10 ** ((log10_M_s - logg_s + 4.4374) / 2)
        Rl = 10 ** ((log10_M_l - logg_l + 4.4374) / 2)

        theta_s = Rs / Ds * 4.65046694812766  #
        theta_l = Rl / Dl * 4.65046694812766  # 2.25461*10**-11*1000*180/np.pi*3600*1000

        rhos = theta_s / theta_E / 1000

        Teffs = 10 ** log10_Teff_s
        Teffl = 10 ** log10_Teff_l

        Source = [np.log10(theta_s), Avs, Teffs, Fe_s, logg_s]
        Lens = [np.log10(theta_l), Avl, Teffl, Fe_l, logg_l]

        local_isochrones_source = None
        local_dist_source = None
        local_isochrones_lens = None
        local_dist_lens = None

        mags_source = []
        mags_lens = []
        mags_baseline = []

        lumi_source = None
        lumi_lens = None

        for key in self.observables.keys():

            if key == 'mags_source':
                # breakpoint()
                observation = self.observables[key]
                if observation is not None:

                    if self.catalog == 'Isochrones':

                        mags_source, lumi_source = (
                            self.reconstruct_mags_from_isochrones(
                            [Ds, Avs, Teffs, Fe_s, logg_s], observation,
                            local_isochrones_source, local_dist_source))
                    else:
                        pass

                    # breakpoint()
                    # waves =  np.concatenate([i[0].wavelength.tolist() for i in
# observation])
                    # waves = np.unique(np.sort(np.r_[waves,np.arange(1000,50000,100)]))
                    # waves = np.arange(1000,50000,100)
                    # flux_source,absolute_flux_source = Spyctres.star_model_new(
# Source,waves,catalog=self.catalog)
                    # lumi_source = np.log10(np.sum(absolute_flux_source[:-1,
# 1]*np.diff(flux_source[:,0]))*Ds**2*31110119154.95548)
                    # lumi_source = self.isoLogL(np.log10(Teffs),Fe_s,logg_s)
                    # scale_lumi = 10**self.isomasslogL(Ms,Fe_s,loggs)/lumi_source

                    # flux_source[:,1] *= scale_lumi

                    # for obs in observation:

                    # flux_source = Spyctres.star_model_new(Source,
# obs[0]._wavelength,catalog='k93models')

                    # ab_mag = obs[0].get_ab_magnitude(flux_source[:,1],flux_source[
# :,0])
                    # vega_mag = ab_mag-obs[1]

                    # if obs[0].name == 'Roman-F087':

                    #    vega_mag = float(self.isoF087(np.log10(Teffs),Fe_s,logg_s))
                    # else:
                    #    vega_mag = float(self.isoF146(np.log10(Teffs),Fe_s,logg_s))

                    # vega_mag += np.sum(obs[
# 0].response*Spyctres.Wang_absorption_law(Avs,obs[0].wavelength/10000))/np.sum(obs[
# 0].response)+5*np.log10(Ds*1000)-5
                    # mags_source.append(float(vega_mag))
                # breakpoint()

            if key == 'mags_lens':
                # breakpoint()
                observation = self.observables[key]
                if observation is not None:

                    if self.catalog == 'Isochrones':

                        mags_lens, lumi_lens = self.reconstruct_mags_from_isochrones(
                            [Dl, Avl, Teffl, Fe_l, logg_l], observation,
                            local_isochrones_lens, local_dist_lens)
                    else:
                        pass

            if key == 'mags_baseline':
                # breakpoint()
                observation = self.observables[key]
                if observation is not None:

                    if self.catalog == 'Isochrones':
                        mags_s, lumi_s = self.reconstruct_mags_from_isochrones(
                            [Ds, Avs, Teffs, Fe_s, logg_s], observation,
                            local_isochrones_lens, local_dist_lens)
                        mags_l, lumi_l = self.reconstruct_mags_from_isochrones(
                            [Dl, Avl, Teffl, Fe_l, logg_l], observation,
                            local_isochrones_lens, local_dist_lens)

                        for ind_mag in range(len(mags_s)):
                            flux_source = 10 ** ((25 - mags_s[ind_mag]) / 2.5)
                            flux_lens = 10 ** ((25 - mags_l[ind_mag]) / 2.5)

                            flux_tot = flux_source + flux_lens

                            mags_baseline.append(25 - 2.5 * np.log10(flux_tot))

                    else:
                        pass

        if self.t0 is not None:

            geo_to_helio_correction = self.Earth_speed_projected_at_t0 * pirel
            mu_rel_vector_hel = mu_rel_vector + geo_to_helio_correction
            log_10_mu_rel_hel = np.log10(np.sqrt(np.sum(mu_rel_vector_hel ** 2)))


        else:

            mu_rel_vector_hel = (None, None)
            log_10_mu_rel_hel = None

        observables = {'log10(M_s)': log10_M_s,
                       'log10(D_s)': log10_D_s,
                       'log10(Teff_s)': np.log10(Teffs),
                       'Fe_s': Fe_s,
                       'logg_s': logg_s,
                       'log10(L_s)': lumi_source,
                       'mu_sN': mu_sN,
                       'mu_sE': mu_sE,
                       'log10(Av_s)': log10_Av_s,
                       'log10(theta_s)': np.log10(theta_s),
                       'log10(R_s)': np.log10(Rs),
                       'log10(M_l)': log10_M_l,
                       'log10(epsilon_D_l)': log10_epsilon_D_l,
                       'log10(D_l)': np.log10(Dl),
                       'log10(Teff_l)': np.log10(Teffl),
                       'Fe_l': Fe_l,
                       'logg_l': logg_l,
                       'log10(L_l)': lumi_lens,
                       'mu_lN': mu_lN,
                       'mu_lE': mu_lE,
                       'log10(epsilon_Av_l)': log10_epsilon_Av_l,
                       'log10(Av_l)': np.log10(Avl),
                       'log10(theta_l)': np.log10(theta_l),
                       'log10(R_l)': np.log10(Rl),
                       'log10(t_E)': np.log10(t_E),
                       'log10(rho_s)': np.log10(rhos),
                       'log10(theta_E)': np.log10(theta_E),
                       'pi_EN': pi_EN,
                       'pi_EE': pi_EE,
                       'log10(pi_E)': np.log10(pi_E),
                       'mu_relN': mu_rel_vector[0],
                       'mu_relE': mu_rel_vector[1],
                       'log10(mu_rel)': np.log10(mu_rel),
                       'mu_rel_hel_N': mu_rel_vector_hel[0],
                       'mu_rel_hel_E': mu_rel_vector_hel[1],
                       'log10(mu_rel_hel)': log_10_mu_rel_hel,
                       'mags_source': mags_source,
                       'mags_lens': mags_lens,
                       'mags_baseline': mags_baseline,
                       }

        return observables, local_dist_source, local_dist_lens

    def generate_GM(self, n_components=None, modes=None):

        true_keys = [i for i in self.observables.keys() if
                     self.observables[i] is not None]
        true_distributions = []

        for key in true_keys:

            if 'mags' in key:

                for obs in self.observables[key].values():
                    true_distributions.append(obs)

            else:

                true_distributions.append(self.observables[key])

        true_distributions = np.c_[true_distributions].T

        if n_components is None:
            n_components = len(true_keys)

        if modes is not None:

            if len(modes) > n_components:
                n_components = len(modes)

            means_init = []

            for i in range(n_components):

                mode = modes[i % len(modes)]
                mode_params = []

                for key in true_keys:

                    if 'mags' in key:

                        for obs in mode[key].values():
                            mode_params.append(obs)

                    else:

                        mode_params.append(mode[key])

                means_init.append(mode_params)
        else:
            means_init = None

        gm = skmix.GaussianMixture(n_components=n_components, means_init=means_init)
        gm.fit(true_distributions)
        self.gm = gm
        self.gm_keys = true_keys.copy()


    def GM_proba(self, parameters):

        observed, dist_s, dist_l = self.generate_observables(parameters)

        to_compare_with_gm = []

        for key in observed.keys():

            if self.observables[key] is not None:

                if 'mags' in key:

                    for obs in observed[key]:
                        to_compare_with_gm.append(obs)

                else:

                    to_compare_with_gm.append(observed[key])

        # breakpoint()
        score = self.gm.score([to_compare_with_gm])  # log-likelihood

        dist_s, dist_l = self.isochrones_score(observed)

        score_tot = np.copy(score)

        # print (score,dist_s,dist_l)

        if dist_s is not None:
            score_tot -= (dist_s.min() * 100)
            # score_tot -= np.mean(np.sort(dist_s)[:10])*1000

        else:
            return -np.inf

        if dist_l is not None:
            score_tot -= (dist_l.min() * 100)
            # score_tot -= np.mean(np.sort(dist_l)[:10])*1000

        # else:
        #    return -np.inf

        return score_tot

    def isochrones_score(self, observed):

        params = [10 ** observed['log10(M_s)'], observed['Fe_s'],
                  observed['log10(Teff_s)'], observed['logg_s'], observed['log10(L_s)'],
                  10 ** observed['log10(M_l)'], observed['Fe_l'],
                  observed['log10(Teff_l)'], observed['logg_l'], observed['log10(L_l)']]

        (Ms, Fe_s, log10_Teff_s, logg_s, log10_L_s, Ml, Fe_l, log10_Teff_l, logg_l,
         log10_L_l) = params

        dist_s = (self.isochrones['logMass'].value - np.log10(Ms)) ** 2 + (
                    self.isochrones['Fe'].value - Fe_s) ** 2 + (
                             self.isochrones['logTe'].value - log10_Teff_s) ** 2 + (
                             self.isochrones['logg'].value - logg_s) ** 2

        try:

            dist_s += (self.isochrones['logL'].value - log10_L_s) ** 2

        except:

            pass

        if self.stellar_lens:

            dist_l = (self.isochrones['logMass'].value - np.log10(Ml)) ** 2 + (
                        self.isochrones['Fe'].value - Fe_l) ** 2 + (
                                 self.isochrones['logTe'].value - log10_Teff_l) ** 2 + (
                                 self.isochrones['logg'].value - logg_l) ** 2

            try:

                dist_l += (self.isochrones['logL'].value - log10_L_l) ** 2

            except:

                pass
        else:

            dist_l = None

        return dist_s, dist_l

    def objective(self, parameters):

        obj = self.GM_proba(parameters)
        return -obj

    def objective_mcmc(self, parameters):

        for ind, param in enumerate(parameters):

            if (param < self.bounds[ind][0]) | (param > self.bounds[ind][1]):
                return -np.inf

        obj = self.GM_proba(parameters)
        return obj

    def mcmc(self, seeds, n_walkers=2, n_chains=10000):

        import emcee

        nwalkers = n_walkers * len(seeds[0])

        ndim = len(seeds[0])

        pos = []
        for i in range(nwalkers):
            choice = np.random.randint(0, len(seeds))

            trial = seeds[choice] + len(seeds[choice]) * [1] * np.random.randn(
                len(seeds[choice])) * 10 ** -4

            pos.append(trial)
        # pos = seed +  len(seed)*[1] * np.random.randn(nwalkers, len(seed))*10**-4
        pos = np.array(pos)
        # with mul.Pool(processes=4) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.objective_mcmc,
                                        moves=[(emcee.moves.DEMove(), 0.8), (
                                            emcee.moves.DESnookerMove(),
                                            0.2)], )  # pool = pool)
        final_positions, final_probabilities, state = sampler.run_mcmc(pos, n_chains,
                                                                       progress=True)

        return sampler

    def de(self, maxiter=5000, popsize=1):

        # breakpoint()

        result = so.differential_evolution(self.objective, self.bounds, maxiter=maxiter,
                                        disp=True, popsize=popsize, tol=0, atol=0.1,
                                        workers=1, polish=False, strategy='best1bin')

        return result
