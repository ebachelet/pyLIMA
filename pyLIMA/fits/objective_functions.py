import numpy as np


def astrometric_residuals(astrometry, astrometric_model):
    residuals = astrometry - astrometric_model

    return residuals


def all_telescope_astrometric_residuals(model, pyLIMA_parameters, norm=False,
                                        rescaling_astrometry_parameters=None):
    """ Compute the residuals of all telescopes according to the model.

    :param object pyLIMA_parameters: object containing the model parameters,
    see microlmodels for more details

    :return: the residuals in flux,
    :rtype: list, a list of array of residuals in flux
    """

    Residuals_ra = []
    err_ra_astrometry = []

    Residuals_dec = []
    err_dec_astrometry = []

    ind = 0

    if rescaling_astrometry_parameters is not None:
        rescaling_astrometry_parameters_ra = rescaling_astrometry_parameters[::2]
        rescaling_astrometry_parameters_dec = rescaling_astrometry_parameters[1::2]

    for telescope in model.event.telescopes:

        if telescope.astrometry is not None:

            # Find the residuals of telescope observation regarding the parameters
            # and model
            astrometry = telescope.astrometry

            astro_ra = astrometry['ra'].value
            astro_dec = astrometry['dec'].value
            microlensing_model = model.compute_the_microlensing_model(telescope,
                                                                      pyLIMA_parameters)

            residus_ra = astrometric_residuals(astro_ra,
                                               microlensing_model['astrometry'][0])
            residus_dec = astrometric_residuals(astro_dec,
                                                microlensing_model['astrometry'][1])

            if rescaling_astrometry_parameters is not None:

                err_ra = astrometry['err_ra'].value * (
                    rescaling_astrometry_parameters_ra[ind])
                err_dec = astrometry['err_dec'].value * (
                    rescaling_astrometry_parameters_dec[ind])

            else:

                err_ra = astrometry['err_ra'].value
                err_dec = astrometry['err_dec'].value

            if norm:
                residus_ra /= err_ra
                residus_dec /= err_dec

            Residuals_ra.append(residus_ra)
            Residuals_dec.append(residus_dec)

            err_ra_astrometry.append(err_ra)
            err_dec_astrometry.append(err_dec)

            ind += 1

    return Residuals_ra, Residuals_dec, err_ra_astrometry, err_dec_astrometry


def photometric_residuals(flux, photometric_model):
    """ Compute the residuals of a telescope lightcurve according to the model.

    :param object telescope: a telescope object. More details in telescopes module.
    :param object pyLIMA_parameters: object containing the model parameters,
    see microlmodels for more details

    :return: the residuals in flux, the priors
    :rtype: array_like, float
    """

    residuals = flux - photometric_model

    return residuals


def all_telescope_photometric_residuals(model, pyLIMA_parameters, norm=False,
                                        rescaling_photometry_parameters=None):
    """ Compute the residuals of all telescopes according to the model.

    :param object pyLIMA_parameters: object containing the model parameters,
    see microlmodels for more details

    :return: the residuals in flux,
    :rtype: list, a list of array of residuals in flux
    """

    residuals = []
    errfluxes = []

    ind = 0

    for telescope in model.event.telescopes:

        if telescope.lightcurve is not None:

            # Find the residuals of telescope observation regarding the parameters
            # and model
            lightcurve = telescope.lightcurve

            flux = lightcurve['flux'].value

            microlensing_model = model.compute_the_microlensing_model(telescope,
                                                                      pyLIMA_parameters)

            residus = photometric_residuals(flux, microlensing_model['photometry'])

            if rescaling_photometry_parameters is not None:

                # err_flux = lightcurve[
                # 'err_flux'].value+rescaling_photometry_parameters[ind] * \
                #           microlensing_model['photometry']
                err_flux = lightcurve['err_flux'].value * \
                           rescaling_photometry_parameters[ind]
            else:

                err_flux = lightcurve['err_flux'].value

            if norm:
                residus /= err_flux

            residuals.append(residus)
            errfluxes.append(err_flux)

            ind += 1

    return residuals, errfluxes


def photometric_chi2(telescope, model, pyLIMA_parameters):
    pass


def all_telescope_photometric_chi2(model, pyLIMA_parameters, rescaling_parameters=None):
    residuals, errfluxes = all_telescope_photometric_residuals(model, pyLIMA_parameters,
                                                               norm=True,
                                                               rescaling_photometry_parameters=rescaling_parameters)

    chi2 = np.sum(np.concatenate(residuals) ** 2)

    return chi2


def all_telescope_photometric_likelihood(model, pyLIMA_parameters,
                                         rescaling_photometry_parameters=None):
    residus, errflux = all_telescope_photometric_residuals(model, pyLIMA_parameters,
                                                           norm=True,
                                                           rescaling_photometry_parameters=rescaling_photometry_parameters)

    ln_likelihood = 0.5 * np.sum(
        np.concatenate(residus) ** 2 + 2 * np.log(np.concatenate(errflux)) + np.log(
            2 * np.pi))

    return ln_likelihood


def photometric_residuals_in_magnitude(telescope, model, pyLIMA_parameters):
    """ Compute the residuals of a telescope lightcurve according to the model.

    :param object telescope: a telescope object. More details in telescopes module.
    :param object pyLIMA_parameters: object containing the model parameters,
    see microlmodels for more details

    :return: the residuals in flux, the priors
    :rtype: array_like, float
    """
    try:

        microlensing_model = model.compute_the_microlensing_model(telescope,
                                                                  pyLIMA_parameters)

        residuals = -2.5*np.log10(telescope.lightcurve['flux'].value
                                  /microlensing_model['photometry'])
        return residuals

    except ValueError:

        return []
