import numpy as np
import scipy.signal as ss


def initial_guess_PSPL(event):
    """
    Function to find initial PSPL guess. This assumes no blending.

    Parameters
    ----------
    event : object, an event object

    Returns
    -------
    guess_model : list, [t0,u0,tE] the PSPL guess
    fs_guess : float, the source flux guess
    """
    import pyLIMA.toolbox.brightness_transformation
    # to estimation
    to_estimations = []
    maximum_flux_estimations = []
    errors_magnitude = []

    for telescope in event.telescopes:

        if telescope.lightcurve_magnitude is not None:
            # Lot of process here, if one fails, just skip
            lightcurve_magnitude = telescope.lightcurve_magnitude
            mean_error_magnitude = np.mean(lightcurve_magnitude['err_mag'].value)
            try:

                # only the best photometry
                good_photometry_indexes = \
                    np.where((lightcurve_magnitude['err_mag'].value <
                              max(0.1, mean_error_magnitude)))[0]
                lightcurve_bis = lightcurve_magnitude[good_photometry_indexes]

                lightcurve_bis['time'] = lightcurve_bis['time'][
                    lightcurve_bis['time'].value.argsort()]
                lightcurve_bis['mag'] = lightcurve_bis['mag'][
                    lightcurve_bis['time'].value.argsort()]
                lightcurve_bis['err_mag'] = lightcurve_bis['err_mag'][
                    lightcurve_bis['time'].value.argsort()]
                mag = lightcurve_bis['mag'].value
                flux = pyLIMA.toolbox.brightness_transformation.magnitude_to_flux(mag)

                # clean the lightcurve using Savitzky-Golay filter on 3 points,
                # degree 1.
                mag_clean = ss.savgol_filter(mag, 3, 1)
                time = lightcurve_bis['time'].value
                flux_clean = pyLIMA.toolbox.brightness_transformation.magnitude_to_flux(
                    mag_clean)
                errmag = lightcurve_bis['err_mag'].value

                flux_source = min(flux_clean)
                good_points = np.where(flux_clean > flux_source)[0]

                while (np.std(time[good_points]) > 5) | (len(good_points) > 100):

                    indexes = \
                        np.where((flux_clean[good_points] > np.median(
                            flux_clean[good_points])) & (
                                         errmag[good_points] <= max(0.1, 2.0 * np.mean(
                                     errmag[good_points]))))[0]

                    if len(indexes) < 1:

                        break

                    else:

                        good_points = good_points[indexes]

                        # gravity = (
                        #   np.median(time[good_points]), np.median(flux_clean[
                        #   good_points]),
                        #    np.mean(errmag[good_points]))

                        # distances = np.sqrt((time[good_points] - gravity[0]) ** 2 /
                        # gravity[0] ** 2)

                to = np.median(time[good_points])
                max_flux = max(flux[good_points])
                to_estimations.append(to)
                maximum_flux_estimations.append(max_flux)
                errors_magnitude.append(
                    np.mean(lightcurve_bis[good_points]['err_mag'].value))

            except ValueError:

                time = lightcurve_magnitude['time'].value
                flux = pyLIMA.toolbox.brightness_transformation.magnitude_to_flux(
                    lightcurve_magnitude['mag'].value)
                to = np.median(time)
                max_flux = max(flux)
                to_estimations.append(to)
                maximum_flux_estimations.append(max_flux)

                errors_magnitude.append(mean_error_magnitude)
    to_guess = sum(np.array(to_estimations) / np.array(errors_magnitude) ** 2) / sum(
        1 / np.array(errors_magnitude) ** 2)
    survey = event.telescopes[0]
    lightcurve = survey.lightcurve_magnitude

    lightcurve = lightcurve[lightcurve['time'].value.argsort()]

    ## fs, uo, tE estimations only one the survey telescope

    time = lightcurve['time'].value
    flux = pyLIMA.toolbox.brightness_transformation.magnitude_to_flux(
        lightcurve['mag'].value)
    errflux = pyLIMA.toolbox.brightness_transformation.error_magnitude_to_error_flux(
        lightcurve['err_mag'].value, flux)

    # fs estimation, no blend

    baseline_flux_0 = np.min(flux)
    baseline_flux = np.median(flux)

    while np.abs(baseline_flux_0 - baseline_flux) > 0.01 * baseline_flux:

        baseline_flux_0 = baseline_flux
        indexes = np.where((flux < baseline_flux))[0].tolist() + np.where(
            np.abs(flux - baseline_flux) < np.abs(errflux))[0].tolist()
        baseline_flux = np.median(flux[indexes])

        if len(indexes) < 100:
            baseline_flux = np.median(flux[flux.argsort()[:100]])
            break

    fs_guess = baseline_flux

    # uo estimation
    max_flux = maximum_flux_estimations[0]
    Amax = max_flux / fs_guess
    if (Amax < 1.0) | np.isnan(Amax):
        Amax = 1.1
    uo_guess = np.sqrt(-2 + 2 * np.sqrt(1 - 1 / (1 - Amax ** 2)))

    # tE estimations
    tE_guesses = []

    # Method 1 : flux(t_demi_amplification) = 0.5 * fs_guess * (Amax + 1)

    half_magnification = 0.5 * (Amax + 1)

    flux_demi_amplification = fs_guess * half_magnification

    index_plus = np.where((time > to_guess) & (flux < flux_demi_amplification))[0]
    index_moins = np.where((time < to_guess) & (flux < flux_demi_amplification))[0]

    if len(index_plus) != 0:

        if len(index_moins) != 0:
            t_demi_amplification = (time[index_plus[0]] - time[index_moins[-1]])
            tE_demi_amplification = t_demi_amplification / (
                    2 * np.sqrt(-2 + 2 * np.sqrt(
                1 + 1 / (half_magnification ** 2 - 1)) - uo_guess ** 2))

            tE_guesses.append(tE_demi_amplification)

        else:
            t_demi_amplification = time[index_plus[0]] - to_guess
            tE_demi_amplification = t_demi_amplification / np.sqrt(
                -2 + 2 * np.sqrt(1 + 1 / (half_magnification ** 2 - 1)) - uo_guess ** 2)

            tE_guesses.append(tE_demi_amplification)
    else:

        if len(index_moins) != 0:
            t_demi_amplification = to_guess - time[index_moins[-1]]
            tE_demi_amplification = t_demi_amplification / np.sqrt(
                -2 + 2 * np.sqrt(1 + 1 / (half_magnification ** 2 - 1)) - uo_guess ** 2)

            tE_guesses.append(tE_demi_amplification)

    # Method 2 : flux(t_E) = fs_guess * (uo^+3)/[(uo^2+1)^0.5*(uo^2+5)^0.5]

    amplification_tE = (uo_guess ** 2 + 3) / (
            (uo_guess ** 2 + 1) ** 0.5 * np.sqrt(uo_guess ** 2 + 5))
    flux_tE = fs_guess * amplification_tE

    index_tE_plus = np.where((flux < flux_tE) & (time > to))[0]
    index_tE_moins = np.where((flux < flux_tE) & (time < to))[0]

    if len(index_tE_moins) != 0:
        index_tE_moins = index_tE_moins[-1]
        tE_moins = to_guess - time[index_tE_moins]

        tE_guesses.append(tE_moins)

    if len(index_tE_plus) != 0:
        index_tE_plus = index_tE_plus[0]
        tE_plus = time[index_tE_plus] - to_guess

        tE_guesses.append(tE_plus)

    # Method 3 : the first points before/after to_guess that reach the baseline. Very
    # rough
    # approximation ot tE.

    index_tE_baseline_plus = \
        np.where((time > to) & (np.abs(flux - fs_guess) < np.abs(errflux)))[0]
    index_tE_baseline_moins = \
        np.where((time < to) & (np.abs(flux - fs_guess) < np.abs(errflux)))[0]

    if len(index_tE_baseline_plus) != 0:
        tEPlus = time[index_tE_baseline_plus[0]] - to_guess

        tE_guesses.append(tEPlus)

    if len(index_tE_baseline_moins) != 0:
        tEMoins = to_guess - time[index_tE_baseline_moins[-1]]

        tE_guesses.append(tEMoins)

    tE_guess = np.median(tE_guesses)

    # safety reason, unlikely
    if (tE_guess < 0.1) | np.isnan(tE_guess):
        tE_guess = 20.0

    # [to,uo,tE], fsource

    return [to_guess, uo_guess, tE_guess], fs_guess


def initial_guess_FSPL(event):
    """
    Function to find initial FSPL guess, i.e. PSPL guess + rho = 0.05

    Parameters
    ----------
    event : object, an event object

    Returns
    -------
    guess_model : list, [t0,u0,tE,rho] the FSPL guess
    fs_guess : float, the source flux guess
    """
    PSPL_guess, fs_guess = initial_guess_PSPL(event)
    # Dummy guess
    rho_guess = 0.05

    FSPL_guess = PSPL_guess + [rho_guess]

    # [to,uo,tE,rho], fsource
    return FSPL_guess, fs_guess


def initial_guess_DSPL(event):
    """
    Function to find initial DSPL guess

    Parameters
    ----------
    event : object, an event object

    Returns
    -------
    guess_model : list, [t0,u0,delta_t0,delta_u0,tE,q_flux] the DSPL guess
    fs_guess : float, the source flux guess
    """
    PSPL_guess, fs_guess = initial_guess_PSPL(event)

    filters = [telescope.filter for telescope in event.telescopes]

    unique_filters = np.unique(filters)

    # Dummy guess
    delta_to_guess = 5  # days
    delta_uo_guess = 0.01
    q_flux_guess = 0.5

    DSPL_guess = PSPL_guess[:2] + [delta_to_guess] + [delta_uo_guess] + \
                 [PSPL_guess[2]] + [q_flux_guess] * len(unique_filters)

    # [to1,uo1,delta_to,uo2,tE,q_F_i], fsource
    return DSPL_guess, fs_guess
