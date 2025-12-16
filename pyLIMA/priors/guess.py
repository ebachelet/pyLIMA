import numpy as np
import scipy.signal as ss


def _get_lightcurve_data(telescope, data_type='time'):
    """
    Extract data from lightcurve, supporting both legacy dict-like and JaxTelescope.
    
    Parameters
    ----------
    telescope : Telescope or JaxTelescope
        Telescope object
    data_type : str
        'time', 'mag', 'err_mag', or 'flux'
    
    Returns
    -------
    data : np.ndarray
        The requested data as numpy array
    """
    lc = telescope.lightcurve
    
    # Try legacy dict-like access first
    try:
        return np.asarray(lc[data_type].value)
    except Exception:
        pass
    
    # Try JaxTelescope attribute access
    try:
        return np.asarray(getattr(lc, data_type))
    except Exception:
        pass
    
    raise ValueError(f"Could not extract '{data_type}' from lightcurve")


def _get_lightcurve_length(telescope):
    """Get number of lightcurve points for both legacy and Jax telescopes."""
    lc = telescope.lightcurve
    
    # Try legacy dict-like length
    try:
        return len(lc['time'].value)
    except Exception:
        pass
    
    # Try JaxTelescope attribute length
    try:
        return len(lc.time)
    except Exception:
        pass
    
    raise ValueError("Could not determine lightcurve length")


    chichidof = np.sum((magnitude-np.median(magnitude))**2/
                       err_magnitude**2)/len(magnitude)

    if chichidof<1.1:

        return False

    else:

        return True

    #signal = np.diff(np.percentile(magnitude,[16,84]))/2

    #if signal<1*np.median(err_magnitude):

    #    return False

    #else:

    #    return True

def initial_guess_PSPL(event):
    """
    Function to find initial PSPL guess. This assumes no blending.
    
    Supports both legacy Telescope (dict-like lightcurve) and JaxTelescope objects.

    Parameters
    ----------
    event : object, an event object

    Returns
    -------
    guess_model : list, [t0,u0,tE] the PSPL guess
    fs_guess : float, the source flux guess
    """

    #THIS NEEDS TO BE REWRITTEN

    import pyLIMA.toolbox.brightness_transformation
    # to estimation
    to_estimations = []
    maximum_flux_estimations = []
    errors_magnitude = []
    index_surveys = []

    for ind, telescope in enumerate(event.telescopes):

        if telescope.lightcurve is not None:
            # Lot of process here, if one fails, just skip
            try:
                # Extract data using unified helper
                err_mag = _get_lightcurve_data(telescope, 'err_mag')
                mag = _get_lightcurve_data(telescope, 'mag')
                time = _get_lightcurve_data(telescope, 'time')
                
                mean_error_magnitude = np.mean(err_mag)

                # only the best photometry
                good_photometry_indexes = \
                    np.where((err_mag <
                              max(0.1, mean_error_magnitude)))[0]
                
                time_subset = time[good_photometry_indexes]
                mag_subset = mag[good_photometry_indexes]
                
                # Sort by time
                sort_idx = np.argsort(time_subset)
                time_sorted = time_subset[sort_idx]
                mag_sorted = mag_subset[sort_idx]
                
                flux = pyLIMA.toolbox.brightness_transformation.magnitude_to_flux(mag_sorted)

                # clean the lightcurve using Savitzky-Golay filter on 3 points,
                # degree 1.
                mag_clean = ss.savgol_filter(mag_sorted, 3, 1)
                flux_clean = pyLIMA.toolbox.brightness_transformation.magnitude_to_flux(
                    mag_clean)
                errmag = err_mag[good_photometry_indexes][sort_idx]

                signal_in_data = check_signal_in_lightcurve(mag_clean, errmag)

                if signal_in_data:

                    flux_source = min(flux_clean)
                    good_points = np.where(flux_clean > flux_source)[0]

                    while (np.std(time_sorted[good_points]) > 5) | (len(good_points) > 100):

                        indexes = \
                            np.where((flux_clean[good_points] > np.median(
                                flux_clean[good_points])) & (
                                             errmag[good_points] <=
                                             max(0.1, 2.0 *
                                                 np.mean(errmag[good_points]))))[0]

                        if len(indexes) < 1:

                            break

                        else:

                            good_points = good_points[indexes]


                    to = np.median(time_sorted[good_points])
                    max_flux = max(flux[good_points])
                    to_estimations.append(to)
                    maximum_flux_estimations.append(max_flux)
                    errors_magnitude.append(np.mean(errmag[good_points]))
                    index_surveys.append(ind)

                else:

                    pass

            except (ValueError, Exception):
               pass

    try:

        to_guess = (sum(np.array(to_estimations) / np.array(errors_magnitude) ** 2) /
                    sum(1 / np.array(errors_magnitude) ** 2))

        n_data = [_get_lightcurve_length(event.telescopes[i]) for i in index_surveys]

        index_survey = index_surveys[np.argmax(n_data)]

        survey = event.telescopes[index_survey]


        print('Selecting '+survey.name+' to estimate u0, tE and fs')

    except (ZeroDivisionError, Exception):

        survey = event.telescopes[0]
        mag_survey = _get_lightcurve_data(survey, 'mag')
        time_survey = _get_lightcurve_data(survey, 'time')
        flux_survey = pyLIMA.toolbox.brightness_transformation.magnitude_to_flux(mag_survey)
        
        to_guess = float(time_survey[np.argmin(mag_survey)])
        maximum_flux_estimations = [float(flux_survey[np.argmin(mag_survey)])]

    # Extract lightcurve data using helper
    time = _get_lightcurve_data(survey, 'time')
    mag = _get_lightcurve_data(survey, 'mag')
    err_mag = _get_lightcurve_data(survey, 'err_mag')
    
    # Sort by time
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    mag = mag[sort_idx]
    err_mag = err_mag[sort_idx]

    ## fs, uo, tE estimations only one the survey telescope

    flux = pyLIMA.toolbox.brightness_transformation.magnitude_to_flux(mag)
    errflux = pyLIMA.toolbox.brightness_transformation.error_magnitude_to_error_flux(
        err_mag, flux)

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

    index_tE_plus = np.where((flux < flux_tE) & (time > to_guess))[0]
    index_tE_moins = np.where((flux < flux_tE) & (time < to_guess))[0]

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
    # approximation ot 3xtE.

    index_tE_baseline_plus = \
        np.where((time > to_guess) & (np.abs(flux - fs_guess) < np.abs(errflux)))[0]
    index_tE_baseline_moins = \
        np.where((time < to_guess) & (np.abs(flux - fs_guess) < np.abs(errflux)))[0]

    if len(index_tE_baseline_plus) != 0:
        tEPlus = time[index_tE_baseline_plus[0]] - to_guess

        tE_guesses.append(tEPlus/3)

    if len(index_tE_baseline_moins) != 0:
        tEMoins = to_guess - time[index_tE_baseline_moins[-1]]

        tE_guesses.append(tEMoins/3)

    tE_guess = np.median(tE_guesses)

    # safety reason, unlikely
    if np.isnan(tE_guess):
        tE_guess = 20.0

    # Return dict with named parameters
    guess = {'t0': float(to_guess), 'u0': float(uo_guess), 'tE': float(tE_guess)}
    return guess, fs_guess


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
    #rho_guess = np.min((np.abs(PSPL_guess[1]),0.05))

    #Amax = np.array(event.telescopes[0].lightcurve['flux'].max()) / fs_guess
    u0 = PSPL_guess['u0']
    Amax = (u0 ** 2 + 2) / (u0 * (u0 ** 2 + 4) ** 0.5)
    rho_guess = np.max((2/(Amax**2-1)**0.5, 2*np.abs(u0)))

    if np.isnan(rho_guess):

        rho_guess = 2*np.abs(u0)

    FSPL_guess = {'t0': float(PSPL_guess['t0']), 'u0': float(PSPL_guess['u0']),
                  'tE': float(PSPL_guess['tE']), 'rho': float(rho_guess)}

    return FSPL_guess, fs_guess

def initial_guess_FSPLarge(event):
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

    #Amax = np.array(event.telescopes[0].lightcurve['flux'].max())/fs_guess
    u0 = PSPL_guess['u0']
    Amax = (u0**2+2)/(u0*(u0**2+4)**0.5)
    rho_guess = np.max((2/(Amax**2-1)**0.5,1.1*np.abs(u0)))

    if np.isnan(rho_guess):

        rho_guess = 2*PSPL_guess['u0']

    FSPL_guess = {'t0': float(PSPL_guess['t0']), 'u0': float(PSPL_guess['u0']),
                  'tE': float(PSPL_guess['tE']), 'rho': float(rho_guess)}

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

    DSPL_guess = {}
    DSPL_guess['t0'] = float(PSPL_guess[0])
    DSPL_guess['u0'] = float(PSPL_guess[1])
    DSPL_guess['delta_t0'] = float(delta_to_guess)
    DSPL_guess['delta_u0'] = float(delta_uo_guess)
    DSPL_guess['tE'] = float(PSPL_guess[2])

    for filt in unique_filters:
        DSPL_guess['q_flux_' + filt] = float(q_flux_guess)

    # [to1,uo1,delta_to,uo2,tE,q_F_i], fsource
    return DSPL_guess, fs_guess


    def guess_dict_to_vector(model, guess_dict):
        """
        Convert a guess dictionary into an ordered parameter list following the
        `model.model_dictionnary` ordering. Missing keys are returned as None.

        Parameters
        ----------
        model : MLmodel
            The model instance whose parameter ordering to follow.
        guess_dict : dict
            Mapping parameter name -> value

        Returns
        -------
        list
            Ordered list of parameter values matching `model.model_dictionnary`.
        """
        parameter_dictionnary = model.model_dictionnary
        vector = []
        for key in parameter_dictionnary.keys():
            vector.append(guess_dict.get(key, None))

        return vector
