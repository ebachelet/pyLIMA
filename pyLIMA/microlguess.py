# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:00:51 2016

@author: ebachelet
"""

import numpy as np
import scipy.signal as ss

import microltoolbox

def initial_guess_PSPL(event):
    """Function to find initial PSPL guess for Levenberg-Marquardt solver (method=='LM').
    This assumes no blending.
    """
    
    # to estimation
    To = []
    Max_flux = []
    Errmag = []
    for telescope in event.telescopes:
        # Lot of process here, if one fails, just skip
        try:

            # only the best photometry
            good_photometry_indexes = np.where((telescope.lightcurve_magnitude[:, 2] < max(0.1, np.mean(telescope.lightcurve_magnitude[:, 2]))))[0]
            lightcurve_bis = telescope.lightcurve_magnitude[good_photometry_indexes]

            
            lightcurve_bis = lightcurve_bis[lightcurve_bis[:, 0].argsort(), :]
        
            mag = lightcurve_bis[:, 1]
            flux =  microltoolbox.magnitude_to_flux(mag)

            # clean the lightcurve using Savitzky-Golay filter on 3 points, degree 1. 
            mag_clean = ss.savgol_filter(mag, 3, 1)
            time = lightcurve_bis[:, 0]
            flux_clean = microltoolbox.flux_to_magnitude(mag_clean)
            errmag = lightcurve_bis[:, 2]


            fs = min(flux_clean)
            good_points = np.where(flux_clean > fs)[0]

            while (np.std(time[good_points]) > 5) | (len(good_points) > 100):

                indexes = np.where((flux_clean[good_points] > np.median(flux_clean[good_points])) & (
                errmag[good_points] <= max(0.1, 2.0 * np.mean(errmag[good_points]))))[0]

                if len(indexes) < 1:

                    break

                else:
                    
                    good_points = good_points[indexes]
                       

                    gravity = (
                            np.median(time[good_points]), np.median(flux_clean[good_points]),
                            np.mean(errmag[good_points]))
                       
                    distances = np.sqrt((time[good_points] - gravity[0]) ** 2 / gravity[0] ** 2)

            to = np.median(time[good_points])
            max_flux = max(flux[good_points])
            To.append(to)
            Max_flux.append(max_flux)
            Errmag.append(np.mean(lightcurve_bis[good_points, 2]))
            
        except:

            time = telescope.lightcurve_magnitude[:, 0]
            flux = microltoolbox.magnitude_to_flux( telescope.lightcurve_magnitude[:, 1])
            to = np.median(time)
            max_flux = max(flux)
            To.append(to)
            Max_flux.append(max_flux)
           
            Errmag.append(np.mean(telescope.lightcurve_magnitude[:, 2]))




    to_guess = sum(np.array(To) / np.array(Errmag) ** 2) / sum(1 / np.array(Errmag) ** 2)
    survey = event.telescopes[0]
    lightcurve = survey.lightcurve_flux
    lightcurve = lightcurve[lightcurve[:, 0].argsort(), :]
    
    ## fs, uo, tE estimations only one the survey telescope    


    time = lightcurve[:, 0]
    flux = lightcurve[:, 1]
    errflux = lightcurve[:, 2]

    # fs estimation, no blend

    baseline_flux_0 = np.min(flux)
    baseline_flux = np.median(flux)
    

    while np.abs(baseline_flux_0 - baseline_flux) > 0.01 * baseline_flux:

        baseline_flux_0 = baseline_flux
        indexes = np.where((flux < baseline_flux))[0].tolist() + np.where(
            np.abs(flux - baseline_flux) < np.abs(errflux))[0].tolist()
        baseline_flux = np.median(flux[indexes])

        if len(indexes) < 100:

                print 'low'
                baseline_flux = np.median(flux[flux.argsort()[:100]])
                break

    fs_guess = baseline_flux

    # uo estimation
    max_flux = Max_flux[0]
    Amax = max_flux / fs_guess
    uo_guess = np.sqrt(-2 + 2 * np.sqrt(1 - 1 / (1 - Amax ** 2)))

    # tE estimations
    tE_guesses = []

    # Method 1 : flux(t_demi_amplification) = 0.5 * fs_guess * (Amax + 1) 

    flux_demi = 0.5 * fs_guess * (Amax + 1)
    flux_tE = fs_guess * (uo_guess ** 2 + 3) / ((uo_guess ** 2 + 1) ** 0.5 * np.sqrt(uo_guess ** 2 + 5))
    index_plus = np.where((time > to_guess) & (flux < flux_demi))[0]
    index_moins = np.where((time < to_guess) & (flux < flux_demi))[0]

    B = 0.5 * (Amax + 1)

    if len(index_plus) != 0:
        if len(index_moins) != 0:
            ttE = (time[index_plus[0]] - time[index_moins[-1]])
            tE1 = ttE / (2 * np.sqrt(-2 + 2 * np.sqrt(1 + 1 / (B ** 2 - 1)) - uo_guess ** 2))

        else:
            ttE = time[index_plus[0]] - to_guess
            tE1 = ttE / np.sqrt(-2 + 2 * np.sqrt(1 + 1 / (B ** 2 - 1)) - uo_guess ** 2)
    else:

        ttE = to_guess- time[index_moins[-1]]
    tE1 = ttE / np.sqrt(-2 + 2 * np.sqrt(1 + 1 / (B ** 2 - 1)) - uo_guess ** 2)
    
    tE_guesses.append(tE1)
    
    # Method 2 : flux(t_E) = fs_guess * (uo^+3)/[(uo^2+1)^0.5*(uo^2+5)^0.5]
 
    flux_tE = fs_guess * (uo_guess ** 2 + 3) / ((uo_guess ** 2 + 1) ** 0.5 * np.sqrt(uo_guess ** 2 + 5))
    indextEplus = np.where((flux < flux_tE) & (time > to))[0]
    indextEmoins = np.where((flux < flux_tE) & (time < to))[0]
    

    if len(indextEmoins) != 0:

        indextEmoins = indextEmoins[-1]
        tEmoins = to_guess- time[indextEmoins]
        tE_guesses.append(tEmoins)

    if len(indextEplus) != 0:

        indextEplus = indextEplus[0]
        tEplus = time[indextEplus] - to_guess
        tE_guesses.append(tEplus)


    # Method 3 : the first points before/after to_guess that reach the baseline. Very rough approximation ot tE.

    indextEPlus = np.where((time > to) & (np.abs(flux - fs_guess) < np.abs(errflux)))[0]
    indextEMoins = np.where((time < to) & (np.abs(flux - fs_guess) < np.abs(errflux)))[0]
   

    if len(indextEPlus) != 0:

        tEPlus = time[indextEPlus[0]] - to_guess
        tE_guesses.append(tEplus)

    if len(indextEMoins) != 0:

        tEMoins = to_guess- time[indextEMoins[-1]]
        tE_guesses.append(tEMoins)

    TE = np.array(tE_guesses)

    tE_guess = np.median(TE)

    # safety reason, unlikely
    if tE_guess < 0.1:

        tE_guess = 20.0
    
    return [to_guess,uo_guess,tE_guess],fs_guess

def initial_guess_FSPL(event): 
  """Function to find initial FSPL guess for Levenberg-Marquardt solver (method=='LM').
    This assumes no blending.
  """
  PSPL_guess, fs_guess = initial_guess_PSPL(event)
    
  rho_guess = 0.05

  FSPL_guess = PSPL_guess+[rho_guess]
    
  return FSPL_guess, fs_guess
