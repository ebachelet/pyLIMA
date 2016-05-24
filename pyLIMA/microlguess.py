# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:00:51 2016

@author: ebachelet
"""

import numpy as np
import scipy.signal as ss

import microltoolbox

def initial_guess_PSPL(event):
    """Function to find initial guess for Levenberg-Marquardt solver (method=='LM').
    Guess are made using the survey telescope for the Paczynski parameters (to,uo,tE).
    This assumes no blending.
    """

    # to estimation
    To = []
    Max_flux = []
    Std = []
    Errmag = []
    for i in event.telescopes:

        try:
                
            # only the best photometry
            good = np.where((i.lightcurve[:, 2] < max(0.1, np.mean(i.lightcurve[:, 2]))))[0]
            lightcurve_bis = i.lightcurve[good]
            mag = lightcurve_bis[:, 1]
            flux =  microltoolbox.magnitude_to_flux(mag)
            lightcurve_bis = lightcurve_bis[lightcurve_bis[:, 0].argsort(), :]
            mag_clean = ss.savgol_filter(mag, 3, 1)
            Time = lightcurve_bis[:, 0]
            flux_clean = microltoolbox.flux_to_magnitude(mag_clean)

            errmag = lightcurve_bis[:, 2]

               

            fs = min(flux_clean)
            index = np.where(flux_clean > fs)[0]
            good = index

            while (np.std(Time[good]) > 5) | (len(good) > 100):

                index = np.where((flux_clean[good] > np.median(flux_clean[good])) & (
                errmag[good] <= max(0.1, 2.0 * np.mean(errmag[good]))))[0]

                if len(index) < 1:

                    break

                else:
                    good = good[index]
                       

                    gravity = (
                            np.median(Time[good]), np.median(flux_clean[good]),
                            np.mean(errmag[good]))
                       
                    distances = np.sqrt((Time[good] - gravity[0]) ** 2 / gravity[0] ** 2)
                      
                to = np.median(Time[good])
                max_flux = max(flux[good])
                std = np.std(lightcurve_bis[good, 0])
                To.append(to)
                Max_flux.append(max_flux)
                Errmag.append(np.mean(lightcurve_bis[good, 2]))
                if std == 0:

                    std = 0.1

                Std.append(std)

        except:

            Time = i.lightcurve[:, 0]
            flux = microltoolbox.magnitude_to_flux( i.lightcurve[:, 1])
            to = np.median(Time)
            max_flux = max(flux)
            To.append(to)
            Max_flux.append(max_flux)
            std = np.std(i.lightcurve[:, 0])
            if std == 0:

                std = 0.1
            Std.append(std)
            Errmag.append(np.mean(i.lightcurve[:, 2]))




    to = sum(np.array(To) / np.array(Errmag) ** 2) / sum(1 / np.array(Errmag) ** 2)
    survey = event.telescopes[0]
    lightcurve = survey.lightcurve_flux
    lightcurve = lightcurve[lightcurve[:, 0].argsort(), :]
    Time = lightcurve[:, 0]
    flux = lightcurve[:, 1]
    errflux = lightcurve[:, 2]

    # fs, no blend

    baseline_flux_0 = np.min(flux)
    baseline_flux = np.median(flux)
    index = []

    while np.abs(baseline_flux_0 - baseline_flux) > 0.01 * baseline_flux:

        baseline_flux_0 = baseline_flux
        index = np.where((flux < baseline_flux))[0].tolist() + np.where(
            np.abs(flux - baseline_flux) < np.abs(errflux))[0].tolist()
        baseline_flux = np.median(flux[index])

        if len(index) < 100:

                print 'low'
                baseline_flux = np.median(flux[flux.argsort()[:100]])
                break

        fs = baseline_flux
        max_flux = Max_flux[0]
        Amax = max_flux / fs
        uo = np.sqrt(-2 + 2 * np.sqrt(1 - 1 / (1 - Amax ** 2)))


        # tE estimations
        flux_demi = 0.5 * fs * (Amax + 1)
        flux_tE = fs * (uo ** 2 + 3) / ((uo ** 2 + 1) ** 0.5 * np.sqrt(uo ** 2 + 5))
        index_plus = np.where((Time > to) & (flux < flux_demi))[0]
        index_moins = np.where((Time < to) & (flux < flux_demi))[0]
        B = 0.5 * (Amax + 1)
        if len(index_plus) != 0:

            if len(index_moins) != 0:

                ttE = (Time[index_plus[0]] - Time[index_moins[-1]])
                tE1 = ttE / (2 * np.sqrt(-2 + 2 * np.sqrt(1 + 1 / (B ** 2 - 1)) - uo ** 2))

            else:

                ttE = Time[index_plus[0]] - to
                tE1 = ttE / np.sqrt(-2 + 2 * np.sqrt(1 + 1 / (B ** 2 - 1)) - uo ** 2)
        else:

            ttE = to - Time[index_moins[-1]]
            tE1 = ttE / np.sqrt(-2 + 2 * np.sqrt(1 + 1 / (B ** 2 - 1)) - uo ** 2)

        indextEplus = np.where((flux < flux_tE) & (Time > to))[0]
        indextEmoins = np.where((flux < flux_tE) & (Time < to))[0]
        tEmoins = 0.0
        tEplus = 0.0

        if len(indextEmoins) != 0:

            indextEmoins = indextEmoins[-1]
            tEmoins = to - Time[indextEmoins]

        if len(indextEplus) != 0:

            indextEplus = indextEplus[0]
            tEplus = Time[indextEplus] - to

        indextEPlus = np.where((Time > to) & (np.abs(flux - fs) < np.abs(errflux)))[0]
        indextEMoins = np.where((Time < to) & (np.abs(flux - fs) < np.abs(errflux)))[0]
        tEPlus = 0.0
        tEMoins = 0.0

        if len(indextEPlus) != 0:

            tEPlus = Time[indextEPlus[0]] - to

        if len(indextEMoins) != 0:

            tEMoins = to - Time[indextEMoins[-1]]

        TE = np.array([tE1, tEplus, tEmoins, tEPlus, tEMoins])
        good = np.where(TE != 0.0)[0]
        tE = np.median(TE[good])

        if tE < 0.1:

            tE = 20.0
      
        return [to,uo,tE],fs

def initial_guess_FSPL(event): 

    PSPL,fs = initial_guess_PSPL(event)       
    
    FSPL = PSPL+[0.05]
    
    return FSPL,fs