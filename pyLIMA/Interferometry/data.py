# Author: Arnaud Cassan.
# Distributed under the terms of the GNU General Public License.

import numpy as np
from microlensing.utils import checkandtimeit, verbosity, printi, printd, printw
import astropy.io.fits as fits

def obs_VIS2(oifits, B, lbd, ref_MJD=0.):
    """Read VIS2 data in OIFITS file
    
        Parameters
        ----------
        oifits : string
            Name of OIFITS file
        B : int
            Index of selected baseline (1-6)
        lbd : int
            Index of selected wavelength (1-6)
        ref_MJD : float
            Reference MJD date, so that: t = MJD - ref_MJD
        
        Returns
        -------
        uvdata : list
            List of lists [u, v, VIS2, VIS2ERR, MJD]
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # Date conventions:
    # DATE_PIONIER = MJD = JD - 2400000.5 : adopted convention
    # DATE_ML = JD - 2450000 (= MJD + 2400000.5 - 2450000 = MJD - 49999.5)
    # DATE_Gaia = JD
    
    # read data
    with fits.open(oifits) as hdulist:
        MJD = hdulist['OI_VIS2'].data['MJD'] - ref_MJD
        LBD = hdulist['OI_WAVELENGTH'].data['EFF_WAVE']
        UCOORD = hdulist['OI_VIS2'].data['UCOORD']
        VCOORD = hdulist['OI_VIS2'].data['VCOORD']
        VIS2 = hdulist['OI_VIS2'].data['VIS2DATA']
        VIS2ERR = hdulist['OI_VIS2'].data['VIS2ERR']
        
        printd(tcol + "Baselines: " + tend + "{0}".format((UCOORD**2 + VCOORD**2)**0.5))
        
        # get number of epochs
        if len(MJD) == 6:
            nep = [1]
        elif len(MJD) == 12:
            nep = [1, 2]
        else:
            raise ValueError("More than 2 epochs in OIFITS (not allowed here)")

    # create (u, v) for all wavelenghts
    ul = np.array([UCOORD / l for l in LBD]).T  # u[obs, lbd]
    vl = np.array([VCOORD / l for l in LBD]).T  # v[obs, lbd]

    # create list of data
    uvdata = []
    for epoch in nep:
        u = ul[6 * (epoch - 1) + B - 1, lbd - 1]
        v = vl[6 * (epoch - 1) + B - 1, lbd - 1]
        mjd = MJD[6 * (epoch - 1) + B - 1]
        vis2 = VIS2[6 * (epoch - 1) + B - 1, lbd - 1]
        vis2err = VIS2ERR[6 * (epoch - 1) + B - 1, lbd - 1]
        
        uvdata.append([u, v, vis2, vis2err, mjd])
      
    return uvdata

def obs_T3PHI(oifits, T, lbd, ref_MJD=0.):
    """Read T3PHI data in OIFITS file
    
        Parameters
        ----------
        oifits : string
            Name of OIFITS file
        T : int
            Index of selected triangle (1-4)
        lbd : int
            Index of selected wavelength (1-6)
        ref_MJD : float
            Reference MJD date, so that: t = MJD - ref_MJD
        
        Returns
        -------
        phidata : list
            List of lists [u1, v1, u2, v2, t3amp, t3phi, t3phierr, mjd]
            
    """
    # Date conventions:
    # DATE_PIONIER = MJD = JD - 2400000.5 : adopted convention
    # DATE_ML = JD - 2450000 (= MJD + 2400000.5 - 2450000 = MJD - 49999.5)
    # DATE_Gaia = JD
    
    # read data
    with fits.open(oifits) as hdulist:
        # read data
        MJD = hdulist['OI_T3'].data['MJD'] - ref_MJD
        LBD = hdulist['OI_WAVELENGTH'].data['EFF_WAVE']
        T3PHI = hdulist['OI_T3'].data['T3PHI']
        T3PHIERR = hdulist['OI_T3'].data['T3PHIERR']
        U1COORD = hdulist['OI_T3'].data['U1COORD']
        V1COORD = hdulist['OI_T3'].data['V1COORD']
        U2COORD = hdulist['OI_T3'].data['U2COORD']
        V2COORD = hdulist['OI_T3'].data['V2COORD']
        T3AMP =  hdulist['OI_T3'].data['T3AMP']
        
        # get number of epochs
        if len(MJD) == 4:
            nep = [1]
        elif len(MJD) == 8:
            nep = [1, 2]
        else:
            raise ValueError('More than 2 epochs in OIFITS')

    # create (u_ij, v_ij) triangles for all wavelenghts
    u1l = np.array([U1COORD / l for l in LBD]).T
    v1l = np.array([V1COORD / l for l in LBD]).T
    u2l = np.array([U2COORD / l for l in LBD]).T
    v2l = np.array([V2COORD / l for l in LBD]).T

    # create list of data
    phidata = []
    for epoch in nep:
        u1 = u1l[4 * (epoch - 1) + T - 1, lbd - 1]
        v1 = v1l[4 * (epoch - 1) + T - 1, lbd - 1]
        u2 = u2l[4 * (epoch - 1) + T - 1, lbd - 1]
        v2 = v2l[4 * (epoch - 1) + T - 1, lbd - 1]
        mjd = MJD[4 * (epoch - 1) + T - 1]
        t3phi = T3PHI[4 * (epoch - 1) + T - 1, lbd - 1]
        t3phierr = T3PHIERR[4 * (epoch - 1) + T - 1, lbd - 1]
        t3amp = T3AMP[4 * (epoch - 1) + T - 1, lbd - 1]
        
        phidata.append([u1, v1, u2, v2, t3amp, t3phi, t3phierr, mjd])

    return phidata

def get_baselines(oifits):
    """Read VIS2 data in OIFITS file
    
        Parameters
        ----------
        oifits : string
            Name of OIFITS file
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
        
    # read data
    with fits.open(oifits) as hdulist:
        UCOORD = hdulist['OI_VIS2'].data['UCOORD']
        VCOORD = hdulist['OI_VIS2'].data['VCOORD']
        
    printi(tcol + "Baselines: " + tend + "{0}".format((UCOORD**2 + VCOORD**2)**0.5))
    
