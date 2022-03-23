# Author: Arnaud Cassan.
# Distributed under the terms of the GNU General Public License.

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import romberg

def mod_VIS(zetac, rho, u, v, model, Nc=20000, LLD=0., Na=8, tol=1e-5):
    """Compute interferometric visibility of an ESPL for a given model.
            
        Parameters
        ----------
        zetac : complex
            Source position [thetaE]. Convention: source travels left
            to write, horizontally.
        rho : float
            Source size [thetaE].
        u, v : floats
            Einstein (u, v) parameters [thetaE^-1].
        model : string
            List of models:
                'ESPL_CONT_SINGEP', 'ESPL_PS_SINGEP', 'ESPL_FLAT_SINGEP', 'ESPL_FULLINT_SINGEP'
                'ESPL_CONT_MULTIEP', 'ESPL_PS_MULTIEP', 'ESPL_FLAT_MULTIEP', 'ESPL_FULLINT_MULTIEP'
            where:
                ESPL: extended-source point-lens.
                CONT: contouring.
                PS: point-source.
                FLAT: flat-images approximation.
                SINGEP: single epoch.
                MULTIEP: multi-epochs.
        Nc : int, optional
            Number of points for images contour uniform sampling.
        LLD : float, optional
            Linear limb-darkening coefficient 'a': I(r) = 1 - a + a cos(theta).
            Link with Gamma: a = 3 Gamma/(2 + Gamma).
        Na : int, optional
            Number of annuli dividing the source for linear limb darkening. The
            source profile is divided in uniform brightness bins.
        tol : float, optional
            Tolerance in integration.
        
        Returns
        -------
        out : complex
            Interferometric visibility.
    """
    # contouring
    if model == 'ESPL_CONT_SINGEP' or model == 'ESPL_CONT_MULTIEP':
        Vp, Vm, Np, Nm = vismag(zetac, rho, u, v, Nc, Nc)
        V = (Vp + Vm) / (Np + Nm)
        
    # full integration
    elif model == 'ESPL_FULLINT_SINGEP' or model == 'ESPL_FULLINT_MULTIEP' or model == 'ESPL_MLENSING_MULTIEP':
        V = fullvis(zetac, rho, u, v, tol=tol)
        
    # point-source approximation
    elif model == 'ESPL_PS_SINGEP' or model == 'ESPL_PS_MULTIEP':
        V = psvis(zetac, u, v)
        
    # flat-images approsimation
    elif model == 'ESPL_FLAT_SINGEP' or model == 'ESPL_FLAT_MULTIEP':
        V = flatvis(zetac, rho, u, v, LLD=LLD, Na=Na, tol=tol)
        
    else:
        raise ValueError("Wrong model ID ({}) in mod_VIS".format(model))
    
    # return global visibility
    return V

def mod_T3PHI(zetac, rho, u_AB, v_AB, u_BC, v_BC, model, Nc=20000, LLD=0., Na=8, tol=1e-5):
    """Compute theoretical interferometric bispectrum,
    ie. modulus and closure phase (ESPL)
    
        Warning
        -------
        Includes variable 'blend' (which is set to 0 by default).
    
        Parameters
        ----------
        zetac : complex
            Source position [thetaE].
            Convention: source travels left to write, horizontally.
        rho : float
            Source size [thetaE].
        u, v : floats
            (u, v) parameters [thetaE^-1].
        Nc_init : int
            Source contour initial sampling
        Nc_fin : int
            Source contour final samplig (uniform on images)
            
        Returns
        -------
        t3phi, t3amp : tuple of float
            t3phi is the closure phase of the given ABC triangle [degree].
            t3amp is the modulus of the bispectrum of the given ABC triangle.
    """
    # baselines list
    B = [(u_AB, v_AB), (u_BC, v_BC), (-u_AB-u_BC, -v_AB-v_BC)]
    
    # compute bispectrum
    Vbis = 1.
    for b in B:
    
        # contouring
        if model == 'ESPL_CONT_SINGEP' or model == 'ESPL_CONT_MULTIEP':
            Vp, Vm, Np, Nm = vismag(zetac, rho, b[0], b[1], Nc, Nc)
            Vbis = Vbis * (Vp + Vm) / (Np + Nm)
            
        # full integration
        elif model == 'ESPL_FULLINT_SINGEP' or model == 'ESPL_FULLINT_MULTIEP' or model == 'ESPL_MLENSING_MULTIEP':
            Vbis = Vbis * fullvis(zetac, rho, b[0], b[1], tol=tol)
            
        # point-source approximation
        elif model == 'ESPL_PS_SINGEP' or model == 'ESPL_PS_MULTIEP':
            Vbis = Vbis * psvis(zetac, b[0], b[1])
            
        # flat-images approsimation
        elif model == 'ESPL_FLAT_SINGEP' or model == 'ESPL_FLAT_MULTIEP':
            Vbis = Vbis * flatvis(zetac, rho, b[0], b[1], LLD=LLD, Na=Na, tol=tol)
            
        else:
            raise ValueError("Wrong model ID ({}) in mod_VIS".format(model))
    
    t3phi = np.angle(Vbis) * 180. / np.pi
    t3amp = np.abs(Vbis)

    # return closure phase and modulus of bispectrum
    return t3phi, t3amp

def flatvis(zetac, rho, u, v, LLD=0., Na=8, tol=1e-5):
    """Compute 'Flat-images approximation' interferometric visibility of an ESPL.
    
        Parameters
        ----------
        zetac : complex
            Source position [thetaE]. Convention: source travels left
            to write, horizontally.
        rho : float
            Source size [thetaE].
        u, v : floats
            Einstein (u, v) parameters [thetaE^-1].
        LLD : float, optional
            Linear limb-darkening coefficient 'a': I(r) = 1 - a + a cos(theta).
            Default is 0 (uniform source). Link with Gamma: a = 3 Gamma/(2 + Gamma).
        Na : int, optional
            Number of annuli dividing the source for linear limb darkening. The
            source profile is divided in uniform brightness bins. Default is 8.
        tol : float, optional
            Tolerance in integration. Default is 1e-4.
                         
        Returns
        -------
        VIS : float
            Flat-images approximation visibitily.
    """
    # parameters
    u0 = np.abs(zetac)
    beta = np.angle(zetac) - np.pi / 2.
    uvE = np.exp(np.complex(0., -beta)) * np.complex(u, v)
    uE, vE = np.real(uvE), np.imag(uvE)
    eta = rho / u0

    # integrande (np.abs to avoid numerical problems)
    def intfunc6(beta, eta, uE, vE):
        return np.sqrt(np.abs(eta**2 - np.sin(beta)**2)) * np.cos(2. * np.pi * vE * np.cos(beta)) * np.cos(2. * np.pi * uE * np.sin(beta))
    
    # integration
    def F(fi, uE, vE):
        betam = np.arcsin(np.min(np.array([fi * eta, 1.])))
        if betam == 0.:
            return 0.
        else:
            return romberg(intfunc6, 0., betam, args=(fi * eta, uE, vE), tol=tol)

    # uniform or LLD
    if LLD == 0.:
        VIS = F(1., uE, vE) / F(1., 0., 0.)
        #  print(F(1., uE, vE))
        #  print(F(1., 0., 0.))
    else:
        Gamma = 2. * LLD / (3. - LLD)
        I = np.linspace(1. + Gamma / 2., 1. - Gamma, Na, endpoint=True)
        f = np.sqrt(np.abs((2. - 2. * I + Gamma) * (-2 + 2 * I + 5. * Gamma))) / 3. / Gamma
        f[0] = 0.
        V, A = 0., 0.
        for i in np.arange(1, Na):
            V += I[i] * (F(f[i], uE, vE) - F(f[i-1], uE, vE))
            A += I[i] * (F(f[i], 0., 0.) - F(f[i-1], 0., 0.))
        #  print(V)
        #  print(A)
        VIS = V / A
        
    return VIS

def fullvis(zetac, rho, u, v, tol=1e-5):
    """Compute full integration interferometric visibility of an ESPL.
    
        Parameters
        ----------
        zetac : complex
            Source position [thetaE]. Convention: source travels left
            to write, horizontally.
        rho : float
            Source size [thetaE].
        u, v : floats
            Einstein (u, v) parameters [thetaE^-1].
        LLD : float, optional
            Linear limb-darkening coefficient 'a': I(r) = 1 - a + a cos(theta).
            Default is 0 (uniform source). Link with Gamma: a = 3 Gamma/(2 + Gamma).
        Na : int, optional
            Number of annuli dividing the source for linear limb darkening. The
            source profile is divided in uniform brightness bins. Default is 8.
        tol : float, optional
            Tolerance in integration. Default is 1e-4.
                         
        Returns
        -------
        VIS : float
            Flat-images approximation visibitily.
    """
    # parameters
    u0 = np.abs(zetac)
    beta = np.angle(zetac) - np.pi / 2.
    uvE = np.exp(np.complex(0., -beta)) * np.complex(u, v)
    uE, vE = np.real(uvE), np.imag(uvE)
    eta = rho / u0
    betam = np.arcsin(np.min(np.array([eta, 1.])))
    
    # integrande
    def intfunc(beta, n, u0, eta, uE, vE):
        # def u_/pm
        A = np.cos(beta)
        B = np.sqrt(np.abs(eta**2 - np.sin(beta)**2))
        u1 = u0 * (A - B)
        u2 = u0 * (A + B)
        
        # general case (uE, vE)
        Omega = - uE * np.sin(beta) + vE * np.cos(beta)
        if n == 1:
            rp = lambda u: 0.5 * (u + np.sqrt(u**2 + 4.))
            if Omega == 0.:
                return 0.5 * (rp(u2)**2 - rp(u1)**2)
            else:
                freal = lambda r: np.cos(2. * np.pi * Omega * r) / (4. * Omega**2 * np.pi**2) + np.sin(2. * np.pi * Omega * r) * r / (2. * Omega * np.pi)
                return freal(rp(u2)) - freal(rp(u1))
        if n == 2:
            rp = lambda u: 0.5 * (u + np.sqrt(u**2 + 4.))
            if Omega == 0.:
                return 0.
            else:
                fimag = lambda r: -np.sin(2. * np.pi * Omega * r) / (4. * Omega**2 * np.pi**2) + np.cos(2. * np.pi * Omega * r) * r / (2. * Omega * np.pi)
                return fimag(rp(u2)) - fimag(rp(u1))
        if n == -1:
            rm = lambda u: 0.5 * (u - np.sqrt(u**2 + 4.))
            if Omega == 0.:
                return 0.5 * (rm(u2)**2 - rm(u1)**2)
            else:
                freal = lambda r: np.cos(2. * np.pi * Omega * r) / (4. * Omega**2 * np.pi**2) + np.sin(2. * np.pi * Omega * r) * r / (2. * Omega * np.pi)
                return freal(rm(u2)) - freal(rm(u1))
        if n == -2:
            rm = lambda u: 0.5 * (u - np.sqrt(u**2 + 4.))
            if Omega == 0.:
                return 0.
            else:
                fimag = lambda r: -np.sin(2. * np.pi * Omega * r) / (4. * Omega**2 * np.pi**2) + np.cos(2. * np.pi * Omega * r) * r / (2. * Omega * np.pi)
                return fimag(rm(u2)) - fimag(rm(u1))
        
    # Integration
    VpR = romberg(intfunc, -betam, betam, args=(1, u0, eta, uE, vE), tol=tol)
    VpI = romberg(intfunc, -betam, betam, args=(2, u0, eta, uE, vE), tol=tol)
    Ap  = romberg(intfunc, -betam, betam, args=(1, u0, eta, 0., 0.), tol=tol)
    
    VmR = romberg(intfunc, -betam, betam, args=(-1, u0, eta, uE, vE), tol=tol)
    VmI = romberg(intfunc, -betam, betam, args=(-2, u0, eta, uE, vE), tol=tol)
    Am  = romberg(intfunc, -betam, betam, args=(-1, u0, eta, 0., 0.), tol=tol)
    
    VIS = (np.complex(VpR, VpI) - np.complex(VmR, VmI)) / (Ap - Am)

    return VIS

def psvis(zetac, u, v):
    """Compute point-source interferometric visibility of an ESPL.
    
        Parameters
        ----------
        zetac : complex
            Source position [thetaE]. Convention: source travels left
            to write, horizontally.
        rho : float
            Source size [thetaE].
        u, v : floats
            Einstein (u, v) parameters [thetaE^-1].
    """
    # parameters
    u0 = np.abs(zetac)
    beta = np.angle(zetac) - np.pi / 2.
    uvE = np.exp(np.complex(0., -beta)) * np.complex(u, v)
    uE, vE = np.real(uvE), np.imag(uvE)
    
    # images positions
    yp = 0.5 * (u0 + np.sqrt(u0**2 + 4.))
    ym = 0.5 * (u0 - np.sqrt(u0**2 + 4.))
    
    # ps magnification
    B = (u0**2 + 2.) / (u0 * np.sqrt(u0**2 + 4.))
    mup = 0.5 * (1. + B)
    mum = 0.5 * (1. - B)

    # Visibility
    Fp = mup * np.exp(np.complex(0., -2. * np.pi * vE * yp))
    Fm = mum * np.exp(np.complex(0., -2. * np.pi * vE * ym))
    
    VIS = (Fp - Fm) / (mup - mum)
    
    return VIS

def newz(zetac, rho, Nc_init, Nc_fin):
    """Resample images contours and compute uniform contouring
    
        Parameters
        ----------
        zetac : complex
            Source position [thetaE].
            Convention: source travels left to write, horizontally.
        rho : float
            Source size [thetaE].
        Nc_init : int
            Source contour initial sampling
        Nc_fin : int
            Source contour final samplig (uniform on images)
            
        Returns
        -------
        nzp, nzm : tuple of complex
            The new contours of the positive (p)
            and negative (m) parity images.
    """
    # intial contour sample at high density (endpoint = FALSE)
    theta = np.linspace(0., 2. * np.pi, Nc_init, endpoint=False)
    
    # get tangents and positions
    zp, zm, dzp, dzm = dz(zetac, rho, theta)
    
    # create curvilinear abcissa
    adzp = np.abs(dzp)
    phip = np.cumsum(adzp)
    phip = phip / np.amax(phip)

    adzm = np.abs(dzm)
    phim = np.cumsum(adzm)
    phim = phim / np.amax(phim)
    
    # interpolation
    intphim = interp1d(phim, theta, bounds_error=False, fill_value=(0., 1.))
    intphip = interp1d(phip, theta, bounds_error=False, fill_value=(0., 1.))
    
    # resampled contour (endpoint = FALSE)
    phi = np.linspace(0., 1., Nc_fin, endpoint=False)
    
    nzp, _, _, _ = dz(zetac, rho, intphip(phi))
    _, nzm, _, _ = dz(zetac, rho, intphim(phi))
    
    # return uniform resampling of the two (p ad m) images
    return nzp, nzm

def dz(zetac, rho, theta):
    """Compute contours and tangents to the contours (ESPL)
        
        Parameters
        ----------
        zetac : complex
            Source position [thetaE].
            Convention: source travels left to write, horizontally.
        rho : float
            Source size [thetaE].
        theta : float
            Angle of sampling points on source contour.
            
        Returns
        -------
        nzp, nzm, dzp, dzm : tuple of complex
            (nzp, nzm) are the new contours of the positive (p) and negative (m) parity images.
             
            corresponding tangents.
    """
    # compute ESPL model
    zp, zm, dzp, dzm = [], [], [], []
    for beta in theta:
        I = np.complex(0., 1.)
        A = rho * np.exp(I * beta)
        zeta = zetac + A
        B = np.sqrt(1. + 4. / np.abs(zeta)**2)
        Ep = 0.5 * zeta * (1. + B)
        Em = 0.5 * zeta * (1. - B)
        Cp = 0.5 * I * A * (1. + B)
        Cm = 0.5 * I * A * (1. - B)
        D = 2. * zeta * np.real(I * A * np.conj(zeta)) / np.abs(zeta)**4 / B
        
        # compute tangents
        dzp.append(Cp - D)
        dzm.append(Cm + D)
        
        # compute positions
        zp.append(Ep)
        zm.append(Em)

    # return positions and tangents
    return np.array(zp), np.array(zm), np.array(dzp), np.array(dzm)

def vismag(zetac, rho, u, v, Nc_init, Nc_fin):
    """Compute model visibilities and magnifications of
    positive and negative parity images (ESPL)
    
        Parameters
        ----------
        zetac : complex
            Source position [thetaE].
            Convention: source travels left to write, horizontally.
        rho : float
            Source size [thetaE].
        u, v : floats
            (u, v) parameters [thetaE^-1].
        Nc_init : int
            Source contour initial sampling
        Nc_fin : int
            Source contour final samplig (uniform on images)
                         
        Returns
        -------
        Vp, Vm, Np, Nm : tuple of complex
            (Vp, Vm) are non-normalized visibilities for p and m images
            (Np, Nm) are p an m images area
    """
    # images position
    nzp, nzm = newz(zetac, rho, Nc_init, Nc_fin)
    xp = np.real(nzp)
    yp = np.imag(nzp)
    xm = np.real(nzm)
    ym = np.imag(nzm)
    
    aVp, aVm = [], []
    for i in range(len(xp)):
        aVp.append(np.exp(np.complex(0., - 2. * np.pi) * (u * xp[i] + v * yp[i])))
        aVm.append(np.exp(np.complex(0., - 2. * np.pi) * (u * xm[i] + v * ym[i])))
    
    # positive parity image
    if (u == 0) and (v == 0):
        Vp, Np = 1., 1.
    elif np.abs(u) > np.abs(v):
        dyp = np.roll(yp, -1) - yp
        aV = np.array(aVp) * np.complex(0., 1.) / u / 2. / np.pi # formule "+", et contour direct
        Vp = np.sum(aV * dyp)
        Np = np.sum(xp * dyp) # formule "+", et contour direct
 #       aVp = np.array(aVp) * np.complex(0., 1.) / u / 2. / np.pi
 #       dyp = np.roll(yp, 1) - yp
 #       Vp = np.sum(aVp * dyp)
 #       Np = np.sum(xp * dyp)
    else:
        dxp = np.roll(xp, -1) - xp
        aV = - np.array(aVp) * np.complex(0., 1.) / v / 2. / np.pi # formule "-", et contour direct
        Vp = np.sum(aV * dxp)
        Np = - np.sum(yp * dxp) # formule "-", et contour direct
        #aVp = - np.array(aVp) * np.complex(0., 1.) / v / 2. / np.pi
        #dxp = np.roll(xp, 1) - xp
        #Vp = np.sum(aVp * dxp)
        #Np = np.sum(yp * dxp)
    
    ## TEST !!!! np.roll(xp, -1) - xp
   # dym = np.roll(ym, -1) - ym
   # aV = - np.array(aVm) * np.complex(0., 1.) / u / 2. / np.pi # formule "+", mais contour inverse
   # Vm = np.sum(aV * dym)
   # Nm = - np.sum(xm * dym) # formule "+", mais contour inverse
   # print("Cas 1:", Vm, Nm, np.abs(Vm))
    
    #dxm = np.roll(xm, -1) - xm
    #aV = np.array(aVm) * np.complex(0., 1.) / v / 2. / np.pi # formule "-", mais contour inverse
    #Vm = np.sum(aV * dxm)
    #Nm = np.sum(ym * dxm) # formule "-", mais contour inverse
    #print("Cas 2:", Vm, Nm, np.abs(Vm))
        
    # negative parity image
    if (u == 0) and (v == 0):
        Vm, Nm = 1., 1.
    elif np.abs(u) > np.abs(v):
        dym = np.roll(ym, -1) - ym
        aV = - np.array(aVm) * np.complex(0., 1.) / u / 2. / np.pi # formule "+", mais contour inverse
        Vm = np.sum(aV * dym)
        Nm = - np.sum(xm * dym) # formule "+", mais contour inverse
#        dym = np.roll(ym, 1) - ym
#        aVm = np.array(aVm) * np.complex(0., 1.) / u / 2. / np.pi
#        Vm = - np.sum(aVm * dym)
#        Nm = - np.sum(xm * dym)
    else:
        dxm = np.roll(xm, -1) - xm
        aV = np.array(aVm) * np.complex(0., 1.) / v / 2. / np.pi # formule "-", mais contour inverse
        Vm = np.sum(aV * dxm)
        Nm = np.sum(ym * dxm) # formule "-", mais contour inverse
        #aVm = - np.array(aVm) * np.complex(0., 1.) / v / 2. / np.pi
        #dxm = np.roll(xm, 1) - xm
        #Vm = - np.sum(aVm * dxm)
        #Nm = - np.sum(ym * dxm)
                
    # return non-normalized visibilities and magnifications
    return Vp, Vm, Np, Nm

def PSPL_VIS(zetac, u, v):
    """Compute theoretical interferometric visibility (PSPL)
    
        Parameters
        ----------
        zetac : complex
            Source position [thetaE].
            Convention: source travels left to write, horizontally.
        u, v : floats
            (u, v) parameters [thetaE^-1].
               
        Returns
        -------
        out : complex
            Interferometric visibility.
    """
    # images position
    zp = 0.5 * zetac * (1. + np.sqrt(1. + 4. / np.abs(zetac)**2))
    zm = 0.5 * zetac * (1. - np.sqrt(1. + 4. / np.abs(zetac)**2))
    
    # magnifications (absolute)
    mup = np.abs(1. / (1. - 1. / np.abs(zp)**4))
    mum = np.abs(1. / (1. - 1. / np.abs(zm)**4))
    mut = mup + mum
    
    # exponential factors of the sum
    Vp = (mup / mut) * np.exp(np.complex(0., - 2. * np.pi) * (u * np.real(zp) + v * np.imag(zp)))
    Vm = (mum / mut) * np.exp(np.complex(0., - 2. * np.pi) * (u * np.real(zm) + v * np.imag(zm)))
        
    # return global complex visibility
    return Vp + Vm


