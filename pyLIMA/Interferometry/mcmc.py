# Author: Arnaud Cassan.
# Distributed under the terms of the GNU General Public License.

import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import emcee
import matplotlib.pylab as plt
from microlensing.utils import checkandtimeit, verbosity, printi, printd, printw
from multiprocessing import Pool
from microlensing.interferometry.ESPL import mod_VIS
from microlensing.interferometry.data import obs_VIS2

class InterferolensingModels():
    """List of interferometric microlensing models
    
        Parameters
        ----------
        model : string
            Model name.
            
        Attributes
        ----------
        model : string
            Model name.
        ndim : int
            Number of parameters.
        params : dict
            Dictionary of parameters (name and value, default values are None).
        pnames : string
            Print format for parameter names.
        lxnames : list
            Parameter names in LaTeX format.
            
        Usage
        -----
        >>> ilm = InterferolensingModels(model)
        >>> ilm.pnames
        >>> ilm.params['thetaE']
        >>> ilm.ndim
        >>> ilm.lxnames
    """
    
    def __init__(self, model):
        self.model = model
    
        # ESPL - single epoch, contouring
        if self.model == 'ESPL_CONT_SINGEP':
            self.ndim = 4
            self.params = {'alpha1':None, 'thetaE':None, 'eta1':None, 'u1':None}
            self.lxnames = [r'$\alpha_1$', r'$\theta_E$', r'$\eta_1$', r'$u_1$']
            
        # ESPL - single epoch, full integration
        elif self.model == 'ESPL_FULLINT_SINGEP':
            self.ndim = 4
            self.params = {'alpha1':None, 'thetaE':None, 'eta1':None, 'u1':None}
            self.lxnames = [r'$\alpha_1$', r'$\theta_E$', r'$\eta_1$', r'$u_1$']
            
        # ESPL - single epoch, point-source approximation
        elif self.model == 'ESPL_PS_SINGEP':
            self.ndim = 3
            self.params = {'alpha1':None, 'thetaE':None, 'u1':None}
            self.lxnames = [r'$\alpha_1$', r'$\theta_E$', r'$u_1$']
            
        # ESPL - single epoch, flat-images approsimation
        elif self.model == 'ESPL_FLAT_SINGEP':
            self.ndim = 3
            self.params = {'alpha1':None, 'thetaE':None, 'eta1':None}
            self.lxnames = [r'$\alpha_1$', r'$\theta_E$', r'$\eta_1$']
            
        # ESPL - multi-epochs, contouring
        elif self.model == 'ESPL_CONT_MULTIEP':
            self.ndim = 6
            self.params = {'alpha':None, 'thetaE':None, 'eta0':None, 't*':None, 't0':None, 'u0':None}
            self.lxnames = [r'$\alpha$', r'$\theta_E$', r'$\eta_0$', r'$t_*$', r'$t_0$', r'$u_0$']
            
        # ESPL - multi-epochs, full integration
        elif self.model == 'ESPL_FULLINT_MULTIEP':
            self.ndim = 6
            self.params = {'alpha':None, 'thetaE':None, 'eta0':None, 't*':None, 't0':None, 'u0':None}
            self.lxnames = [r'$\alpha$', r'$\theta_E$', r'$\eta_0$', r'$t_*$', r'$t_0$', r'$u_0$']
            
        # ESPL - multi-epochs, full integration, microlensing parameters
        elif self.model == 'ESPL_MLENSING_MULTIEP':
            self.ndim = 6
            self.params = {'alpha':None, 'thetaE':None, 'rho':None, 'tE':None, 't0':None, 'u0':None}
            self.lxnames = [r'$\alpha$', r'$\theta_E$', r'$\rho$', r'$t_E$', r'$t_0$', r'$u_0$']
            
        # ESPL - multi-epochs, point-source approximation
        elif self.model == 'ESPL_PS_MULTIEP':
            self.ndim = 5
            self.params = {'alpha':None, 'thetaE':None, 'tE':None, 't0':None, 'u0':None}
            self.lxnames = [r'$\alpha$', r'$\theta_E$', r'$t_E$', r'$t_0$', r'$u_0$']
            
        # ESPL - multi-epochs, flat-images approsimation
        elif self.model == 'ESPL_FLAT_MULTIEP':
            self.ndim = 5
            self.params = {'alpha':None, 'thetaE':None, 'eta0':None, 't*':None, 't0':None}
            self.lxnames = [r'$\alpha$', r'$\theta_E$', r'$\eta_0$', r'$t_*$', r'$t_0$']

        else:
            raise ValueError("Wrong model ID ({}) in InterferolensingModels".format(model))
            
        # print format for parameter names
        self.pnames = '(' + ', '.join([p for p in self.params]) + ')'
        
    
def fit_emcee(oifitslist, model, params, priorfun, samplerfile, nwalkers, chainlen, ncpu, Nc=20000, LLD=0., Na=8, tol=1e-5, ref_MJD=0., ifconverged=False, resume=False):
    """Compute MCMC chains using EMCEE-3
    
        Parameters
        ----------
        oifitslist : list of strings
            List of input OIFITS files
        params : list
            Constains 2-elements lists [[pi_low, pi_high], ...] of parameters bounds
            Ialpha, Irho, IthetaE, Iu0, ItE, It0 : arrays of [float, float]
            Intervals for drawing MCMC samples for each parameter
        Nc : int
            Common source contour initial sampling (global Nc_init)
            and final samplig (global Nc_fin)
        samplerfile : string
            Name of file where the chains are stored (HDF5 format)
        nwalkers : int
            Number of chains in parallel
        chainlen : int
            Individual chain length for one MCMC run
        ncpu : int
            Number of chains computed in parallel (threads)
        ref_MJD : float
            Reference MJD
        ifconverged : boolean
            Automatic stop if convergence
        resume : boolean
            Resume previous run
        hdulists (global) : list
            Contains a list [u, v, VIS2, VIS2ERR, MJD] for each (ep, B, lbd)
        
        Outputs
        -------
        samplerfile : string
            Name of file where the chains are stored (HDF5 format)
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
    
    # get model
    ilm = InterferolensingModels(model)
    ndim = ilm.ndim
    mnames = ilm.pnames
    printi(tcol + "Chosen model : " + tend + "{0}, {1}".format(model, mnames))
    
    # setup
    initial_length = chainlen
    resume_length = chainlen
    
    # read OIFITS files and create data fit-lists
    global hdulists
    hdulists = []
    for oifits in oifitslist:
        for b in range(6):
            for l in range(6):
                hdulists = hdulists + obs_VIS2(oifits, b + 1, l + 1, ref_MJD=ref_MJD)
                
    # initialize run
    if not resume:
        pos = []
        # Initialize the walkers
        for p in params:
            pos.append(np.random.uniform(p[0], p[1], nwalkers))
        pos = np.array(pos).T
                
        # Set up the backend
        backend = emcee.backends.HDFBackend(samplerfile)
        backend.reset(nwalkers, ndim)

        # Initialize the sampler
        with Pool(processes=ncpu) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args=[priorfun], kwargs={'model':model, 'Nc':Nc, 'LLD':LLD ,'Na':Na, 'tol':tol}, backend=backend, pool=pool)
                 
            # Compute average changes of autocorrelation time
            index = 0
            autocorr = np.empty(initial_length)

            # Initialize the converge variable
            taup = np.inf

            # Now we'll sample for up to max_n steps
            for sample in sampler.sample(pos, iterations=initial_length, progress=True):
                if ifconverged:
                    # Convergence test every 300 steps
                    if sampler.iteration % 300: continue

                    # Force computing autocorrelation
                    tau = sampler.get_autocorr_time(tol=0)
                    autocorr[index] = np.mean(tau)
                    index = index + 1

                    # Test convergence
                    converged = np.all(tau * 100 < sampler.iteration)
                    converged &= np.all(np.abs(taup - tau) / tau < 0.01)
                    if converged: break
                    taup = tau
                    
    # resume / continue run
    else:
        backend = emcee.backends.HDFBackend(samplerfile)
        printi(tcol + "Initial size :" + tend + " {0}".format(backend.iteration))
        with Pool(processes=ncpu) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args=[priorfun], kwargs={'model':model, 'Nc':Nc, 'LLD':LLD ,'Na':Na, 'tol':tol}, backend=backend, pool=pool)
            sampler.run_mcmc(None, resume_length, progress=True)
        printi(tcol + "Final size :" + tend + " {0}".format(backend.iteration))

def logprob(x, priorfun, model=None, Nc=None, LLD=None, Na=None, tol=None):
    """Compute gobal chi2 for a set epochs, baselines and wavelenghts,
    and fit for (alpha [deg], rho, thetaE [mas], u0, tE, t0) for time series
    or (alpha [deg], rho, thetaE [mas], u0) for a single epoch
        
        Parameters
        ----------
        hdulists (global) : list
            Contains a list [u, v, VIS2, VIS2ERR, MJD] for each (ep, B, lbd)
        x : list
            Model parameters (alpha, rho, thetaE, u0, tE, t0) for time series
            or (alpha, rho, thetaE, u0) for a single epoch
  
        Returns
        -------
        logp : float
            log(Likelihood) = - chi^2 / 2
    """
    # set I/O shell display
    tcol, tun, tend, tit = "\033[0m\033[34m", "\033[0m\033[1m\033[34m", "\033[0m", "\033[0m\033[3m"
        
    # conversion mas/deg
    mas = np.pi / 180. / 3600. / 1000.
    
    # sum chi2 over all epochs, baselines and wavelenghts
    chi2 = 0.
    for hdulist in hdulists:
        # get individual data
        u, v, VIS2, VIS2ERR, MJD = hdulist
        
        # conversion mas -> rad (thetaE is always x[1])
        u = u * mas * x[1]
        v = v * mas * x[1]
            
        # single epoch, contouring or full integration (alpha1, thetaE, eta1, u1)
        if model == 'ESPL_CONT_SINGEP' or model == 'ESPL_FULLINT_SINGEP':
            zetac = np.complex(0., x[3]) * np.exp(np.complex(0., np.pi / 180. * x[0]))
            VISE2 = np.abs(mod_VIS(zetac, x[2] * x[3], u, v, model, Nc=Nc, LLD=LLD, Na=Na, tol=tol))**2
            
        # single epoch, point-source approximation (alpha1, thetaE, u1)
        elif model == 'ESPL_PS_SINGEP':
            zetac = np.complex(0., x[2]) * np.exp(np.complex(0., np.pi / 180. * x[0]))
            VISE2 = mod_VIS(zetac, 0, u, v, model)**2
            
        # single epoch, flat-images approsimation (alpha1, thetaE, eta1)
        elif model == 'ESPL_FLAT_SINGEP':
            zetac = np.complex(0., 1.) * np.exp(np.complex(0., np.pi / 180. * x[0]))
            VISE2 = mod_VIS(zetac, x[2], u, v, model, LLD=LLD, Na=Na, tol=tol)**2
            
        # multi-epochs, contouring or full integration (alpha, thetaE, eta0, t*, t0, u0)
        elif model == 'ESPL_CONT_MULTIEP' or model == 'ESPL_FULLINT_MULTIEP':
            zetac = np.complex((MJD - x[4]) * x[2] * x[5] / x[3], x[5]) * np.exp(np.complex(0., np.pi / 180. * x[0]))
            VISE2 = np.abs(mod_VIS(zetac, x[2] * x[5], u, v, model, Nc=Nc, LLD=LLD, Na=Na, tol=tol))**2
        
        # multi-epochs, full integration, microlensing parameters (alpha, thetaE, rho, tE, t0, u0)
        elif model == 'ESPL_MLENSING_MULTIEP':
            zetac = np.complex((MJD - x[4]) / x[3], x[5]) * np.exp(np.complex(0., np.pi / 180. * x[0]))
            VISE2 = np.abs(mod_VIS(zetac, x[2], u, v, model, Nc=Nc, LLD=LLD, Na=Na, tol=tol))**2

        # multi-epochs, point-source approximation (alpha, thetaE, tE, t0, u0)
        elif model == 'ESPL_PS_MULTIEP':
            zetac = np.complex((MJD - x[3]) / x[2], x[4]) * np.exp(np.complex(0., np.pi / 180. * x[0]))
            VISE2 = mod_VIS(zetac, 0, u, v, model)**2
                        
        # multi-epochs, flat-images approsimation (alpha, thetaE, eta0, t*, t0)
        elif model == 'ESPL_FLAT_MULTIEP':
            zetac = np.complex((MJD - x[4]) * x[2] / x[3], 1.) * np.exp(np.complex(0., np.pi / 180. * x[0]))
            VISE2 = mod_VIS(zetac, x[2], u, v, model, LLD=LLD, Na=Na, tol=tol)**2
     
        else:
            raise ValueError("Wrong model ID ({}) in logprob".format(model))
            
        # add chi2/pt
        chi2 +=  (VIS2 - VISE2)**2 / VIS2ERR**2
        
    # log probability
    logp = -0.5 * chi2 + priorfun(x)
    
    # print current values of parameters and chi2
    printd(tcol + "Parameters | chi2 | log(P): " + tend + "{0} | {1} | {2}".format(x, chi2, logp))
        
    # return log(prob)
    return logp
