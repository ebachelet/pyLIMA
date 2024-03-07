'''
Welcome to pyLIMA (v2) tutorial 6!

In this tutorial you will learn how to use pyLIMASS on KB0926  and the relevant publication is:
https://iopscience.iop.org/article/10.1088/0004-637X/741/1/22/pdf

Please take some time to familiarize yourself with the pyLIMA documentation.
'''

### Import libraries

import numpy as np
from pyLIMA.pyLIMASS import SourceLensProbabilities
import matplotlib.pyplot as plt


### Collect observables

Hs = 13.78
eHs = 0.03

Is = 15.856
eIs = 0.03

Vs = 17.677
eVs = 0.03

tE = 61.47
etE = 0.4

rho_s = 0.00539
erho_s= 0.02*rho_s

piE = 0.223
epiE = 0.037

phiE = 0.74
ephiE = 0.16

Avs = 2.14
eAvs = 0.3

### Generate samples and create observables

pie = np.random.normal(piE,epiE,10000)
ppie = np.random.normal(phiE,ephiE,10000)

pien = pie*np.cos(ppie)
piee = pie*np.sin(ppie)


mags_source = {'Vmag':np.random.normal(Vs,eVs,10000),'Imag':np.random.normal(Is,eIs,10000),'Hmag':np.random.normal(Hs,eHs,10000)}
mags_baseline = {'Hmag':np.random.normal(13.77,0.05,10000)}


obs = {'log10(pi_E)':np.log10(pie),'phi_E':ppie,'log10(rho_s)':np.log10(np.random.normal(rho_s,erho_s,10000)),'mags_source':mags_source,'mags_baseline':mags_baseline,'log10(Av_s)':np.log10(np.random.normal(Avs,eAvs,10000))}

### Create the pyLIMASS object

SLP = SourceLensProbabilities(obs,stellar_lens=True)

### Create the GM object
SLP.generate_GM()

### Restrict to Ds<15 kpc

SLP.bounds[1] = [0.0,15]
SLP.update_priors()
### Find maximum likelihood region

results = SLP.de(popsize=10)

### Check if model is plausible
metric = SLP.model_is_plausible(results['x'])

### Explore posterior with MCMC
seed = results['x']
sampler = SLP.mcmc([seed],n_chains=20000)
chain = sampler.get_chain()
chains=np.zeros((20000, 32, 17))
chains[:,:,:-1] = chain
chains[:,:,-1] = sampler.get_log_prob()


### Plot results vs Muraki et al.

plt.hist(chain[10000:,:,8].ravel())
plt.axvspan(0.56-0.09,0.56+0.09,alpha=0.5,color='r')
plt.xlabel(r'$M_L~[M_\odot]$')

plt.figure()
Dl = chain[:,:,1]*chain[:,:,9]
plt.hist(Dl.ravel())
plt.axvspan(3.04-0.33,3.04+0.33,alpha=0.5,color='r')
plt.xlabel(r'$D_L~[kpc]$')

plt.show()
### This concludes tutorial 6.

