.. _fits:

Notes on fits
=============

Below are some notes, details and advices about the fits. Most of the fits methods relies on `scipy <https://scipy.org/>`_. While several algorithms are implemented, see `here <https://github.com/ebachelet/pyLIMA/tree/Rebranding/pyLIMA/fits>`_, pyLIMA relies mainly of three main fitting methods. 

Differential Evolution (DE)
---------------------------

DE is a population-based algorithm for global optimization of complex functions by `Storn & Price <https://link.springer.com/article/10.1023/A:1008202821328>`_. There are plenty of litterature discussing the remarkable performance about DE and its derivatives. 

DE is the workhorse of pyLIMA fits and it has proven its reliability and effectiveness on many fits.    

Gradient-Like methods 
---------------------

There are two methods to performs gradient-like fits in pyLIMA, the Trust-Reflective Function (TRF) and Levenberg-Marquardt (LM). They are almost identical, but the former accounts of parameters boundaries (which is desirable when the event is not very well constrained). They are very efficient to find the best models as soon as a minima is found. Jacobian are implemented for simplest models (i.e. PSPL and FSPL without second-order effects).

MCMC
----

MCMC is implemeented via the awesome `emcee <https://emcee.readthedocs.io/en/stable/>`_ package. The number of walkers and links can be adjusted. Uniform priors at the parameters boundaries are set by default.

Parallelization
---------------

DE and MCMC methods can significantly be speed-up by implementing a pool of workers via `multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_:

.. code-block:: python
    
    import multiprocessing as mul
    pool = mul.Pool(processes = 4)

    my_fit.fit(computational_pool = pool)
    
    
Priors
------
pyLIMA now includes the possibility to add user-defined priors. While they are no priors by default, uniform and gaussian priors are `available <https://github.com/ebachelet/pyLIMA/blob/Rebranding/pyLIMA/priors/parameters_priors.py>`_. Users can also define their own functions as long as they return a pdf for a given parameters, for example a Cauchy distribution:


.. code-block:: python

    class CauchyDistribution(object):

        def __init__(self, mean, gamma):
        
            self.mean = mean
            self.gamma = gamma

        def pdf(self, x):
        
            denominator = np.pi*self.ggam*(1+(x-self.mean)**2/self.gamma**2)
            probability = 1 / denominator
            
            return probability
    
    from pyLIMA.models import PSPLmodel
    from pyLIMA.fits import DEfit
    
    model = PSPLmodel(event)
    thefit = DEfit(model)
   
    t0prior =  CauchyDistribution(2459856,0.5)
    u0prior =  CauchyDistribution(0.1,0.5)
    tEprior =  CauchyDistribution(22,0.5)
    
    thefit.priors = [t0prior,u0prior,tEprior]
    
Loss functions
--------------

By default, pyLIMA implements three loss functions:

-   :math:`\chi^2` : the sum of the normed residuals
-   :math:`\log \cal L` : the ln-likelihood, that includes priors
-   soft_l1 : the soft_l1 function is close to the `Huber loss function <https://en.wikipedia.org/wiki/Huber_loss?>`_ and it is very robust against outliers

.. code-block:: python
    
    from pyLIMA.fits import DEfit
    thefit = DEfit(model,loss_function='soft_l1')
    
Fitting algorithms have default loss functions described in :ref:`pyLIMAModules`. The sign of the loss function will depends if the fitting algorithms maximize or minimize the objective function.


Advices on fitting binary lightcurves
-------------------------------------

For fitting binary models, DE has proven to be reliable to locate global minima. However, we recommand to explore  :math:`s\le1` and :math:`s\ge1` separetely, especially to explore carefully the close/wide degeneracy (`see <https://ui.adsabs.harvard.edu/abs/1999A%26A...349..108D/abstract>`_). One the minimas are found, each of them should be explored using MCMC.

We note that some wide binary systems can be hard, if not impossible, to model with the default pyLIMA settings. `OGLE-2015-BLG-0060 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.4603T/abstract>`_ is a good example. In this case, it is recomanded to change the origin of the system, for example to the primary body:

.. code-block:: python
    
    from pyLIMA.models import USBLmodel
    
    usbl = USBLmodel(current_event,origin=['primary',[0,0]])
    



