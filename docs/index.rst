.. pyLIMA documentation master file, created by sphinx-quickstart on Thu May 19 11:03:24 2016.

Welcome to pyLIMA documentation! 
================================

pyLIMA is the first microlensing analysis open-source software, primarly designed to 
fit real data. But more can be done, see the :ref:`examples`. You can find more information
on the `pyLIMA paper(s) <https://arxiv.org/abs/1709.08704>`_.


Quickstart
==========

After the :ref:`installation` step, you can check the version and run a quick test fit after `downloading the data <https://github.com/ebachelet/pyLIMA/blob/master/examples/Survey_1.dat>`_:

.. code-block:: python
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    import pyLIMA
    print(pyLIMA.__version__)
    

    from pyLIMA.fits import TRFfit
    from pyLIMA.models import PSPL_model
    
    from pyLIMA import event
    from pyLIMA import telescopes

    your_event = event.Event()
    your_event.name = 'pyLIMA example'
    
    data_1 = np.loadtxt('path_to_the_data/Survey_1.dat')
    telescope_1 = telescopes.Telescope(name='OGLE',
                                   camera_filter='I',
                                   light_curve=data_1.astype(float),
                                   light_curve_names=['time', 'mag', 'err_mag'],
                                   light_curve_units=['JD', 'mag', 'mag'])

    telescope_1.plot_data()
    plt.show()    
    
    your_event.telescopes.append(telescope_1)
    
    
    pspl = PSPL_model.PSPLmodel(your_event)
    
    my_fit = TRFfit(pspl)
    my_fit.model_parameters_guess = [79.9, 0.008, 10.1]
    my_fit.fit()
    my_fit.fit_outputs()


For more details, check the :ref:`conventions` and :ref:`pyLIMAModules`.

User Guide
----------

.. toctree::
   :maxdepth: 1

   source/Installation
   source/Conventions
   source/Examples
   source/NotesOnFits

pyLIMA modules details
----------------------

Here is the (hopefully up-to-date) documentation
for all submodules.

.. toctree::
   :maxdepth: 1

   source/pyLIMAModules  

