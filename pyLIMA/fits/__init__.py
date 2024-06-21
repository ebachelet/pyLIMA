from .BOOTSTRAP_fit import BOOTSTRAPfit
from .DEMC_fit import DEMCfit
from .DE_fit import DEfit
from .DREAM_fit import DREAMfit
from .GRIDS_fit import GRIDfit
from .LM_fit import LMfit
from .TRF_fit import TRFfit
from .MCMC_fit import MCMCfit
from .MINIMIZE_fit import MINIMIZEfit

import numpy as np
#if int(np.__version__[0]) >= 2:
#    pass #Because pymoo not compatible with numpy2.0.0 21/6/2024
    #raise NotImplementedError("SOME MESSAGE HERE")  # Dummy function, if called will raise NotImplementedError
##else:
#    from .NGSA2_fit import NGSA2fit

# pylima/fits/__init__.py
...
if int(np.__version__[0]) >= 2:
    def NGSA2fit(*args, **kwargs):
        # Dummy function, if called will raise NotImplementedError
        raise NotImplementedError("SOME MESSAGE HERE")
else:
    from .NGSA2_fit import NGSA2fit

__all__ = ["BOOTSTRAPfit", "DEMCfit", "DEfit", "DREAMfit", "GRIDfit", "LMfit",
           "MCMCfit", "MINIMIZEfit",  "TRFfit", "NGSA2fit"]
