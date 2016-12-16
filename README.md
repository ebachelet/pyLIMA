[![Build Status](https://travis-ci.org/ebachelet/pyLIMA.svg?branch=master)](https://travis-ci.org/ebachelet/pyLIMA)
[![Coverage Status](https://coveralls.io/repos/github/ebachelet/pyLIMA/badge.svg?branch=master)](https://coveralls.io/github/ebachelet/pyLIMA?branch=master)

# pyLIMA

Authors : Etienne Bachelet, etibachelet@gmail.com 
	  Rachel Street, rstreet@lcogt.net
	  Valerio Bozza, valboz@sa.infn.it
	  Martin Norbury, mnorbury@lcogt.net
	  and friends!	

# What is that?

pyLIMA is an open source for modeling microlensing events.
It should be flexible enough to handle your data and fit it.
You can also practice by simulating events.

# Installation

Clone the repository or download as a ZIP file. Then, ## DO NOT RUN python install setup.py.
Please use pyLIMA as a external module, by doing a global import :

import sys
sys.path.append(your-path-to-pyLIMA-directory)

from pyLIMA import whatyouneed

pyLIMA also require some python modules installation, see the documentation below.

# What can you do?


This is now in beta!! Here is what you can do :

| Model | Implemented | Examples | Fit Method Advice | 
| :---         |     :---:      |:---: |    ---: |
| Point-Source Point Lens (PSPL)   | Yes     | Yes | Levenberg-Marquardt (LM)     |
| Finite-Source Point Lens (FSPL)   | Yes     | Yes | Levenberg-Marquardt (LM) or Differential Evolution (DE)    |
| Double-Source Point Lens (DSPL)   | Yes     |  Yes | Differential Evolution (DE)    |
| Uniform-Source Binary Lens (USBL)   | In Progress     | No |      |


| Second-Order Effects | Implemented | Examples |Fit Method Advice |
| :---         |     :---:      |   :---: |   ---: |
| Annual parallax   | Yes     | No | Levenberg-Marquardt (LM)     |
| Terrestrial parallax   | Yes     | No | Levenberg-Marquardt (LM) |
| Space parallax   | Yes     |  No| Differential Evolution (DE)    |
| Orbital Motion   | In Progress     | No |       |
| Xallarap   | In Progress     | No |       |

# Documentation
Have a look to the documentation (hopefully up to date):

[Documentation](https://ebachelet.github.io/pyLIMA/)

# How to contribute?

Please email me : etibachelet@gmail.com
