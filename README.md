[![Build Status](https://travis-ci.org/ebachelet/pyLIMA.svg?branch=master)](https://travis-ci.org/ebachelet/pyLIMA)
[![Coverage Status](https://coveralls.io/repos/github/ebachelet/pyLIMA/badge.svg?branch=master)](https://coveralls.io/github/ebachelet/pyLIMA?branch=master)

# pyLIMA

Authors : Etienne Bachelet, etibachelet@gmail.com 
	  Rachel Street, rstreet@lcogt.net
	  Valerio Bozza, valboz@sa.infn.it
	  Martin Norbury, mnorbury@lcogt.net
	  and friends!	

pyLIMA is an open source for modeling microlensing events.
It should be flexible enough to handle your data and fit it.
You can also practice by simulating events.

# Documentation and Installation


[Documentation](https://ebachelet.github.io/pyLIMA/)

### Required materials 
Regular C/C++ and fortran compilers are required for packages installation

You also need [SWIG](http://www.swig.org/download.html)

You need [pip](https://pip.pypa.io/en/stable/installing/) or you can install manually
the required libraries [Documentation](https://ebachelet.github.io/pyLIMA/)


### Installation and use

Clone the repository or download as a ZIP file. Then


python2.7 setup.py build_ext --build-lib=./pyLIMA/subroutines/VBBinaryLensingLibrary
(and, not mandatory, python setup.py clean --all)

The install the required libraries with

pip install -r requirements.txt

This new procedure which should avoid the previous installations headaches!
Successfully test on various UNIX, MAC and Windows! If you encounter any problems,
please contact etibachelet@gmail.com.



Please use pyLIMA as a external module, by doing a global import :

import sys

sys.path.append(your-path-to-pyLIMA-directory)

from pyLIMA import whatyouneed

### Examples
Examples can be found in your pyLIMA directory. Look on the documentation to learn how to run it.



# What can you do?


#### pyLIMA is now in beta!! Here is the status of implemented microlensing models:

| Model | Implemented | Examples | Fit Method Advice | 
| :---         |     :---:      |:---: |    ---: |
| Point-Source Point Lens (PSPL)   | ![Alt text](./doc/HGF.png?raw=true)     | Yes | Levenberg-Marquardt (LM)     |
| Finite-Source Point Lens (FSPL)   |  ![Alt text](./doc/HGF.png?raw=true)      | Yes | Levenberg-Marquardt (LM) or Differential Evolution (DE)    |
| Double-Source Point Lens (DSPL)   | ![Alt text](./doc/HGF.png?raw=true)     |  Yes | Differential Evolution (DE)    |
| Uniform-Source Binary Lens (USBL)   | ![Alt text](./doc/WIP.png?raw=true)  | No |      |

#### pyLIMA can also treat Second Order effects :

| Second-Order Effects | Implemented | Examples |Fit Method Advice |
| :---         |     :---:      |   :---: |   ---: |
| Annual parallax   |  ![Alt text](./doc/HGF.png?raw=true)      | No | Levenberg-Marquardt (LM)     |
| Terrestrial parallax   |  ![Alt text](./doc/HGF.png?raw=true)     | No | Levenberg-Marquardt (LM) |
| Space parallax   |  ![Alt text](./doc/HGF.png?raw=true)      |  No| Levenberg-Marquardt (LM)    |
| Orbital Motion   | ![Alt text](./doc/WIP.png?raw=true)     | No |       |
| Xallarap   | ![Alt text](./doc/WIP.png?raw=true)    | No |       |


# How to contribute?

Want to contribute? Bug detections? Comments?
Please email us : etibachelet@gmail.com, rstreet@lcogt.net, valboz@sa.infn.it
