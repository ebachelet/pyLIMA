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

Clone the repository or download as a ZIP file. Then

#### DO NOT RUN python install setup.py SVP



Please use pyLIMA as a external module, by doing a global import :

import sys

sys.path.append(your-path-to-pyLIMA-directory)

from pyLIMA import whatyouneed


#### Required libraries
pyLIMA also require some python modules installation, see the documentation link above.

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
