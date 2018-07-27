[![Build Status](https://travis-ci.org/ebachelet/pyLIMA.svg?branch=master)](https://travis-ci.org/ebachelet/pyLIMA)
[![Coverage Status](https://coveralls.io/repos/github/ebachelet/pyLIMA/badge.svg?branch=master)](https://coveralls.io/github/ebachelet/pyLIMA?branch=master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.997468.svg)](https://doi.org/10.5281/zenodo.997468)



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

You need [pip](https://pip.pypa.io/en/stable/installing/) or you can install manually
the required libraries [Documentation](https://ebachelet.github.io/pyLIMA/)

pyLIMA should now run both on python2 and python3 !
### Installation and use




Clone the repository or download as a ZIP file. Then

python setup.py install (--user if needed)

pip install -r requirements.txt

This new procedure which should avoid the previous installations headaches!
Successfully test on various UNIX, MAC and Windows! If you encounter any problems,
please contact etibachelet@gmail.com.

You should be able to load pyLIMA as general module :
```python
from pyLIMA import microlmagnification
```
### Examples
Examples can be found in your pyLIMA directory. Look on the documentation to learn how to run it.
There is two version for each examples, one using [Jupyter notebook](https://jupyter.org/) (*.ipynb) or 
classic Python file (*.py).

Example_1 : [HOW TO FIT MY DATA?](https://github.com/ebachelet/pyLIMA/tree/master/examples)

Example_2 : [HOW TO USE YOUR PREFERED PARAMETERS?](https://github.com/ebachelet/pyLIMA/tree/master/examples)

Example_3 : [HOW TO SIMULATE EVENST?](https://github.com/ebachelet/pyLIMA/tree/master/examples)

Example_4 : [HOW TO USE YOUR OWN FITTING ROUTINES?](https://github.com/ebachelet/pyLIMA/tree/master/examples)

Example_5 : [HOW TO FIT PARALLAX?](https://github.com/ebachelet/pyLIMA/tree/master/examples)
# What can you do?


#### pyLIMA is now in beta!! Here is the status of implemented microlensing models:

| Model | Implemented | Examples | Fit Method Advice | 
| :---         |     :---:      |:---: |    ---: |
| Point-Source Point Lens (PSPL)   | ![Alt text](./doc/HGF.png?raw=true)     | Yes | Levenberg-Marquardt (LM)     |
| Finite-Source Point Lens (FSPL)   |  ![Alt text](./doc/HGF.png?raw=true)      | Yes | Levenberg-Marquardt (LM) or Differential Evolution (DE)    |
| Double-Source Point Lens (DSPL)   | ![Alt text](./doc/HGF.png?raw=true)     |  Yes | Differential Evolution (DE)    |
| Uniform-Source Binary Lens (USBL)   | ![Alt text](./doc/HGF.png?raw=true)  | No |      |

#### pyLIMA can also treat Second Order effects :

| Second-Order Effects | Implemented | Examples |Fit Method Advice |
| :---         |     :---:      |   :---: |   ---: |
| Annual parallax   |  ![Alt text](./doc/HGF.png?raw=true)      | No | Levenberg-Marquardt (LM)     |
| Terrestrial parallax   |  ![Alt text](./doc/HGF.png?raw=true)     | No | Levenberg-Marquardt (LM) |
| Space parallax   |  ![Alt text](./doc/HGF.png?raw=true)      |  No| Levenberg-Marquardt (LM)    |
| Orbital Motion   | ![Alt text](./doc/HGF.png?raw=true)     | No |       |
| Xallarap   | ![Alt text](./doc/WIP.png?raw=true)    | No |       |


# How to contribute?

Want to contribute? Bug detections? Comments?
Please email us : etibachelet@gmail.com, rstreet@lcogt.net, valboz@sa.infn.it
