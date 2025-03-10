[![PyPI - Version](https://img.shields.io/pypi/v/pyLIMA.svg)](https://pypi.org/project/pyLIMA)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.997468.svg)](https://doi.org/10.5281/zenodo.997468)
![Build Status](https://github.com/ebachelet/pyLIMA/actions/workflows/actions_unit_tests.yaml/badge.svg)

> [!WARNING]
> **Runing pyLIMA in multiprocessing...**
> 
> The latest version of pyLIMA applies the multiprocessing library to parallelize aspects of its model fitting processes in order to optimize for speed. This has been tested and works under Ubuntu Linux with Python 3.11.
> Users should be aware that the multiprocessing library uses a different method ('spawn') to start threads on the Mac and Windows platforms compared to the method used on Linux ('fork'), as the fork method is considered to be unsafe on these platforms. Unfortunately, this has meant that pyLIMA crashes if run under the latest version of Python (3.11) under a Mac (and likely under Windows), due to the outstanding issue with the multiprocessing library.
We are currently investigating a fix for this issue. In the interim we recommend using an earlier version of Python with the latest pyLIMA.

# pyLIMA

Authors : Etienne Bachelet (etibachelet@gmail.com), Rachel Street (rstreet@lcogt.net),
Valerio Bozza (valboz@sa.infn.it), Yiannis Tsapras (ytsapras@ari.uni-heidelberg.de) 
and friends!

pyLIMA is the first open source software for modeling microlensing events.
It should be flexible enough to handle your data and fit it.
You can also practice by simulating events.

# Documentation and Installation

[Documentation](https://pylima.readthedocs.io/en/latest/)

### Required materials

You need [pip](https://pip.pypa.io/en/stable/installing/) and python, that's it!

### Installation and use


```
>>> pip install pyLIMA
```

You should be able to load pyLIMA as general module :

```python
import pyLIMA
print(pyLIMA.__version__)
```

### Examples

Examples can be found in the pyLIMA directory after cloning this repository. More details can be found in the [Documentation](https://pylima.readthedocs.io/en/latest/)
There is two version for each examples, one
using [Jupyter notebook](https://jupyter.org/) or
classic Python file.

Example_1 : [HOW TO FIT MY DATA?](https://github.com/ebachelet/pyLIMA/tree/master/examples)

Example_2 : [HOW TO USE YOUR PREFERED PARAMETERS?](https://github.com/ebachelet/pyLIMA/tree/master/examples)

Example_3 : [HOW TO SIMULATE EVENST?](https://github.com/ebachelet/pyLIMA/tree/master/examples)

Example_4 : [HOW TO USE YOUR OWN FITTING ROUTINES?](https://github.com/ebachelet/pyLIMA/tree/master/examples)

Example_5 : [HOW TO FIT PARALLAX?](https://github.com/ebachelet/pyLIMA/tree/master/examples)


# How to contribute?

Want to contribute? Bug detections? Comments?
Please email us (etibachelet@gmail.com) or [raise an issue](https://github.com/ebachelet/pyLIMA/issues) (recommended).

# Citations

Please cite [Bachelet et al. 2017](https://ui.adsabs.harvard.edu/abs/2017AJ....154..203B/abstract) (and soon pyLIMA-II Bachelet et al.2025). The BibTeX entry for the paper is::


@ARTICLE{2017AJ....154..203B,
       author = {{Bachelet}, E. and {Norbury}, M. and {Bozza}, V. and {Street}, R.},
        title = "{pyLIMA: An Open-source Package for Microlensing Modeling. I. Presentation of the Software and Analysis of Single-lens Models}",
      journal = {\aj},
     keywords = {gravitational lensing: micro},
         year = 2017,
        month = nov,
       volume = {154},
       number = {5},
          eid = {203},
        pages = {203},
          doi = {10.3847/1538-3881/aa911c},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2017AJ....154..203B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


