Installing pyLIMA under MacOS
=============================
*First, throw your MAC and buy a LINUX computer*

pyLIMA can run very well under MacOS, but there are some subtle differences
relative to the (default) Linux version, as there are some significant differences both
the format of compiled libraries and the framework under which python codes
runs, so a slightly different install process is needed.  

1) Python on MacOS
Python codes like pyLIMA can run under the MacOS python Framework, provided the
the correct versions of various packages (e.g. numpy, scipy, etc) that it requires 
are available.  

However, the default versions of these packages on MacOS can be older and may not 
have all the functions pyLIMA uses.  These can be updated by the user, but bearing
in mind that python is widely used in other applications, it is advisible NOT to
make changes in the system's python Framework.  

Therefore, our recommendation is to use a virtualenv, which creates a distinct
'bubble' python environment within which code can be run and packages installed
without tampering with the system python.  More background on virtualenv can be
found `here <http://sourabhbajaj.com/mac-setup/Python/virtualenv.html>`_

To build the required virtualenv for pyLIMA, follow these steps:

**> sudo pip install virtualenv**

**> cd $DIR**

**> virtualenv venv --distribute --system-site-packages**

**> source $DIR/venv/bin/activate**

where $DIR is a user-owned directory where you wish to install software. 
The 'activate' command allows you to enter the virtualenv 'bubble', and is indicated
by the commandline prompt changing to (venv)>  Packages installed while this is
active will be installed under $DIR/venv/ rather than the system python, and
similarly any python code run will run under the new python install.  

The commandline above initializes the virtualenv with a copy of all packages which
the system python has, since in most cases pyLIMA will use the same versions of
these packages.  Under MacOS Sierra v10.12.3 only the following should need 
installing or updating:

**(venv)> pip install emcee**

**(venv)> pip install astropy**

**(venv)> pip install pyslalib**

**(venv)> pip install matplotlib==2.0.0**

**(venv)> pip install scipy==0.18.1**

When you wish to exit the virtualenv, type

**(venv)> deactivate**

2) Ensure SWIG and PRCE are installed.  These packages are used to wrap the C++
code of VBBinaryLensing and enable it to be called from pyLIMA's python
code. These packages can be downloaded from here:

`pcre <https://sourceforge.net/projects/pcre/?source=typ_redirect>`_

`SWIG <https://sourceforge.net/projects/swig/?source=typ_redirect>`_ 

Both can be built with a "./configure; make; make install" sequence, but note
PRCE needs to be installed first and both should be installed using sudo as they
place files in the systems directories.  

3) Link to the appropriate version of VBBinaryLensing
pyLIMA calls on a pre-built library of the VBBinaryLensing functions, which can
be found at

**$DIR/pyLIMA/pyLIMA/subroutines/VBBinaryLensingLibrary/_VBBinaryLensingLibrary.so**
  
By default, this file refers to a library built in Linux format, but an
additional file, _VBBinaryLensingLibrary.so.osx is also included, compiled under
MacOS Sierra v10.12.3.  To apply the MacOS library:

**> cd $DIR/pyLIMA/pyLIMA/subroutines/VBBinaryLensingLibrary**

**> mv _VBBinaryLensingLibrary.so _VBBinaryLensingLibrary.so.lnx**

**> ln -s _VBBinaryLensingLibrary.so.osx _VBBinaryLensingLibrary.so**

4) **[OPTIONAL]** If you need to re-compile VBBinaryLensing, here is the procedure:

**> cd $DIR/pyLIMA/pyLIMA/subroutines/VBBinaryLensingLibrary**
 
Edit the file VBBinaryLensingLibrary.cpp and remove the line:

**#include "stdafx.h"**

**> swig -c++ -python VBBinaryLensingLibrary.i**

**> g++ -O3 -fPIC -lm -c -m64 VBBinaryLensingLibrary.cpp**

**> g++ -O3 -fPIC -lm -c -m64 VBBinaryLensingLibrary_wrap.cxx -I <path to python include directory>**

The python path depends on your local install, e.g.

**$DIR/venv/include/python2.7**

**> g++ -lpython -shared VBBinaryLensingLibrary.o VBBinaryLensingLibrary_wrap.o -o _VBBinaryLensingLibrary.so.osx**

5) Ensure python matplotlib links to the correct backend to render images
- In your /Users/<UID>/ home directory, create or edit the file
~/.matplotlib/matplotlibrc with the contents "backend: TkAgg"

And you should be ready run pyLIMA on your Mac!
