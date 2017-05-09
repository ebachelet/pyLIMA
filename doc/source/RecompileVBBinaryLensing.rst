Recompile the VBBinaryLensing library
=====================================


First, please install `swig <https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/swig-3.0.12.tar.gz/download?use_mirror=svwh>`_

Then, go into pyLIMA/subroutines/VBBinaryLensingLibrary and :

**> swig -c++ -python VBBinaryLensingLibrary.i**

Edit the file VBBinaryLensingLibrary.cpp and remove the line:

#include “stdafx.h”


Now we can compile all corresponding codes:

**> g++ --fast-math -fPIC -O3 -lm -c VBBinaryLensingLibrary.cpp**

Now we can compile the wrapped version:

**> g++ --fast-math -O3 -fPIC -lm -c VBBinaryLensingLibrary_wrap.cxx**

Finally we create a shared object and the python module:

**> g++ --fast-math -O3 -fPIC -shared VBBinaryLensingLibrary.o VBBinaryLensingLibrary_wrap.o -o _VBBinaryLensingLibrary.so**

This should do the trick. To test this, on the same directory :

**> python **

**> VBBLib=VBBinaryLensingLibrary.VBBinaryLensing()**

**> VBBLibBinaryMag(float(1),float(0.02),float(0.5),float(0.5),float(0.0033),float(0.001))**

**> 1.631172486811772**

Thanks M.Hundertmark!
