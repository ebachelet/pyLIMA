Installation
============

The first installation step is to install pyLIMA in the repository of your choice (could be inside a `<VirtualEnv.html>`_):


**> git clone https://github.com/ebachelet/pyLIMA.git**


Then we need to build the VBBinaryLensingLibrary:

**> python setup.py build_ext --build-lib=./pyLIMA/subroutines/VBBinaryLensingLibrary**

Optional :

**> python setup.py clean --all**

Then you can install the required libraries:


**> pip install -r requirements.txt**



