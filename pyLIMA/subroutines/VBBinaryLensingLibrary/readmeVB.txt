//////////////////
VBBinaryLensing v1.0.4 (released 2016-09-18)
- This code has been developed by Valerio Bozza, University of Salerno.
- Any use of this code for scientific publications should be acknowledged by a citation to: V. Bozza, MNRAS 408 (2010) 2188
//////////////////

The package contains the following files:
- main.cpp // Contains working examples and specific instructions for all functions.
- VBBinaryLensingLibrary.h // Header for the library
- VBBinaryLensingLibrary.cpp // Source code for the library
- makefile.dat // A sample makefile (kindly provided by Wei Zhu)
- OB151212coords.txt // Sample file with event coordinates
- satellite1.txt // Sample table for satellite position (Spitzer)
- satellite2.txt // Sample table for satellite position (Kepler)
- howtopython.txt // Instructions for linking VBBinaryLensing in Python (kindly provided by Markus Hundertmark)
- VBBinaryLensingLibrary.i // Sample swig configuration file for Python

Before compiling, remove #include <stadfx.h> in all cpp files if you do not need it.