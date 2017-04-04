//////////////////
VBBinaryLensing v1.2 (released 2017-03-28)
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

Main change with respect to the 1.0 version is in the parallax calculation,
which now follows http://ssd.jpl.nasa.gov/txt/aprx_pos_planets.pdf.
- Possibility to switch between North-East system and Earth-acceleration parallel/perpendicular system.
- From version 1.2 all limitations to parameters are removed.