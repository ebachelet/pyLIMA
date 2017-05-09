Troubleshooting
===============

We are aware of some problems with pyLIMA installation on different OS, and try to fix these.

**### Troubleshooting due to OS ###**

** LINUX **

pyLIMA is mainly developed on a LINUX platform, so the installation should not present any problems.
Have been tested under Ubuntu and CENtos without problems.

** MAC **

pyLIMA have been tested on different MAC systems and it works. However, it could need more work.
If you facing problems under MAC, we invite you to have a look to `<InstallationForMACUser.html>`_

** WINDOWS **

pyLIMA is currently not supported in Windows, sorry. One way to counter this, is to install a Virtual Machine.

Download and install VirtualBox, `VirtualBox  <https://www.virtualbox.org/wiki/Downloads>`_

Download iso for Ubuntu 16.04.2 LTS, `Ubuntu  <https://www.ubuntu.com/download/desktop>`_ , filename is ubuntu16.04.2-desktop-amd64.iso

Create a new Ubuntu machine on Oracle VM and follow the set-up instructions. 

Here we choosed 3.5 GB of RAM, 3575 MB of RAM (on a machine with 8GB of RAM). Dynamically Allocated virtual disk drive with max size of 100GB (that max size is *FAR* more than need)

After creating the machine, run it for the first time and another set-up wizard should appear.
It will ask for a disk to install from: point it to the Ubuntu iso (ubuntu-16.04.2-desktop-amd64.iso).

Let the virtual machine boot up, and follow the Ubuntu installation instructions.

Then follow the standard install procedure.

**### Troubleshooting due to VBBinaryLensing ###**

pyLIMA is based on the Valerio Bozza package, `VBBinaryLensing <http://www.fisica.unisa.it/GravitationAstrophysics/VBBinaryLensing.htm>`_
This package is C++, and was transformed into a python library using `SWIG <http://www.swig.org/Doc1.3/Python.html>_
This procedure is compiler dependent, and so not flexible between platform.
If you face troubleshooting due to VBBinaryLensing, give a try to `<RecompileVBBinaryLensing.html>`_.

