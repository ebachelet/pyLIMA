.. _conventions:

Conventions
===========

The time given in observations is expected to be in JD. This is not that crucial as long as parallax, and more generally ephemerides, is not considered.

Following the idea of uniform notations in the field, pyLIMA is based on `Gould 2000 <http://adsabs.harvard.edu/abs/2000ApJ...542..785G/>`_. 

The informations in [] represents the units:

-  :math:`t_0` [days]  is define as the time of the minimum impact parameter.
-  :math:`u_0` [ :math:`\theta_E`] is the minimum impact parameter. It is positive if the source pass on the left of the source. It is define on the center of mass of the lens system.
-  :math:`t_E` [days] is the angular Einstein ring crossing time.
-  :math:`\rho` [ :math:`\theta_E`] is the normalised angular source radius.
-  :math:`s` [ :math:`\theta_E`] is the normalised angular separation between the binary lens component.
-  :math:`q` [] is the binary mass ratio, with the smaller body on the right of the system.
-  :math:`\alpha` [rad] is the angle between the lens trajectory and the the binary lens axis (anti-clockwise).  
-  :math:`\pi_{EN}` [ :math:`AU/r_E`] is the North component of the microlensing parallax.
-  :math:`\pi_{EE}` [ :math:`AU/r_E`] is the East component of the microlensing parallax.
-  :math:`\gamma_\parallel` [rad/yr] is the parallel component of the orbital motion (i.e. :math:`\gamma_\parallel=1/sds/dt`) .
-  :math:`\gamma_\perp` [rad/yr] is the perpendicular component of the orbital motion (i.e. :math:`\gamma_\parallel=d\alpha/dt`) .
-  :math:`\gamma_r` [rad/yr] is the radial component of the orbital motion (i.e. :math:`\gamma_\parallel=1/sds_z/dt`).

Then, the lens trajectory :math:`(x_l,y_l)` is define as :

-  :math:`\tau = (t-t_o)/t_E`

-  :math:`x_l = \tau . cos(\alpha)- u_0 . sin(\alpha)`
-  :math:`y_l = \tau . sin(\alpha)+ u_0 . cos(\alpha)`

and therefore the source trajectory :math:`(x_s,y_s)` is simply:


-  :math:`x_s = -x_l`
-  :math:`y_s = -y_l`


In case the parallax is used, the angle :math:`\beta` between the North projected vector and the lens trajectory at t0par is (`Gould 2004 <https://iopscience.iop.org/article/10.1086/382782>`_):

-  :math:`\beta = arctan(\pi_{EE}/\pi_{EN})`

and is accessible with:

.. code-block:: python

   from pyLIMA.parallax import parallax
   
   piEN = 0.8
   piEE = -0.5
   
   beta = parallax.EN_trajectory_angle(piEE,piEN)
   print(beta) #2.12939564...
   
For orbital motion, we used the formalism described in `Bozza 2021 <https://arxiv.org/pdf/2011.04780.pdf>`_ (but also have a look to the excellent `Skowron 2011 <https://iopscience.iop.org/article/10.1088/0004-637X/738/1/87/pdf>`_).

Astrometry
----------

For the time been, astrometry is implemented only for PSPL model without blending. Including astrometry add the following parameters to the models:

-  :math:`\theta_E` [mas]  is the angular Einstein ring radius.
-  :math:`\pi_{source}` [mas] is the parallax of the source
-  :math:`\mu_{source_N}` [x/year] the proper motion of the source in North, in deg/yr or pix/yr depending of the astrometric data unit.
-  :math:`\mu_{source_E}` [x/year] the proper motion of the source in East, in deg/yr or pix/yr depending of the astrometric data unit.
-  :math:`position\_source\_N\_tel` [x] the position of the source in North at t0par seen by the telescope tel, in degree or pixels depending of the astrometric data unit.
-  :math:`position\_source\_E\_tel` [x] the position of the source in East at t0par seen by the telescope tel, in degree or pixels depending of the astrometric data unit.


For more details on other second order effects and models, please have a look to :ref:`pyLIMAModules`.


