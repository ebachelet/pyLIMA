Conventions
===========
pyLIMA use the following conventions.

The time given in observations is expected to be in HJD.

Following the idea of uniform notations in the field, pyLIMA is based on `Gould2000 <http://adsabs.harvard.edu/abs/2000ApJ...542..785G/>`_. 

The informations in [] represents the units :math:`t_{o}` [days]:

-  :math:`t_o` [days]  is define as the time of the minimum impact parameter.
-  :math:`u_o` [ :math:`\theta_E`] is the minimum impact parameter. It is positive if the source pass on the left of the source. It is define on the center of mass of the lens system.
-  :math:`t_E` [days] is the angular Einstein ring crossing time.
-  :math:`\rho` [ :math:`\theta_E`] is the normalised angular source radius.
-  :math:`s` [ :math:`\theta_E`] is the normalised angular separation between the binary lens component.
-  :math:`q` [] is the binary mass ratio, with the smaller body on the right of the system.
-  :math:`\alpha` [rad] is the angle between the source trajectory and the the binary lens axis, counted in trigonometric convention.  
-  :math:`\pi_{EN}` [ :math:`AU/r_E`] is the North component of the microlensing parallax.
-  :math:`\pi_{EE}` [ :math:`AU/r_E`] is the East component of the microlensing parallax.

Then, the source trajectory x,y is define as :

-  :math:`\tau = (t-t_o)/t_E`

-  :math:`x = \tau . cos(\alpha)- u_o . sin(\alpha)`
-  :math:`y = \tau . sin(\alpha)+ u_o . cos(\alpha)`

