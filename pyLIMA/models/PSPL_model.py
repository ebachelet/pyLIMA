import numpy as np

from pyLIMA.models.ML_model import MLmodel
from pyLIMA.magnification import magnification_PSPL
from pyLIMA.astrometry import astrometric_shifts

class PSPLmodel(MLmodel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: PSPL
        :rtype: string
        """
        return 'PSPL'

    def paczynski_model_parameters(self):
        """ Define the PSPL standard parameters, [t0,u0,tE]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'tE': 2}

        return model_dictionary

    def model_astrometry(self, telescope, pyLIMA_parameters):

        """ The astrometric shifts associated to a PSPL model. More details in microlmagnification module.

               :param object telescope: a telescope object. More details in telescope module.
               :param object pyLIMA_parameters: a namedtuple which contain the parameters


               :return: astro_shifts
               :rtype: array_like
        """

        if telescope.astrometry is not None:

            source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters,
                                                                                 data_type='astrometry')

            shifts = astrometric_shifts.PSPL_shifts(source_trajectory_x, source_trajectory_y, pyLIMA_parameters.theta_E)

            angle = np.arctan2(pyLIMA_parameters.piEE, pyLIMA_parameters.piEN)

            #Deltay = shifts[0] * np.cos(angle) - np.sin(angle) * shifts[1]+source_trajectory_y*pyLIMA_parameters.theta_E
            #Deltay -= Deltay[0]
            Deltay = shifts[1]+source_trajectory_y*pyLIMA_parameters.theta_E
            Deltay -= Deltay[0]

            Deltax = shifts[0] + source_trajectory_x * pyLIMA_parameters.theta_E
            Deltax -= Deltax[0]
            #Deltax = shifts[0] * np.sin(angle) + np.cos(angle) * shifts[1]+source_trajectory_x*pyLIMA_parameters.theta_E
            #Deltax -= Deltax[0]
            #import pdb;
            #pdb.set_trace()

            astro_shifts = np.array([Deltax, Deltay])

        else:

            astro_shifts = None

        return astro_shifts

    def model_magnification(self, telescope, pyLIMA_parameters, return_impact_parameter=False):
        """ The magnification associated to a PSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :param boolean return_impact_parameter: if the impact parameter is needed or not

        :return: magnification
        :rtype: array_like
        """

        if telescope.lightcurve_flux is not None:

            source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters,
                                                                                 data_type='photometry')

            magnification = magnification_PSPL.magnification_PSPL(source_trajectory_x, source_trajectory_y,
                                                                              return_impact_parameter)
        else:

            magnification = None

        return magnification

    def magnification_Jacobian(self, telescope, pyLIMA_parameters):
        """ The derivative of a PSPL model lightcurve

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: jacobi
        :rtype: array_like
        """

        # Derivatives of the normalised residuals objective function for PSPL version

        lightcurve = telescope.lightcurve_flux

        time = lightcurve['time'].value

        # Derivative of A = (u^2+2)/(u(u^2+4)^0.5). Amplification[0] is A(t).
        # Amplification[1] is U(t).
        Amplification = self.model_magnification(telescope, pyLIMA_parameters, return_impact_parameter=True)
        dAmplificationdU = (-8) / (Amplification[1] ** 2 * (Amplification[1] ** 2 + 4) ** 1.5)

        # Derivative of U = (uo^2+(t-to)^2/tE^2)^0.5
        dUdt0 = -(time - pyLIMA_parameters.t0) / (pyLIMA_parameters.tE ** 2 * Amplification[1])
        dUdu0 = pyLIMA_parameters.u0 / Amplification[1]
        dUdtE = -(time - pyLIMA_parameters.t0) ** 2 / (pyLIMA_parameters.tE ** 3 * Amplification[1])

        dAdt0 = dAmplificationdU * dUdt0
        dAdu0 = dAmplificationdU * dUdu0
        dAdtE = dAmplificationdU * dUdtE

        fsource_Jacobian = Amplification[0]
        fblend_Jacobian = [1]*len(Amplification[0])

        jacobi = np.array([dAdt0, dAdu0, dAdtE, fsource_Jacobian, fblend_Jacobian])

        return jacobi

    def astrometry_Jacobian(self, telescope, pyLIMA_parameters):
        """ The derivative of a PSPL model lightcurve

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: jacobi
        :rtype: array_like
        """

        pass

