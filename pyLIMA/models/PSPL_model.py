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
        """ Define the PSPL standard parameters, [to,uo,tE]

        :returns: a dictionnary containing the pyLIMA standards
        :rtype: dict
        """
        model_dictionary = {'to': 0, 'uo': 1, 'tE': 2}

        return model_dictionary

    def model_magnification(self, telescope, pyLIMA_parameters, return_impact_parameter=False):
        """ The magnification associated to a PSPL model. More details in microlmagnification module.

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :param boolean return_impact_parameter: if the impact parameter is needed or not

        :return: magnification
        :rtype: array_like
        """

        source_trajectory_x, source_trajectory_y, _ = self.source_trajectory(telescope, pyLIMA_parameters)

        if telescope.astrometry is not None:

            import pyLIMA.magnification.magnification_VBB
            pyLIMA.magnification.magnification_VBB.VBB.astrometry = True

            magnification = magnification_PSPL.magnification_PSPL(source_trajectory_x, source_trajectory_y,
                                                                  return_impact_parameter)

            shifts = astrometric_shifts.PSPL_shifts(source_trajectory_x, source_trajectory_y, pyLIMA_parameters.theta_E)

        else:

            shifts = None
            magnification = magnification_PSPL.magnification_PSPL(source_trajectory_x, source_trajectory_y,
                                                                              return_impact_parameter)

        magnification_model = {'magnification':magnification,'astrometry':shifts}


        return magnification_model

    def light_curve_Jacobian(self, telescope, pyLIMA_parameters):
        """ The derivative of a PSPL model lightcurve

        :param object telescope: a telescope object. More details in telescope module.
        :param object pyLIMA_parameters: a namedtuple which contain the parameters
        :return: jacobi
        :rtype: array_like
        """

        # Derivatives of the normalised residuals objective function for PSPL version

        lightcurve = telescope.lightcurve_flux

        time = lightcurve['time'].value
        errflux = lightcurve['err_flux'].value

        # Derivative of A = (u^2+2)/(u(u^2+4)^0.5). Amplification[0] is A(t).
        # Amplification[1] is U(t).
        Amplification = self.model_magnification(telescope, pyLIMA_parameters, return_impact_parameter=True)
        dAmplificationdU = (-8) / (Amplification[1] ** 2 * (Amplification[1] ** 2 + 4) ** 1.5)

        # Derivative of U = (uo^2+(t-to)^2/tE^2)^0.5
        dUdto = -(time - pyLIMA_parameters.to) / (pyLIMA_parameters.tE ** 2 * Amplification[1])
        dUduo = pyLIMA_parameters.uo / Amplification[1]
        dUdtE = -(time - pyLIMA_parameters.to) ** 2 / (pyLIMA_parameters.tE ** 3 * Amplification[1])

        # Derivative of the model

        dresdto = getattr(pyLIMA_parameters, 'fs_' + telescope.name) * dAmplificationdU * dUdto / errflux
        dresduo = getattr(pyLIMA_parameters, 'fs_' + telescope.name) * dAmplificationdU * dUduo / errflux
        dresdtE = getattr(pyLIMA_parameters, 'fs_' + telescope.name) * dAmplificationdU * dUdtE / errflux

        if self.blend_flux_parameter == 'fb':

            dresdfs = (Amplification[0]) / errflux
            dresdg = 1 / errflux

        if self.blend_flux_parameter == 'g':

            dresdfs = (Amplification[0] + getattr(pyLIMA_parameters, 'g_' + telescope.name)) / errflux
            dresdg = getattr(pyLIMA_parameters, 'fs_' + telescope.name) / errflux

        jacobi = np.array([dresdto, dresduo, dresdtE, dresdfs, dresdg])

        return jacobi
