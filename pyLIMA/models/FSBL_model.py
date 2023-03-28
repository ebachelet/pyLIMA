from pyLIMA.models.USBL_model import USBLmodel
from pyLIMA.magnification import magnification_VBB

class FSBLmodel(USBLmodel):
    @property
    def model_type(self):
        """ Return the kind of microlensing model.

        :returns: FSBL
        :rtype: string
        """
        return 'FSBL'

    def model_magnification(self, telescope, pyLIMA_parameters, return_impact_parameter=None):
        """ The magnification associated to a USBL model.# Derivatives of the normalised residuals objective function for PSPL version

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

            From Bozza  2010 : http://adsabs.harvard.edu/abs/2010MNRAS.408.2188B

            :param object telescope: a telescope object. More details in telescope module.
            :param object pyLIMA_parameters: a namedtuple which contain the parameters
            :return: magnification,
            :rtype: array_like,
        """

        if telescope.lightcurve_flux is not None:

            self.u0_t0_from_uc_tc(pyLIMA_parameters)

            source_trajectoire = self.source_trajectory(telescope, pyLIMA_parameters, data_type='photometry')

            linear_limb_darkening = telescope.gamma * 3 / (2 + telescope.gamma)

            separation = source_trajectoire[2] +  pyLIMA_parameters.separation

            magnification_FSBL = \
                magnification_VBB.magnification_FSBL(separation, pyLIMA_parameters.mass_ratio,
                                                   source_trajectoire[0], source_trajectoire[1],
                                                   pyLIMA_parameters.rho, linear_limb_darkening)


        else:

            magnification_FSBL = None

        if return_impact_parameter:

            return magnification_FSBL, None
        else:
            return magnification_FSBL

