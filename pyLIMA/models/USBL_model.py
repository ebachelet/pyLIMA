import numpy as np
from pyLIMA.caustics import binary_caustics
from pyLIMA.magnification import magnification_VBB
from pyLIMA.models.ML_model import MLmodel


class USBLmodel(MLmodel):

    def __init__(self, event, parallax=['None', 0.0], double_source=['None',0.0],
                 orbital_motion=['None', 0.0], blend_flux_parameter='fblend',
                 origin=['center_of_mass', [0, 0]], fancy_parameters=None):
        """The fit class has to be intialized with an event object."""

        super().__init__(event, parallax=parallax, double_source=double_source,
                         orbital_motion=orbital_motion,
                         blend_flux_parameter=blend_flux_parameter, origin=origin,
                         fancy_parameters=fancy_parameters)

    def model_type(self):

        return 'USBL'

    def paczynski_model_parameters(self):
        """
        [t0,u0,tE,rho,s,q,alpha]
        """
        model_dictionary = {'t0': 0, 'u0': 1, 'tE': 2, 'rho': 3, 'separation': 4,
                            'mass_ratio': 5, 'alpha': 6}

        self.Jacobian_flag = 'Numerical'

        return model_dictionary

    def model_astrometry(self, telescope, pyLIMA_parameters):

        pass

    def model_magnification(self, telescope, pyLIMA_parameters,
                            return_impact_parameter=None):
        """
        The magnification associated to a USBL model.
        See https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.2188B/abstract
            https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5157B/abstract
        """
        if telescope.lightcurve is not None:

            # self.u0_t0_from_uc_tc(pyLIMA_parameters)


            (source1_trajectory_x, source1_trajectory_y,
             source2_trajectory_x, source2_trajectory_y,
             dseparation, dalpha) = self.sources_trajectory(
                telescope, pyLIMA_parameters,
                data_type='photometry')

            separation = dseparation + pyLIMA_parameters['separation']

            source1_magnification = magnification_VBB.magnification_USBL(separation,
                                                     pyLIMA_parameters['mass_ratio'],
                                                     source1_trajectory_x,
                                                     source1_trajectory_y,
                                                     pyLIMA_parameters['rho'])

            if source2_trajectory_x is not None:

                source2_magnification = magnification_VBB.magnification_USBL(separation,
                                                         pyLIMA_parameters[
                                                             'mass_ratio'],
                                                         source2_trajectory_x,
                                                         source2_trajectory_y,
                                                         pyLIMA_parameters['rho_2'])

                blend_magnification_factor = pyLIMA_parameters['q_flux_' +
                                                               telescope.filter]
                effective_magnification = (
                        source1_magnification +
                        source2_magnification *
                        blend_magnification_factor)

                magnification_USBL = effective_magnification

            else:

                magnification_USBL = source1_magnification

        else:

            magnification_USBL = None

        if return_impact_parameter:

            return magnification_USBL, None
        else:
            return magnification_USBL

    def new_origin(self, pyLIMA_parameters=None):
        """

        """

        if 'caustic' in self.origin[0]:

            caustic_regime = binary_caustics.find_2_lenses_caustic_regime(
                pyLIMA_parameters['separation'],
                pyLIMA_parameters['mass_ratio'])

            caustics = binary_caustics.caustic_points_at_phi_0(
                pyLIMA_parameters['separation'],
                pyLIMA_parameters['mass_ratio'])

            caustic = 0 + 0 * 1j

            if caustic_regime == 'resonant':
                caustic = caustics[caustics.real.argmin()]

            if (caustic_regime == 'wide') & (self.origin[0] == 'central_caustic'):
                caustic = caustics[caustics.real.argmin()]

            if (caustic_regime == 'wide') & ((self.origin[0] != 'central_caustic')):
                sorting = caustics.real.argsort()
                caustic = caustics[sorting[2]]

            if (caustic_regime == 'close') & (self.origin[0] == 'central_caustic'):
                sorting = caustics.imag.argsort()
                caustic = caustics[
                    np.where(caustics.real == caustics[sorting[1:3]].real.min())[0]]

            if (caustic_regime == 'close') & (self.origin[0] == 'second_caustic'):
                caustic = caustics[caustics.imag.argmax()]

            if (caustic_regime == 'close') & (self.origin[0] == 'third_caustic'):
                caustic = caustics[caustics.imag.argmin()]

            x_center = caustic.real
            y_center = caustic.imag
            return x_center, y_center

        if 'primary' in self.origin[0]:
            primary_location = -pyLIMA_parameters['separation'] * \
                               pyLIMA_parameters['mass_ratio'] / (
                                       1 + pyLIMA_parameters['mass_ratio'])

            x_center = primary_location
            y_center = 0
            return x_center, y_center

        if 'secondary' in self.origin[0]:
            secondary_location = pyLIMA_parameters['separation'] / (
                    1 + pyLIMA_parameters['mass_ratio'])

            x_center = secondary_location
            y_center = 0
            return x_center, y_center

        x_center = self.origin[1][0]
        y_center = self.origin[1][1]

        return x_center, y_center