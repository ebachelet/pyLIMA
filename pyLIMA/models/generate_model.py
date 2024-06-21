import importlib


def create_model(model_type, event, parallax=['None', 0.0], double_source=['None',0],
                 orbital_motion=['None', 0.0], origin=['center_of_mass', [0, 0]],
                 blend_flux_parameter='fblend',
                 fancy_parameters={}):
    """
    Load a model according to the supplied model_type. Models are expected to be named
    Model<model_type> e.g. ModelPSPL

    :param string model_type: Model type e.g. PSPL
    :return: Model object for given model_type
    """

    try:

        model_module = importlib.import_module('pyLIMA.models.' + model_type + '_model')

    except ValueError:

        return None

    new_model = getattr(model_module, '{}model'.format(model_type))

    return new_model(event, parallax=parallax, double_source=double_source,
                     orbital_motion=orbital_motion,
                     blend_flux_parameter=blend_flux_parameter, origin=origin,
                     fancy_parameters=fancy_parameters)
