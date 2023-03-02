import importlib



def create_model(model_type,event, parallax=['None', 0.0], xallarap=['None'],
                 orbital_motion=['None', 0.0], blend_flux_parameter='fblend'):
    """
    Load a model according to the supplied model_type. Models are expected to be named
    Model<model_type> e.g. ModelPSPL

    :param string model_type: Model type e.g. PSPL
    :return: Model object for given model_type
    """

    try:

        model_module = importlib.import_module('pyLIMA.models.'+model_type+'_model')

    except :


        return None

    new_model = getattr(model_module, '{}model'.format(model_type))

    return new_model(event, parallax,xallarap,orbital_motion,blend_flux_parameter)