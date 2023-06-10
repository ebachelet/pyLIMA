import unittest.mock as mock
import numpy as np

from pyLIMA.models import FSPLmodel
from pyLIMA.toolbox import time_series
def _create_event():

    event = mock.MagicMock()
    event.telescopes = [mock.MagicMock()]
    event.telescopes[0].name = 'Test'
    event.telescopes[0].lightcurve_flux = time_series.construct_time_series(np.array([[0,10,2],[20,200,3]]), ['time','flux','err_flux'],
                                                                                     ['JD','W/m^2','W/m^2'])
    event.telescopes[0].filter = 'I'
    event.telescopes[0].ld_gamma = 0.5
    event.telescopes[0].astrometry = None

    return event

def test_initialize_model():
    event = _create_event()

    Model = FSPLmodel(event)

    assert Model.event == event
    assert Model.parallax_model == ['None', 0.0]
    assert Model.xallarap_model == ['None']
    assert Model.orbital_motion_model == ['None', 0.0]
    assert Model.blend_flux_parameter == 'fblend'
    assert Model.photometry == True
    assert Model.astrometry == False
    assert dict(Model.model_dictionnary) == {'t0': 0, 'u0': 1, 'tE': 2, 'rho': 3, 'fsource_Test': 4, 'fblend_Test': 5}
    assert dict(Model.pyLIMA_standards_dictionnary) == {'t0': 0, 'u0': 1, 'tE': 2, 'rho': 3, 'fsource_Test': 4, 'fblend_Test': 5}
    assert dict(Model.fancy_to_pyLIMA_dictionnary) == {}
    assert dict(Model.pyLIMA_to_fancy_dictionnary) == {}
    assert dict(Model.pyLIMA_to_fancy) == {}
    assert dict(Model.fancy_to_pyLIMA) == {}
    assert Model.standard_parameters_boundaries ==  [(2400000, 2500000), (0.0, 1.0), (0.1, 500), (5e-05, 0.05), (0.0, 200.0), (-200.0, 200.0)]
    assert Model.origin == ['center_of_mass', [0, 0]]
    assert Model.model_type == 'FSPL'

def test_model_magnification():
    event = _create_event()

    Model = FSPLmodel(event)

    params = [0.5,0.002,35,0.05]

    pym = Model.compute_pyLIMA_parameters(params)

    magnification = Model.model_magnification(event.telescopes[0],pym)

    assert np.allclose(magnification,[42.16176577,  1.99919488])

def test_model_magnification_Jacobian():
    event = _create_event()

    Model = FSPLmodel(event)

    params = [0.5,0.002,35,0.05]

    pym = Model.compute_pyLIMA_parameters(params)

    jacob,magnification = Model.model_magnification_Jacobian(event.telescopes[0], pym)

    assert np.allclose(jacob, np.array([[-5.52086665e+00, -2.70522466e+01, 7.88695235e-02,-7.86812979e+02],
                                        [8.25429191e-02, -1.03707770e-02, 4.59881978e-02,7.27955884e-02]]))

    assert np.allclose(magnification,[42.16176577,  1.99919488])

def test_photometric_model_Jacobian():
    event = _create_event()

    Model = FSPLmodel(event)

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)

    jacob = Model.photometric_model_Jacobian(event.telescopes[0], pym)
    assert np.allclose(jacob, np.array([[-0.        ,  0.        ],
                                        [-0.        , -0.        ],
                                        [ 0.        ,  0.        ],
                                        [-0.        ,  0.        ],
                                        [42.16176577,  1.99919488],
                                        [ 1.        ,  1.        ]]))
def test_change_origin():
    event = _create_event()

    Model = FSPLmodel(event,origin=['nothere',[9,8]])

    pym = mock.MagicMock()
    Model.change_origin(pym)

    assert pym.x_center == 9
    assert pym.y_center == 8
def test_compute_the_microlensing_model():
    event = _create_event()

    Model = FSPLmodel(event, origin=['nothere', [9, 8]])

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)

    model = Model.compute_the_microlensing_model(event.telescopes[0],pym)

    assert np.allclose(model['photometry'],[  9.99999999, 200.        ])
    assert model['astrometry'] is None


def test_derive_telescope_flux():
    event = _create_event()

    Model = FSPLmodel(event, origin=['nothere', [9, 8]])

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)
    magnification = Model.model_magnification(event.telescopes[0],pym)

    Model.derive_telescope_flux(event.telescopes[0], pym,magnification)
    assert np.allclose(pym.fsource_Test, -15723296.651212221)
    assert np.allclose(pym.fblend_Test, 15724766.572776612)

def test_find_telescopes_fluxes():
    event = _create_event()

    Model = FSPLmodel(event, origin=['nothere', [9, 8]])

    params = [0.5, 0.002, 35, 0.05]

    fluxes = Model.find_telescopes_fluxes(params)
    assert np.allclose(fluxes.fsource_Test, -15723296.651212221)
    assert np.allclose(fluxes.fblend_Test, 15724766.572776612)

def test_compute_pyLIMA_parameters():
    event = _create_event()

    Model = FSPLmodel(event, origin=['nothere', [9, 8]])

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)
    values =  [getattr(pym,key) for key in pym._fields[:4]]
    assert np.allclose(params,values)


def test_source_trajcetory():
    event = _create_event()

    Model = FSPLmodel(event, origin=['nothere', [9, 8]])

    params = [0.5, 0.002, 35, 0.05]

    pym = Model.compute_pyLIMA_parameters(params)

    X,Y,dS,dA = Model.source_trajectory(event.telescopes[0],pym,data_type='photometry')

    assert np.allclose(X,[-8.98571429, -9.55714286])
    assert np.allclose(Y,[-8.002, -8.002])
    assert np.allclose(dS,[0,0])
    assert np.allclose(dA,[0,0])
def t_star(x):
        return x.rho * x.tE

def tE(x):
        return x.t_star / 10 ** (x.log_rho)
def test_fancy_parameters():

    from pyLIMA.models import fancy_parameters


    setattr(fancy_parameters, 't_star', t_star)
    setattr(fancy_parameters, 'tE', tE)

    my_pars2 = {'log_rho': 'rho', 't_star': 'tE'}

    event = _create_event()

    fspl2 = FSPLmodel(event, fancy_parameters=my_pars2)

    assert dict(fspl2.model_dictionnary) == {'t0': 0, 'u0': 1, 't_star': 2, 'log_rho': 3, 'fsource_Test': 4, 'fblend_Test': 5}
    assert dict(fspl2.fancy_to_pyLIMA_dictionnary) == {'log_rho': 'rho', 't_star': 'tE'}
    assert dict(fspl2.pyLIMA_to_fancy_dictionnary) == {'rho': 'log_rho','tE': 't_star'}
    assert fspl2.pyLIMA_to_fancy['log_rho'] == fancy_parameters.log_rho
    assert fspl2.pyLIMA_to_fancy['t_star'] == t_star
    assert fspl2.fancy_to_pyLIMA['rho'] == fancy_parameters.rho
    assert fspl2.fancy_to_pyLIMA['tE'] == tE

### Need to test submodules now