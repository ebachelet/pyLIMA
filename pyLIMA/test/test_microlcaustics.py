import numpy as np
import mock
import pytest

from pyLIMA import microlcaustics

def test_find_2_lenses_caustics_and_critical_curves():

    separation = 1
    mass_ratio = 1

    caustic_regime, caustics, critical_curves = microlcaustics.find_2_lenses_caustics_and_critical_curves(separation,
                                                                                                          mass_ratio,
                                                                                                          resolution=1000)


    assert caustic_regime == 'resonant'

    for caustic in caustics[:-1]:

        assert caustic is None

    for critical_curve in critical_curves[:-1]:

        assert critical_curve is None

    assert len(caustics[-1]) == 4000
    assert len(critical_curves[-1]) == 4000

    separation = 1.05
    mass_ratio = 10**-6

    caustic_regime, caustics, critical_curves = microlcaustics.find_2_lenses_caustics_and_critical_curves(separation,
                                                                                                          mass_ratio,
                                                                                                          resolution=1000)

    assert caustic_regime == 'wide'

    solution = ['something', None, None, 'something', None]

    for index, caustic in enumerate(caustics):

        if solution[index] is not None:

            assert caustic is not None

        else:

            assert caustic is None

    for index, critical_curve in enumerate(critical_curves):

        if solution[index] is not None:

            assert critical_curve is not None

        else:

            assert critical_curve is None

    assert len(caustics[0]) == 2000
    assert len(critical_curves[0]) == 2000
    assert len(caustics[3]) == 2000
    assert len(critical_curves[3]) == 2000


    separation = 0.65
    mass_ratio = 10**-2

    caustic_regime, caustics, critical_curves = microlcaustics.find_2_lenses_caustics_and_critical_curves(separation,
                                                                                                          mass_ratio,
                                                                                                          resolution=1000)


    assert caustic_regime == 'close'

    solution = ['something', 'something', 'something', None, None]

    for index, caustic in enumerate(caustics):

        if solution[index] is not None:

            assert caustic is not None

        else:

            assert caustic is None

    for index, critical_curve in enumerate(critical_curves):

        if solution[index] is not None:

            assert critical_curve is not None

        else:

            assert critical_curve is None

    assert len(caustics[0]) == 2000
    assert len(critical_curves[0]) == 2000
    assert len(caustics[1]) == 1000
    assert len(critical_curves[1]) == 1000
    assert len(caustics[2]) == 1000
    assert len(critical_curves[2]) == 1000

def test_sort_2lenses_resonant_caustic():

    separation = 1
    mass_ratio = 1

    caustics, critical_curves = microlcaustics.compute_2_lenses_caustics_points(separation, mass_ratio,
                                                                                                resolution = 1000)


    resonant_caustic, resonant_cc =  microlcaustics.sort_2lenses_resonant_caustic(caustics, critical_curves)

    assert len(resonant_caustic) == 4000
    assert len(resonant_cc) == 4000

def test_sort_2lenses_wide_caustic():

    separation = 50.8
    mass_ratio = 0.8

    caustics, critical_curves = microlcaustics.compute_2_lenses_caustics_points(separation, mass_ratio,
                                                                                                resolution = 1000)

    central_caustic, wide_caustic, central_cc, wide_cc = microlcaustics.sort_2lenses_wide_caustics(caustics,
                                                                                                       critical_curves)

    assert len(central_caustic) == 2000
    assert len(central_cc) == 2000
    assert len(wide_caustic) == 2000
    assert len(wide_cc) == 2000

def test_sort_2lenses_close_caustics():

    separation = 0.8
    mass_ratio =10**-7

    caustics, critical_curves = microlcaustics.compute_2_lenses_caustics_points(separation, mass_ratio,
                                                                                                resolution = 1000)

    central_caustic, close_top_caustic, close_bottom_caustic, central_cc, close_top_cc, close_bottom_cc =  \
                                        microlcaustics.sort_2lenses_close_caustics(caustics, critical_curves)

    assert len(central_caustic) == 2000
    assert len(central_cc) == 2000
    assert len(close_top_caustic) == 1000
    assert len(close_top_cc) == 1000
    assert len(close_bottom_caustic) == 1000
    assert len(close_bottom_cc) == 1000

def test_compute_2_lenses_caustics_points():

    separation = 1
    mass_ratio = 1

    caustics, critical_curves = microlcaustics.compute_2_lenses_caustics_points(separation, mass_ratio, resolution=10)

    expected_caustics = np.array([[-0.34062174 +1.09255667e-08j,  0.34062174 -1.09255667e-08j,
         0.00227078 +5.89985517e-01j, -0.00227078 -5.89985517e-01j],
       [-0.32449537 +3.88716328e-03j,  0.32449537 -3.88716328e-03j,
         0.14339958 +6.14341351e-01j, -0.14339958 -6.14341351e-01j],
       [-0.28005368 +3.11265583e-02j,  0.28005368 -3.11265583e-02j,
         0.21110882 +6.50954721e-01j, -0.21110882 -6.50954721e-01j],
       [-0.21727139 +1.09171229e-01j,  0.21727139 -1.09171229e-01j,
         0.20222928 +6.35546930e-01j, -0.20222928 -6.35546930e-01j],
       [-0.16008815 +2.75770061e-01j,  0.16008815 -2.75770061e-01j,
         0.15954411 +5.05653019e-01j, -0.15954411 -5.05653019e-01j],
       [-0.16005182 +5.08537134e-01j,  0.16005182 -5.08537134e-01j,
         0.16064475 +2.72608248e-01j, -0.16064475 -2.72608248e-01j],
       [-0.20274081 +6.36434649e-01j,  0.20274081 -6.36434649e-01j,
         0.21821879 +1.07527110e-01j, -0.21821879 -1.07527110e-01j],
       [-0.21073890 +6.50643458e-01j,  0.21073890 -6.50643458e-01j,
         0.28085509 +3.04521875e-02j, -0.28085509 -3.04521875e-02j],
       [-0.14180538 +6.13757605e-01j,  0.14180538 -6.13757605e-01j,
         0.32494142 +3.72382747e-03j, -0.32494142 -3.72382747e-03j],
       [ 0.00000000 +5.89979840e-01j,  0.00000000 -5.89979840e-01j,
         0.34062502 -1.77493704e-30j, -0.34062502 -4.74549138e-31j]])

    expected_cc = np.array([[ -1.27121765e+00 -4.23740939e-03j,
          1.27121765e+00 +4.23740939e-03j,
          1.13541042e-03 -3.40628297e-01j,
         -1.13541042e-03 +3.40628297e-01j],
       [ -1.21080091e+00 -2.91298067e-01j,
          1.21080091e+00 +2.91298067e-01j,
          7.83343701e-02 -3.56398622e-01j,
         -7.83343701e-02 +3.56398622e-01j],
       [ -1.04131043e+00 -5.30439765e-01j,
          1.04131043e+00 +5.30439765e-01j,
          1.48650889e-01 -3.98841790e-01j,
         -1.48650889e-01 +3.98841790e-01j],
       [ -7.88660316e-01 -6.76012815e-01j,
          7.88660316e-01 +6.76012815e-01j,
          2.17926036e-01 -4.67167930e-01j,
         -2.17926036e-01 +4.67167930e-01j],
       [ -5.07571514e-01 -6.84895332e-01j,
          5.07571514e-01 +6.84895332e-01j,
          3.12934450e-01 -5.72759477e-01j,
         -3.12934450e-01 +5.72759477e-01j],
       [ -3.11073011e-01 -5.70945048e-01j,
          3.11073011e-01 +5.70945048e-01j,
          5.11329251e-01 -6.85863710e-01j,
         -5.11329251e-01 +6.85863710e-01j],
       [ -2.16860118e-01 -4.65962467e-01j,
          2.16860118e-01 +4.65962467e-01j,
          7.92673343e-01 -6.74795373e-01j,
         -7.92673343e-01 +6.74795373e-01j],
       [ -1.47678966e-01 -3.98062197e-01j,
          1.47678966e-01 +3.98062197e-01j,
          1.04441542e+00 -5.27563192e-01j,
         -1.04441542e+00 +5.27563192e-01j],
       [ -7.72733634e-02 -3.55969783e-01j,
          7.72733634e-02 +3.55969783e-01j,
          1.21247860e+00 -2.87407141e-01j,
         -1.21247860e+00 +2.87407141e-01j],
       [  5.38347107e-18 -3.40625019e-01j,
          5.42176191e-17 +3.40625019e-01j,
          1.27122988e+00 -5.72458747e-16j,
         -1.27122988e+00 +2.05998413e-16j]])

    assert np.allclose(caustics, expected_caustics)
    assert np.allclose(critical_curves, expected_cc)


def test_find_area_of_interest_around_caustics():

    separation = 0.98
    mass_ratio = 10**-3.98

    caustics, critical_curves = microlcaustics.compute_2_lenses_caustics_points(separation, mass_ratio, resolution=10)

    area_of_interest = microlcaustics.find_area_of_interest_around_caustics(caustics, secure_factor=0.1)

    expected_area = np.array([[-1.6504915797261186, 0.14854723647294163], [-0.11476431847338857, 0.11477309924840579],
                              [-0.75097217162658847, 4.3903875086154232e-06]])


    assert np.allclose(expected_area, area_of_interest)


def test_find_2_lenses_caustic_regime():

    separation = 0.6
    mass_ratio = 10 ** -3.98

    regime = microlcaustics.find_2_lenses_caustic_regime(separation, mass_ratio)

    assert regime == 'close'

    separation = 1.4
    mass_ratio = 10 ** -0.9

    regime = microlcaustics.find_2_lenses_caustic_regime(separation, mass_ratio)

    assert regime == 'resonant'

    separation = 2.6
    mass_ratio = 10 ** -8

    regime = microlcaustics.find_2_lenses_caustic_regime(separation, mass_ratio)

    assert regime == 'wide'


def test_change_source_trajectory_center_to_caustics_center():

    separation = 0.6
    mass_ratio = 10 ** -3.98


    x_center, y_center = microlcaustics.change_source_trajectory_center_to_caustics_center(separation, mass_ratio)

    assert x_center == -1.0665123701905062
    assert y_center == 0.027521235303172867

    separation = 0.8
    mass_ratio = 0.85

    x_center, y_center = microlcaustics.change_source_trajectory_center_to_caustics_center(separation, mass_ratio)

    assert x_center == 0
    assert y_center == 0

    separation = 6.8
    mass_ratio = 0.85

    x_center, y_center = microlcaustics.change_source_trajectory_center_to_caustics_center(separation, mass_ratio)

    assert x_center == 3.5970400319501321
    assert y_center == 0

    