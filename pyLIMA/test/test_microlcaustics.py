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
    mass_ratio = 10 ** -6

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
    mass_ratio = 10 ** -2

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
                                                                                resolution=1000)

    resonant_caustic, resonant_cc = microlcaustics.sort_2lenses_resonant_caustic(caustics, critical_curves)

    assert len(resonant_caustic) == 4000
    assert len(resonant_cc) == 4000


def test_sort_2lenses_wide_caustic():
    separation = 10.8
    mass_ratio = 0.8

    caustics, critical_curves = microlcaustics.compute_2_lenses_caustics_points(separation, mass_ratio,
                                                                                resolution=1000)

    central_caustic, wide_caustic, central_cc, wide_cc = microlcaustics.sort_2lenses_wide_caustics(caustics,
                                                                                                   critical_curves)

    assert len(central_caustic) == 2000
    assert len(central_cc) == 2000
    assert len(wide_caustic) == 2000
    assert len(wide_cc) == 2000


def test_sort_2lenses_close_caustics():
    separation = 0.8
    mass_ratio = 10 ** -7

    caustics, critical_curves = microlcaustics.compute_2_lenses_caustics_points(separation, mass_ratio,
                                                                                resolution=1000)

    central_caustic, close_top_caustic, close_bottom_caustic, central_cc, close_top_cc, close_bottom_cc = \
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

    expected_caustics = np.array([[0.34062502 - 3.94430453e-31j, -0.34062502 - 3.38963670e-32j,
                                   0.00000000 - 5.89979840e-01j, 0.00000000 + 5.89979840e-01j],
                                  [0.32489215 - 3.74174390e-03j, -0.32489215 + 3.74174390e-03j,
                                   -0.14198333 - 6.13822318e-01j, 0.14198333 + 6.13822318e-01j],
                                  [0.28067735 - 3.06011670e-02j, -0.28067735 + 3.06011670e-02j,
                                   -0.21082253 - 6.50713550e-01j, 0.21082253 + 6.50713550e-01j],
                                  [0.21790294 - 1.08073125e-01j, -0.21790294 + 1.08073125e-01j,
                                   -0.20257117 - 6.36141383e-01j, 0.20257117 + 6.36141383e-01j],
                                  [0.16039580 - 2.74010967e-01j, -0.16039580 + 2.74010967e-01j,
                                   -0.15982495 - 5.07258799e-01j, 0.15982495 + 5.07258799e-01j],
                                  [0.15982495 - 5.07258799e-01j, -0.15982495 + 5.07258799e-01j,
                                   -0.16039580 - 2.74010967e-01j, 0.16039580 + 2.74010967e-01j],
                                  [0.20257117 - 6.36141383e-01j, -0.20257117 + 6.36141383e-01j,
                                   -0.21790294 - 1.08073125e-01j, 0.21790294 + 1.08073125e-01j],
                                  [0.21082253 - 6.50713550e-01j, -0.21082253 + 6.50713550e-01j,
                                   -0.28067735 - 3.06011670e-02j, 0.28067735 + 3.06011670e-02j],
                                  [0.14198333 - 6.13822318e-01j, -0.14198333 + 6.13822318e-01j,
                                   -0.32489215 - 3.74174390e-03j, 0.32489215 + 3.74174390e-03j],
                                  [0.00000000 - 5.89979840e-01j, 0.00000000 + 5.89979840e-01j,
                                   -0.34062502 - 4.74549138e-31j, 0.34062502 - 1.77493704e-30j]])

    expected_cc = np.array([[1.27122988e+00 + 2.89236268e-16j,
                             -1.27122988e+00 + 2.77844475e-17j,
                             -1.48013047e-18 + 3.40625019e-01j,
                             3.30645600e-17 - 3.40625019e-01j],
                            [1.21229332e+00 + 2.87839951e-01j,
                             -1.21229332e+00 - 2.87839951e-01j,
                             -7.73913354e-02 + 3.56017165e-01j,
                             7.73913354e-02 - 3.56017165e-01j],
                            [1.04372694e+00 + 5.28204089e-01j,
                             -1.04372694e+00 - 5.28204089e-01j,
                             -1.47894990e-01 + 3.98235002e-01j,
                             1.47894990e-01 - 3.98235002e-01j],
                            [7.91336472e-01 + 6.75204341e-01j,
                             -7.91336472e-01 - 6.75204341e-01j,
                             -2.17214969e-01 + 4.66363434e-01j,
                             2.17214969e-01 - 4.66363434e-01j],
                            [5.09656858e-01 + 6.85437065e-01j,
                             -5.09656858e-01 - 6.85437065e-01j,
                             -3.11897539e-01 + 5.71750678e-01j,
                             3.11897539e-01 - 5.71750678e-01j],
                            [3.11897539e-01 + 5.71750678e-01j,
                             -3.11897539e-01 - 5.71750678e-01j,
                             -5.09656858e-01 + 6.85437065e-01j,
                             5.09656858e-01 - 6.85437065e-01j],
                            [2.17214969e-01 + 4.66363434e-01j,
                             -2.17214969e-01 - 4.66363434e-01j,
                             -7.91336472e-01 + 6.75204341e-01j,
                             7.91336472e-01 - 6.75204341e-01j],
                            [1.47894990e-01 + 3.98235002e-01j,
                             -1.47894990e-01 - 3.98235002e-01j,
                             -1.04372694e+00 + 5.28204089e-01j,
                             1.04372694e+00 - 5.28204089e-01j],
                            [7.73913354e-02 + 3.56017165e-01j,
                             -7.73913354e-02 - 3.56017165e-01j,
                             -1.21229332e+00 + 2.87839951e-01j,
                             1.21229332e+00 - 2.87839951e-01j],
                            [5.42176191e-17 + 3.40625019e-01j,
                             5.38347107e-18 - 3.40625019e-01j,
                             -1.27122988e+00 + 2.05998413e-16j,
                             1.27122988e+00 - 5.72458747e-16j]])

    assert np.allclose(caustics, expected_caustics)
    assert np.allclose(critical_curves, expected_cc)


def test_find_area_of_interest_around_caustics():
    separation = 0.98
    mass_ratio = 10 ** -3.98

    caustics, critical_curves = microlcaustics.compute_2_lenses_caustics_points(separation, mass_ratio, resolution=10)

    area_of_interest = microlcaustics.find_area_of_interest_around_caustics(caustics, secure_factor=0.1)

    expected_area = np.array([[-0.16472971920662491, 0.1487364691472782], [-0.11476825367160942, 0.11476825367161236],
                              [-0.0079966250296733543, 1.4710455076283324e-15]]
                             )

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


def test_change_source_trajectory_center_to_planetary_caustics_center():
    separation = 0.5
    mass_ratio = 10 ** -3.98

    x_center, y_center = microlcaustics.change_source_trajectory_center_to_planetary_caustics_center(separation,
                                                                                                     mass_ratio)

    assert x_center ==-1.4997160687085691
    assert y_center == 0.035497175763441399

    separation = 0.8
    mass_ratio = 0.85

    x_center, y_center = microlcaustics.change_source_trajectory_center_to_planetary_caustics_center(separation,
                                                                                                     mass_ratio)

    assert x_center == 0
    assert y_center == 0

    separation = 6.8
    mass_ratio = 0.85

    x_center, y_center = microlcaustics.change_source_trajectory_center_to_planetary_caustics_center(separation,
                                                                                                     mass_ratio)

    assert x_center == 3.597042535016731
    assert y_center == 0

def test_change_source_trajectory_center_to_central_caustics_center():
    separation = 0.5
    mass_ratio = 10 ** -1

    x_center, y_center = microlcaustics.change_source_trajectory_center_to_central_caustics_center(separation,
                                                                                                     mass_ratio)

    assert x_center == -0.010300719769916603
    assert y_center == 0

    separation = 0.8
    mass_ratio = 0.85

    x_center, y_center = microlcaustics.change_source_trajectory_center_to_central_caustics_center(separation,
                                                                                                     mass_ratio)

    assert x_center == 0
    assert y_center == 0

    separation = 6.8
    mass_ratio = 0.85

    x_center, y_center = microlcaustics.change_source_trajectory_center_to_central_caustics_center(separation,
                                                                                                     mass_ratio)

    assert x_center == -3.0576148720596557
    assert y_center == 0