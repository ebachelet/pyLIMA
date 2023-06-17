import numpy as np
from pyLIMA.caustics import binary_caustics


def test_find_2_lenses_caustics_and_critical_curves():
    separation = 0.25
    mass_ratio = 0.12

    regime, caustics, crit_curve = \
        binary_caustics.find_2_lenses_caustics_and_critical_curves(
            separation, mass_ratio,
            resolution=5)
    assert regime == 'close'

    assert np.allclose(caustics[0], np.array(
        [0.01628524 + 1.30963236e-32j, 0.00247 - 7.14705680e-03j,
         -0.00115579 - 1.12047766e-02j, -0.0049317 - 2.36808845e-03j,
         -0.00908787 + 1.58696627e-31j, -0.00908787 + 4.54519467e-32j,
         -0.0049317 + 2.36808845e-03j, -0.00115579 + 1.12047766e-02j,
         0.00247 + 7.14705680e-03j, 0.01628524 - 3.08148791e-33j]))

    assert np.allclose(caustics[1],
                       np.array([-2.95002726 + 2.39678545j, -2.94583595 + 2.39903663j,
                                 -2.94527278 + 2.39804629j, -2.9445595 + 2.39425766j,
                                 -2.95002726 + 2.39678545j]))

    assert np.allclose(caustics[2],
                       np.array([-2.95002726 - 2.39678545j, -2.9445595 - 2.39425766j,
                                 -2.94527278 - 2.39804629j, -2.94583595 - 2.39903663j,
                                 -2.95002726 - 2.39678545j]))

    assert caustics[3] is None

    assert caustics[4] is None

    assert np.allclose(crit_curve[0], np.array(
        [1.01181895 - 3.05338437e-17j, 0.71307147 + 6.98075159e-01j,
         -0.00229159 + 9.91501964e-01j, -0.71317538 + 7.02784781e-01j,
         -1.00703013 + 5.89805982e-17j, -1.00703013 + 1.56125113e-17j,
         -0.71317538 - 7.02784781e-01j, -0.00229159 - 9.91501964e-01j,
         0.71307147 - 6.98075159e-01j, 1.01181895 - 8.08950346e-17j]))

    assert np.allclose(crit_curve[1],
                       np.array([0.19403416 - 0.07778688j, 0.19590983 - 0.07507009j,
                                 0.19872016 - 0.07665976j, 0.19705122 - 0.07977972j,
                                 0.19403416 - 0.07778688j]))

    assert np.allclose(crit_curve[2],
                       np.array([0.19403416 + 0.07778688j, 0.19705122 + 0.07977972j,
                                 0.19872016 + 0.07665976j, 0.19590983 + 0.07507009j,
                                 0.19403416 + 0.07778688j]))

    assert crit_curve[3] is None

    assert crit_curve[4] is None


def test_sort_2lenses_resonant_caustic():
    caustic_points = np.array([[0 + 1j, 2.5 - 3.8j, 0.4 + 0.4j]])
    critical_curves_points = np.array([[0.5j, -89 + 78j, 4.5]])

    resonant_caustic, resonant_cc = binary_caustics.sort_2lenses_resonant_caustic(
        caustic_points, critical_curves_points)

    assert np.allclose(resonant_caustic,
                       np.array([0. + 1.j, 0.4 + 0.4j, 0.4 - 0.4j, 0. - 1.j]))
    assert np.allclose(resonant_cc,
                       np.array([0. + 0.5j, 4.5 + 0.j, 4.5 - 0.j, 0. - 0.5j]))


def test_sort_2lenses_close_caustics():
    caustic_points = np.array([[0 + 1j, 2.5 - 3.8j, 0.4 + 0.4j]])
    critical_curves_points = np.array([[0.5j, -89 + 78j, 4.5]])

    central_caustic, close_top_caustic, close_bottom_caustic, central_cc, \
        close_top_cc, close_bottom_cc = \
        binary_caustics.sort_2lenses_close_caustics(caustic_points,
                                                    critical_curves_points)

    assert np.allclose(central_caustic, np.array([0.4 + 0.4j, 0. + 1.j]))
    assert np.allclose(close_top_caustic, np.array([0. + 1.j]))
    assert np.allclose(close_bottom_caustic, np.array([2.5 - 3.8j]))

    assert np.allclose(central_cc, np.array([4.5 + 0.j, 0. + 0.5j]))
    assert np.allclose(close_top_cc, np.array([0. + 0.5j]))
    assert np.allclose(close_bottom_cc, np.array([-89. + 78.j]))


def test_sort_2lenses_wide_caustics():
    caustic_points = np.array([[0 + 1j, 2.5 - 3.8j, 0.4 + 0.4j]])
    critical_curves_points = np.array([[0.5j, -89 + 78j, 4.5]])

    central_caustic, wide_caustic, central_cc, wide_cc = \
        binary_caustics.sort_2lenses_wide_caustics(
            caustic_points,
            critical_curves_points)

    assert np.allclose(central_caustic, np.array([0.4 + 0.4j, 0. + 1.j]))
    assert np.allclose(wide_caustic, np.array([2.5 - 3.8j, 0. + 1.j]))

    assert np.allclose(central_cc, np.array([4.5 + 0.j, 0. + 0.5j]))
    assert np.allclose(wide_cc, np.array([-89. + 78.j, 0. + 0.5j]))


def test_compute_2_lenses_caustics_points():
    separation = 1.24
    mass_ratio = 0.000259

    caustics, critical_curves = binary_caustics.compute_2_lenses_caustics_points(
        separation, mass_ratio, resolution=2)

    assert np.allclose(caustics, np.array(
        [[-2.57080794e-04 + 1.80313310e-31j, 4.66488170e-01 + 3.88809819e-28j,
          3.95061432e-01 - 4.46791095e-28j, 5.35521274e-03 + 9.52564950e-31j],
         [-2.57080794e-04 - 3.23796220e-31j, 4.66488170e-01 + 6.17653437e-29j,
          3.95061432e-01 - 9.69559356e-29j, 5.35521274e-03 - 8.50538812e-32j]]))

    assert np.allclose(critical_curves, np.array(
        [[-1.0002174 - 3.89464757e-17j, 1.26590509 + 1.76169205e-15j,
          1.21118214 - 2.14225729e-15j, 1.00184587 + 3.73444630e-16j],
         [-1.0002174 + 7.30917833e-17j, 1.26590509 + 2.75668099e-16j,
          1.21118214 - 5.30970252e-16j, 1.00184587 - 3.95090328e-17j]]))


def test_find_2_lenses_caustic_regime():
    separation = 0.22
    mass_ratio = 0.22
    regime = binary_caustics.find_2_lenses_caustic_regime(separation, mass_ratio)
    assert regime == 'close'

    separation = 2.22
    mass_ratio = 0.22
    regime = binary_caustics.find_2_lenses_caustic_regime(separation, mass_ratio)
    assert regime == 'wide'

    separation = 1.22
    mass_ratio = 0.22
    regime = binary_caustics.find_2_lenses_caustic_regime(separation, mass_ratio)
    assert regime == 'resonant'


def test_poly_binary_eiphi_0():
    separation = 1.22
    mass_ratio = 0.22
    polynomial = binary_caustics.poly_binary_eiphi_0(separation, mass_ratio)

    assert np.allclose(polynomial, [1.0, -1.56, -0.8316000000000001, 1.9032000000000002,
                                    -0.7799999999999999])


def test_caustic_points_at_phi_0():
    separation = 1.22
    mass_ratio = 0.22
    caustic_points = binary_caustics.caustic_points_at_phi_0(separation, mass_ratio)

    assert np.allclose(caustic_points, np.array([-0.17212486 + 0.j, 0.66178081 + 0.j,
                                                 0.01111935 - 0.23897312j,
                                                 0.01111935 + 0.23897312j]))


def test_lens_equation():
    z = np.array([0 + 1j, -58 + 2j])
    lenses_mass = [0.72, 0.28]
    lenses_pos = [-2.2, 0.78]
    zetas = binary_caustics.lens_equation(z, lenses_mass, lenses_pos)

    assert np.allclose(zetas, np.array(
        [-0.13544576 + 0.70262628j, -57.98235531 + 1.99937622j]))
