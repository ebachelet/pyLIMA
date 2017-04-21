"""
Created on Thu Apr 20 11:19:38 2017

@author: ebachelet
"""
from __future__ import division

import numpy as np
import time
import scipy.optimize as so
import matplotlib.pyplot as plt

def find_2_lenses_caustics_and_critical_curves(separation, mass_ratio, resolution = 1000):

    close_top_caustic = None
    close_bottom_caustic = None
    central_caustic = None
    wide_caustic = None
    resonant_caustic = None

    critical_curve = None

    caustics_points, critical_curve_points = compute_2_lenses_caustics_points(separation, mass_ratio,
                                                                              resolution=resolution)

    critical_curve = critical_curve_points

    caustic_regime = find_2_lenses_caustic_regime(separation, mass_ratio)

    if caustic_regime == 'resonant':

        resonant_caustic = sort_2lenses_resonant_caustic(caustics_points)

    if caustic_regime == 'close':

        central_caustic, close_top_caustic, close_bottom_caustic =  sort_2lenses_close_caustics(caustics_points)

    if caustic_regime == 'wide':
        central_caustic, wide_caustic = sort_2lenses_wide_caustics(caustics_points)

    caustics = [central_caustic, close_top_caustic, close_bottom_caustic, wide_caustic, resonant_caustic]

    return caustic_regime, caustics, critical_curve


def sort_2lenses_resonant_caustic(caustic_points):

    keep = []
    medians = []
    for count in xrange(4):

        slice = caustic_points[count::4][2:-2]

        good = np.where(slice[:,0].imag>0)[0]

        keep.append(slice[good])
        medians.append(np.median(slice[good], axis=0))

    good_order = [keep[0]]
    del keep[0]
    while len(keep) != 0:

        dists = []
        reference = good_order[-1][-1,0]
        arrays = []
        for i in xrange(len(keep)):

            distance = ((reference.real-keep[i][0,0].real)**2+(reference.imag-keep[i][0,0].imag)**2)**0.5
            arrays.append(keep[i])
            dists.append(distance)
            distance = ((reference.real - keep[i][::-1][0,0].real) ** 2 + (reference.imag
                                                                            - keep[i][::-1][0,0].imag) ** 2) ** 0.5
            dists.append(distance)
            arrays.append(keep[i][::-1])


        min_index = np.argmin(dists)
        good_order.append(arrays[min_index])
        del keep[int(min_index/2)]

    top_caustic = np.r_[good_order[0],good_order[1],good_order[2],good_order[3]]


    resonant_caustic = top_caustic

    return resonant_caustic[:,0]

def sort_2lenses_close_caustics(caustic_points):

    median_x = np.median(caustic_points[:,0].real)

    central_caustic = []

    close_top_index = np.where((caustic_points[:,0].real<median_x) & (caustic_points[:,0].imag>0))[0]
    close_top_caustic = caustic_points[close_top_index,0]

    close_bottom_index = np.where((caustic_points[:, 0].real < median_x) & (caustic_points[:, 0].imag < 0))[0]
    close_bottom_caustic = caustic_points[close_bottom_index, 0]
    central_caustic_index = [i for i in xrange(len(caustic_points[:,0])) if (i not in close_bottom_index)
                             & (i not in close_top_index)][:]

    central_caustic = caustic_points[central_caustic_index,0]
    top = np.where(central_caustic.imag>0)[0]
    bot = np.where(central_caustic.imag<0)[0]

    central_caustic = np.r_[central_caustic[top],central_caustic[bot][:-1]]
    return central_caustic, close_top_caustic, close_bottom_caustic


def sort_2lenses_wide_caustics(caustic_points):

    median_x = np.median(caustic_points[:, 0].real)

    top_central_caustic_index = np.where((caustic_points[:,0].real<median_x) & (caustic_points[:,0].imag>0))[0][:-3]
    top_central_caustic = caustic_points[top_central_caustic_index, 0]
    bottom_central_caustic = np.conj(caustic_points[top_central_caustic_index, 0])[::-1]

    central_caustic = np.r_[top_central_caustic, bottom_central_caustic]

    top_wide_caustic_index = np.where((caustic_points[:, 0].real > median_x) & (caustic_points[:, 0].imag > 0))[0]
    top_wide_caustic = caustic_points[top_wide_caustic_index, 0]
    bottom_wide_caustic = np.conj(caustic_points[top_wide_caustic_index, 0])
    wide_caustic = np.r_[top_wide_caustic, bottom_wide_caustic]


    return central_caustic, wide_caustic

def compute_2_lenses_caustics_points(separation, mass_ratio, resolution = 1000):

    caustics = []
    critical_curves = []

    center_of_mass = mass_ratio / (1 + mass_ratio)

    # Witt&Mao magic numbers
    total_mass = 0.5
    mass_1 = 1 / (1 + mass_ratio)
    mass_2 = mass_ratio * mass_1
    delta_mass = (mass_2 - mass_1) / 2
    lens_1 = -separation / 2.0
    lens_2 = separation / 2.0
    lens_1_conjugate = np.conj(lens_1)
    lens_2_conjugate = np.conj(lens_2)

    theta = np.arange(0, 2 * np.pi, 2 * np.pi / resolution)

    sols = []
    for angle in theta:

        e_phi = np.cos(-angle) + 1j * np.sin(-angle)  # See Witt & Mao

        wm_0 = -2.0 * total_mass * lens_1 ** 2 + e_phi * lens_1 ** 4
        wm_1 = 4.0 * lens_1 * delta_mass
        wm_2 = -2.0 * total_mass - 2 * e_phi * lens_1 ** 2
        wm_3 = 0.0
        wm_4 = e_phi

        polynomial_coefficients = [wm_4, wm_3, wm_2, wm_1, wm_0]

        polynomial_roots = np.roots(polynomial_coefficients)
        polynomial_roots = np.sort(polynomial_roots)

        for count,root in enumerate(polynomial_roots):

            root_conjugate = np.conj(root)
            zeta_caustic = root + mass_1 / (lens_1_conjugate - root_conjugate) + mass_2 / (
                lens_2_conjugate - root_conjugate)

            caustics.append([zeta_caustic,angle])
            critical_curves.append([root,angle])

    caustics = np.array(caustics)
    critical_curves = np.array(critical_curves)


    #shift into center of mass referentiel

    caustics[:,0] += separation / 2 - center_of_mass
    critical_curves[:,0] += separation / 2 - center_of_mass

    return caustics, critical_curves




def find_area_of_interest_around_caustics(caustics,secure_factor = 0.1):



    min_X = np.min(caustics[:,0].real)-secure_factor
    max_X = np.max(caustics[:,0]).real+secure_factor

    min_Y = np.min(caustics[:, 0].imag)-secure_factor
    max_Y = np.max(caustics[:, 0].imag)+secure_factor


    center_of_the_box_X = min_X + (max_X-min_X)/2
    center_of_the_box_Y = min_Y + (max_Y-min_Y)/2



    area_of_interest = [[min_X, max_X], [min_Y, max_Y], [center_of_the_box_X, center_of_the_box_Y]]


    return area_of_interest



def find_2_lenses_caustic_regime(separation, mass_ratio):

    #from Cassan 2008

    #compute close limit
    mass_ratio_constant = (1+mass_ratio)**2/(27*mass_ratio)

    constant_1 = (1-3*mass_ratio_constant)/(mass_ratio_constant)
    constant_2 = 3

    polynomial_coefficients = [1,0,0,0,constant_1,0,0,0,constant_2,0,0,0,-1]
    polynomial_roots = np.roots(polynomial_coefficients)

    s_close = [i.real for i in polynomial_roots if (i.imag==0) & (i.real>0)][0]

    #compute wide limit

    s_wide = ((1+mass_ratio**(1/3))**3/(1+mass_ratio))**0.5


    caustic_regime = 'resonant'

    if separation > s_wide:

        caustic_regime = 'wide'

    if separation < s_close:

        caustic_regime = 'close'

    return caustic_regime


def dx(function, x,args):
    return abs(0-function(x,args))

def newtons_method(function, jaco, x0,  args = None, e=0.1):
    delta = dx(function, x0, args)
    while delta > e:
        x0 = x0 - function(x0,args)/jaco(x0,args)
        delta = dx(function, x0,args)
    return x0

def foo(x,pp):

    pp1 = pp[0]
    pp2 = pp[1]
    pp3 = pp[2]
    pp4 = pp[3]
    pp5 = pp[4]
    aa =  pp1 * x ** 4 + pp2 * x ** 3 + pp3 * x**2 + pp4 * x + pp5

    return aa

def jaco(x,pp):
    pp1 = pp[0]
    pp2 = pp[1]
    pp3 = pp[2]
    pp4 = pp[3]
    pp5 = pp[4]
    aa = 4*pp1*x**3+3*pp2*x**2+2*pp3*x+pp4

    return aa

def hessian(x,pp1,pp2,pp3,pp4,pp5):

    aa = 12*pp1*x**2+6*pp2*x+2*pp3

    return aa