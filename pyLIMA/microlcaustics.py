"""
Created on Thu Apr 20 11:19:38 2017

@author: ebachelet
"""
from __future__ import division

import numpy as np
import time
import scipy.spatial as ss
import matplotlib.pyplot as plt

def find_2_lenses_caustics_and_critical_curves(separation, mass_ratio, resolution = 1000):

    close_top_caustic = None
    close_top_cc = None
    close_bottom_caustic = None
    close_bottom_cc = None
    central_caustic = None
    central_cc = None
    wide_caustic = None
    wide_cc = None
    resonant_caustic = None
    resonant_cc = None

    caustics_points, critical_curve_points = compute_2_lenses_caustics_points(separation, mass_ratio,
                                                                              resolution=resolution)

    critical_curve = critical_curve_points

    caustic_regime = find_2_lenses_caustic_regime(separation, mass_ratio)

    if caustic_regime == 'resonant':

        resonant_caustic, resonant_cc = sort_2lenses_resonant_caustic(caustics_points, critical_curve_points)

    if caustic_regime == 'close':

        result =  sort_2lenses_close_caustics(caustics_points, critical_curve_points)
        central_caustic, close_top_caustic, close_bottom_caustic, central_cc, close_top_cc, close_bottom_cc = result

    if caustic_regime == 'wide':
        central_caustic, wide_caustic, central_cc , wide_cc = sort_2lenses_wide_caustics(caustics_points,
                                                                                         critical_curve_points)

    caustics = [central_caustic, close_top_caustic, close_bottom_caustic, wide_caustic, resonant_caustic]
    critical_curve = [central_cc, close_top_cc, close_bottom_cc, wide_cc, resonant_cc]

    return caustic_regime, caustics, critical_curve


def sort_2lenses_resonant_caustic(caustic_points, critical_curves_points):




    median_x = np.median(caustic_points[:, :].real, axis=0)
    median_y = np.median(caustic_points[:, :].imag, axis=0)

    global_median_x = np.median(caustic_points[:, :-1].real, axis=(0, 1))




    starting_points = caustic_points[0]
    order =  np.where(median_y>0)[0]

    if starting_points[order[0]].imag< starting_points[order[1]].imag:

        resonant_caustic = caustic_points[:,order[0]]
        resonant_cc = critical_curves_points[:,order[0]]
        resonant_caustic = np.r_[resonant_caustic, caustic_points[:,order[1]]]
        resonant_cc = np.r_[resonant_cc, critical_curves_points[:, order[1]]]

    else:

        resonant_caustic = caustic_points[:,order[1]]
        resonant_cc = critical_curves_points[:,order[1]]

        resonant_caustic = np.r_[resonant_caustic, caustic_points[:,order[0]]]
        resonant_cc = np.r_[resonant_cc, critical_curves_points[:, order[0]]]

    order = np.where(median_y < 0)[0]

    if starting_points[order[0]].imag < starting_points[order[1]].imag:

        resonant_caustic = np.r_[resonant_caustic,caustic_points[:, order[1]]]
        resonant_caustic = np.r_[resonant_caustic, caustic_points[:, order[0]]]
        resonant_cc = np.r_[resonant_cc, critical_curves_points[:, order[1]]]
        resonant_cc = np.r_[resonant_cc, critical_curves_points[:, order[0]]]



    else:

        resonant_caustic = np.r_[resonant_caustic,caustic_points[:, order[0]]]
        resonant_caustic = np.r_[resonant_caustic, caustic_points[:, order[1]]]
        resonant_cc = np.r_[resonant_cc, critical_curves_points[:, order[0]]]
        resonant_cc = np.r_[resonant_cc, critical_curves_points[:, order[1]]]

    return resonant_caustic, resonant_cc

def sort_2lenses_close_caustics(caustic_points, critical_curves_points):

    median_x = np.median(caustic_points[:, :].real, axis=0)
    median_y = np.median(caustic_points[:, :].imag, axis=0)

    global_median_x = np.median(caustic_points[:, :].real, axis=(0, 1))

    top_caustic = np.where((median_x<global_median_x) & (median_y>0))[0]
    close_top_caustic = caustic_points[:, top_caustic]
    close_top_cc = critical_curves_points[:, top_caustic]

    bottom_caustic = np.where((median_x < global_median_x) & (median_y < 0))[0]
    close_bottom_caustic = caustic_points[:, bottom_caustic]
    close_bottom_cc = critical_curves_points[:, bottom_caustic]

    index_left = np.arange(0,4).tolist()
    index_left.remove(top_caustic)
    index_left.remove(bottom_caustic)
    central_caustic = np.r_[caustic_points[:, index_left[0]], caustic_points[:, index_left[1]]]
    central_cc = np.r_[critical_curves_points[:, index_left[0]],critical_curves_points[:, index_left[1]]]

    return central_caustic, close_top_caustic, close_bottom_caustic, central_cc, close_top_cc, close_bottom_cc


def sort_2lenses_wide_caustics(caustic_points, critical_curves_points):


    median_x = np.median(caustic_points[:, :].real, axis=0)

    global_median_x = np.median(caustic_points[:, :].real, axis=(0, 1))

    wide_caustic_index = np.where((median_x < global_median_x))[0]
    wide_caustic = np.r_[caustic_points[:, wide_caustic_index[0]], caustic_points[:, wide_caustic_index[1]]]
    wide_cc = np.r_[critical_curves_points[:, wide_caustic_index[0]],
                    critical_curves_points[:, wide_caustic_index[1]]]

    index_left = np.arange(0, 4).tolist()
    index_left.remove(wide_caustic_index[0])
    index_left.remove(wide_caustic_index[1])
    central_caustic = np.r_[caustic_points[:, index_left[0]], caustic_points[:, index_left[1]]]
    central_cc = np.r_[critical_curves_points[:, index_left[0]],
                       critical_curves_points[:, index_left[1]]]

    return central_caustic, wide_caustic, central_cc, wide_cc

def compute_2_lenses_caustics_points(separation, mass_ratio, resolution = 1000):

    caustics = []
    critical_curves = []

    center_of_mass = mass_ratio / (1 + mass_ratio)*separation

    # Witt&Mao magic numbers
    total_mass = 0.5
    mass_1 = 1 / (1 + mass_ratio)
    mass_2 = mass_ratio * mass_1
    delta_mass = (mass_2 - mass_1) / 2
    lens_1 = -separation / 2.0
    lens_2 = separation / 2.0
    lens_1_conjugate = np.conj(lens_1)
    lens_2_conjugate = np.conj(lens_2)

    phi = np.linspace(0.0, 2*np.pi, resolution)
    roots = []
    wm_1 = 4.0 * lens_1 * delta_mass
    wm_3 = 0.0
    slice_1 = []
    slice_2 = []
    slice_3 = []
    slice_4 = []
    slices = [slice_1, slice_2, slice_3, slice_4]
    for angle in phi:

        e_phi = np.cos(-angle) + 1j * np.sin(-angle)  # See Witt & Mao

        wm_0 = -2.0 * total_mass * lens_1 ** 2 + e_phi * lens_1 ** 4
        wm_2 = -2.0 * total_mass - 2 * e_phi * lens_1 ** 2
        wm_4 = e_phi

        polynomial_coefficients = [wm_4, wm_3, wm_2, wm_1, wm_0]

        polynomial_roots = np.roots(polynomial_coefficients)

        if len(roots) == 0 :

            pol_roots = polynomial_roots

        else:

            aa = np.c_[polynomial_roots.real, polynomial_roots.imag]
            bb = np.c_[roots[-1].real, roots[-1].imag]

            distances = ss.distance.cdist(aa, bb)
            good_order = [0,0,0,0]
            for i in xrange(4):
                index = np.argmin(distances[i])
                good_order[index] = polynomial_roots[i]

            pol_roots = np.array(good_order)

        roots.append(pol_roots)



        images_conjugate = np.conj(pol_roots)
        zeta_caustics = pol_roots + mass_1 / (lens_1_conjugate -  images_conjugate) + mass_2 / (
            lens_2_conjugate -  images_conjugate)

        if len(caustics) == 0:
            caustics = zeta_caustics
            critical_curves = pol_roots
        else:
            caustics = np.vstack((caustics,zeta_caustics))
            critical_curves = np.vstack((critical_curves, pol_roots))







    #shift into center of mass referentiel

    caustics += -center_of_mass+separation/2
    critical_curves += -center_of_mass+separation/2

    return caustics, critical_curves




def find_area_of_interest_around_caustics(caustics,secure_factor = 0.1):

    all_points = []
    for caustic in caustics:

        if caustic is not None:
            all_points += caustic.ravel().tolist()
    all_points = np.array(all_points)
    min_X = np.min(all_points.real)-secure_factor
    max_X = np.max(all_points.real)+secure_factor

    min_Y = np.min(all_points.imag)-secure_factor
    max_Y = np.max(all_points.imag)+secure_factor


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

    return abs(0-function(x,args[0],args[1],args[2],args[3],args[4]))

def newtons_method(function, jaco, x0,  args = None, e=10**-10):
    delta = dx(function, x0, args)

    while np.max(delta) > e:
        x0 = x0 - function(x0,args[0],args[1],args[2],args[3],args[4])/jaco(x0,args[0],args[1],args[2],args[3],args[4])
        delta = dx(function, x0,args)
    return x0

def foo(x,pp1,pp2,pp3,pp4,pp5):

    x_2 = x**2
    x_3 = x_2*x
    x_4 = x_2*x_2
    aa =  pp1 * x_4 + pp2 * x_3 + pp3 * x_2 + pp4 * x + pp5

    return aa


def foo2(X,pp1,pp2,pp3,pp4,pp5):

    x = X[0]+1j*X[1]



    x_2 = x**2
    x_3 = x_2*x
    x_4 = x_2*x_2
    aa =  pp1 * x_4 + pp2 * x_3 + pp3 * x_2 + pp4 * x + pp5


    return [aa.real, aa.imag]


def jaco(x,pp1,pp2,pp3,pp4,pp5):

    x_2 = x ** 2
    x_3 = x_2 * x
    aa = 4*pp1*x_3+3*pp2*x_2+2*pp3*x+pp4

    return aa


def hessian(x,pp1,pp2,pp3,pp4,pp5):

    aa = 12*pp1*x**2+6*pp2*x+2*pp3

    return aa

def Jacobian(zi,mtot,deltam,z1) :

    ziconj = np.conj(zi)

    dzeta=(mtot-deltam)/(np.conjugate(z1)-ziconj)**2+(mtot+deltam)/(-np.conjugate(z1)-ziconj)**2
    dzetaconj=np.conj(dzeta)

    dZETA=dzeta*dzetaconj

    detJ=1-dZETA.real
    return np.sign(detJ)


def find_slices(slice_1,slice_2,points):

    centroid_1 = np.median(slice_1.real) +1j*np.median(slice_1.imag)
    centroid_2 = np.median(slice_2.real) +1j*np.median(slice_2.imag)

