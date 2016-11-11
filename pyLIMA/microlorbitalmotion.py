import numpy as np

def orbital_motion_2D_trajectory_shift(to_om, time, dalpha_dt):


    dalpha =  dalpha_dt*(time-to_om)

    return dalpha

def orbital_motion_2D_separation_shift(to_om, time, ds_dt):

    dseparation = ds_dt*(time - to_om)

    return dseparation
