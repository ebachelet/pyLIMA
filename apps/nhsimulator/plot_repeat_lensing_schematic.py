# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:42:23 2018

@author: rstreet
"""
import astropy.units as u
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors
from matplotlib import cm, rcParams

def plot_repeat_lensing_schematic():
    """Function to draw a schematic diagram to illustrate how the probability
    of a repeat lensing event is calculated."""
    
    source_loc = (0.0,5.0,0.0)
    lens_loc = (10.0,5.0,0.0)
    earth_loc = (20.0,5.0,0.0)
    nh_loc = (20.0,4.5,0.0)
    
    
    rcParams.update({'font.size': 18})
    
    fig = plt.figure(2,(10,10),frameon=False)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter([lens_loc[0]], [lens_loc[1]], [lens_loc[2]], c='k', marker='o',s=100.0)
    ax.text(10.0,5.1,-0.25,'Lens')
    ax.scatter([source_loc[0]], [source_loc[1]], [source_loc[2]], c='r', marker='*',s=100.0)
    ax.text(0.0,5.1,-0.25,'Source')
    ax.scatter([earth_loc[0]], [earth_loc[1]], [earth_loc[2]], c='b', marker='o',s=100.0)
    ax.text(20.0,5.1,-0.25,'Earth')
    ax.scatter([nh_loc[0]], [nh_loc[1]], [nh_loc[2]], c='k', marker='s',s=100.0)
    ax.text(20.0,4.0,0.2,'New Horizons')
    x = np.array([nh_loc[0],nh_loc[0]])
    y = np.array([nh_loc[1],4.3])
    z = np.array([nh_loc[2],0.18])
    ax.plot(x, y, z,'k-')
    
    # Draw the NH-source line
    n = 30
    dx = abs(source_loc[0] - nh_loc[0])/n
    ns_s_x = np.arange(source_loc[0],nh_loc[0],dx)
    dy = abs(source_loc[1] - nh_loc[1])/n
    ns_s_y = np.arange(source_loc[1],nh_loc[1],-dy)
    dz = (source_loc[2] - nh_loc[2])/n
    ns_s_z = np.zeros(len(ns_s_x))
    ax.plot(ns_s_x, ns_s_y, ns_s_z,'k-')
    
    idx = np.where((ns_s_x-lens_loc[0]) == 0.0)
    
    zoff = 0.5
    x = np.array([lens_loc[0],lens_loc[0]])
    y = np.array([lens_loc[1],ns_s_y[idx]])
    z = np.array([zoff,zoff])
    ax.plot(x, y, z,'k-')
    x = np.array([lens_loc[0],lens_loc[0]])
    y = np.array([lens_loc[1],lens_loc[1]])
    z = np.array([zoff-0.05,zoff+0.05])
    ax.plot(x, y, z,'k-')
    x = np.array([ns_s_x[idx],ns_s_x[idx]])
    y = np.array([ns_s_y[idx],ns_s_y[idx]])
    z = np.array([zoff-0.05,zoff+0.05])
    ax.plot(x, y, z,'k-')
    dy = lens_loc[1] - (2.0*abs(lens_loc[1]-ns_s_y[idx]))/3.0
    ax.text(lens_loc[0],dy,zoff-0.1,r'$x_{L}$')
    
    
    
    # Draw cone:
    n = 30
    cone_radius = 0.1
    dtheta = (2.0*np.pi)/200.0
    theta = np.arange(0,(2.0*np.pi)+dtheta,dtheta)
    rcone = np.linspace(0,cone_radius,len(ns_s_x))
    xx = np.zeros([len(ns_s_x),len(theta)])
    yy = np.zeros([len(ns_s_x),len(theta)])
    zz = np.zeros([len(ns_s_x),len(theta)])
    for i,x in enumerate(ns_s_x):
        xx[i,:].fill(x)
        yy[i,:] = ns_s_y[i] + (rcone[i] * np.cos(theta))
        zz[i,:] = ns_s_z[i] + (rcone[i] * np.sin(theta))
    
    colors = np.empty(xx.shape, dtype=str)
    colors.fill('r')
    
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False,
                           facecolors=colors,alpha=0.05)
    
    x = np.array([nh_loc[0],nh_loc[0]])
    y = np.array([nh_loc[1],nh_loc[1]+cone_radius])
    z = np.array([-0.25,-0.25])
    ax.plot(x, y, z,'k-')
    x = np.array([nh_loc[0],nh_loc[0]])
    y = np.array([nh_loc[1],nh_loc[1]])
    z = np.array([-0.3,-0.20])
    ax.plot(x, y, z,'k-')
    x = np.array([nh_loc[0],nh_loc[0]])
    y = np.array([nh_loc[1]+cone_radius,nh_loc[1]+cone_radius])
    z = np.array([-0.3,-0.20])
    ax.plot(x, y, z,'k-')
    ax.text(nh_loc[0],nh_loc[1],-0.45,r'$\tilde{r_{\rm{E}}}$')
    
    # Draw Earth-source line
    n = 20
    dx = abs(source_loc[0] - earth_loc[0])/n
    x = np.arange(source_loc[0],earth_loc[0],dx)
    y = np.zeros(len(x))
    y.fill(source_loc[1])
    dz = abs(source_loc[2] - earth_loc[2])/n
    z = np.zeros(len(x))
    ax.plot(x, y, z,'k-')
    
    yoff = 6.0
    zoff = -0.7
    dx = abs(source_loc[0] - earth_loc[0])/n
    x = np.linspace(source_loc[0],earth_loc[0],2)
    y = np.zeros(len(x))
    y.fill(yoff)
    z = np.zeros(len(x))
    z.fill(zoff)
    ax.plot(x, y, z,'k-')
    x = np.array([source_loc[0],source_loc[0]])
    y = np.array([yoff,yoff])
    z = np.array([zoff-0.05,zoff+0.05])
    ax.plot(x, y, z,'k-')
    x = np.array([earth_loc[0],earth_loc[0]])
    y = np.array([yoff,yoff])
    z = np.array([zoff-0.05,zoff+0.05])
    ax.plot(x, y, z,'k-')
    ax.text(lens_loc[0],yoff,zoff-0.1,r'$D_{s}$')
    
    
    # Earth-NH line:
    zoff = -0.7
    x = np.array([earth_loc[0],nh_loc[0]])
    y = np.array([earth_loc[1],nh_loc[1]])
    z = np.array([zoff,zoff])
    ax.plot(x, y, z,'k-')
    x = np.array([earth_loc[0],earth_loc[0]])
    y = np.array([earth_loc[1],earth_loc[1]])
    z = np.array([zoff-0.05,zoff+0.05])
    ax.plot(x, y, z,'k-')
    x = np.array([nh_loc[0],nh_loc[0]])
    y = np.array([nh_loc[1],nh_loc[1]])
    z = np.array([zoff-0.05,zoff+0.05])
    ax.plot(x, y, z,'k-')
    dy = earth_loc[1] - (2.0*abs(earth_loc[1]-nh_loc[1]))/3.0
    ax.text(nh_loc[0],dy,zoff-0.1,r'$a_{NH}$')
    
    
    # Lens plane
    xx = np.zeros([len(ns_s_x),len(theta)])
    y = np.linspace(4.0,6.0,20)
    z = np.linspace(-1.0,1.0,20)
    yy, zz = np.meshgrid(y, z)
    xx = np.zeros([len(yy),len(zz)])
    xx.fill(lens_loc[0])
    colors = np.empty(xx.shape, dtype=str)
    colors.fill('b')
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False,
                           facecolors=colors,alpha=0.05)
    ax.text(lens_loc[0],y.mean(),z.max(),'Lens plane',color='b')
    
    # Draw line indicating lensing range of motion:
    x = np.array([lens_loc[0],lens_loc[0]])
    y = np.array([lens_loc[1],nh_loc[1]])
    z = np.linspace(0,rcone.max(),2)
    ax.plot(x, y, z,'k-')
    x = np.array([lens_loc[0],lens_loc[0]])
    y = np.array([lens_loc[1],nh_loc[1]])
    z = np.linspace(0,-rcone.max(),2)
    ax.plot(x, y, z,'k-')
    
    # Plot arc indicating lensing range:
    r = 0.1
    dtheta = np.pi/200.0
    theta = np.arange(0.5*np.pi,1.5*np.pi,dtheta)
    y = nh_loc[1] + (r * np.cos(theta))
    z = nh_loc[2] + (r * np.sin(theta))
    x = np.zeros(len(y))
    x.fill(lens_loc[0])
    ax.plot(x, y, z,'b-')
    
    x = np.array([10.0,10.0])
    y = np.array([nh_loc[1],4.35])
    z = np.array([nh_loc[2],0.1])
    ax.plot(x, y, z,'b-')
    ax.text(10.0,4.25,0.1,r'$\xi$')
    
    show_axis = False
    if show_axis:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    
    else:
        ax.axis('off')
        
    ax.view_init(azim=30,elev=20)
    fig.tight_layout()
    fig.subplots_adjust(left=-0.1,right=1.1,top=1.1,bottom=-0.1)
    
    plt.savefig('range_motion_lensing.png')
    
if __name__ == '__main__':
    
    plot_repeat_lensing_schematic()
    