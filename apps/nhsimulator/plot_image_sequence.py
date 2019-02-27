# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:20:38 2019

@author: rstreet
"""

from os import path
from sys import argv
from astropy.io import fits
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize

def generate_image_sequence(input_file):
    
    params = read_input_file(input_file)
    
    (fig, axes) = plt.subplots(nrows=2,ncols=len(params['image_list']))
    plt.subplots_adjust(wspace=0.2, hspace=-0.6)
    plt.rcParams.update({'font.size': 16})
    
    ref_image_section = crop_image(params, params['image_list'][params['ref_index']][0])
    ref_diff_section = crop_image(params, params['image_list'][params['ref_index']][1])
    norm_image = ImageNormalize(ref_image_section, interval=ZScaleInterval())
    norm_diff_image = ImageNormalize(ref_diff_section, interval=ZScaleInterval())
    
    for icol, pair in enumerate(params['image_list']):
        
        image_section = crop_image(params, pair[0])
        diff_image_section = crop_image(params, pair[1])
        
        #norm_image = ImageNormalize(image_section, interval=ZScaleInterval())
        #norm_diff_image = ImageNormalize(diff_image_section, interval=ZScaleInterval())
                      
        axes[0,icol].imshow(image_section, norm=norm_image)
        axes[1,icol].imshow(diff_image_section, norm=norm_diff_image)
        
        
        #axes[0,icol].set_xlabel('x [pixels]')
        axes[1,icol].set_xlabel('x [pixels]')
        
        plt.gca().set_aspect('equal', adjustable='box')
        
        if icol == 0:
            
            axes[0,icol].set_ylabel('y [pixels]')
            axes[1,icol].set_ylabel('y [pixels]')
            
        elif icol > 0:

            axes[0,icol].get_yaxis().set_ticks([])
            axes[1,icol].get_yaxis().set_ticks([])
    
        axes[0,icol].get_xaxis().set_ticks([])
    
    plt.savefig(path.join(params['data_dir'],'image_sequence.png'),
                bbox_inches='tight')
    plt.savefig(path.join(params['data_dir'],'image_sequence.eps'),
                bbox_inches='tight')
    
def read_input_file(input_file):
    
    params = {}
    
    try:
        flines = open(input_file,'r').readlines()
        
        pos = flines[0].replace('\n','').split()
        
        params['xcen'] = float(pos[0])
        params['ycen'] = float(pos[1])
        
        box = flines[1].replace('\n','').split()
        
        params['width'] = float(box[0])
        params['height'] = float(box[1])
        
        params['data_dir'] = flines[2].replace('\n','')
        
        params['ref_index'] = int(flines[3].replace('\n',''))
        
        image_list = []
        
        for l in flines[4:]:
            
            if '#' not in l:
                
                (image, diff_image) = l.replace('\n','').split()
                
                image_list.append( ( image, diff_image) )
        
        params['image_list'] = image_list
        
    except IOError:
        
        raise IOError('Cannot find input file '+input_file)
        exit()
    
    params['xmin'] = int(params['xcen'] - (params['width']/2.0))
    params['xmax'] = int(params['xcen'] + (params['width']/2.0))
    params['ymin'] = int(params['ycen'] - (params['height']/2.0))
    params['ymax'] = int(params['ycen'] + (params['height']/2.0))
    
    return params
    
def crop_image(params, image_file):
    
    image = fits.getdata(path.join(params['data_dir'],image_file))

    image_section = image[params['ymin']:params['ymax'],params['xmin']:params['xmax']]
    
    return image_section
    
if __name__ == '__main__':
    
    if len(argv) == 1:
        input_file = raw_input('Please enter the path to the input file: ')
    else:
        input_file = argv[1]
        
    generate_image_sequence(input_file)
    