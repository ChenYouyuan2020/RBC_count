import numpy as np 

def preprocess_imgs(imgs, cut_range_x=[100,924], cut_range_y=[100,924]):
    new_images = imgs[:,:,cut_range_x[0]:cut_range_x[1],cut_range_y[0]:cut_range_y[1]]
    return new_images