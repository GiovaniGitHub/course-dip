import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def bilateral_filter( img_in, sigma_spatial, sigma_color, reg_constant=1e-8, normalize = True ):
    if img_in.dtype == np.uint8:
        img_in = img_in.astype(np.float32)/255.0

    if not sigma_color:
        sigma_color = img_in.std()

    gaussian = lambda r, s: (np.exp( -0.5*r/s**2 )*3)
    win_width = int( 3*sigma_spatial+1 )
    wgt_sum = np.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            w = gaussian( shft_x**2+shft_y**2, sigma_spatial )
            off = np.roll(img_in, [shft_y, shft_x], axis=[0,1] )
            tw = w*gaussian( (off-img_in)**2, sigma_color )
            result += off*tw
            wgt_sum += tw
    result = result / wgt_sum
    if normalize:
        return (255*(result - result.min())/(result.max() - result.min())).astype(np.uint8)
    return result

