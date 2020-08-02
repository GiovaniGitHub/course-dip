import numpy as np

def bilateral_filter( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):
    gaussian = lambda r, s: (np.exp( -0.5*r/s**2 )*3).astype(int)*1.0/3.0

    win_width = int( 3*sigma_s+1 )

    wgt_sum = np.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):

            w = gaussian( shft_x**2+shft_y**2, sigma_s )

            off = np.roll(img_in, [shft_y, shft_x], axis=[0,1] )

            tw = w*gaussian( (off-img_in)**2, sigma_v )

            result += off*tw
            wgt_sum += tw
