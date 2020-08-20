from utils import convolve
import numpy as np
from scipy.signal import convolve2d
def richardson_lucy(image, kernel, iterations=50):

    image = image.astype(np.float)
    kernel = kernel.astype(np.float)
    im_deconv = np.full(image.shape, 0.5)
    kernel_mirror = np.flip(kernel)

    for i in range(iterations):
        conv = convolve2d(im_deconv, kernel, mode='same')
        relative_blur = image / conv
        im_deconv *= convolve2d(relative_blur, kernel_mirror, mode='same')

    im_deconv[im_deconv > 1] = 1
    im_deconv[im_deconv < -1] = -1

    return im_deconv
