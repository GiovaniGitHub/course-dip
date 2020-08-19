from utils import convolve
import numpy as np

def richardson_lucy(image, kernel, iterations=50):

    image = image.astype(np.float)
    kernel = kernel.astype(np.float)
    im_deconv = np.full(image.shape, 0.5)
    kernel_mirror = np.flip(kernel)

    for i in range(iterations):
        print(f'iteration: {i}')
        conv = convolve(im_deconv, kernel)
        relative_blur = image / conv
        im_deconv *= convolve(relative_blur, kernel_mirror)

    im_deconv[im_deconv > 1] = 1
    im_deconv[im_deconv < -1] = -1

    return im_deconv