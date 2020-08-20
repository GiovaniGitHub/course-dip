import numpy as np
from utils import convolve_fft

def gaussian_kernel(n,mean=0,sigma=1):
    if n % 2 == 0:
        axis_x = list(range(-int((n)/2),int((n+2)/2),1))
        axis_x.remove(0)
    else:
        axis_x = list(range(-int((n-1)/2),int((n+1)/2),1))
    kernel_1d = st.norm.pdf(axis_x,mean,sigma)

    kernel = np.outer(kernel_1d,kernel_1d)

    return kernel/kernel.sum()


def gaussian_filter(img, n=3, sd=1):
    kernel = gaussian_kernel(n=n,mean=0,sigma=sd)

    img_blur = convolve_fft(img,kernel)

    return img_blur
