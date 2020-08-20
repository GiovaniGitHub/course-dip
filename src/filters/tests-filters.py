from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import numpy as np
from butterworth_filter import butterworth
from bilateral_filter import bilateral_filter
from wavelet_filter import wavelet_threshold
from richardson_lucy import richardson_lucy
from matplotlib import pyplot as plt
from utils import *

def normalize_image(image):
    return (255*(image - image.min())/(image.max() - image.min())).astype(np.uint8)

def test_butterworth(image):
    image_noisy = noisy('gaussian',)
    image_filtered = butterworth(image)
    image_filtered = normalize_image(image_filtered)
    plt.imshow(np.concatenate((image, image_filtered),axis=1))
    plt.show()

def test_bilateral_filter(image, sigma_s, sigma_v):
    image_filtered = bilateral_filter(image, sigma_s, sigma_v)
    image_filtered = normalize_image(image_filtered)
    plt.imshow(np.concatenate((image, image_filtered),axis=1))
    plt.show()


def test_richardson_lucy(image):
    kernel = np.ones((5, 5)) / 25

    image_noisy = image.copy()
    image_noisy = convolve2d(image, kernel, mode='same')
    image_noisy += (np.random.poisson(lam=25, size=image.shape) - 10) / 255.

    image_filtered = richardson_lucy(image_noisy, kernel, iterations = 10)

    mult_plot([image, image_noisy, image_filtered], ['original','noisy','filtered'])

if __name__ == "__main__":
    image = plt.imread('../../slides/lena_gray.png')

    test_richardson_lucy(image)
