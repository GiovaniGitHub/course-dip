from matplotlib import pyplot as plt
import numpy as np
from butterworth_filter import butterworth
from bilateral_filter import bilateral_filter
from matplotlib import pyplot as plt


def normalize_image(img):
    return (255*(img - img.min())/(img.max() - img.min())).astype(np.uint8)

def gaussian_noise(image):
    pass

def salt_and_pepper(image):
    s_vs_p = 0.5
    amount = 0.016
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 1
    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0

    return out

def test_butterworth(img):
    img_filtered = butterworth(img)
    img_filtered = normalize_image(img_filtered)
    plt.imshow(np.concatenate((img, img_filtered),axis=1))
    plt.show()

def test_bilateral_filter(img, sigma_s, sigma_v):
    img_filtered = bilateral_filter(img, sigma_s, sigma_v)
    img_filtered = normalize_image(img_filtered)
    plt.imshow(np.concatenate((img, img_filtered),axis=1))
    plt.show()


if __name__ == "__main__":
    img = plt.imread('../../slides/lena_gray.png')
    img_noised = salt_and_pepper(img)
    img_filtered = bilateral_filter(img_noised, 10, 0.1, normalize = True)
    plt.imshow(img_filtered, cmap='gray')
    plt.show()
