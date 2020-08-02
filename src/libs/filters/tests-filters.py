from matplotlib import pyplot as plt
import numpy as np
from butterworth_filter import butterworth

def normalize_image(img):
    return (255*(img - img.min())/(img.max() - img.min())).astype(np.uint8)

def test_butterworth(img):
    img_filtered = butterworth(img)
    img_filtered = normalize_image(img_filtered)
    plt.imshow(np.concatenate((img, img_filtered),axis=1))
    plt.show()
