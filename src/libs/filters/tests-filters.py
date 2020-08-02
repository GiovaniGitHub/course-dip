from matplotlib import pyplot as plt
import numpy as np
from butterworth_filter import butterworth

# from . import normalize_image
def normalize_image(img):
    return (255*(img - img.min())/(img.max() - img.min())).astype(np.uint8)

def test_butterworth():
    img = plt.imread('/media/nobrega/df72d85f-0be8-42e6-b5c1-6e674c8e2097/nobrega/Documentos/Python/Giovani/Fundamentos de PDI/messi.jpg')
    img_filtered = butterworth(img)
    img_filtered = normalize_image(img_filtered)
    plt.imshow(np.concatenate((img, img_filtered),axis=1))
    plt.show()


if __name__ == "__main__":
    test_butterworth()
