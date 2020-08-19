import numpy as np
from matplotlib import pyplot as plt

def mult_plot(imgs, labels):
    fig, axes = plt.subplots(figsize=(16, 16), nrows=1, ncols=len(labels))
    fig.tight_layout(pad=2.0)

    for i in range(len(axes)):
        axes[i].imshow(imgs[i], cmap = 'gray')
        axes[i].set(title=labels[i])
    plt.show()

def gaussian_noisy(image, mean=None, sigma=None):
    if not mean:
        mean = image.mean()

    if not sigma:
        sigma = image.std()

    gauss = np.random.normal(mean,sigma,image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = image + gauss

    return noisy

def salt_and_pepper_noisy(image, amount = 0.004):

    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1
    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (0.5))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0

    return out

def poisson_noisy(image):
    vals = 2 ** np.ceil(np.log2(len(np.unique(image))))
    noisy = np.random.poisson(image * vals) / float(vals)

    return noisy + image

def speckle_noisy(image):
    if len(image.shape) == 2:
        gauss = image.std()*np.random.randn(image.shape[0],image.shape[1]) + image.mean()
        gauss = gauss.reshape(image.shape[0],image.shape[1])

    elif len(image.shape) == 3:
        gauss = image.std()*np.random.randn(image.shape[0],image.shape[1],image.shape[2]) + image.mean()
        gauss = gauss.reshape(image.shape[0],image.shape[1],image.shape[2])

    noisy = image + image * gauss
    return noisy

def convolve(image, kernel):
    output = np.zeros_like(image)
    if kernel.shape[0] % 2 != 0:
        w_x = int((kernel.shape[0] - 1)/2)
        w_y = int((kernel.shape[1] - 1)/2)
    else:
        w_x = int((kernel.shape[0])/2)
        w_y = int((kernel.shape[1])/2)

    padded_x = image.shape[0] + (kernel.shape[0] - 1)
    padded_y =  image.shape[1] +  (kernel.shape[1] - 1)
    image_padded = np.zeros((padded_x,padded_y))


    image_padded[w_x:-w_x, w_y:-w_y] = image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            x_end = x+kernel.shape[0]
            y_end = y+kernel.shape[1]
            output[x,y]=(kernel*image_padded[x:x_end,y:y_end]).sum()

    return output
