import numpy as np


def noisy(noise_typ, image):

    if noise_typ == "gauss":
        mean = 0
        var = 0.008
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,image.shape)
        gauss = gauss.reshape(image.shape)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":

        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0

        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        if len(image.shape) == 2:
            gauss = image.std()*np.random.randn(image.shape[0],image.shape[1]) + image.mean()
            gauss = gauss.reshape(image.shape[0],image.shape[1])
            noisy = image + image * gauss
            return noisy
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
