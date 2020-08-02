import numpy as np

def butterworth_kernel(l, h):
    u = np.array(range(l))
    v = np.array(range(h))

    idx = u[u > l/2]
    u[idx] = u[idx] - l

    idy = v[v > h/2]
    v[idy] = v[idy] - h

    [V, U] = np.meshgrid(v, u)

    return np.sqrt(U**2 + V**2)

def butterworth2D(img, d0 = 50, n = 2):
    l, h = img.shape
    f = np.fft.fft2(img)

    D = butterworth_kernel(l, h)
    H = 1/(1 + (D/d0)**(2*n))

    G = H*f

    output_image = (np.fft.ifft2(G)).astype(np.float)

    return output_image

def butterworth(img, d0 = 50, n = 2):
    if len(img.shape) > 2:
        return np.dstack((
            butterworth2D(img[:,:,0],d0,n),
            butterworth2D(img[:,:,1],d0,n),
            butterworth2D(img[:,:,2],d0,n)
        ))

    return butterworth2D(img,d0,n)
