import cv2
import numpy as np
from matplotlib import pyplot as plt
import mahotas

def plot_surf(data):
    if data.shape[0] > 100:
        data = cv2.resize(data,None,fx=100.0/data.shape[0],fy=100.0/data.shape[1])
    X, Y = np.mgrid[0:data.shape[0], 0:data.shape[1]]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, data, rstride=1, cstride=1, cmap=plt.cm.gray,
        linewidth=0)
    plt.show()
