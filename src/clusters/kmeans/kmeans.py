import numpy as np
import sys

def get_labels(data, centroids):
    labels = []
    for d in data:
        labels.append(
            np.argmin([np.linalg.norm(d - centroid) for centroid in centroids])
        )
    return labels

def kmeans(data, k):
    centroids = []
    centroids.append(data[np.random.randint(
            data.shape[0]), :])

    for c_id in range(k - 1):

        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize

            for j in range(len(centroids)):
                temp_dist = np.linalg.norm(point - centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
    return centroids

