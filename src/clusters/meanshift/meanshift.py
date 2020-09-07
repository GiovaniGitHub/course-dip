import numpy as np

MIN_DISTANCE = 1e-1

def meanshift( data,n_iterations = 5, look_distance = 1, kernel_bandwidth = 1):

    gaussian_value = lambda r,s: (1/(s*np.sqrt(2*np.pi))) * np.exp(-0.5*((r / s))**2)
    labels = np.array([-1]*data.shape[0])
    X = np.copy(data)
    past_X = []

    def neighbourhood(X, x_centroid, distance = 1):
        eligible_X = []
        for i, x in enumerate(X):
            distance_between = np.linalg.norm(x - x_centroid)
            if distance_between <= distance:
                eligible_X.append(x)
                labels[i] = X.tolist().index(x_centroid.tolist())
        return eligible_X

    for it in range(n_iterations):
        for i, x in enumerate(X):
            neighbours = neighbourhood(X, x, look_distance)
            numerator = 0
            denominator = 0
            for neighbour in neighbours:
                distance = np.linalg.norm(neighbour - x)
                weight = gaussian_value(distance, kernel_bandwidth)
                numerator += (weight * neighbour)
                denominator += weight

            new_x = numerator / denominator

            X[i] = new_x

        past_X.append(np.copy(X))

    return labels


