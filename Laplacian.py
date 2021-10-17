import numpy as np
from scipy.io import loadmat
from scipy import sparse
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
import matplotlib.pyplot as plt

# calculates normalized Laplacian
def Laplacian(data, k, sigma=1):
    def gaussian(x, sigma):
        denom = 2.0*(sigma**2.0)
        E = np.exp(-(x**2.0)/denom)
        return E
        
    func = np.vectorize(gaussian, excluded=['sigma'])
    NN = NearestNeighbors(n_neighbors=k,
                          algorithm='auto',
                          metric='euclidean',
                          n_jobs= 1) # setting jobs higher might be faster,
                                     # though it might also cause isses with
                                     # determinism?
    NN.fit(data)

    W = NN.kneighbors_graph(mode='distance')
    actual_sigma = W[W != 0].std()
    W[W != 0] = func(W[W != 0], actual_sigma)

    components = W.sum(axis=1)
    DPM_components = np.power(components, -0.5)

    D = np.diagflat(components)
    DPM = np.diagflat(DPM_components)
    L =  D - W
    DLD = np.linalg.multi_dot([DPM, L, DPM])

    return DLD

if __name__ == "__main__":
    dataset = loadmat('coil20.mat')
    data = dataset['X']
    k = 2
    L = Laplacian(data, k)