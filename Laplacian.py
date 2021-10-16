import numpy as np
from scipy.io import loadmat
from scipy import sparse
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

# calculates normalized Laplacian
def Laplacian(data, k=10, sigma=1):
    knn_dist_graph = kneighbors_graph(X=data,
                                    n_neighbors=k,
                                    mode='distance',
                                    metric='euclidean',
                                    n_jobs=6)

    W = sparse.csr_matrix(knn_dist_graph.shape)
    nonzeroindices = knn_dist_graph.nonzero()
    W[nonzeroindices] = np.exp(-np.asarray(knn_dist_graph[nonzeroindices])**2 / 2.0 * sigma**2)
    W = 0.5 * (W + W.T)
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
    k = 10
    L = Laplacian(data, k)