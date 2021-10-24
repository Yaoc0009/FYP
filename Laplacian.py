import numpy as np
from scipy.io import loadmat
from scipy import sparse
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
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
                          n_jobs= 1)
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

def Laplacian2():
    n_neighbors = 10
    knn_dist_graph = kneighbors_graph(X=data,
                                    n_neighbors=n_neighbors,
                                    mode='distance',
                                    metric='euclidean',
                                    n_jobs=1)

    sigma = 1
    similarity_graph = sparse.csr_matrix(knn_dist_graph.shape)
    nonzeroindices = knn_dist_graph.nonzero()

    similarity_graph[nonzeroindices] = np.exp(-np.asarray(knn_dist_graph[nonzeroindices])**2 / 2.0 * sigma**2)

    similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)

    degree_matrix = similarity_graph.sum(axis=1)
    diagonal_matrix = np.diag(np.asarray(degree_matrix).reshape(len(data),))

    L =  diagonal_matrix - similarity_graph

if __name__ == "__main__":
    dataset = loadmat('coil20.mat')
    data = dataset['X']
    k = 2
    L = Laplacian(data, k)