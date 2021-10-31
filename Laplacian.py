import numpy as np
from scipy.io import loadmat
from scipy import sparse
from scipy.sparse import spdiags, eye
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

def adjacency(data, n_neighbors, sigma):
    W = kneighbors_graph(data, n_neighbors, mode='distance', include_self=False)
    W = W.maximum(W.T)
    W = sparse.csr_matrix((np.exp(-W.data**2 / 2 / sigma**2), W.indices, W. indptr), shape=(len(data), len(data)))
    return W.A

def laplacian(X_lab, X_unlab, n_neighbours, sigma):
    data = np.vstack([X_lab, X_unlab])
    W = adjacency(data, n_neighbours, sigma)
    D = np.sum(W, axis=1)
    D[D != 0] = np.sqrt(1 / D[D != 0])
    D = spdiags(D, 0, len(W), len(W))
    W = D @ W @ D
    L = eye(len(W)) - W # L = I - D^-1/2*W*D^-1/2
    L = L.A
    return L

if __name__ == "__main__":
    dataset = loadmat('coil20.mat')
    data = dataset['X']
    k = 2
    L = Laplacian(data, k)