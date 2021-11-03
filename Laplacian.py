import numpy as np
from scipy import sparse
from scipy.sparse import spdiags, eye
from sklearn.neighbors import kneighbors_graph

# calculates normalized Laplacian
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