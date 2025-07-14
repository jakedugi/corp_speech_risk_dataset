import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import linear_kernel

def cosine_sparse(a: csr_matrix, b: csr_matrix) -> float:
    """
    Compute the cosine similarity between two sparse vectors.
    """
    num   = a.multiply(b).sum()
    denom = np.sqrt(a.multiply(a).sum()) * np.sqrt(b.multiply(b).sum())
    return float(num / denom)

def kernel_kmeans(K: np.ndarray, k=20, max_iter=100):
    """
    Kernel k-means clustering using a precomputed kernel matrix K.
    Returns cluster assignments for each row.
    """
    from sklearn.cluster import KMeans
    n = K.shape[0]
    H = np.random.randn(n, k)
    for _ in range(max_iter):
        D = np.diag(K)[:, None] - 2*K.dot(H) + np.sum(H*H, axis=0)
        y = D.argmin(axis=1)
        H = np.eye(k)[y]
    return y
