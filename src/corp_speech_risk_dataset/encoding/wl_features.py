import networkx as nx
from grakel import GraphKernel
from .parser import to_dependency_graph

# 1-hop WL counts are usually enough; set h=2 for extra context
_wl = GraphKernel(kernel={'name': 'weisfeiler_lehman', 'n_iter': 1}, normalize=True)

def to_grakel_graph(nx_g: nx.DiGraph):
    """
    Convert a NetworkX DiGraph to GraKeL's (edges, labels) format.
    GraKeL expects undirected graphs with integer labels.
    """
    labels = {n: nx_g.nodes[n]['pos'] for n in nx_g.nodes}
    edges  = [(u, v) for u, v in nx_g.edges]
    return (edges, labels)

def wl_vector(text: str):
    """
    Compute the Weisfeiler-Lehman subtree feature vector for the given text.
    Returns a sparse SciPy row vector (CSR matrix).
    """
    g_nx = to_dependency_graph(text)
    g_gk = to_grakel_graph(g_nx)
    dense = _wl.fit_transform([g_gk])[0]           # 1-D NumPy
    from scipy import sparse
    return sparse.csr_matrix(dense)                # 1 × n CSR row 