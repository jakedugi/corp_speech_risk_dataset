"""Tests for Weisfeiler-Lehman graph kernel features."""

import pytest
import networkx as nx
from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.corp_speech_risk_dataset.encoding.wl_features import (
    _pos_id,
    to_grakel_graph,
    wl_vector,
)
from src.corp_speech_risk_dataset.encoding.parser import to_dependency_graph


def test_pos_id_mapping():
    """Test POS tag to integer ID mapping."""
    # Same tag should get same ID
    assert _pos_id("NOUN") == _pos_id("NOUN")
    
    # Different tags should get different IDs
    noun_id = _pos_id("NOUN")
    verb_id = _pos_id("VERB")
    assert noun_id != verb_id
    
    # IDs should be integers
    assert isinstance(noun_id, int)
    assert isinstance(verb_id, int)


def test_grakel_graph_conversion():
    """Test NetworkX to GraKeL graph conversion."""
    # Create a simple dependency graph
    text = "The quick brown fox jumps."
    g_nx = to_dependency_graph(text)
    edges, labels = to_grakel_graph(g_nx)
    
    # Check node indices are contiguous 0...n-1
    n_nodes = len(g_nx.nodes())
    node_indices = set(labels.keys())
    assert node_indices == set(range(n_nodes))
    
    # Check edge indices are valid
    for u, v in edges:
        assert 0 <= u < n_nodes
        assert 0 <= v < n_nodes
    
    # Check labels are integers
    assert all(isinstance(label, int) for label in labels.values())


def test_wl_vector_shape():
    """Test WL feature vector shape and sparsity."""
    text = "The quick brown fox jumps over the lazy dog."
    vec = wl_vector(text)
    
    # Should be a 1 × n sparse matrix
    assert isinstance(vec, sparse.csr_matrix)
    assert vec.shape[0] == 1
    
    # No stray indices
    if vec.nnz > 0:  # only check if non-empty
        assert vec.shape[1] == vec.indices.max() + 1


def test_wl_vector_deterministic():
    """Test WL features are deterministic for same input."""
    text = "Corporate governance requirements."
    
    # Multiple runs should give identical features
    vec1 = wl_vector(text)
    vec2 = wl_vector(text)
    assert np.array_equal(vec1.toarray(), vec2.toarray())


def test_wl_vector_different():
    """Test WL features differ for different inputs."""
    # Test pairs of similar but different sentences
    test_pairs = [
        (
            "The quick brown fox jumps over the lazy dog.",
            "The lazy brown dog sleeps under the tall tree."
        ),
        (
            "Corporate governance requirements.",
            "Corporate finance regulations."
        ),
        (
            "SEC Form 10-K filing.",
            "SEC Form 8-K report."
        )
    ]
    
    for text1, text2 in test_pairs:
        vec1 = wl_vector(text1)
        vec2 = wl_vector(text2)
        
        # Convert to dense for cosine similarity
        dense1 = vec1.toarray()
        dense2 = vec2.toarray()
        
        # Compute cosine similarity
        sim = cosine_similarity(dense1, dense2)[0, 0]
        
        # Should be different but potentially similar
        assert sim < 1.0, f"Vectors for '{text1}' and '{text2}' are identical"
        
        # For debugging
        if sim > 0.99:
            print(f"Warning: Very high similarity ({sim:.3f}) between:")
            print(f"  Text 1: {text1}")
            print(f"  Text 2: {text2}")


@pytest.mark.parametrize("text", [
    "Simple short text.",
    "A longer sentence with more complex structure and multiple clauses.",
    "§12(b) compliance requirements for corporate governance.",
    "First sentence. Second sentence.",  # Multiple sentences
    "",  # Empty text
    "   ",  # Just whitespace
])
def test_wl_vector_robustness(text):
    """Test WL feature extraction is robust to various inputs."""
    try:
        vec = wl_vector(text)
        assert isinstance(vec, sparse.csr_matrix)
        assert vec.shape[0] == 1
        if text.strip():  # non-empty text should have features
            assert vec.shape[1] > 0
        else:  # empty text gets 1×1 zero matrix
            assert vec.shape == (1, 1)
    except Exception as e:
        pytest.fail(f"Failed on text: {text!r}\nError: {str(e)}")


def test_empty_graph_handling():
    """Test handling of empty graphs and edge cases."""
    # Empty graph
    g = nx.DiGraph()
    edges, labels = to_grakel_graph(g)
    assert edges == []
    assert labels == {}
    
    # Empty text
    vec = wl_vector("")
    assert vec.shape == (1, 1)
    assert vec.nnz == 0  # should be all zeros
    
    # Whitespace
    vec = wl_vector("   \n\t   ")
    assert vec.shape == (1, 1)
    assert vec.nnz == 0


def test_wl_vector_similarity():
    """Test that similar sentences have higher similarity than different ones."""
    # Similar pairs (should have high similarity)
    similar_pairs = [
        (
            "The company filed SEC Form 10-K.",
            "The company submitted SEC Form 10-K."
        ),
        (
            "Board approved the governance policy.",
            "Board accepted the governance policy."
        )
    ]
    
    # Different pairs (should have lower similarity)
    different_pairs = [
        (
            "The company filed SEC Form 10-K.",
            "Revenue increased by 15% this quarter."
        ),
        (
            "Board approved the governance policy.",
            "Shareholders received annual dividends."
        )
    ]
    
    def get_similarity(text1: str, text2: str) -> float:
        vec1 = wl_vector(text1).toarray()
        vec2 = wl_vector(text2).toarray()
        return cosine_similarity(vec1, vec2)[0, 0]
    
    # Similar pairs should have higher similarity
    similar_sims = [get_similarity(t1, t2) for t1, t2 in similar_pairs]
    different_sims = [get_similarity(t1, t2) for t1, t2 in different_pairs]
    
    avg_similar = np.mean(similar_sims)
    avg_different = np.mean(different_sims)
    
    assert avg_similar > avg_different, (
        f"Similar pairs ({avg_similar:.3f}) should have higher "
        f"similarity than different pairs ({avg_different:.3f})"
    ) 