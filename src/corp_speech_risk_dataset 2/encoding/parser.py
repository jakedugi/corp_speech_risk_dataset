import spacy
import networkx as nx
from typing import Any

_nlp = spacy.load("en_core_web_sm")

def to_dependency_graph(text: str) -> nx.DiGraph:
    """
    Parse text with spaCy and return a NetworkX DiGraph representing the dependency tree.
    Each node has 'pos' and 'text' as attributes; each edge has dependency label.
    """
    doc = _nlp(text)
    g = nx.DiGraph()
    for token in doc:
        # Store both POS tag and original text for richer labeling
        g.add_node(token.i, pos=token.pos_, text=token.text)
        if token.head.i != token.i:
            g.add_edge(token.head.i, token.i, dep=token.dep_)
    return g
