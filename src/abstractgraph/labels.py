"""Default label, attribute, and edge functions for AbstractGraph."""

import networkx as nx
import numpy as np
from typing import Optional, Callable, Any, List, Iterable, Tuple, Dict, Set
from abstractgraph.hashing import hash_bounded, hash_graph


DEFAULT_NBITS = 14


def _get_mapped_subgraph(node_attrs: dict) -> Optional[nx.Graph]:
    mapped_subgraph = node_attrs.get("mapped_subgraph")
    if mapped_subgraph is not None:
        return mapped_subgraph
    return node_attrs.get("association")

#==========================================================================================
# Label functions for AbstractGraph
#==========================================================================================
def graph_hash_label_function_factory(nbits: int = DEFAULT_NBITS) -> Callable[[dict], int]:
    """
    Build a label function that hashes the mapped base subgraph.

    Args:
        nbits: The number of bits for the hash output (default: 14).

    Returns:
        Callable[[dict], int]: Label function mapping node attrs to an integer hash.
    """
    def label_fn(node_attrs: dict) -> int:
        subgraph = _get_mapped_subgraph(node_attrs)
        if subgraph is None:
            raise ValueError("Node attributes must contain a 'mapped_subgraph' key.")
        return hash_graph(subgraph, nbits=nbits)
    label_fn.nbits = nbits # Attach nbits as an attribute
    label_fn.label_mode = "graph_hash"
    return label_fn

def graph_structure_hash_label_function_factory(nbits: int = DEFAULT_NBITS) -> Callable[[dict], int]:
    """
    Build a label function that hashes only mapped-subgraph structure.

    Args:
        nbits: The number of bits for the hash output (default: 14).

    Returns:
        Callable[[dict], int]: Label function using structure-only hashing.
    """
    def label_fn(node_attrs: dict) -> int:
        subgraph = _get_mapped_subgraph(node_attrs)
        if subgraph is None:
            raise ValueError("Node attributes must contain a 'mapped_subgraph' key.")

        # Copy and sanitize node and edge labels
        structure_graph = subgraph.copy()
        for node in structure_graph.nodes:
            structure_graph.nodes[node]["label"] = "-"
        for u, v in structure_graph.edges:
            structure_graph.edges[u, v]["label"] = "-"

        return hash_graph(structure_graph, nbits=nbits)
    label_fn.nbits = nbits # Attach nbits as an attribute
    label_fn.label_mode = "graph_structure_hash"
    return label_fn



def source_function_hash_label_function_factory(nbits: int = DEFAULT_NBITS) -> Callable[[dict], int]:
    """
    Build a label function that hashes the source function metadata.

    Args:
        nbits: The number of bits to use for the hash output (e.g. 8 → 0-255).

    Returns:
        Callable[[dict], int]: Label function mapping metadata to a bounded hash.
    """
    def label_fn(node_attrs: dict) -> int:
        # Extract the source function identifier from metadata; default to 'unknown'
        source = node_attrs.get("meta", {}).get("source_function", "unknown")
        # Use stable, bounded hashing for reproducibility and column consistency
        return hash_bounded(source, nbits=nbits)
    label_fn.nbits = nbits # Attach nbits as an attribute
    label_fn.label_mode = "source_function_hash"
    return label_fn

def source_chain_hash_label_function_factory(nbits: int = DEFAULT_NBITS) -> Callable[[dict], int]:
    """
    Build a label function that hashes the full operator source chain.

    Args:
        nbits: The number of bits to use for the hash output (e.g. 8 → 0-255).

    Returns:
        Callable[[dict], int]: Label function mapping source_chain to a bounded hash.
    """
    def label_fn(node_attrs: dict) -> int:
        meta = node_attrs.get("meta", {})
        source_chain = meta.get("source_chain", None)
        if source_chain is None:
            raise ValueError("Node attributes must contain meta['source_chain'].")
        if not isinstance(source_chain, str):
            source_chain = str(source_chain)
        return hash_bounded(source_chain, nbits=nbits)
    label_fn.nbits = nbits # Attach nbits as an attribute
    label_fn.label_mode = "source_chain_hash"
    return label_fn

def name_hash_label_function_factory(nbits: int = DEFAULT_NBITS) -> Callable[[dict], int]:
    """
    Build a label function that hashes a user-defined interpretation-node name.

    Falls back to meta["source_chain"] so it can be used without the name operator.

    Args:
        nbits: The number of bits to use for the hash output (e.g. 8 → 0-255).

    Returns:
        Callable[[dict], int]: Label function mapping a name to a bounded hash.
    """
    def label_fn(node_attrs: dict) -> int:
        meta = node_attrs.get("meta", {})
        name = meta.get("user_name")
        if name is None:
            name = meta.get("source_chain")
        if name is None:
            raise ValueError("Node attributes must contain meta['user_name'] or meta['source_chain'].")
        if not isinstance(name, str):
            name = str(name)
        return hash_bounded(name, nbits=nbits)
    label_fn.nbits = nbits # Attach nbits as an attribute
    label_fn.label_mode = "operator_hash"
    return label_fn

#==================================================================================================
# Attribute functions for AbstractGraph
#==================================================================================================

def sum_attribute_function(subgraph: nx.Graph) -> np.ndarray:
    """
    Sum node attributes across a subgraph.

    Args:
        subgraph: The NetworkX subgraph from which to aggregate attributes.

    Returns:
        np.ndarray: Sum of node attributes (or ones if missing).
    """
    attr_list = [data.get('attribute', np.array([1.])) for _, data in subgraph.nodes(data=True)]
    return np.sum(attr_list, axis=0)
    

def mean_attribute_function(subgraph: nx.Graph) -> np.ndarray:
    """
    Compute the mean of node attributes across a subgraph.

    Args:
        subgraph: The NetworkX subgraph from which to aggregate attributes.

    Returns:
        np.ndarray: Mean of node attributes (or ones if missing).
    """
    attr_list = [data.get('attribute', np.array([1.])) for _, data in subgraph.nodes(data=True)]
    return np.mean(attr_list, axis=0)

#==================================================================================================
# Edge functions for AbstractGraph
#==================================================================================================

def intersection_edge_function(abstract_graph: "AbstractGraph") -> "AbstractGraph":
    """
    Add interpretation-graph edges for intersecting mapped subgraphs.

    Args:
        abstract_graph: The AbstractGraph instance to update with new edges.

    Returns:
        AbstractGraph: The updated AbstractGraph instance.
    """
    nodes = list(abstract_graph.interpretation_graph.nodes)
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i + 1 :]:
            subgraph1 = _get_mapped_subgraph(abstract_graph.interpretation_graph.nodes[node1])
            subgraph2 = _get_mapped_subgraph(abstract_graph.interpretation_graph.nodes[node2])
            shared_count = len(set(subgraph1.nodes) & set(subgraph2.nodes))
            if shared_count > 0:
                abstract_graph.interpretation_graph.add_edge(
                    node1,
                    node2,
                    shared_base_nodes=shared_count,
                    shared_preimage_nodes=shared_count,
                )
    return abstract_graph

def null_edge_function(abstract_graph: "AbstractGraph") -> "AbstractGraph":
    """
    No-op edge function.

    Args:
        abstract_graph: The AbstractGraph instance.

    Returns:
        AbstractGraph: The unchanged AbstractGraph instance.
    """
    return abstract_graph
