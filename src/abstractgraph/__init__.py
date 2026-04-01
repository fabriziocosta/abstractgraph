"""Core AbstractGraph namespace."""

from abstractgraph.display import (
    display,
    display_decomposition_graph,
    display_graph,
    display_graphs,
    display_grouped_graphs,
    display_mappings,
)
from abstractgraph.artificial import ArtificialGraphDatasetConstructor, RandomGraphConstructor
from abstractgraph.graphs import (
    AbstractGraph,
    get_interpretation_label_to_mapped_subgraphs,
    get_mapped_subgraph,
    graph_to_abstract_graph,
    graphs_to_abstract_graphs,
    set_mapped_subgraph,
)
from abstractgraph.hashing import hash_bounded, hash_graph, hash_sequence, hash_set, hash_value
from abstractgraph.labels import (
    DEFAULT_NBITS,
    graph_hash_label_function_factory,
    graph_structure_hash_label_function_factory,
    intersection_edge_function,
    mean_attribute_function,
    name_hash_label_function_factory,
    null_edge_function,
    source_chain_hash_label_function_factory,
    source_function_hash_label_function_factory,
    sum_attribute_function,
)
from abstractgraph.operators import *  # noqa: F401,F403
from abstractgraph.vectorize import AbstractGraphNodeTransformer, AbstractGraphTransformer, vectorize
from abstractgraph.utils import *  # noqa: F401,F403
from abstractgraph.xml import *  # noqa: F401,F403

__all__ = [
    "AbstractGraph",
    "get_interpretation_label_to_mapped_subgraphs",
    "get_mapped_subgraph",
    "graph_to_abstract_graph",
    "graphs_to_abstract_graphs",
    "set_mapped_subgraph",
    "hash_graph",
    "hash_bounded",
    "hash_sequence",
    "hash_set",
    "hash_value",
    "DEFAULT_NBITS",
    "graph_hash_label_function_factory",
    "graph_structure_hash_label_function_factory",
    "name_hash_label_function_factory",
    "source_chain_hash_label_function_factory",
    "source_function_hash_label_function_factory",
    "sum_attribute_function",
    "mean_attribute_function",
    "null_edge_function",
    "intersection_edge_function",
    "vectorize",
    "AbstractGraphTransformer",
    "AbstractGraphNodeTransformer",
    "display",
    "display_graph",
    "display_graphs",
    "display_grouped_graphs",
    "display_mappings",
    "display_decomposition_graph",
    "RandomGraphConstructor",
    "ArtificialGraphDatasetConstructor",
    "remove_redundant_mapped_subgraphs",
    "number_of_interpretation_graph_nodes",
    "number_of_interpretation_graph_edges",
]
