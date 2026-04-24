"""
Deterministic hashing utilities for graphs and general Python values.

Goals
- Deterministic across runs and platforms (no use of Python's built-in hash).
- Order handling by intent: provide both order-aware (sequences) and
  order-independent (multisets) combinators.
- Compact, bounded indices for downstream feature maps (via nbits).

Core building blocks
- canonicalize / canonical_bytes: Convert arbitrary values into a stable,
  type-tagged, JSON-serializable form. Handles scalars, containers, numpy arrays,
  bytes, and dicts (sorted by canonicalized key). This prevents collisions like 1 vs "1" and
  avoids platform-dependent stringification.
- hash_value: SHA-256 over canonical_bytes(value) → big integer (base 16).
- hash_sequence: Order-aware combiner for lists/tuples.
- hash_set: Order-independent combiner for multisets by hashing elements first
  and sorting the resulting integers before combining.
- hash_bounded: Map a large hash to a fixed bit-width with a reserved low range
  (returns in [2, 2**nbits - 1]), leaving 0/1 available for special features.

Graph hashing (label-aware)
- hash_node: Hash of a node’s own label combined with an order-independent
  multiset of (neighbor label, edge label) pairs. Uses degree==0 to detect
  isolated nodes.
- hash_rooted_graph: For each root, perform BFS to group nodes by distance and
  combine their node hashes per layer; then combine layers in order to form a
  rooted-subgraph hash.
- canonical_dfs_graph_signature: Build a relabeling-invariant DFS certificate
  on top of rooted hashes. Candidate traversal order is driven by node hashes,
  edge labels, and direction tags; tied candidates are enumerated and the
  lexicographically smallest certificate is kept.
- hash_graph: Aggregate information across the whole graph by combining rooted
  subgraph hashes along edges with edge labels, a canonical DFS certificate,
  plus a multiset hash of all node labels. Finally bound to nbits using
  hash_bounded.

Notes and limitations
- Only the 'label' attribute is used for nodes/edges. For MultiGraph/DiGraph,
  adapt edge iteration to handle keys/direction explicitly.
- canonicalize does not special-case NetworkX graphs; use `hash_graph` directly
  when you intend to hash a graph object.
- Complexity: hashing all rooted subgraphs is roughly O(V·(V+E)); the canonical
  DFS certificate can add branching in tied neighborhoods. For large symmetric
  graphs, consider caching, limiting radius, or alternative summaries.
- Collisions: Cryptographic hashing minimizes but does not eliminate collisions;
  reducing to nbits increases collision risk. Choose nbits to balance size and
  collision tolerance.
"""

import hashlib
import json
import math
from base64 import b64encode
from collections import defaultdict
from typing import Any, Optional, List, Tuple, Dict, Union, Set
import numpy as np
import networkx as nx
try:
    import multiprocessing_on_dill as mp
except Exception:
    import multiprocessing as mp


def _edge_orientation_tag(graph: nx.Graph, u: Any, v: Any) -> str:
    """Return a stable orientation tag for one incident edge."""
    if not nx.is_directed(graph):
        return "undirected"
    return "out"


def _iter_incident_edge_payloads(graph: nx.Graph, node_idx: Any) -> List[tuple[str, Any, Any]]:
    """Return incident edge payloads with explicit direction tags."""
    payloads: List[tuple[str, Any, Any]] = []
    if nx.is_directed(graph):
        for neighbor_idx in graph.successors(node_idx):
            payloads.append(
                (
                    "out",
                    graph.nodes[neighbor_idx].get("label", ""),
                    graph.edges[node_idx, neighbor_idx].get("label", ""),
                )
            )
        for neighbor_idx in graph.predecessors(node_idx):
            payloads.append(
                (
                    "in",
                    graph.nodes[neighbor_idx].get("label", ""),
                    graph.edges[neighbor_idx, node_idx].get("label", ""),
                )
            )
        return payloads

    for neighbor_idx in graph.neighbors(node_idx):
        payloads.append(
            (
                "undirected",
                graph.nodes[neighbor_idx].get("label", ""),
                graph.edges[node_idx, neighbor_idx].get("label", ""),
            )
        )
    return payloads


def canonicalize(value: Any) -> Any:
    """
    Produce a deterministic, JSON-serializable, type-tagged representation of `value`.

    Ensures stability across Python versions and platforms, preserves ordering for
    ordered containers, and sorts unordered containers. Prevents collisions across
    differing types (e.g., 1 vs "1").

    Args:
        value: Arbitrary Python value to canonicalize.

    Returns:
        Any: Canonical, JSON-serializable representation.
    """
    # None and booleans
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", int(value))

    # Numeric types (integers, floats)
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            return ("float", "NaN")
        if math.isinf(x):
            return ("float", "Inf" if x > 0 else "-Inf")
        # repr maintains precision for typical floats
        return ("float", repr(x))

    # Strings and bytes-like
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("bytes", b64encode(bytes(value)).decode("ascii"))

    # NumPy arrays
    if isinstance(value, np.ndarray):
        # Encode dtype, shape, and raw bytes to avoid large JSON lists
        return ("ndarray", str(value.dtype), tuple(value.shape), b64encode(value.tobytes()).decode("ascii"))

    # Containers
    if isinstance(value, (list, tuple)):
        tag = "list" if isinstance(value, list) else "tuple"
        return (tag, [canonicalize(v) for v in value])
    if isinstance(value, (set, frozenset)):
        elems = [canonicalize(v) for v in value]
        elems.sort()
        tag = "set" if isinstance(value, set) else "frozenset"
        return (tag, elems)
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            k_canon = canonicalize(k)
            # Use JSON string of canonical key to ensure a stable, comparable key
            k_str = json.dumps(k_canon, separators=(",", ":"), ensure_ascii=False)
            items.append((k_str, canonicalize(v)))
        items.sort(key=lambda kv: kv[0])
        return ("dict", items)

    # Fallback: type name + repr
    return ("object", type(value).__name__, repr(value))


def canonical_bytes(value: Any) -> bytes:
    """
    Serialize a canonicalized value to UTF-8 JSON bytes.

    Args:
        value: Arbitrary Python value to serialize.

    Returns:
        bytes: UTF-8 JSON representation of the canonicalized value.
    """
    return json.dumps(canonicalize(value), separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hash_value(value: Any) -> int:
    """
    Generates a consistent hash for a given value using SHA-256 over a
    canonical, type-tagged JSON serialization.

    Args:
        value (Any): The value to hash.

    Returns:
        int: The integer representation of the SHA-256 hash.
    """
    sha256_hash = hashlib.sha256(canonical_bytes(value)).hexdigest()
    return int(sha256_hash, 16)


def hash_sequence(iterable: List[Any]) -> int:
    """
    Hashes a list by converting it to a tuple and then applying a consistent hash.

    Args:
        iterable (List[Any]): The list to hash.

    Returns:
        int: The consistent hash value of the tuple.
    """
    # Convert the list to a tuple to make it immutable and hashable
    tuple_representation = tuple(iterable)
    # Generate a consistent hash for the tuple
    return hash_value(tuple_representation)


def hash_set(iterable: List[Any]) -> int:
    """
    Hash a multiset order-independently by hashing elements first and sorting the hashes.

    Args:
        iterable (List[Any]): Elements to hash order-independently.

    Returns:
        int: Consistent hash of the multiset contents.
    """
    hashed = sorted(hash_value(e) for e in iterable)
    return hash_value(tuple(hashed))


def hash_bounded(value: Any, nbits: int = 10) -> int:
    """
    Hashes a value, applies a bitmask to limit its size, and ensures the result is within a specified range.

    This function combines the functionality of masking and range limitation into a single step.
    It first generates a consistent hash of the input value, applies a bitmask based on `nbits` to limit
    the hash size, and then adjusts the result to ensure it falls within the range [2, 2**nbits - 1].

    Args:
        value (Any): The value to hash.
        nbits (int, optional): Number of bits to limit the hash. Defaults to 10.

    Returns:
        int: The hashed value limited to `nbits`, and in the range [2, 2**nbits - 1]

    Raises:
        ValueError: If `nbits` is less than 2.
    """
    if nbits < 2:
        raise ValueError("nbits must be at least 2 to ensure a valid hash range.")

    # Calculate the maximum value based on the number of bits
    max_index = 2 ** nbits
    # Create a bitmask with the specified number of bits (e.g., nbits=10 -> mask=0b1111111111)
    mask = (1 << nbits) - 1
    # Apply the bitmask to the consistent hash using bitwise AND to limit the hash size
    masked_hash = hash_value(value) & mask
    # Use modulo to ensure the masked hash is within [0, max_index - 3]
    limited_hash = masked_hash % (max_index - 2)
    # Adjust the hash to ensure it is at least 2
    final_hash = limited_hash + 2
    return final_hash

def hash_node(node_idx: int, graph: nx.Graph) -> int:
    """
    Computes a hash for a node based on its label and the labels of its neighbors.

    Args:
        node_idx (int): The index of the node in the graph.
        graph (nx.Graph): The graph containing the node.

    Returns:
        int: The computed hash for the node.
    """
    # Retrieve the label of the current node.
    # If the node doesn't have a 'label' attribute, default to an empty string.
    node_label = graph.nodes[node_idx].get('label', '')
    # Generate a consistent hash for the node's label using the hash_value function.
    node_label_hash = hash_value(node_label)
    
    # Fast check for isolated node: degree 0
    if graph.degree[node_idx] == 0:
        return node_label_hash
    
    # Initialize a list to store hashes related to the node's neighborhood.
    neighborhood_hashes = []
    
    # Track edge direction explicitly so reversed arcs hash differently.
    for direction_tag, neighbor_label, edge_label in _iter_incident_edge_payloads(graph, node_idx):
        neighborhood_hashes.append(
            hash_sequence(
                [
                    hash_value(direction_tag),
                    hash_value(neighbor_label),
                    hash_value(edge_label),
                ]
            )
        )
    
    # After processing all neighbors, combine all neighborhood-related hashes.
    # The hash_set function ensures that the combination is order-independent.
    neighborhood_hash = hash_set(neighborhood_hashes)
    
    # Create a sequence containing the neighborhood hash and the node's own label hash.
    combined_sequence = [neighborhood_hash, node_label_hash]
    # Generate the final hash for the node by hashing the combined sequence.
    node_hash = hash_sequence(combined_sequence)
    
    # Return the computed hash for the node.
    return node_hash


def compute_node_labels_set_hash(graph: nx.Graph) -> int:
    """
    Computes a hash value based on the set of node labels in the graph.
    
    The function retrieves the 'label' attribute from each node (using '-' as a default),
    sorts them to ensure order-independence, and then computes the hash using hash_set.
    
    Args:
        graph (nx.Graph): A NetworkX graph with nodes that may have a 'label' attribute.
    
    Returns:
        int: The hash value representing the set of node labels in the graph.
    """
    labels = []
    for node in graph.nodes():
        node_label = graph.nodes[node].get('label', '-')
        labels.append(node_label)
    return hash_set(labels)


def hash_rooted_graph(node_idx: int, graph: nx.Graph, node_hash_dict: Dict[int, int]) -> int:
    """
    Computes a hash for the subgraph rooted at a given node, considering all reachable nodes.

    Args:
        node_idx (int): The index of the root node.
        graph (nx.Graph): The graph containing the node.
        node_hash_dict (Dict[int, int]): Auxiliary dictionary containing precomputed hashes for each node.

    Returns:
        int: The hash representing the rooted subgraph.
    """
    def invert_dict(d: Dict[Any, Any]) -> Dict[Any, List[Any]]:
        """
        Inverts a dictionary mapping keys to single values into a dictionary mapping values to lists of keys.

        Args:
            d (Dict[Any, Any]): The dictionary to invert.

        Returns:
            Dict[Any, List[Any]]: The inverted dictionary where each key maps to a list of original keys.
        """
        # Initialize a defaultdict of lists to collect keys for each value
        inverted = defaultdict(list)
        # Iterate over each key-value pair in the original dictionary
        for key, value in d.items():
            # Append the key to the list corresponding to the value in the inverted dictionary
            inverted[value].append(key)
        # Convert defaultdict back to a regular dict before returning
        return dict(inverted)
    # Find all nodes reachable from the root node along with their shortest path lengths
    node_distance_dict = nx.single_source_shortest_path_length(graph, node_idx)

    # Invert the dictionary to group nodes by their distance from the root
    distance_to_nodes = invert_dict(node_distance_dict)

    # Initialize a list to store hash codes for each distance level
    distance_hashes = []
    # Iterate over each distance level in sorted order
    for distance, nodes_at_distance in sorted(distance_to_nodes.items()):
        # Collect the precomputed hashes of all nodes at the current distance
        node_hashes = [node_hash_dict[node] for node in nodes_at_distance]
        # Sort the node hashes to ensure consistent ordering
        sorted_node_hashes = sorted(node_hashes)
        # Hash the sorted list of node hashes to get a combined hash for this distance level
        distance_combined_hash = hash_sequence(sorted_node_hashes)
        # Append the combined hash to the list
        distance_hashes.append(distance_combined_hash)

    # Hash the list of distance-based hashes to obtain the final subgraph hash
    subgraph_hash = hash_sequence(distance_hashes)
    return subgraph_hash


def _iter_traversal_edges(graph: nx.Graph, node_idx: Any) -> List[Tuple[str, Any, Any]]:
    """Return incident traversal edges as (direction, neighbor, edge_label)."""
    if nx.is_directed(graph):
        edges: List[Tuple[str, Any, Any]] = []
        for neighbor_idx in graph.successors(node_idx):
            edges.append(("out", neighbor_idx, graph.edges[node_idx, neighbor_idx].get("label", "")))
        for neighbor_idx in graph.predecessors(node_idx):
            edges.append(("in", neighbor_idx, graph.edges[neighbor_idx, node_idx].get("label", "")))
        return edges

    return [
        ("undirected", neighbor_idx, graph.edges[node_idx, neighbor_idx].get("label", ""))
        for neighbor_idx in graph.neighbors(node_idx)
    ]


def _traversal_edge_key(direction_tag: str, edge_label: Any, neighbor_hash: int) -> Tuple[Any, ...]:
    """Return a relabeling-invariant ordering key for a traversal edge."""
    return (direction_tag, canonicalize(edge_label), neighbor_hash)


def _canonical_dfs_signature_from_root(
    graph: nx.Graph,
    root: Any,
    node_hash_dict: Dict[Any, int],
) -> Tuple[Any, ...]:
    """Build a canonical DFS signature, enumerating only tied next-candidate choices.

    Candidate ordering uses the already-computed node hashes, edge labels, and
    direction tags. When multiple unvisited neighbors share the same key, each
    tied choice is explored and the lexicographically smallest resulting
    certificate is kept. Node ids are used only as dictionary keys, never as
    ordering tie-breakers, so the signature remains stable under relabeling.
    """

    def visit(
        node_idx: Any,
        visited: Set[Any],
        discovery_index: Dict[Any, int],
    ) -> Tuple[Tuple[Any, ...], Set[Any], Dict[Any, int]]:
        visited_edges = []
        for direction_tag, neighbor_idx, edge_label in _iter_traversal_edges(graph, node_idx):
            if neighbor_idx in discovery_index:
                visited_edges.append(
                    (
                        direction_tag,
                        canonicalize(edge_label),
                        discovery_index[neighbor_idx],
                    )
                )
        visited_edges.sort()

        suffix, visited, discovery_index = process_node(node_idx, visited, discovery_index)
        signature = (
            ("node", node_hash_dict[node_idx]),
            ("visited_edges", tuple(visited_edges)),
            *suffix,
        )
        return signature, visited, discovery_index

    def process_node(
        node_idx: Any,
        visited: Set[Any],
        discovery_index: Dict[Any, int],
    ) -> Tuple[Tuple[Any, ...], Set[Any], Dict[Any, int]]:
        candidates = []
        for direction_tag, neighbor_idx, edge_label in _iter_traversal_edges(graph, node_idx):
            if neighbor_idx in visited:
                continue
            key = _traversal_edge_key(direction_tag, edge_label, node_hash_dict[neighbor_idx])
            candidates.append((key, neighbor_idx))

        if not candidates:
            return (), visited, discovery_index

        min_key = min(key for key, _ in candidates)
        tied_next_nodes = []
        seen_next_nodes = set()
        for key, neighbor_idx in candidates:
            if key == min_key and neighbor_idx not in seen_next_nodes:
                tied_next_nodes.append(neighbor_idx)
                seen_next_nodes.add(neighbor_idx)

        best_branch = None
        best_visited = None
        best_discovery_index = None
        for neighbor_idx in tied_next_nodes:
            next_visited = set(visited)
            next_discovery_index = dict(discovery_index)
            next_visited.add(neighbor_idx)
            next_discovery_index[neighbor_idx] = len(next_discovery_index)

            child_signature, child_visited, child_discovery_index = visit(
                neighbor_idx,
                next_visited,
                next_discovery_index,
            )
            rest_signature, final_visited, final_discovery_index = process_node(
                node_idx,
                child_visited,
                child_discovery_index,
            )
            branch = (("tree_edge", min_key, child_signature), *rest_signature)
            if best_branch is None or branch < best_branch:
                best_branch = branch
                best_visited = final_visited
                best_discovery_index = final_discovery_index

        return best_branch or (), best_visited or visited, best_discovery_index or discovery_index

    def process_remaining_components(
        visited: Set[Any],
        discovery_index: Dict[Any, int],
    ) -> Tuple[Tuple[Any, ...], Set[Any], Dict[Any, int]]:
        remaining = [node_idx for node_idx in graph.nodes() if node_idx not in visited]
        if not remaining:
            return (), visited, discovery_index

        min_hash = min(node_hash_dict[node_idx] for node_idx in remaining)
        tied_roots = [node_idx for node_idx in remaining if node_hash_dict[node_idx] == min_hash]

        best_branch = None
        best_visited = None
        best_discovery_index = None
        for next_root in tied_roots:
            next_visited = set(visited)
            next_discovery_index = dict(discovery_index)
            next_visited.add(next_root)
            next_discovery_index[next_root] = len(next_discovery_index)

            component_signature, component_visited, component_discovery_index = visit(
                next_root,
                next_visited,
                next_discovery_index,
            )
            rest_signature, final_visited, final_discovery_index = process_remaining_components(
                component_visited,
                component_discovery_index,
            )
            branch = (("component", component_signature), *rest_signature)
            if best_branch is None or branch < best_branch:
                best_branch = branch
                best_visited = final_visited
                best_discovery_index = final_discovery_index

        return best_branch or (), best_visited or visited, best_discovery_index or discovery_index

    initial_visited = {root}
    initial_discovery_index = {root: 0}
    root_signature, visited_after_root, discovery_after_root = visit(
        root,
        initial_visited,
        initial_discovery_index,
    )
    remaining_signature, _, _ = process_remaining_components(visited_after_root, discovery_after_root)
    return (
        "canonical_dfs",
        "directed" if nx.is_directed(graph) else "undirected",
        ("root", root_signature),
        *remaining_signature,
    )


def canonical_dfs_graph_signature(graph: nx.Graph, node_hash_dict: Dict[Any, int]) -> Tuple[Any, ...]:
    """Return a relabeling-invariant DFS certificate for the whole graph."""
    rooted_signatures = [
        _canonical_dfs_signature_from_root(graph, node_idx, node_hash_dict)
        for node_idx in graph.nodes()
    ]
    return tuple(sorted(rooted_signatures))


def hash_graph(graph: nx.Graph, nbits: int = 19) -> int:
    """
    Computes a hash for the entire graph by hashing all rooted subgraphs.

    Args:
        graph (nx.Graph): The graph to hash.
        nbits (int, optional): Number of bits to limit the final hash. Defaults to 19.

    Returns:
        int: The hash representing the entire graph.

    Raises:
        ValueError: If `nbits` is not a positive integer.
    """
    # Validate that nbits is a positive integer
    if not isinstance(nbits, int) or nbits <= 0:
        raise ValueError("nbits must be a positive integer")

    # Create a dictionary mapping each node to its computed hash
    node_hashes: Dict[int, int] = {node: hash_node(node, graph) for node in graph.nodes()}

    # Create a dictionary mapping each node to the hash of its rooted subgraph
    rooted_subgraph_hashes: Dict[int, int] = {
        node: hash_rooted_graph(node, graph, node_hashes) for node in graph.nodes()
    }

    # Initialize a list to store hashes of all edges in the graph
    hashes_list = []
    node_labels_set_hash = compute_node_labels_set_hash(graph)
    hashes_list.append(node_labels_set_hash)
    hashes_list.append(hash_value("directed" if nx.is_directed(graph) else "undirected"))
    hashes_list.append(hash_value(canonical_dfs_graph_signature(graph, rooted_subgraph_hashes)))
    
    # Iterate over each edge in the graph
    for u, v in graph.edges():
        # Retrieve and hash the edge's label
        edge_label = graph.edges[u, v].get('label', '')
        edge_label_hash = hash_value(edge_label)
        if nx.is_directed(graph):
            combined_edge_hash = hash_sequence(
                [
                    hash_value("directed_edge"),
                    rooted_subgraph_hashes[u],
                    rooted_subgraph_hashes[v],
                    edge_label_hash,
                ]
            )
        else:
            combined_node_hashes = hash_set([rooted_subgraph_hashes[u], rooted_subgraph_hashes[v]])
            combined_edge_hash = hash_sequence(
                [hash_value("undirected_edge"), combined_node_hashes, edge_label_hash]
            )
        # Append the combined edge hash to the list
        hashes_list.append(combined_edge_hash)

    # Hash the set of all edge hashes to obtain the final graph hash, bounded by nbits
    final_graph_hash = hash_bounded(hash_set(hashes_list), nbits=nbits)
    return final_graph_hash


def _parallel_map(func, items, processes):
    """
    Map a function over items using a multiprocessing pool.

    Args:
        func: Callable to apply.
        items: Iterable of items to process.
        processes: Optional process count override.

    Returns:
        list: List of mapped results.
    """
    if not items:
        return []
    if processes is None:
        processes = mp.cpu_count()
    with mp.Pool(processes) as pool:
        return pool.map(func, items)


def _hash_all(graphs, hash_func, parallel, processes):
    """
    Hash a sequence of graphs with optional parallelism.

    Args:
        graphs: Iterable of graphs to hash.
        hash_func: Hashing function to apply.
        parallel: Whether to use multiprocessing.
        processes: Optional process count override.

    Returns:
        list: Hash values in input order.
    """
    if parallel:
        return _parallel_map(hash_func, graphs, processes)
    return [hash_func(graph) for graph in graphs]


def _build_index(keys, values):
    """
    Build a dictionary from keys and values.

    Args:
        keys: Iterable of keys.
        values: Iterable of values.

    Returns:
        dict: Mapping of keys to values.
    """
    return {key: value for key, value in zip(keys, values)}


def _check_len(first, second, name_first, name_second):
    """
    Validate that two sequences have the same length.

    Args:
        first: First sequence.
        second: Second sequence.
        name_first: Name to use for error reporting.
        name_second: Name to use for error reporting.

    Returns:
        None.
    """
    if len(first) != len(second):
        raise ValueError(f"{name_first} and {name_second} must have the same length")


class GraphHashDeduper(object):
    """Deduplicate graphs using a hash function."""

    def __init__(self, hash_func=hash_graph, parallel=True, processes=None):
        """
        Initialize the deduper.

        Args:
            hash_func: Hash function to use.
            parallel: Whether to use multiprocessing.
            processes: Optional process count override.

        Returns:
            None.
        """
        self.parallel = parallel
        self.hash_func = hash_func
        self.processes = processes

    def __repr__(self):
        """
        Return a concise representation of the deduper.

        Args:
            None.

        Returns:
            str: String representation with configuration values.
        """
        infos = ['%s:%s' % (key, value) for key, value in self.__dict__.items()]
        infos = ', '.join(infos)
        return '%s(%s)' % (self.__class__.__name__, infos)

    def fit(self, graphs):
        """
        Build an index over the provided graphs.

        Args:
            graphs: Iterable of graphs to index.

        Returns:
            GraphHashDeduper: Fitted instance.
        """
        self.index = self.build_index(graphs)
        return self

    def build_index(self, graphs):
        """
        Create a hash-to-graph index.

        Args:
            graphs: Iterable of graphs to index.

        Returns:
            dict: Mapping of hash values to graphs.
        """
        hashes = _hash_all(graphs, self.hash_func, self.parallel, self.processes)
        return _build_index(hashes, graphs)

    def filter(self, graphs):
        """
        Filter graphs not present in the fitted index.

        Args:
            graphs: Iterable of graphs to filter.

        Returns:
            list: Graphs whose hashes are not in the index.
        """
        new_index = self.build_index(graphs)
        unique_graphs = [
            new_index[hash_key]
            for hash_key in new_index
            if hash_key not in self.index
        ]
        return unique_graphs

    def fit_filter(self, graphs):
        """
        Fit the index and return unique graphs.

        Args:
            graphs: Iterable of graphs to index.

        Returns:
            list: Unique graphs after hashing.
        """
        self.index = self.build_index(graphs)
        return list(self.index.values())


def zip_lists(*lists):
    """
    Zip multiple lists into a list of tuples.

    Args:
        *lists: Lists to zip.

    Returns:
        list: Zipped tuples.
    """
    return list(zip(*lists))


def unzip_tuples(items, count=None):
    """
    Unzip tuples into separate lists.

    Args:
        items: Iterable of tuples to unzip.
        count: Optional expected number of output lists.

    Returns:
        list: Lists grouped by tuple position.
    """
    if not items:
        if count is None:
            return []
        return [[] for _ in range(count)]
    return [list(group) for group in zip(*items)]


class GraphHashAuxDeduper(object):
    """Deduplicate graphs while carrying auxiliary data."""

    def __init__(self, hash_func=hash_graph, parallel=True, processes=None):
        """
        Initialize the deduper with auxiliary data handling.

        Args:
            hash_func: Hash function to use.
            parallel: Whether to use multiprocessing.
            processes: Optional process count override.

        Returns:
            None.
        """
        self.parallel = parallel
        self.hash_func = hash_func
        self.processes = processes

    def __repr__(self):
        """
        Return a concise representation of the deduper.

        Args:
            None.

        Returns:
            str: String representation with configuration values.
        """
        infos = ['%s:%s' % (key, value) for key, value in self.__dict__.items()]
        infos = ', '.join(infos)
        return '%s(%s)' % (self.__class__.__name__, infos)

    def fit(self, graphs, auxiliary_infos):
        """
        Build an index over graphs and auxiliary infos.

        Args:
            graphs: Iterable of graphs to index.
            auxiliary_infos: Iterable of auxiliary data aligned with graphs.

        Returns:
            GraphHashAuxDeduper: Fitted instance.
        """
        self.index = self.build_index(graphs, auxiliary_infos)
        return self

    def build_index(self, graphs, auxiliary_infos=None):
        """
        Create a hash-to-(graph, aux) index.

        Args:
            graphs: Iterable of graphs to index.
            auxiliary_infos: Iterable of auxiliary data aligned with graphs.

        Returns:
            dict: Mapping of hash values to (graph, aux) tuples.
        """
        if auxiliary_infos is None:
            pairs = graphs
            graphs, auxiliary_infos = unzip_tuples(pairs, count=2)
        _check_len(graphs, auxiliary_infos, "graphs", "auxiliary_infos")
        hashes = _hash_all(graphs, self.hash_func, self.parallel, self.processes)
        pairs = zip_lists(graphs, auxiliary_infos)
        return _build_index(hashes, pairs)

    def filter(self, graphs, auxiliary_infos):
        """
        Filter graph/aux pairs not present in the fitted index.

        Args:
            graphs: Iterable of graphs to filter.
            auxiliary_infos: Iterable of auxiliary data aligned with graphs.

        Returns:
            Tuple[list, list]: Unique graphs and aligned auxiliary infos.
        """
        new_index = self.build_index(graphs, auxiliary_infos)
        unique_data = [
            new_index[hash_key]
            for hash_key in new_index
            if hash_key not in self.index
        ]
        if not unique_data:
            return [], []
        unique_graphs, unique_auxiliary_infos = unzip_tuples(unique_data, count=2)
        return unique_graphs, unique_auxiliary_infos

    def fit_filter(self, graphs, auxiliary_infos):
        """
        Fit the index and return unique graph/aux pairs.

        Args:
            graphs: Iterable of graphs to index.
            auxiliary_infos: Iterable of auxiliary data aligned with graphs.

        Returns:
            Tuple[list, list]: Unique graphs and aligned auxiliary infos.
        """
        self.index = self.build_index(graphs, auxiliary_infos)
        unique_data = list(self.index.values())
        if not unique_data:
            return [], []
        unique_graphs, unique_auxiliary_infos = unzip_tuples(unique_data, count=2)
        return unique_graphs, unique_auxiliary_infos
