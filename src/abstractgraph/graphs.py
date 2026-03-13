"""Core AbstractGraph data structure and conversion utilities."""

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import warnings
from typing import Optional, Callable, Any, List, Iterable, Sequence, Tuple, Dict, Set 
from abstractgraph.labels import (
    DEFAULT_NBITS,
    graph_hash_label_function_factory,
    mean_attribute_function,
    name_hash_label_function_factory,
    null_edge_function,
)
try:
    import multiprocessing_on_dill as mp
except Exception:
    import multiprocessing as mp

class AbstractGraph:
    """
    Represents an abstract graph derived from a preimage graph.

    Image nodes store subgraph associations along with computed labels and attributes,
    while optional edge functions add relations between image nodes.

    Attributes:
        preimage_graph (nx.Graph): The original graph from which this abstract graph is derived.
        image_graph (nx.Graph): The abstract graph itself, where each node represents a set of equivalent subgraphs.
        label_function (Callable[[nx.Graph], Any]): Function to compute labels for the image nodes based on their subgraphs.
        attribute_function (Callable[[nx.Graph], np.ndarray]): Function to compute attributes for the image nodes based on their subgraphs.
        edge_function (Callable[["AbstractGraph"], "AbstractGraph"]): Function to generate edges in the image graph based on the subgraph structure.
    """
    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        abstract_graph: Optional["AbstractGraph"] = None,
        label_function: Optional[Callable[[nx.Graph], Any]] = None,
        attribute_function: Optional[Callable[[nx.Graph], np.ndarray]] = None,
        edge_function: Optional[Callable[["AbstractGraph"], "AbstractGraph"]] = None,
        nbits: int = DEFAULT_NBITS,
    ) -> None:
        """
        Initialize an AbstractGraph.

        Args:
            graph: Optional NetworkX graph to use as the preimage graph.
            abstract_graph: Optional AbstractGraph to copy from.
            label_function: Optional label function for image nodes.
            attribute_function: Optional attribute aggregation function for image nodes.
            edge_function: Optional edge generation function for the image graph.
            nbits: Hash bit width used by default label functions.

        Returns:
            None.
        """
        self.preimage_graph: nx.Graph = nx.Graph()
        self.image_graph: nx.Graph = nx.Graph()

        # Use explicitly provided functions, or inherit from abstract_graph, or fall back to default
        self.label_function = (
            label_function
            if label_function is not None
            else (abstract_graph.label_function if abstract_graph else graph_hash_label_function_factory(nbits))
        )
        self.attribute_function = (
            attribute_function
            if attribute_function is not None
            else (abstract_graph.attribute_function if abstract_graph else mean_attribute_function)
        )
        self.edge_function = (
            edge_function
            if edge_function is not None
            else (abstract_graph.edge_function if abstract_graph else null_edge_function)
        )

        if graph:
            self.from_graph(graph)
        elif abstract_graph:
            self.from_abstract_graph(abstract_graph)

    def copy(self) -> "AbstractGraph":
        """
        Return a deep copy of this AbstractGraph.

        Args:
            None.

        Returns:
            AbstractGraph: Copied AbstractGraph with the same functions and graphs.
        """
        # The easiest way is to call your own from_abstract_graph
        new = AbstractGraph(
            label_function=self.label_function,
            attribute_function=self.attribute_function,
            edge_function=self.edge_function,
        )
        # Copy the two graphs
        new.preimage_graph = self.preimage_graph.copy()
        new.image_graph    = self.image_graph.copy()
        return new

    def from_graph(self, graph: nx.Graph) -> "AbstractGraph":
        """
        Initialize the AbstractGraph from a given preimage graph.

        Args:
            graph: The standard NetworkX graph to copy.

        Returns:
            AbstractGraph: The updated AbstractGraph instance.
        """
        self.preimage_graph = graph.copy()
        return self
    
    def from_abstract_graph(self, abstract_graph: "AbstractGraph") -> "AbstractGraph":
        """
        Copy the preimage and image graphs from another AbstractGraph.

        Args:
            abstract_graph: The AbstractGraph instance to copy.

        Returns:
            AbstractGraph: The updated AbstractGraph instance.
        """
        self.preimage_graph = abstract_graph.preimage_graph.copy()
        self.image_graph = abstract_graph.image_graph.copy()
        return self
    
    def _add_image_node(self, association: nx.Graph, meta: Optional[dict] = None) -> int:
        """
        Add a new image node to the image graph.

        Args:
            association: Subgraph associated with the new image node.
            meta: Optional metadata dictionary to attach to the image node.

        Returns:
            int: The newly added image node id.
        """
        node_id = len(self.image_graph)
        self.image_graph.add_node(
            node_id,
            association=association,
            label=None,
            attribute=None,
            meta=meta or {}
        )
        return node_id

    def create_default_image_node(self) -> None:
        """
        Create a single image node associated with the full preimage graph.

        Args:
            None.

        Returns:
            AbstractGraph: The updated AbstractGraph instance.
        """
        subgraph = self.preimage_graph.copy()
        self._add_image_node(association=subgraph)
        return self

    def create_image_node_with_subgraph_from_nodes(self, nodes: Iterable[Any], meta: Optional[dict] = None) -> None:
        """
        Create an image node from a node-induced subgraph.

        Args:
            nodes: Iterable of preimage node ids to include.
            meta: Optional metadata dictionary to attach to the image node.

        Returns:
            None.
        """
        subgraph = self.preimage_graph.subgraph(set(nodes)).copy()
        self._add_image_node(association=subgraph, meta=meta)

    def create_image_node_with_subgraph_from_edges(self, edges: Iterable[Tuple[Any, Any]], meta: Optional[dict] = None) -> None:
        """
        Create an image node from an edge-induced subgraph.

        Args:
            edges: Iterable of preimage edges to include.
            meta: Optional metadata dictionary to attach to the image node.

        Returns:
            None.
        """
        subgraph = self.preimage_graph.edge_subgraph(list(edges)).copy()
        self._add_image_node(association=subgraph, meta=meta)

    def create_image_node_with_subgraph_from_subgraph(self, subgraph: nx.Graph, meta: Optional[dict] = None) -> None:
        """
        Create an image node from a supplied subgraph.

        Args:
            subgraph: Preimage subgraph to associate with the new image node.
            meta: Optional metadata dictionary to attach to the image node.

        Returns:
            None.
        """
        self._add_image_node(association=subgraph, meta=meta)

    def apply_label_function(self) -> None:
        """
        Apply the label function to each image node.

        Args:
            None.

        Returns:
            AbstractGraph: The updated AbstractGraph instance.
        """
        if self.label_function is None:
            raise ValueError("No label_function provided during initialization")
        for node in self.image_graph.nodes:
            node_attributes = self.image_graph.nodes[node]
            self.image_graph.nodes[node]["label"] = self.label_function(node_attributes)
        return self

    def apply_attribute_function(self) -> None:
        """
        Apply the attribute function to each image node association.

        Args:
            None.

        Returns:
            AbstractGraph: The updated AbstractGraph instance.
        """
        if self.attribute_function is None:
            raise ValueError("No attribute_function provided during initialization")
        for node in self.image_graph.nodes:
            subgraph = self.image_graph.nodes[node]["association"]
            self.image_graph.nodes[node]["attribute"] = self.attribute_function(subgraph)
        return self

    def apply_edge_function(self) -> None:
        """
        Apply the edge function to generate image-graph edges.

        Args:
            None.

        Returns:
            AbstractGraph: The updated AbstractGraph instance.
        """
        self.edge_function(self)
        return self

    def update(self) -> None:
        """
        Update labels, attributes, and edges for the image graph.

        Args:
            None.

        Returns:
            AbstractGraph: The updated AbstractGraph instance.
        """
        self.apply_label_function()
        self.apply_attribute_function()
        self.apply_edge_function()
        return self

    def get_preimage_nodes_inverse_associations(self) -> List[nx.Graph]:
        """
        Computes the inverse association mapping from preimage nodes to image nodes.

        For each node 'p' in the preimage graph, it identifies all image graph nodes 'q'
        such that 'p' is part of the subgraph associated with 'q'. It then creates the
        induced subgraph of the image_graph containing these nodes 'q'. This induced
        subgraph is stored in the preimage node's data dictionary under the key
        'inverse_association'.

        The function returns a list of these induced subgraphs, where the i-th graph
        corresponds to the i-th node in `list(self.preimage_graph.nodes())`.

        Args:
            None.

        Returns:
            List[nx.Graph]: Induced image-graph subgraphs per preimage node.
        """
        # Step 1: Build the map from preimage node ID to set of image node IDs
        inverse_associations_map: Dict[Any, Set[Any]] = {
            node: set() for node in self.preimage_graph.nodes()
        }

        # Iterate through image nodes and their associated subgraphs
        for image_node_id, image_node_data in self.image_graph.nodes(data=True):
            association_subgraph = image_node_data.get("association")

            if association_subgraph is not None:
                # Iterate through nodes within the association subgraph
                for preimage_node_id in association_subgraph.nodes():
                    # Check if the preimage node exists in our map (initialized from preimage_graph)
                    # and add the image node ID to its inverse association set
                    if preimage_node_id in inverse_associations_map:
                        inverse_associations_map[preimage_node_id].add(image_node_id)
                    # else: # Optional: Warn if an association contains nodes not in the main preimage graph
                    #     warnings.warn(f"Node {preimage_node_id} from association of image node {image_node_id} "
                    #                   f"not found in preimage graph. Skipping inverse mapping for this instance.")

        # Step 2: Create induced subgraphs, store them in preimage nodes, and collect them in a list
        result_subgraphs: List[nx.Graph] = []
        preimage_node_list = list(self.preimage_graph.nodes()) # Ensure consistent order

        for preimage_node_id in preimage_node_list:
            # Get the set of associated image node IDs for the current preimage node
            associated_image_nodes = inverse_associations_map.get(preimage_node_id, set())

            # Create the induced subgraph from the image graph using the collected IDs
            # Use .copy() to get an independent graph object, not just a view
            induced_subgraph = self.image_graph.subgraph(associated_image_nodes).copy()

            # Store the induced subgraph in the corresponding preimage node's attributes
            # We iterate through preimage_node_list, so the node must exist.
            self.preimage_graph.nodes[preimage_node_id]['inverse_association'] = induced_subgraph

            # Add the created subgraph to the result list
            result_subgraphs.append(induced_subgraph)

        return result_subgraphs

    def get_image_nodes_associations(self) -> List[nx.Graph]:
        """
        Return the association subgraphs for all image nodes.

        Args:
            None.

        Returns:
            List[nx.Graph]: Subgraphs associated with image nodes.
        """
        return [data["association"] for _, data in self.image_graph.nodes(data=True)]
        
    def __add__(self, other: object) -> "AbstractGraph":
        """
        Combine two AbstractGraphs.

        Args:
            other: Another AbstractGraph or compatible object to merge.

        Returns:
            AbstractGraph: A new AbstractGraph with composed preimage and disjoint image graphs.
        """
        if other is None or other == 0:
            return self

        if not isinstance(other, AbstractGraph):
            if not (hasattr(other, "preimage_graph") and hasattr(other, "image_graph")):
                return NotImplemented

        new_qg = AbstractGraph(
            label_function=self.label_function,
            attribute_function=self.attribute_function,
            edge_function=self.edge_function,
        )

        new_qg.preimage_graph = nx.compose(self.preimage_graph, other.preimage_graph)
        new_qg.image_graph = nx.disjoint_union(self.image_graph, other.image_graph)

        return new_qg

    def __repr__(self) -> str:
        """
        Return a formatted string representation of the AbstractGraph.

        Args:
            None.

        Returns:
            str: Human-readable representation of the preimage and image graphs.
        """

        def graph_repr(graph: nx.Graph, indent: int = 0) -> str:
            """
            Build a formatted string for a NetworkX graph.

            Args:
                graph: The NetworkX graph to represent.
                indent: The current indentation level.

            Returns:
                str: Formatted graph contents.
            """
            indent_str = "    " * indent
            lines = []
            lines.append(f"{indent_str}Nodes:")
            for node, data in graph.nodes(data=True):
                attr_parts = []
                for key, value in data.items():
                    if key == "association" and isinstance(value, nx.Graph):
                        # Recursively represent the subgraph with increased indentation.
                        subgraph_str = graph_repr(value, indent=indent+2)
                        attr_parts.append(f"{key}:\n{subgraph_str}")
                    else:
                        attr_parts.append(f"{key}: {value}")
                attr_str = "{" + ", ".join(attr_parts) + "}"
                lines.append(f"{indent_str}  {node}: {attr_str}")
            lines.append(f"{indent_str}Edges:")
            for u, v, edata in graph.edges(data=True):
                edata_str = "{" + ", ".join(f"{k}: {v}" for k, v in edata.items()) + "}"
                lines.append(f"{indent_str}  ({u}, {v}): {edata_str}")
            return "\n".join(lines)
        
        lines = []
        lines.append("AbstractGraph:")
        lines.append("Preimage Graph:")
        lines.append(graph_repr(self.preimage_graph, indent=1))
        lines.append("Image Graph:")
        lines.append(graph_repr(self.image_graph, indent=1))
        return "\n".join(lines)

    def to_graph(self, connection_label: str = "abstract") -> nx.Graph:
        """
        Converts the AbstractGraph into a single NetworkX graph with integer node IDs.

        - Preimage nodes are numbered from 0 to N-1 (same as in self.preimage_graph).
        - Image nodes are numbered starting from N.
        - Edges from image nodes to preimage nodes in their subgraphs are added with 'label' = connection_label.

        Args:
            connection_label: Label to use for association edges.

        Returns:
            nx.Graph: A combined graph with preimage and image nodes.
        """
        G = nx.Graph()

        # Map original preimage node ids to consistent 0...N-1 ids
        preimage_nodes = list(self.preimage_graph.nodes())
        preimage_id_map = {orig_id: i for i, orig_id in enumerate(preimage_nodes)}
        next_node_id = len(preimage_nodes)

        # 1. Add preimage nodes and edges
        for orig_id, data in self.preimage_graph.nodes(data=True):
            G.add_node(preimage_id_map[orig_id], **data, kind="preimage", original_id=orig_id)
        for u, v, edata in self.preimage_graph.edges(data=True):
            G.add_edge(preimage_id_map[u], preimage_id_map[v], **edata)

        # 2. Add image nodes and edges
        image_id_map = {}  # map image graph node -> global node id
        for img_node, data in self.image_graph.nodes(data=True):
            image_id = next_node_id
            image_id_map[img_node] = image_id
            next_node_id += 1
            G.add_node(image_id, **{k: v for k, v in data.items() if k != "association"}, kind="image", original_id=img_node)

        for u, v, edata in self.image_graph.edges(data=True):
            G.add_edge(image_id_map[u], image_id_map[v], **edata)

        # 3. Connect image nodes to preimage nodes (association membership)
        for img_node, data in self.image_graph.nodes(data=True):
            subgraph = data.get("association", nx.Graph())
            image_node_id = image_id_map[img_node]
            for orig_pre_id in subgraph.nodes():
                if orig_pre_id in preimage_id_map:
                    G.add_edge(image_node_id, preimage_id_map[orig_pre_id], label=connection_label, kind="abstract")

        return G

    def to_array(self) -> csr_matrix:
        """
        Generate a sparse CSR matrix that sums image-node attributes per label.

        The output shape is (n_preimage_nodes, n_labels * attribute_dim), where
        ``n_labels = 2**nbits`` and each label slice accumulates the attribute vectors
        from image nodes that project to that label.

        Args:
            None.

        Returns:
            scipy.sparse.csr_matrix: Flattened label × attribute counts per base node.
        """
        if self.label_function is None:
            raise ValueError("Cannot generate array without a label_function.")
        self.apply_label_function()
        self.apply_attribute_function()

        nbits = getattr(self.label_function, "nbits", None)
        if nbits is None:
            raise ValueError(
                "Could not automatically determine 'nbits' from the label_function. "
                "Ensure the label_function was created using a provided helper "
                "(e.g., graph_hash_label_function) or manually assign an 'nbits' attribute."
            )
        m = 2**nbits

        preimage_nodes = list(self.preimage_graph.nodes())
        node_to_index = {node: i for i, node in enumerate(preimage_nodes)}
        n = len(preimage_nodes)

        def _normalize_attribute(value: Any) -> np.ndarray:
            """
            Normalize an attribute to a 1D float array.

            Args:
                value: Attribute value (scalar or array-like).

            Returns:
                np.ndarray: 1D float array representation.
            """
            arr = np.asarray(1.0 if value is None else value, dtype=float)
            if arr.ndim == 0:
                arr = arr.reshape(1,)
            elif arr.ndim != 1:
                raise TypeError(
                    f"Attribute must be a 1-D array or scalar; got ndim={arr.ndim}."
                )
            return arr

        attribute_dim: Optional[int] = None
        image_attribute_cache: Dict[Any, np.ndarray] = {}
        for image_node_id, image_node_data in self.image_graph.nodes(data=True):
            attr_val = image_node_data.get("attribute", None)
            attr_arr = _normalize_attribute(attr_val)
            if attribute_dim is None:
                attribute_dim = attr_arr.shape[0]
            elif attr_arr.shape[0] != attribute_dim:
                raise ValueError(
                    f"Inconsistent attribute dimension for image node {image_node_id}: "
                    f"{attr_arr.shape[0]} vs expected {attribute_dim}."
                )
            image_attribute_cache[image_node_id] = attr_arr

        if attribute_dim is None:
            attribute_dim = 1

        total_features = m * attribute_dim
        count_matrix = lil_matrix((n, total_features), dtype=float)

        inverse_associations_map: Dict[Any, Set[Any]] = {
            node: set() for node in self.preimage_graph.nodes()
        }
        for image_node_id, image_node_data in self.image_graph.nodes(data=True):
            association_subgraph = image_node_data.get("association")
            if association_subgraph is None:
                continue
            for preimage_node_id in association_subgraph.nodes():
                if preimage_node_id in inverse_associations_map:
                    inverse_associations_map[preimage_node_id].add(image_node_id)

        for preimage_node_id, associated_image_node_ids in inverse_associations_map.items():
            preimage_node_index = node_to_index.get(preimage_node_id)
            if preimage_node_index is None:
                warnings.warn(
                    f"Preimage node {preimage_node_id} not found in node_to_index map. Skipping."
                )
                continue

            for image_node_id in associated_image_node_ids:
                if image_node_id not in self.image_graph:
                    warnings.warn(
                        f"Associated image node {image_node_id} not found in image_graph. Skipping."
                    )
                    continue

                image_node_data = self.image_graph.nodes[image_node_id]
                image_node_label = image_node_data.get("label")
                if image_node_label is None:
                    warnings.warn(
                        f"Image node {image_node_id} has no label. Skipping count for this node."
                    )
                    continue

                try:
                    label_int = int(image_node_label)
                except (ValueError, TypeError) as e:
                    warnings.warn(
                        f"Image node {image_node_id} has non-integer label '{image_node_label}' (Error: {e}). Skipping."
                    )
                    continue

                if not (0 <= label_int < m):
                    warnings.warn(
                        f"Image node {image_node_id} has label {label_int} outside the expected range [0, {m-1}). Skipping."
                    )
                    continue

                attr_vec = image_attribute_cache.get(image_node_id)
                if attr_vec is None:
                    attr_vec = _normalize_attribute(image_node_data.get("attribute", None))
                    image_attribute_cache[image_node_id] = attr_vec

                col_start = label_int * attribute_dim
                col_end = col_start + attribute_dim
                count_matrix[
                    preimage_node_index, col_start:col_end
                ] += attr_vec.reshape(1, -1)

        return count_matrix.tocsr()

def graph_to_abstract_graph(
    graph: nx.Graph,
    decomposition_function: Callable[[AbstractGraph], AbstractGraph],
    nbits: int,
    label_function: Optional[Callable[[dict], Any]] = None,
    label_mode: str = "graph_hash",
) -> AbstractGraph:
    """
    Build and update an AbstractGraph from a preimage graph and decomposition.

    Args:
        graph: Preimage NetworkX graph.
        decomposition_function: Decomposition function to produce image nodes.
        nbits: Hash bit width for the default label function.
        label_function: Optional label function for image nodes. If provided, overrides nbits.
        label_mode: Label factory selector when label_function is None.
            - "graph_hash": use graph_hash_label_function_factory(nbits)
            - "operator_hash": use name_hash_label_function_factory(nbits)

    Returns:
        AbstractGraph: The updated AbstractGraph instance.
    """
    if label_function is None:
        if label_mode == "graph_hash":
            label_function = graph_hash_label_function_factory(nbits)
        elif label_mode == "operator_hash":
            label_function = name_hash_label_function_factory(nbits)
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")
    abstract_graph = AbstractGraph(graph=graph, label_function=label_function)
    abstract_graph.create_default_image_node()
    abstract_graph = decomposition_function(abstract_graph)
    abstract_graph.update()
    return abstract_graph


def _graph_to_abstract_graph_worker(args):
    """
    Worker wrapper for multiprocessing graph_to_abstract_graph.

    Args:
        args: Tuple of (graph, decomposition_function, nbits, label_function, label_mode).

    Returns:
        AbstractGraph: Converted AbstractGraph instance.
    """
    graph, decomposition_function, nbits, label_function, label_mode = args
    return graph_to_abstract_graph(graph, decomposition_function, nbits, label_function, label_mode)

def _graphs_to_abstract_graphs(
    graphs: Sequence[nx.Graph],
    decomposition_function: Callable[[AbstractGraph], AbstractGraph],
    nbits: int,
    label_function: Optional[Callable[[dict], Any]] = None,
    label_mode: str = "graph_hash",
) -> Sequence[AbstractGraph]:
    """
    Convert a sequence of graphs to AbstractGraphs serially.

    Args:
        graphs: Input graphs to convert.
        decomposition_function: Decomposition function to apply.
        nbits: Hash bit width for the label function.
        label_function: Optional label function for image nodes.
        label_mode: Label factory selector when label_function is None.

    Returns:
        Sequence[AbstractGraph]: Converted AbstractGraph instances.
    """
    abstract_graphs = []
    for graph in graphs:
        abstract_graphs.append(graph_to_abstract_graph(graph, decomposition_function, nbits, label_function, label_mode))
    return abstract_graphs

def graphs_to_abstract_graphs(
    graphs: Sequence[nx.Graph],
    decomposition_function: Callable[[AbstractGraph], AbstractGraph],
    nbits: int,
    n_jobs: int = -1,
    label_function: Optional[Callable[[dict], Any]] = None,
    label_mode: str = "graph_hash",
) -> Sequence[AbstractGraph]:
    """Parallel version of graphs_to_abstract_graphs.

    Args:
        graphs: Input graphs to convert.
        decomposition_function: AbstractGraph decomposition function to build image nodes.
        nbits: Hash bit width used by graph_to_abstract_graph.
        n_jobs: Number of worker processes; -1 uses all available CPUs.
        label_function: Optional label function for image nodes.
        label_mode: Label factory selector when label_function is None.
            - "graph_hash": use graph_hash_label_function_factory(nbits)
            - "operator_hash": use name_hash_label_function_factory(nbits)

    Returns:
        A list of AbstractGraph instances.
    """
    if n_jobs in (None, 1):
        return _graphs_to_abstract_graphs(graphs, decomposition_function, nbits, label_function, label_mode)
    if n_jobs < 0:
        n_jobs = max(1, mp.cpu_count())
    else:
        n_jobs = max(1, int(n_jobs))
    with mp.Pool(processes=n_jobs) as pool:
        args = [(graph, decomposition_function, nbits, label_function, label_mode) for graph in graphs]
        return pool.map(_graph_to_abstract_graph_worker, args)
