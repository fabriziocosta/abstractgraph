"""Core AbstractGraph data structure and conversion utilities."""

import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

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


def _warn_deprecated(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"`{old_name}` is deprecated and will be removed in a future release; use `{new_name}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def get_mapped_subgraph(node_data: Dict[str, Any]) -> Optional[nx.Graph]:
    """Return the canonical mapped subgraph payload with legacy fallback."""
    mapped_subgraph = node_data.get("mapped_subgraph")
    if mapped_subgraph is not None:
        return mapped_subgraph
    return node_data.get("association")


def set_mapped_subgraph(node_data: Dict[str, Any], mapped_subgraph: Optional[nx.Graph]) -> None:
    """Write both canonical and legacy payload keys during the migration window."""
    node_data["mapped_subgraph"] = mapped_subgraph
    node_data["association"] = mapped_subgraph


class AbstractGraph:
    """
    Represents an abstract graph derived from a base graph.

    Interpretation nodes store mapped base subgraphs along with computed labels
    and attributes, while optional edge functions add relations between
    interpretation nodes.
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
        """Initialize an AbstractGraph."""
        self.base_graph: nx.Graph = nx.Graph()
        self.interpretation_graph: nx.Graph = nx.Graph()

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

        if graph is not None:
            self.from_graph(graph)
        elif abstract_graph is not None:
            self.from_abstract_graph(abstract_graph)

    @property
    def preimage_graph(self) -> nx.Graph:
        _warn_deprecated("preimage_graph", "base_graph")
        return self.base_graph

    @preimage_graph.setter
    def preimage_graph(self, value: nx.Graph) -> None:
        _warn_deprecated("preimage_graph", "base_graph")
        self.base_graph = value

    @property
    def image_graph(self) -> nx.Graph:
        _warn_deprecated("image_graph", "interpretation_graph")
        return self.interpretation_graph

    @image_graph.setter
    def image_graph(self, value: nx.Graph) -> None:
        _warn_deprecated("image_graph", "interpretation_graph")
        self.interpretation_graph = value

    def copy(self) -> "AbstractGraph":
        """Return a deep copy of this AbstractGraph."""
        new = AbstractGraph(
            label_function=self.label_function,
            attribute_function=self.attribute_function,
            edge_function=self.edge_function,
        )
        new.base_graph = self.base_graph.copy()
        new.interpretation_graph = self.interpretation_graph.copy()
        return new

    def from_graph(self, graph: nx.Graph) -> "AbstractGraph":
        """Initialize the AbstractGraph from a given base graph."""
        self.base_graph = graph.copy()
        return self

    def from_abstract_graph(self, abstract_graph: "AbstractGraph") -> "AbstractGraph":
        """Copy the base and interpretation graphs from another AbstractGraph."""
        self.base_graph = abstract_graph.base_graph.copy()
        self.interpretation_graph = abstract_graph.interpretation_graph.copy()
        return self

    def _add_interpretation_node(self, mapped_subgraph: nx.Graph, meta: Optional[dict] = None) -> int:
        """Add a new interpretation node to the interpretation graph."""
        node_id = len(self.interpretation_graph)
        self.interpretation_graph.add_node(
            node_id,
            mapped_subgraph=mapped_subgraph,
            association=mapped_subgraph,
            label=None,
            attribute=None,
            meta=meta or {},
        )
        return node_id

    def _add_image_node(self, association: nx.Graph, meta: Optional[dict] = None) -> int:
        _warn_deprecated("_add_image_node", "_add_interpretation_node")
        return self._add_interpretation_node(mapped_subgraph=association, meta=meta)

    def create_default_interpretation_node(self) -> "AbstractGraph":
        """Create a single interpretation node mapped to the full base graph."""
        mapped_subgraph = self.base_graph.copy()
        self._add_interpretation_node(mapped_subgraph=mapped_subgraph)
        return self

    def create_default_image_node(self) -> "AbstractGraph":
        _warn_deprecated("create_default_image_node", "create_default_interpretation_node")
        return self.create_default_interpretation_node()

    def create_interpretation_node_with_subgraph_from_nodes(
        self, nodes: Iterable[Any], meta: Optional[dict] = None
    ) -> None:
        """Create an interpretation node from a node-induced base subgraph."""
        mapped_subgraph = self.base_graph.subgraph(set(nodes)).copy()
        self._add_interpretation_node(mapped_subgraph=mapped_subgraph, meta=meta)

    def create_image_node_with_subgraph_from_nodes(self, nodes: Iterable[Any], meta: Optional[dict] = None) -> None:
        _warn_deprecated(
            "create_image_node_with_subgraph_from_nodes",
            "create_interpretation_node_with_subgraph_from_nodes",
        )
        self.create_interpretation_node_with_subgraph_from_nodes(nodes, meta=meta)

    def create_interpretation_node_with_subgraph_from_edges(
        self, edges: Iterable[Tuple[Any, Any]], meta: Optional[dict] = None
    ) -> None:
        """Create an interpretation node from an edge-induced base subgraph."""
        mapped_subgraph = self.base_graph.edge_subgraph(list(edges)).copy()
        self._add_interpretation_node(mapped_subgraph=mapped_subgraph, meta=meta)

    def create_image_node_with_subgraph_from_edges(
        self, edges: Iterable[Tuple[Any, Any]], meta: Optional[dict] = None
    ) -> None:
        _warn_deprecated(
            "create_image_node_with_subgraph_from_edges",
            "create_interpretation_node_with_subgraph_from_edges",
        )
        self.create_interpretation_node_with_subgraph_from_edges(edges, meta=meta)

    def create_interpretation_node_with_subgraph_from_subgraph(
        self, subgraph: nx.Graph, meta: Optional[dict] = None
    ) -> None:
        """Create an interpretation node from a supplied base subgraph."""
        self._add_interpretation_node(mapped_subgraph=subgraph, meta=meta)

    def create_image_node_with_subgraph_from_subgraph(
        self, subgraph: nx.Graph, meta: Optional[dict] = None
    ) -> None:
        _warn_deprecated(
            "create_image_node_with_subgraph_from_subgraph",
            "create_interpretation_node_with_subgraph_from_subgraph",
        )
        self.create_interpretation_node_with_subgraph_from_subgraph(subgraph, meta=meta)

    def apply_label_function(self) -> "AbstractGraph":
        """Apply the label function to each interpretation node."""
        if self.label_function is None:
            raise ValueError("No label_function provided during initialization")
        for node in self.interpretation_graph.nodes:
            node_attributes = self.interpretation_graph.nodes[node]
            self.interpretation_graph.nodes[node]["label"] = self.label_function(node_attributes)
        return self

    def apply_attribute_function(self) -> "AbstractGraph":
        """Apply the attribute function to each mapped subgraph."""
        if self.attribute_function is None:
            raise ValueError("No attribute_function provided during initialization")
        for node in self.interpretation_graph.nodes:
            mapped_subgraph = get_mapped_subgraph(self.interpretation_graph.nodes[node])
            self.interpretation_graph.nodes[node]["attribute"] = self.attribute_function(mapped_subgraph)
        return self

    def apply_edge_function(self) -> "AbstractGraph":
        """Apply the edge function to generate interpretation-graph edges."""
        self.edge_function(self)
        return self

    def update(self) -> "AbstractGraph":
        """Update labels, attributes, and edges for the interpretation graph."""
        self.apply_label_function()
        self.apply_attribute_function()
        self.apply_edge_function()
        return self

    def get_base_nodes_inverse_mappings(self) -> List[nx.Graph]:
        """
        Compute the inverse mapping from base nodes to interpretation nodes.
        """
        inverse_mappings: Dict[Any, Set[Any]] = {node: set() for node in self.base_graph.nodes()}
        for interpretation_node_id, interpretation_node_data in self.interpretation_graph.nodes(data=True):
            mapped_subgraph = get_mapped_subgraph(interpretation_node_data)
            if mapped_subgraph is None:
                continue
            for base_node_id in mapped_subgraph.nodes():
                if base_node_id in inverse_mappings:
                    inverse_mappings[base_node_id].add(interpretation_node_id)

        result_subgraphs: List[nx.Graph] = []
        base_node_list = list(self.base_graph.nodes())
        for base_node_id in base_node_list:
            associated_interpretation_nodes = inverse_mappings.get(base_node_id, set())
            induced_subgraph = self.interpretation_graph.subgraph(associated_interpretation_nodes).copy()
            self.base_graph.nodes[base_node_id]["inverse_mapping"] = induced_subgraph
            self.base_graph.nodes[base_node_id]["inverse_association"] = induced_subgraph
            result_subgraphs.append(induced_subgraph)
        return result_subgraphs

    def get_preimage_nodes_inverse_associations(self) -> List[nx.Graph]:
        _warn_deprecated("get_preimage_nodes_inverse_associations", "get_base_nodes_inverse_mappings")
        return self.get_base_nodes_inverse_mappings()

    def get_interpretation_nodes_mapped_subgraphs(self) -> List[nx.Graph]:
        """Return the mapped subgraphs for all interpretation nodes."""
        return [get_mapped_subgraph(data) for _, data in self.interpretation_graph.nodes(data=True)]

    def get_image_nodes_associations(self) -> List[nx.Graph]:
        _warn_deprecated("get_image_nodes_associations", "get_interpretation_nodes_mapped_subgraphs")
        return self.get_interpretation_nodes_mapped_subgraphs()

    def __add__(self, other: object) -> "AbstractGraph":
        """Combine two AbstractGraphs."""
        if other is None or other == 0:
            return self
        if not isinstance(other, AbstractGraph):
            if not (hasattr(other, "base_graph") or hasattr(other, "preimage_graph")):
                return NotImplemented

        other_base = other.base_graph if hasattr(other, "base_graph") else other.preimage_graph
        other_interpretation = (
            other.interpretation_graph if hasattr(other, "interpretation_graph") else other.image_graph
        )

        new_qg = AbstractGraph(
            label_function=self.label_function,
            attribute_function=self.attribute_function,
            edge_function=self.edge_function,
        )
        new_qg.base_graph = nx.compose(self.base_graph, other_base)
        new_qg.interpretation_graph = nx.disjoint_union(self.interpretation_graph, other_interpretation)
        return new_qg

    def __repr__(self) -> str:
        """Return a formatted string representation of the AbstractGraph."""

        def graph_repr(graph: nx.Graph, indent: int = 0) -> str:
            indent_str = "    " * indent
            lines = [f"{indent_str}Nodes:"]
            for node, data in graph.nodes(data=True):
                attr_parts = []
                for key, value in data.items():
                    if key == "association" and "mapped_subgraph" in data:
                        continue
                    if key in {"mapped_subgraph", "association"} and isinstance(value, nx.Graph):
                        subgraph_str = graph_repr(value, indent=indent + 2)
                        attr_parts.append(f"{key}:\n{subgraph_str}")
                    else:
                        attr_parts.append(f"{key}: {value}")
                lines.append(f"{indent_str}  {node}: " + "{" + ", ".join(attr_parts) + "}")
            lines.append(f"{indent_str}Edges:")
            for u, v, edata in graph.edges(data=True):
                edata_str = "{" + ", ".join(f"{k}: {v}" for k, v in edata.items()) + "}"
                lines.append(f"{indent_str}  ({u}, {v}): {edata_str}")
            return "\n".join(lines)

        lines = ["AbstractGraph:", "Base Graph:", graph_repr(self.base_graph, indent=1)]
        lines.append("Interpretation Graph:")
        lines.append(graph_repr(self.interpretation_graph, indent=1))
        return "\n".join(lines)

    def to_graph(self, connection_label: str = "abstract") -> nx.Graph:
        """Convert the AbstractGraph into a single NetworkX graph with integer node ids."""
        graph_out = nx.Graph()

        base_nodes = list(self.base_graph.nodes())
        base_id_map = {orig_id: i for i, orig_id in enumerate(base_nodes)}
        next_node_id = len(base_nodes)

        for orig_id, data in self.base_graph.nodes(data=True):
            graph_out.add_node(base_id_map[orig_id], **data, kind="base", original_id=orig_id)
        for u, v, edata in self.base_graph.edges(data=True):
            graph_out.add_edge(base_id_map[u], base_id_map[v], **edata)

        interpretation_id_map: Dict[Any, int] = {}
        for interpretation_node, data in self.interpretation_graph.nodes(data=True):
            interpretation_id = next_node_id
            interpretation_id_map[interpretation_node] = interpretation_id
            next_node_id += 1
            graph_out.add_node(
                interpretation_id,
                **{k: v for k, v in data.items() if k not in {"mapped_subgraph", "association"}},
                kind="interpretation",
                original_id=interpretation_node,
            )

        for u, v, edata in self.interpretation_graph.edges(data=True):
            graph_out.add_edge(interpretation_id_map[u], interpretation_id_map[v], **edata)

        for interpretation_node, data in self.interpretation_graph.nodes(data=True):
            mapped_subgraph = get_mapped_subgraph(data) or nx.Graph()
            interpretation_node_id = interpretation_id_map[interpretation_node]
            for orig_base_id in mapped_subgraph.nodes():
                if orig_base_id in base_id_map:
                    graph_out.add_edge(
                        interpretation_node_id,
                        base_id_map[orig_base_id],
                        label=connection_label,
                        kind="abstract",
                    )

        return graph_out

    def to_array(self) -> csr_matrix:
        """Generate a sparse CSR matrix that sums interpretation-node attributes per label."""
        if self.label_function is None:
            raise ValueError("Cannot generate array without a label_function.")
        self.apply_label_function()
        self.apply_attribute_function()

        nbits = getattr(self.label_function, "nbits", None)
        if nbits is None:
            raise ValueError(
                "Could not automatically determine 'nbits' from the label_function. "
                "Ensure the label_function was created using a provided helper "
                "(e.g., graph_hash_label_function_factory) or manually assign an 'nbits' attribute."
            )
        m = 2**nbits

        base_nodes = list(self.base_graph.nodes())
        node_to_index = {node: i for i, node in enumerate(base_nodes)}
        n = len(base_nodes)

        def _normalize_attribute(value: Any) -> np.ndarray:
            arr = np.asarray(1.0 if value is None else value, dtype=float)
            if arr.ndim == 0:
                arr = arr.reshape(1,)
            elif arr.ndim != 1:
                raise TypeError(f"Attribute must be a 1-D array or scalar; got ndim={arr.ndim}.")
            return arr

        attribute_dim: Optional[int] = None
        interpretation_attribute_cache: Dict[Any, np.ndarray] = {}
        for interpretation_node_id, interpretation_node_data in self.interpretation_graph.nodes(data=True):
            attr_val = interpretation_node_data.get("attribute", None)
            attr_arr = _normalize_attribute(attr_val)
            if attribute_dim is None:
                attribute_dim = attr_arr.shape[0]
            elif attr_arr.shape[0] != attribute_dim:
                raise ValueError(
                    f"Inconsistent attribute dimension for interpretation node {interpretation_node_id}: "
                    f"{attr_arr.shape[0]} vs expected {attribute_dim}."
                )
            interpretation_attribute_cache[interpretation_node_id] = attr_arr

        if attribute_dim is None:
            attribute_dim = 1

        total_features = m * attribute_dim
        count_matrix = lil_matrix((n, total_features), dtype=float)

        inverse_mappings: Dict[Any, Set[Any]] = {node: set() for node in self.base_graph.nodes()}
        for interpretation_node_id, interpretation_node_data in self.interpretation_graph.nodes(data=True):
            mapped_subgraph = get_mapped_subgraph(interpretation_node_data)
            if mapped_subgraph is None:
                continue
            for base_node_id in mapped_subgraph.nodes():
                if base_node_id in inverse_mappings:
                    inverse_mappings[base_node_id].add(interpretation_node_id)

        for base_node_id, associated_interpretation_node_ids in inverse_mappings.items():
            base_node_index = node_to_index.get(base_node_id)
            if base_node_index is None:
                warnings.warn(f"Base node {base_node_id} not found in node_to_index map. Skipping.")
                continue

            for interpretation_node_id in associated_interpretation_node_ids:
                if interpretation_node_id not in self.interpretation_graph:
                    warnings.warn(
                        f"Associated interpretation node {interpretation_node_id} not found in interpretation_graph. Skipping."
                    )
                    continue

                interpretation_node_data = self.interpretation_graph.nodes[interpretation_node_id]
                interpretation_node_label = interpretation_node_data.get("label")
                if interpretation_node_label is None:
                    warnings.warn(
                        f"Interpretation node {interpretation_node_id} has no label. Skipping count for this node."
                    )
                    continue

                try:
                    label_int = int(interpretation_node_label)
                except (ValueError, TypeError) as e:
                    warnings.warn(
                        f"Interpretation node {interpretation_node_id} has non-integer label "
                        f"'{interpretation_node_label}' (Error: {e}). Skipping."
                    )
                    continue

                if not (0 <= label_int < m):
                    warnings.warn(
                        f"Interpretation node {interpretation_node_id} has label {label_int} "
                        f"outside the expected range [0, {m - 1}). Skipping."
                    )
                    continue

                attr_vec = interpretation_attribute_cache.get(interpretation_node_id)
                if attr_vec is None:
                    attr_vec = _normalize_attribute(interpretation_node_data.get("attribute", None))
                    interpretation_attribute_cache[interpretation_node_id] = attr_vec

                col_start = label_int * attribute_dim
                col_end = col_start + attribute_dim
                count_matrix[base_node_index, col_start:col_end] += attr_vec.reshape(1, -1)

        return count_matrix.tocsr()


def graph_to_abstract_graph(
    graph: nx.Graph,
    decomposition_function: Callable[[AbstractGraph], AbstractGraph],
    nbits: int,
    label_function: Optional[Callable[[dict], Any]] = None,
    label_mode: str = "graph_hash",
) -> AbstractGraph:
    """Build and update an AbstractGraph from a base graph and decomposition."""
    if label_function is None:
        if label_mode == "graph_hash":
            label_function = graph_hash_label_function_factory(nbits)
        elif label_mode == "operator_hash":
            label_function = name_hash_label_function_factory(nbits)
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")
    abstract_graph = AbstractGraph(graph=graph, label_function=label_function)
    abstract_graph.create_default_interpretation_node()
    abstract_graph = decomposition_function(abstract_graph)
    abstract_graph.update()
    return abstract_graph


def _graph_to_abstract_graph_worker(args):
    """Worker wrapper for multiprocessing graph_to_abstract_graph."""
    graph, decomposition_function, nbits, label_function, label_mode = args
    return graph_to_abstract_graph(graph, decomposition_function, nbits, label_function, label_mode)


def _graphs_to_abstract_graphs(
    graphs: Sequence[nx.Graph],
    decomposition_function: Callable[[AbstractGraph], AbstractGraph],
    nbits: int,
    label_function: Optional[Callable[[dict], Any]] = None,
    label_mode: str = "graph_hash",
) -> Sequence[AbstractGraph]:
    """Convert a sequence of graphs to AbstractGraphs serially."""
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
    """Parallel version of graphs_to_abstract_graphs."""
    if n_jobs in (None, 1):
        return _graphs_to_abstract_graphs(graphs, decomposition_function, nbits, label_function, label_mode)
    if n_jobs < 0:
        n_jobs = max(1, mp.cpu_count())
    else:
        n_jobs = max(1, int(n_jobs))
    with mp.Pool(processes=n_jobs) as pool:
        args = [(graph, decomposition_function, nbits, label_function, label_mode) for graph in graphs]
        return pool.map(_graph_to_abstract_graph_worker, args)
