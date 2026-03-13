"""Vectorization helpers for AbstractGraph features."""

import numpy as np
from abstractgraph.hashing import hash_graph
from joblib import Parallel, delayed
from scipy.sparse import vstack  # For stacking sparse matrices
from typing import Any, Callable, List, Optional, Union, Dict
from scipy.sparse import lil_matrix, csr_matrix
from abstractgraph.graphs import AbstractGraph
from abstractgraph.labels import graph_hash_label_function_factory

def vectorize(abstract_graph: "AbstractGraph", nbits: int = 10, return_dense: bool = True) -> Union[np.ndarray, csr_matrix]:
    """
    Vectorize an AbstractGraph into node-level feature rows.

    The first column is a bias term of ones, and the second column stores
    preimage node degree. Remaining columns are hashed label buckets from
    AbstractGraph.to_array().

    Args:
        abstract_graph: AbstractGraph instance to vectorize.
        nbits: Hash bit width; number of features is 2**nbits.
        return_dense: If True returns a dense array, else a CSR matrix.

    Returns:
        Union[np.ndarray, csr_matrix]: Feature matrix with bias and degree columns.

    Raises:
        ValueError: If nbits < 2 or if to_array fails.
    """
    if nbits < 2:
        raise ValueError("nbits must be at least 2 to accommodate bias and degree features.")

    # 1. Set the label function with the correct nbits attribute.
    #    This is crucial for to_array to infer the correct matrix dimensions.
    abstract_graph.label_function = graph_hash_label_function_factory(nbits=nbits)

    # 2. Call to_array to get the base count matrix (always returns csr_matrix).
    #    to_array internally calls apply_label_function.
    try:
        M_sparse = abstract_graph.to_array()
    except ValueError as e:
        # Re-raise with a more specific context if needed, or just let it propagate.
        raise ValueError(f"Failed to generate base array using to_array. Original error: {e}")

    # 3. Get preimage graph info AND degrees in the correct order
    base_graph = abstract_graph.preimage_graph
    # Ensure the node order matches the row order implicitly used by to_array
    base_nodes: List[Any] = list(base_graph.nodes())
    n = len(base_nodes)

    # Handle empty graph case where M_sparse might be (0, m)
    if n == 0:
        # Return the matrix as is, converted to dense if requested
        return M_sparse.toarray() if return_dense else M_sparse

    # Calculate degrees in the order corresponding to matrix rows
    degrees = np.array([base_graph.degree[node] for node in base_nodes])

    # 4. Modify columns 0 and 1 using vectorized/optimized operations
    if return_dense:
        M = M_sparse.toarray()
        M[:, 0] = 1           # Assign 1 to the entire first column
        M[:, 1] = degrees     # Assign degrees to the second column
    else:
        # Use LIL format for efficient column assignment in sparse matrices
        M_lil = M_sparse.tolil()
        M_lil[:, 0] = 1           # Assign 1 to the entire first column
        # Assign degrees to the second column (reshape needed for LIL column assignment)
        M_lil[:, 1] = degrees.reshape(-1, 1)
        M = M_lil.tocsr()       # Convert back to CSR

    return M

class AbstractGraphTransformer:
    """Graph-level vectorizer using a decomposition function."""

    def __init__(self, 
                 nbits: int, 
                 decomposition_function: Callable[[AbstractGraph], AbstractGraph],
                 return_dense: bool = True, 
                 n_jobs: int = -1,
                 backend: Optional[str] = None) -> None:
        """
        Initialize the transformer.

        Args:
            nbits: Hash bit width used by vectorize.
            decomposition_function: Function that decomposes an AbstractGraph.
            return_dense: Whether to return dense arrays.
            n_jobs: Joblib parallelism setting.
            backend: Optional joblib backend (e.g., "threading").

        Returns:
            None.
        """
        self.nbits = nbits
        self.decomposition_function = decomposition_function
        self.return_dense = return_dense
        self.n_jobs = n_jobs
        self.backend = backend

    def fit(self, X: List[Any], y: Optional[Any] = None) -> "AbstractGraphTransformer":
        """
        No-op fit for pipeline compatibility.

        Args:
            X: Input graphs (unused).
            y: Optional targets (unused).

        Returns:
            AbstractGraphTransformer: Self.
        """
        return self

    def fit_transform(self, X: List[Any], y: Optional[Any] = None) -> Any:
        """
        Fit to data, then transform it.

        Args:
            X: List of graphs.
            y: Optional targets (unused).

        Returns:
            Any: Stacked array or sparse matrix of graph-level features.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def _process_graph(self, graph: Any) -> Any:
        """
        Decompose and vectorize a single graph into a feature row.

        Args:
            graph: Input graph.

        Returns:
            Any: 1 x n_features dense array or CSR matrix.
        """
        # Create the AbstractGraph from the input graph using the provided graph.
        # The following call creates a AbstractGraph and populates its image_graph with a default node.
        ag = AbstractGraph(graph=graph)
        ag.create_default_image_node()
        # Apply the provided decomposition function.
        ag = self.decomposition_function(ag)
        # Vectorize the abstract graph.
        arr = vectorize(ag, nbits=self.nbits, return_dense=self.return_dense)
        # Sum over rows to get a single feature vector per graph.
        arr = arr.sum(axis=0)
        if not self.return_dense:             # i.e. we promised a sparse output
            arr = csr_matrix(arr)             # 1 × n_features CSR row
        return arr

    def transform(self, X: List[Any], y: Optional[Any] = None) -> Any:
        """
        Transform a list of graphs into a stacked feature array.
        
        Args:
            X: List of graphs.
            y: Optional targets (unused).

        Returns:
            Any: Stacked dense array or stacked CSR matrix.
        """
        if self.backend == "dill":
            try:
                import multiprocessing_on_dill as mp
            except Exception as e:
                raise ImportError("backend='dill' requires multiprocessing_on_dill") from e
            n_jobs = self.n_jobs
            try:
                n_jobs = int(n_jobs)
            except Exception:
                n_jobs = 1
            if n_jobs < 0:
                n_jobs = mp.cpu_count()
            n_jobs = max(1, int(n_jobs))
            with mp.Pool(processes=n_jobs) as pool:
                arrays = pool.map(self._process_graph, X)
        else:
            arrays = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(self._process_graph)(graph) for graph in X
            )
        if not arrays:
            n_features = 2 ** self.nbits
            if self.return_dense:
                return np.zeros((0, n_features))
            return csr_matrix((0, n_features))
        if self.return_dense:
            return np.stack(arrays)
        else:
            return vstack(arrays)

class AbstractGraphNodeTransformer:
    """Node-level vectorizer using a decomposition function."""

    def __init__(self, nbits: int, decomposition_function: Callable[[AbstractGraph], AbstractGraph],
                 return_dense: bool = True, n_jobs: int = -1) -> None:
        """
        Initialize the node-level transformer.

        Args:
            nbits: Hash bit width used by vectorize.
            decomposition_function: Function that decomposes an AbstractGraph.
            return_dense: Whether to return dense arrays.
            n_jobs: Joblib parallelism setting.

        Returns:
            None.
        """
        self.nbits = nbits
        self.decomposition_function = decomposition_function
        self.return_dense = return_dense
        self.n_jobs = n_jobs

    def fit(self, X: List[Any], y: Optional[Any] = None) -> "AbstractGraphNodeTransformer":
        """
        No-op fit for pipeline compatibility.

        Args:
            X: Input graphs (unused).
            y: Optional targets (unused).

        Returns:
            AbstractGraphNodeTransformer: Self.
        """
        return self

    def fit_transform(self, X: List[Any], y: Optional[Any] = None) -> List[Any]:
        """
        Fit to data, then transform it.

        Args:
            X: List of graphs.
            y: Optional targets (unused).

        Returns:
            List[Any]: Feature matrices per input graph.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def _process_graph(self, graph: Any) -> Any:
        """
        Decompose and vectorize a single graph into node features.

        Args:
            graph: Input graph.

        Returns:
            Any: Dense array or CSR matrix of node features.
        """
        # Create the AbstractGraph from the input graph.
        ag = AbstractGraph(graph=graph)
        ag.create_default_image_node()
        # Apply the provided decomposition function.
        ag = self.decomposition_function(ag)
        # Vectorize the abstract graph.
        arr = vectorize(ag, nbits=self.nbits, return_dense=self.return_dense)
        return arr

    def transform(self, X: List[Any], y: Optional[Any] = None) -> List[Any]:
        """
        Transform a list of graphs into a list of feature matrices.
        
        Args:
            X: List of graphs.
            y: Optional targets (unused).

        Returns:
            List[Any]: Feature matrices per input graph.
        """
        if not X:
            return []
        arrays = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_graph)(graph) for graph in X
        )
        return arrays
