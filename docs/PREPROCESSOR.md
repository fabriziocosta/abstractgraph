# AbstractGraph Preprocessor — Attention‑Derived Preimage Graphs

This module provides a neural, attention‑driven path to build “abstract graphs” directly from token embeddings, complementing the symbolic/operator path used elsewhere in this repository.

- File: `abstract_graph_preprocessor.py`
- Scope: lightweight Transformer encoder, attention aggregation → iterated MST + DP foresting → consensus co‑clustering → robust token‑level (preimage) edges; optional dataset‑level node clustering; scikit‑style wrapper.
- Output: per‑instance NetworkX preimage graphs with token nodes carrying embeddings and robust edges; graph attribute stores the co‑clustering matrix.

The preprocessor does not depend on `AbstractGraph` types from `type.py`. It constructs token‑level preimage graphs directly from attention, exposing a minimal, learned structure for inspection, clustering, or downstream modeling. If you want an `AbstractGraph`, pass the produced NetworkX graph to `AbstractGraph(graph=...)`.


## Where It Fits In

- Symbolic/operator pipeline (core repo):
  - Build image nodes by explicit graph decompositions (`abstract_graph_operators.py`) over a NetworkX preimage graph; vectorize with `vectorize.py`; train with `estimator/graph_estimator.py` or `estimator/neural.py`.
- Neural/preprocessor pipeline (this module):
  - Learn token embeddings with a small Transformer, aggregate attention across layers/heads, then induce robust token‑level adjacency via consensus clustering over iterated maximum spanning trees. You get a preimage graph over tokens with embeddings and optional dataset‑level type clusters.

Use this module when you have token matrices (shape `(n_tokens, d_in)`) and want a learned, attention‑based preimage graph without committing to explicit structural operators. It complements, rather than replaces, the operator pipeline.


## Quick Start

```python
import numpy as np
from AbstractGraph.abstract_graph_preprocessor import AbstractGraphPreprocessor, ImageNodeClusterer

# Toy dataset: list of (n_tokens, d_in) arrays and labels
d_in = 16
X = [np.random.randn(np.random.randint(8, 16), d_in) for _ in range(20)]
y = np.random.choice(["A", "B"], size=len(X))

# Optional: function to label image nodes from raw tokens
# signature: (instance_array: np.ndarray, token_indices: List[int]) -> Any
label_fn = lambda arr, idx: len(idx)

# Optional: dataset‑level node clusterer (KMeans over pooled node embeddings)
clusterer = ImageNodeClusterer(n_clusters=16, random_state=0)

pp = AbstractGraphPreprocessor(
    d_model=64, n_heads=4, num_layers=2, n_epochs=5, lr=1e-3,
    K_mst=3, alpha=1.0, beta=0.5, rho=0.6,
    device="auto", label_fn=label_fn, node_clusterer=clusterer,
)

# Fit on instance labels (supervision improves token embeddings)
pp.fit(X, y)

# Extract attention‑derived preimage graphs (NetworkX)
graphs = pp.transform(X)

# Optionally assign dataset‑level cluster ids to each token node
graphs = pp.assign_node_cluster_labels(graphs)

# Inspect first graph (NetworkX)
g0 = graphs[0]
print(g0)                           # nx.Graph with nodes/edges
print(g0.graph['co_cluster'].shape) # (N_tokens, N_tokens)
print(g0.nodes[0]['embedding'].shape)  # (d_model,)
```


## Module Overview

- Transformer backbone
  - `MultiHeadSelfAttention`: minimal multi‑head self‑attention (no positional encodings).
  - `TransformerEncoderLayerCustom`: attention + FFN with residual + LayerNorm.
  - `SimpleTransformerEncoder`: stacks `num_layers` custom encoder layers; projects `d_in → d_model`.

- Graph extraction helpers
  - `maximum_spanning_tree_edges(W)`: Kruskal‑style MST on dense symmetric weights.
  - `dp_forest_on_tree(N, edges, alpha, beta)`: light tree DP; keep edge if `alpha*w >= beta`, then connected components form a forest of clusters.
  - `build_preimage_edges_from_attention(W, K_mst, alpha, beta, rho)`:
    - Same pipeline, but retains token‑level edges via consensus adjacency and DP‑kept MST edges.
    - Returns a list of `(u, v, attr)` edges and the `co_cluster_matrix`.

- Dataset‑level node clustering
  - `ImageNodeClusterer(n_clusters, cluster_method='kmeans', random_state)`: clusters pooled node embeddings across the dataset; exposes `fit`, `predict`, `fit_predict`.

- Scikit‑style wrapper
  - `AbstractGraphPreprocessor(...)`:
    - `fit(X, y)`: learns token embeddings with a simple Transformer and a graph‑level classifier head.
    - `transform(X)`: aggregates attention to a symmetric matrix `W`; builds NetworkX preimage graphs with robust edges and node embeddings.
    - `fit_transform(X, y)`: convenience.
    - `predict(X)`: graph‑level predictions using the classifier head.
    - `extract_node_embeddings(graphs)`: stacks node embeddings across graphs.
    - `assign_node_cluster_labels(graphs)`: applies `ImageNodeClusterer` and writes `cluster_id` per node.


## Output Schema

Output of `transform(X)` is a list of NetworkX graphs:

- Node attributes
  - `embedding`: np.ndarray shape `(d_model,)`
  - `label`: Any — optional value from `label_fn(instance_array, [token_index])`
  - `cluster_id`: int — only after `assign_node_cluster_labels`
- Edge attributes
  - `weight`: float — primary weight (consensus frequency if available, otherwise mean kept‑MST weight)
  - `consensus`: float — consensus frequency (optional)
  - `mst_count`: int — how many DP‑kept MSTs contributed (optional)
  - `mst_weight_mean`: float — mean of contributing MST weights (optional)
- Graph attributes
  - `co_cluster`: np.ndarray shape `(N, N)` — co‑clustering frequency across forests


## Hyperparameters and Practical Tips

- Attention aggregation
  - Layers/heads: attention is averaged over heads per layer, then averaged across layers; result symmetrized `W = 0.5*(W + W.T)`.
  - No positional encodings are used by default; tokens are treated as sets.

- Graph extraction
  - `K_mst` (default 3): number of disjoint MST iterations; higher can improve robustness up to diminishing returns.
  - `alpha`, `beta`: keep MST edge if `alpha*w >= beta`; larger `beta` yields more/smaller groups; `alpha` rescales the threshold.
  - `rho` (default 0.6): consensus threshold on co‑clustering frequencies; higher `rho` yields tighter groups.

- Transformer training
  - `d_model`, `n_heads`, `num_layers`, `dim_feedforward`, `dropout`: capacity/regularization knobs.
  - `n_epochs`, `lr`: simple Adam loop with batch size 1 to avoid padding; works for small datasets; consider more sophistication for scale.
  - `device`: `'auto' | 'cpu' | 'cuda'`.

- Node labeling and clustering
  - `label_fn(instance_array, token_indices)`: attach domain labels to image nodes (e.g., aggregate token metadata).
  - `ImageNodeClusterer`: requires scikit‑learn for KMeans; set `random_state` for reproducibility.


## Integration With The Rest Of The Repo

- Relationship to `AbstractGraph`
  - This preprocessor constructs a token‑level preimage graph (NetworkX). If you want to create an `AbstractGraph`, pass that graph to `AbstractGraph(graph=your_graph)` and proceed with operators/labeling.
  - When you do have a domain preimage graph already, prefer the operator + vectorizer stack (`abstract_graph_operators.py`, `vectorize.py`, `type.py`).

- Using with `estimator/graph_estimator.py` and `estimator/neural.py`
  - `GraphEstimator` expects features from `AbstractGraphTransformer`; it is not wired to consume the preprocessor’s output dicts.
  - `NeuralGraphEstimator` expects per‑node feature matrices from `AbstractGraphNodeTransformer` and builds its own Transformer; it is separate from this module.
  - If you want to plug attention‑derived groups into a downstream model, you can:
    - Use `nodes[*]['embedding']` as features for a custom classifier, or
    - Convert `nodes/edges` into a NetworkX graph for visualization or further processing.

- Minimal conversion to NetworkX for visualization

```python
import networkx as nx

G = nx.Graph()
for n in g0['nodes']:
    G.add_node(n['id'], size=len(n['token_indices']))
for e in g0['edges']:
    G.add_edge(e['source'], e['target'], weight=e['weight'])
# Now use your own plotting or adapt parts of `display.py`.
```


## Design Notes and Limitations

- Simplified DP: the “DP forest” reduces to a local keep/cut rule on MST edges. It is fast and works well with consensus but is not a full dynamic program over trees.
- Token‑only view: the method operates on token indices and their learned interactions; it does not consult a domain preimage graph unless you provide one via your `label_fn`.
- Small‑scale training loop: the built‑in loop trades sophistication for clarity; for larger tasks, consider batching with padding/masks and more epochs/schedules.
- Optional dependencies: `ImageNodeClusterer` uses scikit‑learn’s `KMeans`; import is guarded and only required if you instantiate/use the clusterer.


## Reference API

- `build_preimage_edges_from_attention(W, K_mst=3, alpha=1.0, beta=0.5, rho=0.6)` → `(edges, co_cluster)`
- `class ImageNodeClusterer(BaseEstimator)` with `fit`, `predict`, `fit_predict` over node embeddings `Z`.
- `class AbstractGraphPreprocessor(BaseEstimator, TransformerMixin)` with
  - `fit(X, y)` — train encoder + classifier
  - `transform(X)` — produce NetworkX preimage graphs
  - `fit_transform(X, y)`
  - `predict(X)` — graph‑level labels
  - `extract_node_embeddings(graphs)` — stack pooled node embeddings
  - `assign_node_cluster_labels(graphs)` — add `cluster_id` per node


## Related Files

- `abstract_graph_operators.py`: symbolic decomposition operators over `AbstractGraph`.
- `type.py`: core `AbstractGraph` data structure and `to_array()` vectorizer backend.
- `vectorize.py`: feature extractors built on `AbstractGraph`.
- `estimator/graph_estimator.py`: classical estimator wrapper with optional feature selection + manifold.
- `estimator/neural.py`: Transformer models over node‑level features from `AbstractGraphNodeTransformer`.

If you want this preprocessor’s output to interoperate more tightly with the symbolic pipeline, consider adding a small adapter that maps token groups to preimage subgraphs when you know the token↔preimage mapping in your domain.
