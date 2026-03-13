# Abstract Graph Module — Codebase Summary

This document summarizes the code under `AbstractGraph`. It provides a high‑level overview, file map, main abstractions and APIs, external dependencies, extension points, and notable design choices. It is intended to be used as context by LLMs instead of scanning the entire codebase.

## Purpose and Overview

- The module builds, composes, visualizes, serializes, and vectorizes Abstract Graphs (AGs).
- An Abstract Graph represents subgraphs from a base (preimage) graph as nodes in an image graph. Each image node stores its associated subgraph, a label, and aggregated attributes, with optional edges between image nodes derived from relations between their associated subgraphs.
- The toolkit includes:
  - The core `AbstractGraph` data structure and conversions to arrays/graphs.
  - Default label/attribute/edge functions.
  - A rich operator library for graph decomposition, filtering, and composition.
  - Visualization utilities for AGs and for operator pipelines.
  - XML round‑trip of operator pipelines for portability and reproducibility.
  - Vectorization to feature matrices for ML workflows.
    - `vectorize.py` calls `AbstractGraph.to_array()` to aggregate image-node attributes per hashed label, then reserves column 0 for bias and column 1 for degree.

## Directory Tree

```
AbstractGraph/
├─ definitions.py          # Default label/attribute/edge functions (hashing, aggregation, intersection edges)
├─ display.py              # Visualization of AGs and operator decomposition graphs (matplotlib/graphviz)
├─ abstract_graph_operators.py # Composable operators: composition, conditionals, loops, decompositions, filters, combinations
├─ abstract_graph_xml.py    # XML serialization/deserialization of operator pipelines; registries for ops/combiners
├─ type.py                 # Core `AbstractGraph` class; image/preimage graphs; update/apply; to_graph; to_array
├─ vectorize.py            # Vectorization helpers and transformers (graph/node level) using `AbstractGraph.to_array`
├─ topk.py                 # Top-K ranking/ROC helpers that build select-top operators and plot their performance
├─ importance.py           # Annotates graphs with ranked importance scores and renders importance grids
├─ interpolate.py          # Graph interpolation via AbstractGraph rewrites and embedding paths
├─ interpolation_generator.py # Generation loop for interpolation + model-based filtering
├─ feasibility.py          # Feasibility estimators for graph constraints and feature presence
```

## Main Modules, Classes, and Functions

### Core Type: `type.py`
- `class AbstractGraph`
  - Holds two NetworkX graphs:
    - `preimage_graph`: the original base graph.
    - `image_graph`: the abstract graph; nodes represent subgraph equivalence classes and store:
      - `association`: a NetworkX subgraph of the preimage graph.
      - `label`: computed by `label_function`.
      - `attribute`: aggregated by `attribute_function`.
  - Functional hooks:
    - `label_function(node_attrs) -> Any` (e.g., hash of `association`).
    - `attribute_function(subgraph) -> np.ndarray` (e.g., sum of node attributes).
    - `edge_function(ag) -> ag` (e.g., add edges for intersecting subgraphs).
  - Key methods:
    - Construction: `from_graph`, `from_abstract_graph`, `copy`.
    - Image node creation: `create_default_image_node` (whole preimage), `create_image_node_with_subgraph_from_nodes/edges/subgraph`.
    - Update: `apply_label_function`, `apply_attribute_function`, `apply_edge_function`, `update`.
    - Inverse associations: `get_preimage_nodes_inverse_associations()` builds induced subgraphs of image nodes for each preimage node.
    - Conversion: `to_graph(connection_label='abstract')` merges preimage+image into one graph; `to_array()` builds a sparse row per preimage node counting image labels.
    - Combination: `__add__` merges two AGs (`nx.compose` on preimage, `nx.disjoint_union` on image).

Interaction model:
- Operators (from `abstract_graph_operators.py`) receive a `AbstractGraph`, manipulate its image nodes/edges, and return a new or modified QG. Label/attribute/edge functions influence `update`/`to_array`/visualization.

### Defaults: `definitions.py`
- Label factories (attach `nbits` attribute used by `to_array` to size feature space 2**nbits):
  - `graph_hash_label_function_factory(nbits)` — hash of association subgraph (labels preserved).
  - `graph_structure_hash_label_function_factory(nbits)` — structure‑only hash (labels sanitized to `'-'`).
  - `source_function_hash_label_function_factory(nbits)` — bounded, stable hash of operator source identifier (from metadata).
- Attribute aggregation:
  - `sum_attribute_function(subgraph)` — sums `node['attribute']` arrays.
- Edge generation:
  - `intersection_edge_function(ag)` — connect image nodes whose associations share preimage nodes.
  - `null_edge_function(ag)` — no‑op default.

### Operators: `abstract_graph_operators.py`
Composable transformations and selectors over `AbstractGraph`. Many are `toolz.curry`‑compatible and carry metadata attributes for introspection and XML round‑trip (e.g., `operator_type`, `chain`, `decomposition_functions`, `combiner`).

- Higher‑order composition:
  - `add(*fns)` — apply each function on the same input QG; union outputs via `__add__`.
  - `compose(*fns)` — reverse‑order function composition (right‑to‑left).
  - `forward_compose(*fns)` — forward composition (left‑to‑right).
  - `compose_product(combiner, *fns)` — run in parallel and reduce with `combiner`.
- Conditionals and loops (curried):
  - `if_then_else`, `if_then_elif_else`, `for_loop`, `while_loop` — control flow around QG transformations.
- Structural primitives and decompositions (selected examples; full set is extensive):
  - `identity`, `node`, `edge` — basic passes/selectors.
  - Connected components: `connected_component_decomposition_function`, `connected_component`.
  - Degree‑based: `degree_decomposition_function`, `degree`.
  - Partitioning: `split_decomposition_function`, `split`; community: `kernighan_lin_bisection` usage.
  - Neighborhood/BFS helpers: `get_reachable_nodes_bfs`, `neighborhood`.
  - Cycles/trees/paths: `cycle_*`, `tree`, `path_*`.
  - Graphlets/cliques: `graphlet_*`, `clique_*`.
  - Set/transform utilities: `complement`, `merge`, label munging (`unlabel`, `prepend_label`).
- Combination and distance‑based pairing:
  - Distance utilities: `get_distance`, `get_distance_matrix`, `all_distances_are_feasible`.
  - `combination_decomposition_function`, `combination` and `binary_combination_*` with distance constraints.
- Filters:
  - By connected components, nodes, edges, node label, and quantiles/min/max on subgraph sizes.

Interaction model:
- Operators often construct new image nodes whose `association` subgraphs reference the preimage graph; later passes can label, aggregate attributes, and generate image‑image edges.
- High‑level pipelines are composed via `compose`/`add`/`compose_product`, visualized in `display.py`, and serializable with `abstract_graph_xml.py`.

### Visualization: `display.py`
- Color and drawing helpers:
  - `stable_hash` (MD5) and `get_color` map labels to colors deterministically via continuous colormaps (default `'hsv'`).
  - `display_graph(graph, ...)` draws a NetworkX graph onto a Matplotlib Axes with styling.
- abstract Graph view:
  - `display(qgraph, ...)` renders preimage (left) and image (right) with dashed connections from image nodes to the preimage nodes in their `association` subgraphs.
  - `display_mappings(qgraph, ...)` groups image nodes by label and renders representative subgraphs sorted by frequency.
- Decomposition graph (operator pipeline) visualization:
  - `decomposition_to_graph(comp_func)` builds a directed graph of an operator pipeline (functions, operators, parameters).
  - `display_decomposition_graph(comp_func_or_graph, ...)` renders the decomposition graph using Graphviz `dot` via `pygraphviz` and shows the image.

### XML Round‑Trip: `abstract_graph_xml.py`
- Registries:
  - `OPERATOR_REGISTRY`, `COMBINER_REGISTRY`, and helpers `register_operator`, `register_combiner`, `register_from_module` (bulk register from a module, e.g., `abstract_graph_operators.py`).
- Serialization:
  - `operator_to_xml_element/op_string(op)` emits `<Operator type=...>` with sorted bound kwargs, child operators, and combiner either as a legacy `combiner` attribute (registered named) or nested `<Combiner><Operator/></Combiner>` for full param round‑trip.
- Deserialization:
  - `operator_from_xml_element/op_string(xml)` rebuilds curried/normal operators, handling `compose`, `forward_compose`, `add`, `compose_product` (legacy `product` accepted), conditionals/loops, and leaf curried ops; wraps builders expecting `ag` as first arg.

Interaction model:
- Pipelines defined with functions from `abstract_graph_operators.py` can be serialized to XML, shared, versioned, then reconstructed back into callable pipelines after registering the referenced names.

### Vectorization: `vectorize.py`
- `vectorize(ag, nbits, return_dense)`
  - Ensures `ag.label_function` uses the requested `nbits` (so `to_array()` sizes to 2**nbits).
  - Calls `ag.to_array()` to get a CSR count matrix `(n_preimage_nodes × 2**nbits)` of image labels per preimage node.
  - Overwrites column 0 with ones (bias) and column 1 with preimage node degrees; returns dense or CSR.
- Transformers (joblib‑parallel):
  - `AbstractGraphTransformer(nbits, decomposition_function, return_dense=True, n_jobs=-1)` — for graph‑level features; applies a decomposition to each graph, vectorizes, and sums rows to one vector per graph; stacks outputs.
  - `AbstractGraphNodeTransformer(...)` — returns per‑graph matrices (no graph‑level sum), one matrix per input graph.

Interaction model:
- Designed for ML pipelines: select a decomposition recipe, vectorize graphs into fixed-length features, and feed into downstream models.

### Top-K Ranking & Evaluation: `topk.py`
- `make_topk_df` builds Top-K decomposition pipelines by tracking feature importances across CV folds, ranking features, and composing `select_top_by_feature_ranking`.
- `compute_topk_roc_results` and `plot_topk_roc_curve` evaluate ROC-AUC distributions per Top-K operator using injected estimator/metric helpers.
- `plot_topk_roc_curves` wraps the previous helpers to compare multiple decomposition recipes on a shared axes with consistent IQR shading and legends.

### Importance Visualization: `importance.py`
- `annotate_graph_node_saliency` maps ranked feature labels back to the preimage graph, normalizes scores to [0,1], and stores per-node/edge importance attributes.
- `plot_graph_node_saliency` renders those annotations with customized colormaps/widths and reuses `display.py`’s color hashing for consistent node colors.
- `plot_graph_node_saliency_with_estimator`/`plot_graph_node_saliency_grid` combine annotation + plotting for single or grid layouts, handy for inspecting multiple molecules or subgraphs.

### Interpolation: `interpolate.py`
This module implements graph interpolation using AbstractGraph rewrites. It builds AbstractGraph representations for a source graph and a set of donors, computes cut-signature compatibility, and performs image-node swaps to produce structurally consistent mutations. Interpolation uses an embedding space (from a `GraphTransformer`) with MST+mutual-kNN neighborhood graphs and shortest-path routing to guide stepwise rewrites.

### Generation: `interpolation_generator.py`
This module implements the interpolation-driven generation loop. `InterpolationGenerator` iteratively expands a pool of candidates using a predictive model to score and retain high-quality intermediates.

### Feasibility Estimators: `feasibility.py`
Feasibility utilities provide scikit-style estimators that check structural constraints (e.g., label presence, size ranges, connectivity) and feature-based conditions. These estimators expose `fit`, `predict`, and `number_of_violations` to integrate with filtering or generative loops, and include helpers that compute AbstractGraph-derived features (via `AbstractGraphTransformer`) to enforce "must exist" or "cannot exist" feature constraints.

## Key External Libraries

- NetworkX (`networkx`): graph data structures, subgraphing, layouts, algorithms (connected components, cycles, etc.).
- NumPy (`numpy`): numeric arrays and aggregation.
- SciPy sparse matrices (`scipy.sparse`): `lil_matrix`, `csr_matrix`, `vstack` for efficient count matrices.
- Matplotlib (`matplotlib.pyplot`, `matplotlib.cm`): graph visualization; color maps.
- PyGraphviz (`networkx.drawing.nx_agraph.to_agraph`): Graphviz `dot` layout and rendering of decomposition graphs.
- Toolz (`toolz.curry`): currying operators and preserving metadata.
- Joblib (`joblib.Parallel`, `delayed`): parallel transform over input graphs.
- XML/AST (`xml.etree.ElementTree`, `ast.literal_eval`): safe XML parameter encoding/decoding.
- Stdlib helpers: `hashlib` (stable hash), `inspect` (introspection/closures), `itertools` (combinations/product), `warnings`.
- Internal hashing: `hash_graph`, `hash_bounded` (stable, bounded hashes for label spaces).

## Extension and Modification Points

- Label/attribute/edge behavior:
  - Provide custom `label_function(node_attrs)`, `attribute_function(subgraph)`, and `edge_function(ag)` when constructing `AbstractGraph`.
  - Create new label factories that attach `.nbits` for compatibility with `to_array()`.
- Operators:
  - Add new decomposition/transformation functions in `abstract_graph_operators.py` following existing patterns (optionally `@curry`).
  - Compose higher‑level pipelines with `compose`/`add`/`compose_product`.
- XML round‑trip:
  - Register new operators/combiners via `register_operator` / `register_combiner` (or `register_from_module`) to serialize/deserialize pipelines.
  - For complex combiners, prefer nested `<Combiner>` encoding to preserve parameters.
- Visualization:
  - Extend node/edge styling in `display_graph`/`display` or add new visual encodings for labels/attributes.
- Vectorization:
  - Customize `vectorize` (e.g., different reserved columns or attribute aggregations) or add new transformers.

## Notable Conventions, Assumptions, and Design Choices

- Image node association:
  - Each image node stores an `association` subgraph of the preimage graph; connections for visualization and counts derive from these associations.
- Feature space sizing:
  - Label functions used by `to_array()` are expected to carry an `nbits` attribute; feature dimension is `2**nbits`. The provided hash‑based factories set `.nbits`.
- Reserved feature columns:
  - `vectorize` reserves column 0 for bias and column 1 for preimage node degree; `nbits >= 2` is enforced.
- Graph merging semantics:
  - `AbstractGraph.__add__` composes preimage graphs (`nx.compose`) and disjoint‑unions image graphs; image node indices are re‑based accordingly.
- Operator metadata:
  - Operators carry introspection attributes (`operator_type`, `chain`, `decomposition_functions`, `combiner`) enabling visualization and XML round‑trip.
- XML encoding of callables:
  - Callable parameter values are encoded as `ref:<name>` and resolved via registries; only Python literals are embedded directly using `repr`/`literal_eval`.
- Color stability:
  - Visualization uses MD5‑based stable hashing and continuous colormaps to minimize collisions across sessions.
- Empty graph safety:
  - Layout, display, and vectorization paths include guardrails for empty preimage/image graphs.

## Practical Usage Pattern

1) Build a `AbstractGraph` from a base NetworkX graph and create one or more image nodes (e.g., `create_default_image_node`).
2) Apply a decomposition pipeline (operators from `abstract_graph_operators.py`).
3) Update or vectorize:
   - `ag.update()` to populate labels/attributes/edges as needed.
   - `vectorize(ag, nbits=..., return_dense=...)` to get features for ML.
4) Visualize with `display(ag, ...)` and/or export pipeline with `abstract_graph_xml`.

---

This summary covers the public surface and interaction patterns most relevant for analysis and extension by an LLM.
