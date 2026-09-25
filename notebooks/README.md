# Core notebooks

This folder contains notebooks centered on the core AbstractGraph data model,
operators, XML round-trips, preprocessing, and vectorization behavior.

Layout:
- `examples/` for user-facing core workflows
- `research/` for exploratory core-only notebooks

Recommended sequence:
- `examples/graph_hashing_directed_vs_undirected.ipynb`
  Small labeled graphs showing how graph hashes differ for undirected and
  directed edge semantics.
- `examples/01_unary_decompositions.ipynb`
  Step 1: unary decomposition operators and visual intuition.
- `examples/02_composition_and_add.ipynb`
  Step 2: `compose`, `forward_compose`, and additive unions.
- `examples/03_merge_and_complements.ipynb`
  Step 3: `merge`, `complement`, and `edge_complement` for aggregate and outside-context views.
- `examples/04_filters_and_selection.ipynb`
  Step 4: structural filters, label filters, and deterministic subsampling.
- `examples/05_binary_and_combination_operators.ipynb`
  Step 5: combinations, intersections, binary operators, and shortest-path unions.
- `examples/06_control_flow_and_conditionals.ipynb`
  Step 6: use branching and loop operators to build conditional graph programs.
- `examples/07_xml_and_operator_serialization.ipynb`
  Step 7: XML registration, serialization, deserialization, and round-trips.
- `examples/08_vectorization_and_features.ipynb`
  Step 8: node-level vectorization, graph-level aggregation, and batch transformers.
- `examples/09_preprocessor_attention_pipeline.ipynb`
  Step 9: attention-derived base-graph construction through
  `abstractgraph-graphicalizer` and handoff into operators.
- `examples/10_feature_inspection_and_subgraphs.ipynb`
  Step 10: inspect hashed feature labels by mapping them back to representative subgraphs.

Reference notebook:
- `examples/overview.ipynb`
  Broad operator sampler covering unary, compositional, filtering, XML, and meta operators.
- `examples/custom_operator_scaffold.ipynb`
  How to write custom node-induced and edge-induced operators with the public scaffold helpers.

Bootstrap behavior:
- Notebooks use `notebooks/_bootstrap.py` to locate the repo root.
- They prepend available sibling `src/` directories to `sys.path`.
- They normalize the working directory to the repo root so relative paths are
  stable across Jupyter launch locations.
