# Core notebooks

This folder contains notebooks centered on the core AbstractGraph data model,
operators, XML round-trips, preprocessing, and vectorization behavior.

Layout:
- `examples/` for user-facing core workflows
- `research/` for exploratory core-only notebooks

Recommended sequence:
- `examples/example_abstract_graph_operators_01_unary_decompositions.ipynb`
  Step 1: unary decomposition operators and visual intuition.
- `examples/example_abstract_graph_operators_02_composition_and_add.ipynb`
  Step 2: `compose`, `forward_compose`, and additive unions.
- `examples/example_abstract_graph_operators_03_merge_and_complements.ipynb`
  Step 3: `merge`, `complement`, and `edge_complement` for aggregate and outside-context views.
- `examples/example_abstract_graph_operators_04_filters_and_selection.ipynb`
  Step 4: structural filters, label filters, and deterministic subsampling.
- `examples/example_abstract_graph_operators_05_binary_and_combination_operators.ipynb`
  Step 5: combinations, intersections, binary operators, and shortest-path unions.
- `examples/example_abstract_graph_operators_06_control_flow_and_conditionals.ipynb`
  Step 6: use branching and loop operators to build conditional graph programs.
- `examples/example_abstract_graph_operators_07_xml_and_operator_serialization.ipynb`
  Step 7: XML registration, serialization, deserialization, and round-trips.
- `examples/example_abstract_graph_operators_08_vectorization_and_features.ipynb`
  Step 8: node-level vectorization, graph-level aggregation, and batch transformers.
- `examples/example_abstract_graph_operators_09_preprocessor_attention_pipeline.ipynb`
  Step 9: attention-derived base-graph construction through
  `abstractgraph-graphicalizer` and handoff into operators.
- `examples/example_abstract_graph_operators_10_feature_inspection_and_subgraphs.ipynb`
  Step 10: inspect hashed feature labels by mapping them back to representative subgraphs.

Reference notebook:
- `examples/example_abstract_graph_operators_overview.ipynb`
  Broad operator sampler covering unary, compositional, filtering, XML, and meta operators.

Bootstrap behavior:
- Notebooks use `notebooks/_bootstrap.py` to locate the repo root.
- They prepend available sibling `src/` directories to `sys.path`.
- They normalize the working directory to the repo root so relative paths are
  stable across Jupyter launch locations.
