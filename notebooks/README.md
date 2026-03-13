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
- `examples/example_abstract_graph_operators_03_filters_and_selection.ipynb`
  Step 3: structural filters, label filters, and deterministic subsampling.
- `examples/example_abstract_graph_operators_04_binary_and_combination_operators.ipynb`
  Step 4: combinations, intersections, binary operators, and shortest-path unions.
- `examples/example_abstract_graph_operators_05_xml_and_operator_serialization.ipynb`
  Step 5: XML registration, serialization, deserialization, and round-trips.
- `examples/example_abstract_graph_operators_06_vectorization_and_features.ipynb`
  Step 6: node-level vectorization, graph-level aggregation, and batch transformers.
- `examples/example_abstract_graph_operators_07_preprocessor_attention_pipeline.ipynb`
  Step 7: attention-derived preimage graph construction and handoff into operators.
- `examples/example_abstract_graph_operators_08_feature_inspection_and_subgraphs.ipynb`
  Step 8: inspect hashed feature labels by mapping them back to representative subgraphs.

Reference notebook:
- `examples/example_abstract_graph_operators_overview.ipynb`
  Broad operator sampler covering unary, compositional, filtering, XML, and meta operators.
