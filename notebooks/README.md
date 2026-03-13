# Core notebooks

This folder contains notebooks centered on the core AbstractGraph data model,
operators, XML round-trips, preprocessing, and vectorization behavior.

Layout:
- `examples/` for user-facing core workflows
- `research/` for exploratory core-only notebooks

Notable examples:
- `examples/example_abstract_graph_operators_overview.ipynb`
  Broad operator sampler covering unary, compositional, filtering, XML, and meta operators.
- `examples/example_abstract_graph_operators_01_unary_decompositions.ipynb`
  First staged walkthrough focused only on unary decomposition operators and their visualizations.
- `examples/example_abstract_graph_operators_02_composition_and_add.ipynb`
  Second staged walkthrough focused on `compose`, `forward_compose`, order sensitivity, and additive operator unions.
- `examples/example_abstract_graph_operators_03_filters_and_selection.ipynb`
  Third staged walkthrough focused on structural filters, label-based filters, and deterministic subsampling.
