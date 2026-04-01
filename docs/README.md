# abstractgraph docs

This document is the package-level map for the standalone `abstractgraph` repo.
It replaces the old monorepo-oriented overview and focuses on the core package
surface that other repos depend on.

## Purpose

`abstractgraph` provides the core intermediate representation used across the
split repositories:

- `abstractgraph`: core graph abstraction and utilities
- `abstractgraph-ml`: estimators and analysis built on top of the core
- `abstractgraph-generative`: rewrite, autoregressive, interpolation, and story
tooling built on top of the core and ML layers

The core package is responsible for:
- representing abstract graphs
- decomposing and transforming graphs through operators
- serializing operator pipelines
- visualizing graph structures and decompositions
- hashing and vectorizing subgraph structure
- adapting non-graph inputs into base NetworkX graphs

## Ecosystem

Sibling repositories:

- `abstractgraph`
  Path: `/home/fabrizio/work/abstractgraph`
- `abstractgraph-ml`
  Path: `/home/fabrizio/work/abstractgraph-ml`
- `abstractgraph-generative`
  Path: `/home/fabrizio/work/abstractgraph-generative`
- `abstractgraph-graphicalizer`
  Path: `/home/fabrizio/work/abstractgraph-graphicalizer`

See [../ECOSYSTEM.md](../ECOSYSTEM.md) for the install order and dependency
direction.

## Module map

- `abstractgraph.graphs`
  `AbstractGraph`, `graph_to_abstract_graph`, `graphs_to_abstract_graphs`
- `abstractgraph.labels`
  default label, attribute, and edge functions
- `abstractgraph.hashing`
  stable bounded hashes, graph hashes, dedup helpers
- `abstractgraph.operators`
  decomposition operators and composition helpers
- `abstractgraph.xml`
  XML round-trip for decomposition pipelines
- `abstractgraph.display`
  rendering for base graphs, abstract graphs, grouped graph families,
  mappings, and operator DAGs
- `abstractgraph.vectorize`
  graph-level and node-level vectorizers
- `abstractgraph.feature_subgraphs`
  feature-to-subgraph inspection helpers
- `abstractgraph.utils`
  plotting and utility helpers shared by examples
- `abstractgraph.to_graph`
  graph-construction adapters such as NLP dependency parsing

## Key concepts

- Base graph:
  the original NetworkX graph
- Interpretation graph:
  a graph whose nodes store mapped base subgraphs
- Mapped subgraph:
  the base subgraph stored on an interpretation node, canonically in
  `node['mapped_subgraph']`
- Operator:
  a pure `AbstractGraph -> AbstractGraph` transformation
- Vectorization:
  hashing abstract subgraphs into a bounded feature space

## Typical workflow

1. Start from a base NetworkX graph.
2. Convert to `AbstractGraph`.
3. Apply one or more decomposition operators.
4. Call `update()` if labels/attributes/edges need recomputation.
5. Visualize with `abstractgraph.display`.
6. Vectorize with `abstractgraph.vectorize`.
7. Hand the result to `abstractgraph-ml` or `abstractgraph-generative`.

## Related docs

- [OPERATORS.md](OPERATORS.md)
- [VECTORIZATION.md](VECTORIZATION.md)
- [WHITE_PAPER.md](WHITE_PAPER.md)
- [RELEASE_TERMINOLOGY_MIGRATION.md](RELEASE_TERMINOLOGY_MIGRATION.md)
- [MIGRATION_TERMINOLOGY.md](MIGRATION_TERMINOLOGY.md)

## Notebook sequence

If you want a guided path through the package rather than a flat list of
examples, use the staged notebook sequence in `notebooks/examples/`:

1. `example_abstract_graph_operators_01_unary_decompositions.ipynb`
   Unary decomposition primitives.
2. `example_abstract_graph_operators_02_composition_and_add.ipynb`
   Composition order and additive unions.
3. `example_abstract_graph_operators_03_filters_and_selection.ipynb`
   Structural filtering and selection.
4. `example_abstract_graph_operators_04_binary_and_combination_operators.ipynb`
   Combination, intersection, and binary operators.
5. `example_abstract_graph_operators_05_control_flow_and_conditionals.ipynb`
   Control-flow operators for branching and bounded or predicate-driven iteration.
6. `example_abstract_graph_operators_06_xml_and_operator_serialization.ipynb`
   XML serialization and round-trip of operator programs.
7. `example_abstract_graph_operators_07_vectorization_and_features.ipynb`
   Node-level and graph-level vectorization.
8. `example_abstract_graph_operators_08_preprocessor_attention_pipeline.ipynb`
   Attention-derived base-graph construction through
   `abstractgraph-graphicalizer` and downstream handoff.
9. `example_abstract_graph_operators_09_feature_inspection_and_subgraphs.ipynb`
   Inspection of hashed feature labels via recurring representative subgraphs.

The older `example_abstract_graph_operators_overview.ipynb` remains useful as a
compact sampler, but it is no longer the best first notebook.
