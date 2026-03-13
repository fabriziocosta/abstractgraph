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
- building attention-derived preimage graphs
- adapting non-graph inputs into base NetworkX graphs

## Ecosystem

Sibling repositories:

- `abstractgraph`
  Path: `/home/fabrizio/work/abstractgraph`
- `abstractgraph-ml`
  Path: `/home/fabrizio/work/abstractgraph-ml`
- `abstractgraph-generative`
  Path: `/home/fabrizio/work/abstractgraph-generative`

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
  rendering for base graphs, abstract graphs, mappings, and operator DAGs
- `abstractgraph.vectorize`
  graph-level and node-level vectorizers
- `abstractgraph.preprocessor`
  attention-driven preimage graph builder
- `abstractgraph.feature_subgraphs`
  feature-to-subgraph inspection helpers
- `abstractgraph.utils`
  plotting and utility helpers shared by examples
- `abstractgraph.to_graph`
  graph-construction adapters such as NLP dependency parsing

## Key concepts

- Preimage graph:
  the original NetworkX graph
- Image graph:
  a graph whose nodes store associated subgraphs of the preimage
- Association:
  the subgraph stored on an image node, usually in `node['association']`
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
- [PREPROCESSOR.md](PREPROCESSOR.md)
- [WHITE_PAPER.md](WHITE_PAPER.md)
