# Terminology Migration

This document defines the ecosystem-wide terminology migration from the old
`preimage/image/association` vocabulary to the new
`base/interpretation/mapped_subgraph` vocabulary.

## Canonical Terms

- `preimage graph` -> `base graph`
- `preimage_graph` -> `base_graph`
- `image graph` -> `interpretation graph`
- `image_graph` -> `interpretation_graph`
- `association` -> `mapped_subgraph`
- "a preimage subgraph is associated to an image node" ->
  "a base subgraph is mapped to an interpretation node"

## Core Model

The canonical model is:

- `base graph`
  The original graph under study.
- `interpretation graph`
  The graph whose nodes represent mapped base subgraphs and whose edges
  represent relations between those interpreted structures.
- `AbstractGraph`
  The pair of `(base_graph, interpretation_graph)`.
- `mapped_subgraph`
  The base subgraph stored on an interpretation node.

## Compatibility Policy

The migration is staged, not immediate.

Currently supported deprecated aliases include:

- `AbstractGraph.preimage_graph`
- `AbstractGraph.image_graph`
- interpretation-node payload key `association`
- `create_default_image_node(...)`
- `create_image_node_with_subgraph_from_nodes(...)`
- `create_image_node_with_subgraph_from_edges(...)`
- `create_image_node_with_subgraph_from_subgraph(...)`
- `get_image_nodes_associations()`
- `get_preimage_nodes_inverse_associations()`

Deprecated aliases should remain thin wrappers over the canonical API. New code
should not introduce fresh uses of the deprecated vocabulary.

## Current Canonical Entry Points

In core `abstractgraph`:

- `AbstractGraph.base_graph`
- `AbstractGraph.interpretation_graph`
- interpretation-node payload key `mapped_subgraph`
- `create_default_interpretation_node(...)`
- `create_interpretation_node_with_subgraph_from_nodes(...)`
- `create_interpretation_node_with_subgraph_from_edges(...)`
- `create_interpretation_node_with_subgraph_from_subgraph(...)`
- `get_interpretation_nodes_mapped_subgraphs()`
- `get_base_nodes_inverse_mappings()`

In current generative code:

- constructor args `base_cut_radius` and `interpretation_cut_radius`
- generator property `interpretation_graph_pool`
- `generate(..., interpretation_graphs=...)`

The older generative names still work as compatibility aliases:

- `preimage_cut_radius`
- `image_cut_radius`
- `image_graph_pool`
- `generate(..., image_graphs=...)`

## Scope

This migration applies across:

- `abstractgraph`
- `abstractgraph-ml`
- `abstractgraph-generative`

Explicit legacy namespaces may retain old vocabulary temporarily for backward
compatibility, but active non-legacy code and active documentation should
converge on the canonical terms above.

## Removal Planning

Before removing deprecated aliases, the project should explicitly decide:

- which release is the last compatibility release
- which public aliases will be removed together
- whether current generative public names like `target_image_graph` and
  `fixed_image_graph` are part of that same removal wave

See [TERMINOLOGY_AUDIT.md](TERMINOLOGY_AUDIT.md) for the current remaining
surface.
