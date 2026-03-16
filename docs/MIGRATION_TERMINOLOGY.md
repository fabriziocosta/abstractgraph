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

The current policy is:

- the current release line is the compatibility release line
- deprecated aliases remain supported across the current release line
- new code, new examples, and active documentation should use only canonical
  names
- alias removal should happen in the next planned breaking cleanup pass after
  the compatibility release line

Until a versioned release process is defined more formally, treat the next
breaking cleanup pass as the point where deprecated aliases may be removed.

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

## Compatibility Matrix

| Old name | Canonical name | Supported now | Warning behavior | Planned status |
| --- | --- | --- | --- | --- |
| `preimage graph` | `base graph` | docs compatibility only | no runtime warning | remove from active docs |
| `preimage_graph` | `base_graph` | yes | deprecated on public property access | remove after compatibility window |
| `image graph` | `interpretation graph` | docs compatibility only | no runtime warning | remove from active docs |
| `image_graph` | `interpretation_graph` | yes | deprecated on public property access | remove after compatibility window |
| `association` | `mapped_subgraph` | yes | compatibility fallback in readers; warnings only on explicit deprecated entry points | remove fallback after compatibility window |
| `create_default_image_node(...)` | `create_default_interpretation_node(...)` | yes | deprecated wrapper | remove after compatibility window |
| `create_image_node_with_subgraph_from_nodes(...)` | `create_interpretation_node_with_subgraph_from_nodes(...)` | yes | deprecated wrapper | remove after compatibility window |
| `create_image_node_with_subgraph_from_edges(...)` | `create_interpretation_node_with_subgraph_from_edges(...)` | yes | deprecated wrapper | remove after compatibility window |
| `create_image_node_with_subgraph_from_subgraph(...)` | `create_interpretation_node_with_subgraph_from_subgraph(...)` | yes | deprecated wrapper | remove after compatibility window |
| `get_image_nodes_associations()` | `get_interpretation_nodes_mapped_subgraphs()` | yes | deprecated wrapper | remove after compatibility window |
| `get_preimage_nodes_inverse_associations()` | `get_base_nodes_inverse_mappings()` | yes | deprecated wrapper | remove after compatibility window |
| `preimage_cut_radius` | `base_cut_radius` | yes | deprecated alias warning | remove after compatibility window |
| `image_cut_radius` | `interpretation_cut_radius` | yes | deprecated alias warning | remove after compatibility window |
| `image_graph_pool` | `interpretation_graph_pool` | yes | deprecated alias warning | remove after compatibility window |
| `generate(..., image_graphs=...)` | `generate(..., interpretation_graphs=...)` | yes | deprecated alias warning | remove after compatibility window |
| `fixed_image_graph` | `fixed_interpretation_graph` | yes | deprecated alias warning | remove after compatibility window |
| `return_image_steps` | `return_interpretation_steps` | yes | deprecated alias warning | remove after compatibility window |

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
- `generate_pruning_sequences(..., fixed_interpretation_graph=...)`
- `generate_pruning_sequences(..., return_interpretation_steps=...)`

The older generative names still work as compatibility aliases:

- `preimage_cut_radius`
- `image_cut_radius`
- `image_graph_pool`
- `generate(..., image_graphs=...)`
- `fixed_image_graph`
- `return_image_steps`

## Scope

This migration applies across:

- `abstractgraph`
- `abstractgraph-ml`
- `abstractgraph-generative`

Explicit legacy namespaces may retain old vocabulary temporarily for backward
compatibility, but active non-legacy code and active documentation should
converge on the canonical terms above.

## Removal Plan

The intended removal grouping is:

- core aliases:
  `preimage_graph`, `image_graph`, `association`,
  `create_*image_node*`, `get_image_nodes_associations()`,
  `get_preimage_nodes_inverse_associations()`
- generative aliases:
  `preimage_cut_radius`, `image_cut_radius`, `image_graph_pool`,
  `generate(..., image_graphs=...)`, `fixed_image_graph`,
  `return_image_steps`

Private helper names do not define the compatibility contract. They should be
cleaned opportunistically, but the removal plan is driven by public API and
user-facing documentation first.

Before the removal pass, the project should confirm:

- the last compatibility release line
- the exact set of aliases removed together
- whether any remaining compatibility reads should start warning first

See [TERMINOLOGY_AUDIT.md](TERMINOLOGY_AUDIT.md) for the current remaining
surface.
