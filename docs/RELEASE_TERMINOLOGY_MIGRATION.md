# Terminology Migration Summary

This note summarizes the ecosystem-wide rename from the old
`preimage/image/association` vocabulary to the canonical
`base/interpretation/mapped_subgraph` vocabulary.

## Canonical Vocabulary

- `preimage graph` -> `base graph`
- `image graph` -> `interpretation graph`
- `association` -> `mapped_subgraph`
- `AbstractGraph` -> the pair `(base_graph, interpretation_graph)`

## What Changed

Across `abstractgraph`, `abstractgraph-ml`, and `abstractgraph-generative`:

- core `AbstractGraph` APIs now expose `base_graph` and
  `interpretation_graph`
- interpretation nodes now canonically store `mapped_subgraph`
- vectorization, feature inspection, display, and operator code now use the
  canonical ontology internally
- active docs and active example notebooks now teach the canonical terms
- current generative APIs now provide canonical entry points such as
  `base_cut_radius`, `interpretation_cut_radius`,
  `interpretation_graph_pool`, `interpretation_graphs`,
  `fixed_interpretation_graph`, and `return_interpretation_steps`

## What Still Works

The migration is staged. Deprecated aliases still work in the compatibility
release line, including:

- `preimage_graph`
- `image_graph`
- `association`
- `create_*image_node*`
- `get_image_nodes_associations()`
- `get_preimage_nodes_inverse_associations()`
- `preimage_cut_radius`
- `image_cut_radius`
- `image_graph_pool`
- `generate(..., image_graphs=...)`
- `fixed_image_graph`
- `return_image_steps`

These aliases are compatibility shims. New code should not introduce fresh uses
of them.

## What To Update First

If you maintain downstream code, update in this order:

1. replace direct property access with `base_graph` and
   `interpretation_graph`
2. replace node payload reads of `association` with `mapped_subgraph`
3. replace old generative keyword arguments with canonical names
4. update notebooks, scripts, and docs so they stop teaching deprecated terms

## Compatibility Horizon

The current release line is the compatibility release line.

Deprecated aliases are intended to stay available for this release line and
then be removed together in the next breaking cleanup pass, once downstream
usage has been updated.

## Related Docs

- [MIGRATION_TERMINOLOGY.md](MIGRATION_TERMINOLOGY.md)
- [TERMINOLOGY_AUDIT.md](TERMINOLOGY_AUDIT.md)
