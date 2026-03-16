# Terminology Audit

This document records the remaining old-terminology surface after the staged
rename migration.

The goal is not to eliminate every old token immediately. The goal is to
separate:

- deliberate compatibility aliases
- active public APIs not yet renamed
- documentation debt
- explicit legacy boundaries

## Category 1: Deliberate Compatibility Aliases

These are expected and currently intentional.

Core compatibility in [graphs.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/graphs.py):

- deprecated properties `preimage_graph` and `image_graph`
- deprecated helper methods `create_image_node_*`
- deprecated query method `get_image_nodes_associations()`
- payload fallback `association`

ML and display compatibility reads:

- [importance.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph-ml/src/abstractgraph_ml/importance.py)
- [neural.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph-ml/src/abstractgraph_ml/neural.py)
- [display.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/display.py)

These modules still read `mapped_subgraph` first and then fall back to
`association`.

## Category 2: Current Active APIs Still Teaching Old Terms

These are active non-legacy APIs whose names or docstrings still expose old
terminology.

Generative public surface in
[conditional.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph-generative/src/abstractgraph_generative/conditional.py):

- `_image_node_type(...)`
- image-oriented docstrings around node ordering and target selection
- parameter names like `target_image` in internal helper calls

Generative public surface in
[autoregressive.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph-generative/src/abstractgraph_generative/autoregressive.py):

- image-oriented pruning terminology
- deprecated compatibility aliases `fixed_image_graph` and `return_image_steps`

These should be handled in an explicit staged API pass, not opportunistically.

## Category 3: Documentation Debt

These are explanatory materials that still teach the old vocabulary even when
the implementation path is already canonical.

Largest remaining core cluster:

- [operators.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/operators.py)

This file now uses the canonical implementation paths internally, but many
docstrings still describe operators in terms of `preimage graph`, `image node`,
and `association`.

Remaining generative documentation debt:

- [rewrite.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph-generative/src/abstractgraph_generative/rewrite.py)
- lower-priority method docstrings in
  [conditional.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph-generative/src/abstractgraph_generative/conditional.py)
- [README_CONTITIONAL_AUTOREGRESSIVE.md](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph-generative/src/abstractgraph_generative/README_CONTITIONAL_AUTOREGRESSIVE.md)
  still has some compatibility-facing mentions of `preimage_cut_radius`

Notebook/example debt still tied to old public names:

- core notebooks that still use `number_of_image_graph_nodes(...)`
- generative notebooks that still use `preimage_cut_radius`,
  `image_cut_radius`, `fixed_image_graph`, or `return_image_steps`
- notebook prose that still says `preimage` / `image` when explaining active flows

## Category 4: Explicit Legacy Boundary

These are intentionally excluded from full migration for now.

- `abstractgraph_generative.legacy.*`
- `abstractgraph_generative/conditional_v0_1/*`
- research notebooks and scratch notebooks

Old terminology inside those locations is not currently considered a migration
bug.

## Recommended Next Passes

1. Finish the docstring rewrite in
   [operators.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/operators.py).
2. Finish the remaining current generative docstring cleanup in
   [conditional.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph-generative/src/abstractgraph_generative/conditional.py)
   and
   [rewrite.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph-generative/src/abstractgraph_generative/rewrite.py).
3. Update notebooks that still intentionally demonstrate deprecated helper
   names once the corresponding public alias plan is final.
