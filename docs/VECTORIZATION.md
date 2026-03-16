# Vectorization

This document explains the main idea behind vectorization in the current
`abstractgraph` codebase.

The vectorization stack turns an `AbstractGraph` into machine-learning-ready
features by counting how interpretation structures attach back to base-graph
nodes.

## Main Idea

The central representation is not a flat fingerprint computed directly from the
base graph.

Instead, the pipeline is:

1. start from a base graph
2. build an `AbstractGraph`
3. generate interpretation nodes with operators
4. assign each interpretation node a bounded integer label by hashing its
   associated subgraph
5. project those labeled interpretation nodes back onto the base nodes they
   touch
6. count the resulting label occurrences into a fixed-width feature space

So the vectorizer does not ask only:

> what structures exist in the graph?

It asks:

> for each base node, which interpreted structures contain it?

This is the key design choice.

## Core Representation

The relevant implementation lives in:

- [vectorize.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/vectorize.py)
- [graphs.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/graphs.py)
- [labels.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/labels.py)
- [hashing.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/hashing.py)
- [feature_subgraphs.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/feature_subgraphs.py)

An `AbstractGraph` stores:

- `base_graph`: the original graph
- `interpretation_graph`: a graph whose nodes store mapped subgraphs of the
  base graph

Each interpretation node typically has:

- `mapped_subgraph`: the mapped base subgraph
- `label`: a bounded integer feature id
- `attribute`: a scalar or vector weight contributed by that interpretation node
- `meta`: operator provenance or user metadata

## How Labels Are Built

The default vectorization path uses
`graph_hash_label_function_factory(nbits)`.

That label function hashes each interpretation node's `mapped_subgraph` with
`hash_graph(...)` and maps it into a bounded range determined by `nbits`.

Important implementation detail:

- the total feature space size is `2**nbits`
- hash values are restricted to `[2, 2**nbits - 1]`
- columns `0` and `1` are intentionally reserved

The reserved columns are used later for:

- column `0`: constant node-existence feature
- column `1`: base-node degree

This means the structural hashed features start at column `2`.

## What `to_array()` Actually Computes

The core counting logic is in
[graphs.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/graphs.py).

`AbstractGraph.to_array()` builds a sparse matrix with:

- rows = base-graph nodes
- columns = label buckets, optionally expanded by attribute dimension

The process is:

1. apply the interpretation-node label function
2. apply the interpretation-node attribute function
3. for each base node, collect the set of interpretation nodes whose mapped
   subgraphs contain that base node
4. for each such interpretation node, add its attribute vector into the feature slice
   corresponding to its label

So each row describes the interpretation context of one base node.

If an interpretation node has scalar attribute `1`, it contributes a count.
If it has a vector attribute, it contributes that vector into the slice for its
label.

Conceptually, this is a counting measure over interpreted structures incident
to each base node.

## What `vectorize()` Adds

The public `vectorize(...)` function in
[vectorize.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/vectorize.py)
starts from `to_array()` and then overwrites the first two columns:

- column `0` becomes all ones
- column `1` becomes the degree of each base node in the base graph

The result is a node-feature matrix where each row contains:

- a constant node-existence feature
- a local structural baseline from graph degree
- hashed counts of interpreted subgraphs touching that node

Column `0` matters for two reasons.

First, when node rows are pooled into a graph-level vector, summing that column
produces the number of instantiated nodes in the graph. So column `0` is not
just an abstract intercept term; it is also a direct graph-size feature under
sum pooling.

Second, in generative settings this column can act as a node-existence signal.
When a model allocates a fixed maximum number of node slots and uses padding,
column `0` helps distinguish actual instantiated nodes from empty or inactive
slots.

This is the main node-level representation used by the library.

## Node-Level vs Graph-Level Vectorization

There are two public transformer classes in
[vectorize.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/vectorize.py).

`AbstractGraphNodeTransformer`

- converts each input graph into an `AbstractGraph`
- creates a default interpretation node
- applies a decomposition function
- returns the node-level feature matrix produced by `vectorize(...)`

`AbstractGraphTransformer`

- runs the same decomposition and node-level vectorization
- then sums the node rows over the whole graph
- returns one feature vector per input graph

So graph-level features are not computed independently. They are the aggregate
of node-level interpretation counts.

This gives a simple hierarchy:

- node level: which interpreted structures touch this base node?
- graph level: how much total interpreted structure of each type exists in this
  graph?

## Why Hashing Is Used

The system allows potentially many generated subgraphs, so it needs a bounded
feature space.

`hash_graph(...)` provides a deterministic graph hash based on:

- node labels
- edge labels
- rooted neighborhood structure
- order-independent aggregation where appropriate

The final result is reduced to a bounded integer index using `nbits`.

This gives:

- fixed feature width
- deterministic labels across runs
- a practical way to feed structural programs into standard ML models

The tradeoff is collision risk. Different subgraphs can map to the same bucket,
especially when `nbits` is small.

## How Features Stay Interpretable

The codebase does not stop at hashing.

[feature_subgraphs.py](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/abstractgraph/src/abstractgraph/feature_subgraphs.py)
groups representative mapped subgraphs by interpretation-node label across a set of
graphs.

That module lets you inspect:

- which hashed labels appear
- which mapped subgraphs produced them
- recurring representative subgraphs for a given label

So the intended workflow is:

1. generate hashed structural features for learning
2. inspect representative subgraphs for active labels afterward

This is the code-level bridge between vectorization and interpretation.

## Practical Reading of the Design

The main idea of vectorization in this repo is:

> build interpretation nodes first, then count their incidence over base nodes

This is different from treating the base graph as a bag of local patterns from
the start.

The representation is explicitly two-stage:

- operator programs generate structural objects
- vectorization turns those objects into bounded count features

That separation matters because it keeps the semantic object in the
interpretation graph, while the vector is only the downstream ML encoding.

## Current Limitations and Notes

- Some compatibility aliases still exist in the API for one migration window.
- The default label function hashes the full associated subgraph, so feature
  identity depends on the graph hashing scheme.
- Collisions are unavoidable in bounded hashing.
- The graph-level transformer simply sums node rows, so large graphs naturally
  produce larger raw counts unless downstream normalization is added elsewhere.
- `feature_subgraphs.py` is the main mechanism for recovering representative
  subgraphs from hashed labels.

## Short Version

Vectorization in `abstractgraph` is a projection-and-count pipeline:

- generate interpretation nodes from mapped base subgraphs
- hash each interpreted subgraph into a bounded label
- project those labels back to the base nodes contained in each subgraph
- count them into node-level feature rows
- optionally sum node rows to obtain graph-level vectors

The vector is therefore a compressed summary of how generated structural
interpretations are distributed over the base graph.
