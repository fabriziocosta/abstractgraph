<p align="center">
  <img src="docs/assets/AG_Logo.png" alt="AbstractGraph logo" width="220">
</p>

# abstractgraph

`abstractgraph` is the semantic core of the AbstractGraph ecosystem.

It defines the representation used to treat an ordinary graph as an object that
can be decomposed, transformed, serialized, compared, vectorized, and inspected.
The package is intentionally focused on graph semantics rather than model
training, raw data ingestion, or generation.

For package layout, local setup, validation commands, and documentation index,
see [docs/ORGANIZATION.md](docs/ORGANIZATION.md).

## Core Idea

An Abstract Graph has two connected levels:

- a base graph, usually an original NetworkX graph
- an interpretation graph, whose nodes represent mapped subgraphs of the base
  graph

This two-level structure lets graph fragments become first-class objects. They
can be named, filtered, composed, hashed, serialized, converted into features,
or passed to downstream learning and generation layers.

## Semantic Role

`abstractgraph` answers questions such as:

- What does it mean to decompose a graph into meaningful mapped subgraphs?
- How can graph transformations be composed into reusable operator programs?
- How can graph fragments be compared, hashed, serialized, or vectorized?
- How can structural features remain traceable back to the subgraphs that
  produced them?

## Ecosystem Boundaries

This repository owns the shared abstract graph vocabulary.

- Raw domain objects become base graphs in
  [`abstractgraph-graphicalizer`](../abstractgraph-graphicalizer/README.md).
- Estimators, feasibility checks, and feature importance live in
  [`abstractgraph-ml`](../abstractgraph-ml/README.md).
- Rewriting, generation, interpolation, optimization, and repair live in
  [`abstractgraph-generative`](../abstractgraph-generative/README.md).

## Main Concepts

### Operators

Operators describe graph decompositions and transformations. They are the
composable language for turning a base graph into an interpretation graph and
for refining which structural fragments matter.

### Serialization

Operator pipelines can be serialized and loaded again, which makes graph
programs portable across examples, experiments, and downstream packages.

### Hashing and Vectorization

Mapped subgraphs can be converted into stable feature identities and feature
matrices. This is the bridge from graph semantics into machine learning.

### Display and Inspection

Display helpers keep abstract graph structure interpretable by showing base
graphs, mapped subgraphs, and operator outputs.

## Ecosystem

See the [AbstractGraph ecosystem README](../../README.md) for how this
repository fits with the sibling repositories.
