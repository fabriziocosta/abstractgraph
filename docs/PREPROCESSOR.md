# Preprocessor migration

The attention-driven preprocessor has been extracted from `abstractgraph` into
`abstractgraph-graphicalizer`.

Use the new import path:

- `abstractgraph_graphicalizer.attention`

The old import path remains available temporarily through a compatibility shim:

- `abstractgraph.preprocessor`

## What it produces

`AbstractGraphPreprocessor.transform(X)` returns a list of NetworkX graphs whose
nodes represent input tokens and whose edges come from robust attention-derived
co-clustering.

These output graphs are plain NetworkX graphs. If you want to continue with the
operator pipeline, wrap them with:

```python
from abstractgraph.graphs import AbstractGraph

ag = AbstractGraph(graph=preimage_graph)
```

## Main classes

- `AbstractGraphPreprocessor`
  Transformer-like wrapper that learns token embeddings and induces robust
  token-level graph structure.
- `ImageNodeClusterer`
  Optional dataset-level clustering over node embeddings.

## Position in the split repos

- `abstractgraph`
  keeps a temporary compatibility shim only
- `abstractgraph-graphicalizer`
  owns the preprocessor implementation
- `abstractgraph-ml`
  can consume graphs or derived features downstream
- `abstractgraph-generative`
  may consume the produced graphs as preimages for later decomposition

## Minimal usage

```python
import numpy as np
from abstractgraph_graphicalizer.attention import AbstractGraphPreprocessor

X = [np.random.randn(12, 16), np.random.randn(10, 16)]
y = [0, 1]

pp = AbstractGraphPreprocessor(d_model=64, n_heads=4, num_layers=2, n_epochs=5)
pp.fit(X, y)
graphs = pp.transform(X)
```

## Output schema

Node attributes typically include:
- `embedding`
- `label` when `label_fn` is provided
- `cluster_id` after clustering

Graph attributes include:
- `co_cluster`

## When to use it

Use the preprocessor when:
- your raw inputs are token or sequence embeddings
- you want a learned preimage graph before any symbolic decomposition
- you want to inspect induced local structure before moving into the core
  operator/vectorizer stack

Use the operator pipeline instead when:
- you already have a meaningful base graph
- you want explicit motif definitions such as cycles, paths, neighborhoods, or
  cliques
