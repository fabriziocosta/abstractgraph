# abstractgraph

`abstractgraph` is the core package for building, transforming, serializing,
visualizing, and vectorizing Abstract Graphs.

An Abstract Graph has two levels:
- a preimage graph: the original NetworkX graph
- an image graph: nodes that represent associated subgraphs of the preimage

This repo contains the core abstractions and utilities only. Estimators live in
`abstractgraph-ml`. Generators and story-graph tooling live in
`abstractgraph-generative`.

## Ecosystem

This repo is one part of a three-repo stack:

- `abstractgraph`
  Path: `/home/fabrizio/work/abstractgraph`
- `abstractgraph-ml`
  Path: `/home/fabrizio/work/abstractgraph-ml`
- `abstractgraph-generative`
  Path: `/home/fabrizio/work/abstractgraph-generative`

See [ECOSYSTEM.md](ECOSYSTEM.md) for the dependency graph and install order.

## Package layout

- `src/abstractgraph/graphs.py`
  Core `AbstractGraph` type and conversion helpers.
- `src/abstractgraph/operators.py`
  Decomposition and transformation operators plus composition helpers.
- `src/abstractgraph/xml.py`
  XML serialization/deserialization of operator pipelines.
- `src/abstractgraph/hashing.py`
  Stable bounded hashing and graph hashing helpers.
- `src/abstractgraph/labels.py`
  Default label, attribute, and edge functions.
- `src/abstractgraph/display.py`
  Visualization for graphs, mappings, and operator pipelines.
- `src/abstractgraph/vectorize.py`
  Graph- and node-level vectorizers.
- `src/abstractgraph/preprocessor.py`
  Attention-derived preimage graph preprocessor.
- `src/abstractgraph/to_graph/`
  Adapters that build base graphs from external data.

## Documentation

- [docs/README.md](docs/README.md)
- [docs/OPERATORS.md](docs/OPERATORS.md)
- [docs/PREPROCESSOR.md](docs/PREPROCESSOR.md)
- [docs/WHITE_PAPER.md](docs/WHITE_PAPER.md)
- [ECOSYSTEM.md](ECOSYSTEM.md)

## Notebooks

- `notebooks/examples/` contains core usage examples.
- `notebooks/research/` contains exploratory core notebooks.

## Local validation

```bash
python -m pip install -e . --no-deps
python scripts/smoke_test.py
```
