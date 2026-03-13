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

Recommended sequence:

1. `notebooks/examples/example_abstract_graph_operators_01_unary_decompositions.ipynb`
   Start with the core unary decomposition vocabulary.
2. `notebooks/examples/example_abstract_graph_operators_02_composition_and_add.ipynb`
   Learn composition order and additive unions.
3. `notebooks/examples/example_abstract_graph_operators_03_filters_and_selection.ipynb`
   Constrain decompositions with structural and label-based filters.
4. `notebooks/examples/example_abstract_graph_operators_04_binary_and_combination_operators.ipynb`
   Build new subgraphs from combinations and overlaps.
5. `notebooks/examples/example_abstract_graph_operators_05_xml_and_operator_serialization.ipynb`
   Serialize and round-trip operator pipelines.
6. `notebooks/examples/example_abstract_graph_operators_06_vectorization_and_features.ipynb`
   Convert abstract graphs into ML-ready feature matrices.
7. `notebooks/examples/example_abstract_graph_operators_07_preprocessor_attention_pipeline.ipynb`
   See how attention-derived preimage graphs feed into the same pipeline.
8. `notebooks/examples/example_abstract_graph_operators_08_feature_inspection_and_subgraphs.ipynb`
   Inspect which hashed feature labels correspond to which recurring subgraphs.

Reference notebook:
- `notebooks/examples/example_abstract_graph_operators_overview.ipynb`
  Broad operator sampler kept as a compact survey after the staged sequence.

## Local validation

```bash
python -m pip install -e . --no-deps
python scripts/smoke_test.py
```
