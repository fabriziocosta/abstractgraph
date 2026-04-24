# abstractgraph

`abstractgraph` is the core package for building, transforming, serializing,
visualizing, and vectorizing Abstract Graphs.

An Abstract Graph has two levels:
- a base graph: the original NetworkX graph
- an interpretation graph: nodes that represent mapped subgraphs of the base graph

This repo contains the core abstractions and utilities only. Estimators live in
`abstractgraph-ml`. Generators and story-graph tooling live in
`abstractgraph-generative`. Raw-data-to-graph graphicalizers now live in
`abstractgraph-graphicalizer`.

## Ecosystem

This repo is one part of a four-repo stack:

- `abstractgraph`
  Path: `/home/fabrizio/work/abstractgraph`
- `abstractgraph-ml`
  Path: `/home/fabrizio/work/abstractgraph-ml`
- `abstractgraph-generative`
  Path: `/home/fabrizio/work/abstractgraph-generative`
- `abstractgraph-graphicalizer`
  Path: `/home/fabrizio/work/abstractgraph-graphicalizer`

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
- `src/abstractgraph/to_graph/`
  Adapters that build base graphs from external data.

## Documentation

- [docs/README.md](docs/README.md)
- [docs/OPERATORS.md](docs/OPERATORS.md)
- [docs/VECTORIZATION.md](docs/VECTORIZATION.md)
- [docs/WHITE_PAPER.md](docs/WHITE_PAPER.md)
- [docs/RELEASE_TERMINOLOGY_MIGRATION.md](docs/RELEASE_TERMINOLOGY_MIGRATION.md)
- [docs/MIGRATION_TERMINOLOGY.md](docs/MIGRATION_TERMINOLOGY.md)
- [ECOSYSTEM.md](ECOSYSTEM.md)

## Notebooks

- `notebooks/examples/` contains core usage examples.
- `notebooks/research/` contains exploratory core notebooks.
- Example and research notebooks now bootstrap imports and normalize the
  working directory automatically for the standard ecosystem layout.

Recommended sequence:

1. `notebooks/examples/example_abstract_graph_operators_01_unary_decompositions.ipynb`
   Start with the core unary decomposition vocabulary.
2. `notebooks/examples/example_abstract_graph_operators_02_composition_and_add.ipynb`
   Learn composition order and additive unions.
3. `notebooks/examples/example_abstract_graph_operators_03_filters_and_selection.ipynb`
   Constrain decompositions with structural and label-based filters.
4. `notebooks/examples/example_abstract_graph_operators_04_binary_and_combination_operators.ipynb`
   Build new subgraphs from combinations and overlaps.
5. `notebooks/examples/example_abstract_graph_operators_05_control_flow_and_conditionals.ipynb`
   Learn how `if_then_else`, `if_then_elif_else`, `for_loop`, and `while_loop` turn pipelines into graph programs.
6. `notebooks/examples/example_abstract_graph_operators_06_xml_and_operator_serialization.ipynb`
   Serialize and round-trip operator pipelines.
7. `notebooks/examples/example_abstract_graph_operators_07_vectorization_and_features.ipynb`
   Convert abstract graphs into ML-ready feature matrices.
8. `notebooks/examples/example_abstract_graph_operators_08_preprocessor_attention_pipeline.ipynb`
   See how `abstractgraph-graphicalizer` attention backends feed base
   graphs into the same pipeline.
9. `notebooks/examples/example_abstract_graph_operators_09_feature_inspection_and_subgraphs.ipynb`
   Inspect which hashed feature labels correspond to which recurring subgraphs.

Reference notebook:
- `notebooks/examples/example_abstract_graph_operators_overview.ipynb`
  Broad operator sampler kept as a compact survey after the staged sequence.

## Local validation

```bash
python -m pip install -e . --no-deps
python scripts/smoke_test.py
```

## Ecosystem

See the [AbstractGraph ecosystem README](../../README.md) for how this
repository fits with the sibling repositories.
