# abstractgraph Organization

This document covers code organization, local setup, validation, and supporting
documentation for `abstractgraph`.

For the semantic role of this repository, see [../README.md](../README.md).

## Package Layout

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

- [README.md](README.md)
- [OPERATORS.md](OPERATORS.md)
- [HASHING.md](HASHING.md)
- [VECTORIZATION.md](VECTORIZATION.md)
- [WHITE_PAPER.md](WHITE_PAPER.md)

## Install

Standalone editable install:

```bash
python -m pip install -e .
```

When working inside the `abstractgraph-ecosystem` superproject, install this
repository first so sibling packages can import the core API:

```bash
python -m pip install -e repos/abstractgraph --no-deps
```

Use `--no-deps` in the superproject when dependencies are managed by the shared
environment. Omit it for a standalone checkout when pip should resolve runtime
dependencies.

## Dependencies

Runtime dependencies declared in `pyproject.toml`:

- `networkx`
- `numpy`
- `scipy`
- `joblib`
- `scikit-learn`
- `matplotlib`
- `toolz`

`display_decomposition_graph` uses Graphviz through `pygraphviz` when rendering
operator decomposition graphs. Install Graphviz and `pygraphviz` in the active
environment if those visualizations are needed.

## Caveats

- This is the foundational package. Do not depend on sibling repositories from
  core modules unless the dependency direction is intentionally changed at the
  ecosystem level.
- Bounded graph hashes are intended to be stable feature identifiers, not
  cryptographic hashes.
- Graphviz rendering depends on native libraries; prefer conda-forge packages
  for `graphviz` and `pygraphviz` when using conda environments.

## Notebooks

- `notebooks/examples/` contains core usage examples.
- `notebooks/research/` contains exploratory core notebooks.
- Example and research notebooks bootstrap imports and normalize the working
  directory automatically for the standard ecosystem layout.

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
   Learn how control-flow operators turn pipelines into graph programs.
6. `notebooks/examples/example_abstract_graph_operators_06_xml_and_operator_serialization.ipynb`
   Serialize and round-trip operator pipelines.
7. `notebooks/examples/example_abstract_graph_operators_07_vectorization_and_features.ipynb`
   Convert abstract graphs into ML-ready feature matrices.
8. `notebooks/examples/example_abstract_graph_operators_08_preprocessor_attention_pipeline.ipynb`
   See how graphicalizer attention backends feed base graphs into the same
   pipeline.
9. `notebooks/examples/example_abstract_graph_operators_09_feature_inspection_and_subgraphs.ipynb`
   Inspect which hashed feature labels correspond to recurring subgraphs.

Reference notebook:

- `notebooks/examples/example_abstract_graph_operators_overview.ipynb`
  Broad operator sampler kept as a compact survey after the staged sequence.

## Local Validation

```bash
python -m pip install -e .
python scripts/smoke_test.py
```
