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
- [VECTORIZATION.md](VECTORIZATION.md)
- [WHITE_PAPER.md](WHITE_PAPER.md)
- [RELEASE_TERMINOLOGY_MIGRATION.md](RELEASE_TERMINOLOGY_MIGRATION.md)
- [MIGRATION_TERMINOLOGY.md](MIGRATION_TERMINOLOGY.md)

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
python -m pip install -e . --no-deps
python scripts/smoke_test.py
```
