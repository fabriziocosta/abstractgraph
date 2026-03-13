# Operators in `abstractgraph`

Operators are the main mechanism for turning a base graph into a structured set
of associated subgraphs.

In the standalone `abstractgraph` package they live in:
- `abstractgraph.operators`

They are designed as pure `AbstractGraph -> AbstractGraph` transformations and
can be composed, serialized, visualized, and reused across ML and generative
workflows.

## Core idea

An operator reads the current image-node associations and emits a new
`AbstractGraph` whose image graph contains transformed or newly decomposed
associations.

Examples:
- `node()`: singleton-node associations
- `edge()`: 2-node edge associations
- `neighborhood(radius=...)`: ego-graph associations
- `cycle()`: cycle associations
- `clique(number_of_nodes=...)`: clique associations
- `path(number_of_edges=...)`: bounded simple paths

## Composition helpers

- `add(f, g, ...)`
  Run multiple operators on the same input and union the outputs.
- `compose(f, g, ...)`
  Reverse-order composition.
- `forward_compose(f, g, ...)`
  Forward-order composition.
- `compose_product(combiner, f, g, ...)`
  Run multiple operators in parallel and combine their outputs.

## Common workflow

```python
from abstractgraph.graphs import graph_to_abstract_graph
from abstractgraph.operators import forward_compose, node, neighborhood

df = forward_compose(
    node(),
    neighborhood(radius=1),
)

ag = graph_to_abstract_graph(graph, decomposition_function=df, nbits=12)
```

## Operator families

- Structural decomposition:
  `node`, `edge`, `connected_component`, `neighborhood`, `cycle`, `tree`,
  `path`, `graphlet`, `clique`
- Partitioning and centrality:
  `split`, `betweenness_centrality`, `betweenness_centrality_split`
- Combination and overlap:
  `intersection`, `intersection_edges`, `combination`, `binary_combination`
- Normalization and metadata:
  `unlabel`, `prepend_label`, metadata/tagging helpers
- Filtering:
  `filter_by_number_of_nodes`, `filter_by_number_of_edges`,
  `filter_by_number_of_connected_components`, `filter_by_node_label`

## XML round-trip

Operators can be serialized through `abstractgraph.xml`.

Typical pattern:

```python
import abstractgraph.operators as ag_ops
from abstractgraph.xml import register_from_module, operator_to_xml_string

register_from_module(ag_ops)
xml_text = operator_to_xml_string(df, pretty=True)
```

## Where operators fit

- `abstractgraph`
  defines and executes operators
- `abstractgraph-ml`
  uses operators to define feature vocabularies
- `abstractgraph-generative`
  uses operators to define image-node structure for rewriting and generation

For the conceptual background behind these choices, see [WHITE_PAPER.md](WHITE_PAPER.md).
