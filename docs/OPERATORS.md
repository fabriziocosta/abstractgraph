# Operators in `abstractgraph`

Operators are the main mechanism for turning a base graph into a structured set
of mapped subgraphs.

In the standalone `abstractgraph` package they live in:
- `abstractgraph.operators`

They are designed as pure `AbstractGraph -> AbstractGraph` transformations and
can be composed, serialized, visualized, and reused across ML and generative
workflows.

## Core idea

An operator reads the current interpretation-node mapped subgraphs and emits a
new `AbstractGraph` whose interpretation graph contains transformed or newly
decomposed mapped subgraphs.

Examples:
- `node()`: singleton-node mapped subgraphs
- `edge()`: 2-node edge mapped subgraphs
- `neighborhood(radius=...)`: ego-graph mapped subgraphs
- `cycle()`: cycle mapped subgraphs
- `clique(number_of_nodes=...)`: clique mapped subgraphs
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

## Directedness support

Graph directedness is owned by the NetworkX graph type: use `nx.Graph` for
undirected graphs and `nx.DiGraph` for directed graphs. Operators do not add a
second directedness parameter. Instead, every built-in operator declares a
`directed_support` metadata value that explains how it handles directed inputs.

The supported values are:

- `agnostic`
  Higher-order or control-flow operators that do not inspect topology directly.
  They validate and compose the support of their child operators.
- `preserve`
  Operators that accept directed and undirected graphs and preserve the current
  mapped-subgraph edge orientation.
- `weak`
  Operators that accept directed graphs but intentionally use weak or
  undirected connectivity semantics for the relevant structural step.
- `directed`
  Operators that use directed semantics when the base graph is directed and
  undirected semantics when the base graph is undirected.
- `undirected_only`
  Operators whose semantics are undirected in this package. They raise a clear
  `ValueError` when used on a directed base graph.

Examples:

```python
from abstractgraph.operators import clique, connected_component, cycle

connected_component.directed_support  # "weak"
cycle.directed_support                # "directed"
clique.directed_support               # "undirected_only"
```

Curried operator instances keep the same metadata. Use
`get_directed_support()` when inspecting a partially applied operator, because
`toolz.curry` stores the metadata on the wrapped function:

```python
from abstractgraph.operators import get_directed_support

get_directed_support(clique(number_of_nodes=3))  # "undirected_only"
```

Use `get_operator_registry()` when you need to inspect or document the built-in
operators programmatically.

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
  uses operators to define interpretation-node structure for rewriting and generation

For the conceptual background behind these choices, see [WHITE_PAPER.md](WHITE_PAPER.md).
