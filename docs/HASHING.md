# Graph Hashing

This document describes the deterministic graph hashing implemented in
`src/abstractgraph/hashing.py`.

The hashing code is used to assign stable integer identifiers to graph
structures and subgraph-derived features. It is designed for reproducible
feature extraction, deduplication, and comparison across runs. It is not a
cryptographic identity guarantee.

## Design Goals

- Deterministic across Python runs and platforms.
- Stable under node relabeling.
- Sensitive to node labels and edge labels.
- Sensitive to graph structure beyond local degree profiles.
- Explicitly distinguish directed and undirected graphs.
- Produce bounded feature ids through `nbits`.

Python's built-in `hash(...)` is intentionally not used because it is salted per
process and therefore not stable across runs.

## Canonical Value Hashing

All low-level hashing starts with `canonicalize(value)`.

`canonicalize` converts Python values into type-tagged, JSON-serializable
representations before hashing. This avoids collisions such as integer `1` and
string `"1"` being treated as the same value.

Supported value families include:

- `None`, booleans, integers, and floats
- strings and bytes-like values
- NumPy arrays, including dtype, shape, and raw bytes
- lists and tuples, preserving order
- sets and frozensets, sorted canonically
- dictionaries, sorted by canonicalized key
- fallback objects, represented by type name and `repr(...)`

The canonical representation is serialized by `canonical_bytes(value)` and then
hashed with SHA-256 by `hash_value(value)`.

## Combinators

The implementation uses two main hash combinators:

- `hash_sequence(values)` is order-aware.
- `hash_set(values)` is order-independent and multiset-aware.

`hash_set` hashes each element first, sorts the resulting integers, and hashes
the sorted tuple. Equal repeated elements are preserved because duplicates
remain present in the sorted list.

The final public graph hash is reduced by `hash_bounded(value, nbits)`, which
returns an integer in:

```text
[2, 2**nbits - 1]
```

The values `0` and `1` are reserved for special feature uses elsewhere in the
library.

## Inputs Used

Graph hashing currently uses the NetworkX graph topology plus the `label`
attribute on nodes and edges.

For a node:

```python
graph.nodes[node].get("label", "")
```

For an edge:

```python
graph.edges[u, v].get("label", "")
```

Other node and edge attributes are ignored by `hash_graph`.

## Node Hashes

`hash_node(node, graph)` computes a local label-aware neighborhood hash.

For an isolated node, the node hash is just the hash of its own label.

For a non-isolated node, the hash combines:

- the node's own label
- a multiset of incident edge payloads

Each incident edge payload contains:

- a direction tag
- the neighbor node label
- the edge label

For undirected graphs the direction tag is:

```text
undirected
```

For directed graphs, outgoing and incoming arcs are separated:

```text
out
in
```

This means a node with an outgoing edge to label `B` hashes differently from a
node with an incoming edge from label `B`.

## Rooted BFS Hashes

`hash_rooted_graph(root, graph, node_hashes)` computes a rooted structural
summary from every root node.

The function runs a shortest-path traversal from the root:

```python
nx.single_source_shortest_path_length(graph, root)
```

Nodes are grouped by distance from the root. For each distance layer:

1. collect the precomputed `hash_node(...)` values of nodes in that layer
2. sort the node hashes to remove dependence on node ids or iteration order
3. hash the ordered layer list

The rooted graph hash is then the ordered sequence of layer hashes.

This BFS-style summary is good at capturing label and distance structure, but
some non-isomorphic or differently wired graphs can produce identical layer
sets from every root. That limitation is why the current implementation also
adds a canonical DFS certificate.

## Canonical DFS Certificate

`canonical_dfs_graph_signature(graph, rooted_hashes)` adds a relabeling-stable
DFS certificate on top of the rooted BFS hashes.

The purpose is to distinguish difficult cases where every root sees the same
BFS distance-layer hash sets, but the actual wiring differs.

Candidate traversal order is based only on relabeling-invariant information:

- edge direction tag
- canonicalized edge label
- rooted hash of the neighbor

Node ids are never used as ordering tie-breakers.

When candidates have distinct keys, DFS follows the smallest key. When multiple
candidates have the same key, the implementation enumerates only those tied
candidates and keeps the lexicographically smallest resulting certificate.

This is the key compromise:

- Most graphs avoid expensive exhaustive canonicalization.
- Symmetric/tied neighborhoods still get enough search to avoid arbitrary node
  id dependence.

The DFS signature also records visited/back edges through discovery indices.
That allows cross-edge structure to affect the certificate, not only the tree
edges selected by DFS.

For disconnected graphs, remaining components are processed in canonical order.
If several remaining component roots have the same rooted hash, those tied
roots are enumerated in the same way.

The signature begins with a graph-type marker:

```text
directed
undirected
```

so the same topology does not hash the same when represented as directed versus
undirected.

## Whole-Graph Hash

`hash_graph(graph, nbits=19)` combines several graph-level summaries.

The current aggregate includes:

1. a multiset hash of all node labels
2. a graph-type marker: `directed` or `undirected`
3. the canonical DFS graph signature
4. a multiset of edge hashes

For each edge, the edge hash includes the edge label and endpoint rooted hashes.

For undirected graphs, endpoint rooted hashes are combined order-independently:

```text
undirected_edge, hash_set([rooted_hash(u), rooted_hash(v)]), edge_label_hash
```

For directed graphs, endpoint order is preserved:

```text
directed_edge, rooted_hash(source), rooted_hash(target), edge_label_hash
```

The final aggregate is reduced by:

```python
hash_bounded(hash_set(hashes_list), nbits=nbits)
```

This final step gives a compact integer suitable for vector indices and feature
labels.

## Directed vs Undirected Support

The hashing algorithm explicitly supports both `nx.Graph` and `nx.DiGraph`.

The same visual path:

```text
A --step-- B --step-- C
```

and directed path:

```text
A ->step-> B ->step-> C
```

are different to the hash function because:

- node neighborhood payloads use `undirected` versus `out` / `in`
- the whole-graph aggregate includes `undirected` versus `directed`
- edge hashes for directed graphs preserve source-target order
- the canonical DFS signature includes the graph-type marker and direction tags

Reversing a directed edge can also change the hash. For example:

```text
A -> B
```

and:

```text
B -> A
```

are different when labels and structure make that orientation observable.

## Stability Under Relabeling

Node ids are treated as implementation handles, not semantic content.

The algorithm avoids using node ids as ordering tie-breakers. It sorts or
canonicalizes by hashes, labels, direction tags, edge labels, and DFS discovery
indices instead.

As a result, relabeling nodes should preserve the hash when the labeled graph
structure is otherwise unchanged.

## Collision Model

There are two collision sources:

- SHA-256 collisions in the unbounded intermediate hashes, which are
  practically negligible for this use.
- Bounded collisions introduced by `nbits`.

The bounded hash is intentionally compact. A smaller `nbits` value gives fewer
feature columns but more collisions. A larger `nbits` value reduces collisions
at the cost of a larger possible feature space.

Use larger values such as `nbits=19` or `nbits=31` when collision risk matters.

## Complexity

The rooted BFS stage computes a rooted summary from every node, which is roughly:

```text
O(V * (V + E))
```

The canonical DFS certificate is usually modest, but tied neighborhoods can add
branching. The implementation limits enumeration to tied candidates rather than
trying every possible traversal order.

Highly symmetric graphs can still be more expensive than asymmetric graphs.

## Limitations

- Only the `label` attribute is hashed.
- The implementation is intended for simple `Graph` and `DiGraph` use. Multi-edge
  graphs may need additional handling for edge keys and parallel edge labels.
- Hashes are feature identifiers, not proofs of graph isomorphism.
- Bounded hashes can collide by construction.

## Related Tests

The regression tests in `tests/test_terminology_migration.py` cover:

- directed and undirected versions of the same path hashing differently
- reversed directed paths hashing differently
- difficult same-label, same-degree graph pairs hashing differently
- relabeling stability for the difficult graph pair
