
# AbstractGraph 

Note: Operators are defined in `abstract_graph_operators.py` (the older `operator.py` name is not used in this repo).

AbstractGraph is a two‑layer representation designed to describe “what parts of a graph we are looking at” and to keep that information composable and serializable.

- Preimage graph: the original input graph (e.g., molecular graph, social network). It stays unchanged throughout a pipeline.
- Image graph (associations): a dynamic set of image nodes, where each image node is associated with a subgraph of the preimage. Decomposition operators rewrite this set by emitting new subgraphs (cycles, paths, ego‑graphs, cliques, etc.).
- Provenance metadata: every operator call records its name and parameters on emitted image nodes, enabling inspection, filtering, and XML round‑trips.
- Operators: pure functions from AbstractGraph → AbstractGraph that either decompose (emit new subgraphs) or transform (normalize labels, complement, merge, filters). Composition helpers (`add`, `compose`, `forward_compose`, `compose_product`) let you build rich, multi‑stage workflows.

This separation (fixed preimage, evolving image associations) makes structural queries explicit, auditable, and easy to vectorize.

## From Decompositions To Predictors

You turn decomposition pipelines into features with the vectorizers in `vectorize.py`, then train classical or neural models.

1) Author a decomposition function DF
   - Compose operators to emit the subgraphs you care about (e.g., `forward_compose(node(), neighborhood(radius=1))`, `add(cycle(), tree())`, `compose_product(binary_combination(distance=(0,1)), cycle(), tree())`).
2) Vectorize
   - Global (graph‑level): `AbstractGraphTransformer(decomposition_function=DF, ...)` maps each input graph to a fixed‑length feature vector using stable hashing of subgraphs and optional counts/weights.
   - Node‑level: `AbstractGraphNodeTransformer(decomposition_function=DF, ...)` maps each input graph to a matrix of per‑node features (variable length), suitable for neural encoders.
3) Learn
   - Classical/scikit: `GraphEstimator(transformer=..., manifold=PCA(n_components=2), estimator=RandomForestClassifier(...), n_selected_features=...)`.
     - Trains the downstream estimator on raw transformer features.
     - Optionally selects top‑k features using estimator importances/coefficients and fits the manifold on the selected subset for visualization and downstream transforms.
   - Neural: `NeuralGraphEstimator(node_vectorizer=..., mode='classification'|'regression', ...)`.
     - Uses the node‑level vectorizer to produce per‑node embeddings, then encodes with a Transformer encoder to a graph embedding and predicts (classification/regression).
     - Provides `.transform()` for graph embeddings and `.plot()` for 2D visuals (UMAP/t‑SNE/PCA fallback).
4) Iterate
   - Inspect feature importances (classical) or attention/embeddings (neural) and refine DF: tighten motif sizes/radii, add/remove motifs, or add distance‑based relations.
   - Keep counts controlled with early filters (e.g., `filter_by_number_of_nodes/edges/components`) and bounded enumeration.

Benefits
- Interpretable structure: features come from explicit subgraphs with provenance.
- Composability: mix motifs/views and distance relations with `add` and `compose_product`.
- Portability: XML registration enables saving/loading operators for reproducible pipelines.

## Core Decomposition Operators

- node(): emit one image node per singleton vertex in each association.
- edge(): emit one image node per edge (2‑node induced subgraph).
- connected_component(): emit one image node per connected component of each association.
- degree(value): emit one image node per association containing only vertices whose degree lies within `value`.
  - value: int or (min,max) inclusive bounds.
- split(n_parts=2, seed=0): recursively bipartition the largest current part (Kernighan–Lin) until reaching `n_parts` or until no further split is possible; keep `seed` fixed for reproducible partitions.
- neighborhood(radius): emit ego‑graphs for each node for all BFS radii in `radius`.
  - radius: int or (min,max) inclusive bounds.
- cycle(): emit one image node per simple cycle (cycle basis) of each association.
- tree(): emit one image node per acyclic connected component (after removing cycle edges) of each association.
- path(number_of_edges): emit one image node per simple path whose edge length is within bounds.
  - number_of_edges: int or (min,max) inclusive bounds on path length.
- graphlet(radius, number_of_nodes): emit connected induced subgraphs (graphlets) inside ego neighborhoods.
  - radius: ego radius; number_of_nodes: int or (min,max).
- clique(number_of_nodes): emit one image node per clique (fully connected subgraph) within size bounds.
  - number_of_nodes: int or (min,max).

## Composition Helpers

- add(f, g, ...): run multiple decomposition functions on the same input and union their outputs (parallel additive view).
- compose(...): reverse‑order composition; applies functions from right to left to the quotient graph.
- forward_compose(...): forward‑order composition; applies functions from left to right.
- compose_product(combiner, ...): run functions in parallel and combine their outputs with a `combiner` (e.g., `binary_combination`).

## Auxiliary/Transform Operators Frequently Used With Decompositions

- intersection_edges(...): keep intersections of subgraphs’ node sets with optional size/connectivity filters.
- combination(...), binary_combination(...): combine two subgraph sets when their pairwise distance is within bounds.
- intersection(...): compute pairwise intersections (nodes) with optional constraints.
- complement(): replace each associated subgraph by its node complement within the preimage.
- edge_complement(): replace each associated subgraph by the edge-induced complement within the preimage.
- merge(): merge associations according to heuristic rules (see file docstring for details).
- betweenness_centrality(number_of_nodes): keep nodes with top betweenness centrality.
- betweenness_centrality_split(number_of_nodes=5): rank nodes by betweenness centrality, split ranked nodes into chunks, and emit one induced subgraph per chunk.
- betweenness_centrality_hop_split(n_hops=1): start from high-betweenness anchors, emit connected components from overlapping BFS hop windows ([0,n_hops], [n_hops,2*n_hops], ...).
- remove_redundant_associations(): drop image nodes whose associated subgraphs are fully covered by other image nodes.
- unlabel(label='-'): set all node/edge labels in the preimage to a constant.
- prepend_label(label): prefix all node/edge labels in the preimage.
- Filters: filter_by_number_of_nodes, filter_by_number_of_edges, filter_by_number_of_connected_components, filter_by_node_label.

## Notes

- All decomposition operators read the current set of image‑node associations and emit new associations in the output AbstractGraph. Labels/attributes are not computed during decomposition; call `update()` downstream as needed.
- Many operators accept either a scalar or a (min,max) tuple—scalars are treated as equal bounds.
- Enumeration operators (path, graphlet, clique) can grow combinatorially; constrain parameters and use filters to manage output size.

## Using Operators For Predictive Tasks

This section outlines practical patterns to compose operators to surface informative subgraphs that become features for downstream models.

High‑level workflow
- Define a decomposition function (DF): pick one or more structural primitives (e.g., node, edge, cycle, tree, path, graphlet, neighborhood) and compose them to generate candidate subgraphs.
- Constrain search: apply filters (filter_by_* and degree/size bounds) early to avoid combinatorial blow‑up; prefer small radius/size first.
- Combine views: use add(...) to union complementary motifs (e.g., cycles + trees), or compose_product(...) with a combiner (e.g., binary_combination) to relate two views by a distance constraint.
- Normalize labels when needed: unlabel() before structural extraction to avoid label leakage; prepend_label() to namespace sources before merging.
- Select representatives: intersection_edges(...) helps collapse/clean overlaps and enforce connectivity/size constraints on the resulting set.
- Vectorize: pass the DF to AbstractGraphTransformer or AbstractGraphNodeTransformer to embed the subgraphs/graphs.
- Train/evaluate: fit a predictive model; inspect importances to refine the DF (iterate).

Common composition patterns
- Local motifs around entities: forward_compose(node(), neighborhood(radius=(1,2))) produces ego‑graphs; add degree(value=...) to limit hubs/leaves.
- Cyclic vs. acyclic structure: forward_compose(connected_component(), add(cycle(), tree())) to separate rings and chains.
- Constrained paths: forward_compose(connected_component(), path(number_of_edges=(2,4))) to capture chain fragments of bounded length.
- Clique/graphlet mining: add(clique(number_of_nodes=(3,4)), graphlet(radius=1, number_of_nodes=(3,4))) for dense motifs; combine with filters to control count.
- Multi‑view with relations: compose_product(binary_combination(distance=(0,1)), cycle(), tree()) to pair cycles with nearby trees; distance bounds gate spurious matches.
- Post‑processing intersections: compose(intersection_edges(size_threshold=k, must_be_connected=True), add(cycle(), tree())) to keep only persistent/central overlaps.

Parameter‑tuning tips
- Start small: narrow bounds (radii, sizes) and relax gradually as needed.
- Push filters upstream: the earlier you prune (e.g., filter_by_number_of_nodes/edges/components), the more tractable later enumeration becomes.
- Control label bias: when structural signal matters, unlabel() first; re‑introduce semantics later if useful.
- Iterate with model feedback: inspect estimator feature importances to drop noisy motifs or to specialize ranges.

Vectorization and modeling
- Global graph features: AbstractGraphTransformer(decomposition_function=DF, ...) turns the DF’s subgraphs into a fixed‑length vector per input graph.
- Node‑level features: AbstractGraphNodeTransformer(decomposition_function=DF, ...) produces per‑node embeddings (useful for neural models in estimator/neural.py).
- Manifold/selection: GraphEstimator can fit a manifold on selected features from an estimator’s importances to improve visualization and learning.

Validation
- Ablations: compare single‑motif DFs vs. combined (add/compose_product) to quantify gain.
- Robustness: vary bounds (radius/size/distance) and ensure performance is stable.
- Complexity: monitor subgraph counts; add early filters or reduce bounds if counts explode.

### General Principles
- Explore combinations of motif types at different distances. Use compose_product(binary_combination(distance=... ), A(), B(), C()) to test whether a single motif (e.g., cycles), pairs (cycle+tree), or triplets (cycle+tree+clique) are jointly predictive.
- Sweep distance windows to learn spatial dependence: near (0–1) vs. moderate (2–3) vs. loose (4+). If performance peaks at small distances, co‑localization matters; if larger distances work, motifs can act at range.
- Run ablations per component: compare A only, B only, A+B, A+C, B+C, and A+B+C to quantify marginal contribution of each motif and their interactions.

## Label‑Aware Selection (Chemoinformatics)

Many workflows need to include or exclude specific atom types or functional groups when mining motifs. AbstractGraph supports this via label functions and label‑based filters.

Label sources
- You can normalize labels with `unlabel()` (structure‑only analysis) or namespace them with `prepend_label()` when merging sources.

Selecting/avoiding labeled nodes
- Use `filter_by_node_label(must_have_one_of=[...], cannot_have_any_in=[...])` to keep only those associations whose node sets contain desired labels and avoid forbidden labels.
- Place the filter:
  - Upstream (before decomposition) to restrict the search space (faster, fewer candidates).
  - Downstream (after e.g., `cycle()`, `path()`) to enforce constraints on the final motifs (e.g., “cycles containing N and no S”).

Examples
- Keep neighborhoods around heteroatoms (N/O) and drop metals:
  - `compose(filter_by_node_label(must_have_one_of=['N','O'], cannot_have_any_in=['Na','K','Ca','Mg']), neighborhood(radius=(1,2)))`
- Aromatic rings containing nitrogen but not sulfur:
  - `compose(filter_by_node_label(must_have_one_of=['N'], cannot_have_any_in=['S']), cycle())`
- Halogen‑proximal chains (2–3 bonds from halogens):
  - `compose(filter_by_node_label(must_have_one_of=['F','Cl','Br','I']), path(number_of_edges=(2,3)))`
- Two‑motif proximity: cycle near carbonyl oxygen (assuming labels annotate O in C=O):
  - `compose_product(binary_combination(distance=(0,2)), cycle(), compose(filter_by_node_label(must_have_one_of=['O']), neighborhood(radius=1)))`

Tips
- Structural vs. semantic signal: run ablations with and without `unlabel()` to measure how much labels contribute beyond topology.
- Group identities via relabelling: first select nodes with specific labels (e.g., `['N','O']`), then drop their original semantics with `unlabel()` and assign an explicit group tag with `prepend_label('hetero_')`. Repeat for other sets (e.g., halogens with `prepend_label('hal_')`) and combine views with `add(...)` or relate them with `compose_product(...)`.
- Combine with size/degree filters to avoid overly large/degenerate subgraphs before vectorization.
