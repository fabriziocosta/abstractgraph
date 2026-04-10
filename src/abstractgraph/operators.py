#====================================================================================================
# SECTIONS:
# NOTE: Module renamed from operator.py to abstract_graph_operators.py
# to avoid shadowing Python's stdlib 'operator' during imports.
#====================================================================================================
# UTILITIES: value_to_2tuple  build_meta_from_function_context
# HIGHER ORDER OPERATORS: add  compose  forward_compose  compose_product
# CONDITIONAL OPERATORS: if_then_else  if_then_elif_else
# ITERATION OPERATORS: for_loop  while_loop
# UNARY OPERATORS: identity  random_part  node  edge  connected_component  degree  split  neighborhood  cycle  tree  path  spine  graphlet  clique  complement  local_complement  edge_complement  local_edge_complement  betweenness_centrality  betweenness_centrality_split  betweenness_centrality_hop_split  low_cut_partition  merge  deduplicate  remove_redundant_associations  intersection  combination  union_of_shortest_paths
# META OPERATORS: name
# EDGE OPERATORS: intersection_edges
# FILTER OPERATORS: filter_by_number_of_connected_components  filter_by_number_of_nodes  filter_by_number_of_edges  filter_by_node_label  filter_by_edge_label  select_top_by_feature_ranking  filter_by_sampling
# BINARY OPERATORS:  binary_combination  binary_intersection
# BASE GRAPH OPERATORS: unlabel  prepend_label  restore_label
# SCALAR OPERATORS: number_of_image_graph_nodes  number_of_image_graph_edges  quantile_number_of_subgraph_nodes  quantile_number_of_subgraph_edges  max_number_of_subgraph_nodes  min_number_of_subgraph_nodes  max_number_of_subgraph_edges  min_number_of_subgraph_edges
#====================================================================================================

import networkx as nx
import numpy as np
import copy
from collections import defaultdict
from toolz import curry
from typing import Callable, Any, Dict, List, Tuple, Optional, Union
import inspect
import itertools
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import combinations, product
from networkx.algorithms.community import kernighan_lin_bisection
from abstractgraph.graphs import AbstractGraph, get_mapped_subgraph
from abstractgraph.hashing import hash_set
from abstractgraph.xml import operator_to_xml_string
import random


#====================================================================================================
# UTILITIES
#====================================================================================================

_SOURCE_CHAIN_XML: ContextVar[Optional[str]] = ContextVar("source_chain_xml", default=None)
_ADD_INCLUDE_SELF: ContextVar[bool] = ContextVar("add_include_self", default=False)


def _get_source_chain_xml() -> Optional[str]:
    """
    Return the current source-chain XML stored in context, if any.

    Args:
        None.

    Returns:
        Optional[str]: XML string or None.
    """
    return _SOURCE_CHAIN_XML.get()


@contextmanager
def _source_chain_context(xml: Optional[str]):
    """
    Temporarily set the source-chain XML for downstream metadata capture.

    Args:
        xml: XML string to store in context.

    Returns:
        None.
    """
    token = _SOURCE_CHAIN_XML.set(xml)
    try:
        yield
    finally:
        _SOURCE_CHAIN_XML.reset(token)


@contextmanager
def _add_include_self_context(flag: bool):
    """
    Temporarily mark add as an inner operator so it should include itself.

    Args:
        flag: True if add should include itself in source-chain XML.

    Returns:
        None.
    """
    token = _ADD_INCLUDE_SELF.set(flag)
    try:
        yield
    finally:
        _ADD_INCLUDE_SELF.reset(token)


def _get_add_include_self() -> bool:
    """
    Read the add include-self flag from context.

    Args:
        None.

    Returns:
        bool: True if add should include itself in the source chain.
    """
    return _ADD_INCLUDE_SELF.get()


def _operator_xml_string(op: Any) -> str:
    """
    Serialize an operator to a compact XML string for stable hashing.

    Args:
        op: Operator to serialize.

    Returns:
        str: Compact XML string.
    """
    return operator_to_xml_string(op, pretty=False)


def _is_add_operator(func: Any) -> bool:
    """
    Detect if a callable is an add operator.

    Args:
        func: Callable to test.

    Returns:
        bool: True if func is an add operator.
    """
    return getattr(func, "operator_type", None) == "add" or getattr(func, "__name__", None) == "add"


def _call_decomposition(func: Callable[['AbstractGraph'], 'AbstractGraph'], abstract_graph: 'AbstractGraph') -> 'AbstractGraph':
    """
    Call a decomposition function, marking add operators as inner.

    Args:
        func: Decomposition callable.
        abstract_graph: AbstractGraph input.

    Returns:
        AbstractGraph: Result of applying func.
    """
    if _is_add_operator(func):
        with _add_include_self_context(True):
            return func(abstract_graph)
    return func(abstract_graph)


def value_to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Converts a value to a tuple of two identical integers.
    """
    if isinstance(value, tuple):
        return value
    elif isinstance(value, int):
        return (value, value)
    else:
        raise ValueError(f"Invalid value: {value}. Expected an int or a tuple.")
    

def build_meta_from_function_context(exclude_keys: Tuple[str, ...] = ("abstract_graph",)) -> Dict[str, Any]:
    """
    Attempts to construct a meta dictionary including:
    - source_function: inferred from the calling function (even if curried)
    - params: all parameters except excluded ones (like abstract_graph)
    - source_chain: XML string for the full operator chain when available

    Returns:
        dict: metadata with 'source_function' and 'params'
    """
    frame = inspect.currentframe()
    caller_frame = frame.f_back

    # Step 1: Get parameter values (excluding things like 'abstract_graph')
    args, _, _, values = inspect.getargvalues(caller_frame)
    params_dict = {k: values[k] for k in args if k not in exclude_keys}

    # Step 2: Try to infer function name from calling frame
    source_function = "unknown"
    try:
        func_obj = values.get(args[0], None)
        source_function = inspect.unwrap(func_obj).__name__
    except Exception:
        # fallback: use the code object
        try:
            source_function = caller_frame.f_code.co_name
        except Exception:
            pass

    meta = {
        "source_function": source_function,
        "params": params_dict
    }
    source_chain_xml = _get_source_chain_xml()
    if source_chain_xml is None:
        source_chain_xml = source_function
    meta["source_chain"] = source_chain_xml
    parent_subgraph = values.get("subgraph")
    if isinstance(parent_subgraph, nx.Graph):
        meta["parent_mapped_subgraph"] = parent_subgraph.copy()
    return meta


#====================================================================================================
# HIGHER ORDER OPERATORS
#====================================================================================================
def add(*decomposition_functions, dedup: bool = True):
    """
    Build an operator that adds (unions) the outputs of multiple decomposition functions.
    """
    def composed(abstract_graph: 'AbstractGraph'):
        """Additive composition of decomposition outputs over a shared base graph.
        Summary
            Given one AbstractGraph, run several decomposition functions on it and
            add their resulting interpretation graphs together using the graph’s `+`
            semantics, producing a single aggregate AbstractGraph.

        Semantics
            - Input AG state:
                Uses the provided `abstract_graph` as the common source for all
                decompositions; reads its `label_function`, `attribute_function`,
                and `edge_function` to seed the result.
            - Output AG state:
                Returns a new AbstractGraph whose interpretation graph is the additive
                combination of all per-function outputs; the base graph is
                aligned with the input (as defined by `__add__` on AbstractGraph).
            - Determinism:
                Deterministic given the input graph and the ordered list of
                decomposition functions.

        Parameters
            abstract_graph : AbstractGraph
                The graph to be decomposed by each function in `decomposition_functions`.

        Returns
            AbstractGraph
                A single quotient graph equivalent to
                `func_1(ag) + func_2(ag) + ... + func_m(ag)`, with operator settings
                (label/attribute/edge functions) preserved from the input.

        Algorithm
            1. Initialise an empty/base AbstractGraph carrying the input’s functional
               settings (label/attribute/edge functions).
            2. For each decomposition function `f` in order:
               a) Compute `f(abstract_graph)` → a AbstractGraph.
               b) Add it to the running `result` via `result = result + f(abstract_graph)`.
            3. Return the accumulated `result`.

        Complexity
            Let m be the number of functions, T_f the cost of each decomposition,
            and A the cost of a `+` merge:
              - Time:  Σ T_f  +  (m − 1)·A
              - Memory: proportional to the size of the union of interpretation-node sets and
                any metadata/materialised edges produced by the functions.

        Interactions
            - Pairs naturally with decomposition functions such as:
              `connected_components_decomposition`, `cycle_decomposition`,
              `clique_decomposition`, `filter_by_*`.
            - Often followed by consolidation steps like `deduplicate`, `merge`,
              or `project` to normalise/aggregate overlapping interpretation nodes.
            - Order matters if `+` is not strictly commutative/associative in the
              implementation (e.g., metadata precedence rules).

        Examples
            # Combine connected components and simple cycles into one operator
            cc_plus_cycles = add(connected_components_decomposition,
                                 cycle_decomposition)
            qg_out = cc_plus_cycles(qg_in)

            # Add several filters
            fused = add(filter_by_label('Person'),
                        filter_by_attribute('weight', '>', 70.0))(qg_in)

        Domain Analogies
            - Chemistry: union of detected motifs (e.g., rings + functional groups).
            - Social networks: merge of multiple community detections (by interests,
              by interaction frequency) into one layer.
            - Vision: combine edge maps from different detectors into a single feature layer.

        Failure Modes
            - Empty input list (`add()` with no functions) returns a base/empty
              quotient graph carrying only operator settings.
            - Incompatible outputs: if `__add__` requires matched base graphs
              or settings, and a decomposition violates those assumptions, a merge
              error may occur.
            - Non-idempotent `+`: repeated addition of overlapping interpretation nodes may
              duplicate structures unless `__add__` handles deduplication.
            - Exceptions raised in any decomposition function propagate to the caller.
        """
        # Preserve the functional settings from the input graph
        base = AbstractGraph(
            label_function=abstract_graph.label_function,
            attribute_function=abstract_graph.attribute_function,
            edge_function=abstract_graph.edge_function,
        )

        include_self = _get_add_include_self()
        existing_chain = _get_source_chain_xml()
        xml_self = _operator_xml_string(composed)

        result = base
        if include_self:
            xml_to_use = existing_chain or xml_self
            with _source_chain_context(xml_to_use):
                for func in decomposition_functions:
                    out = _call_decomposition(func, abstract_graph)
                    if dedup:
                        out = deduplicate(out)
                    result = result + out
        else:
            for func in decomposition_functions:
                child_xml = _operator_xml_string(func)
                with _source_chain_context(child_xml):
                    out = _call_decomposition(func, abstract_graph)
                if dedup:
                    out = deduplicate(out)
                result = result + out
        return result

    composed.__name__ = "add"
    composed.decomposition_functions = decomposition_functions
    composed.operator_type = "add"
    composed.params = {"dedup": dedup}
    return composed

#--------------------------------------------------------------------------------
def compose(*decomposition_functions, dedup: bool = True):
    def composed(abstract_graph: 'AbstractGraph'):
        """Reverse-order composition of decomposition functions on a AbstractGraph.
        Summary
            Applies a chain of decomposition functions to a quotient graph,
            evaluating them from right to left (last provided function runs first).

        Semantics
            - Input AG state:
                Receives one AbstractGraph as input.
            - Output AG state:
                Returns the transformed AbstractGraph after sequentially applying
                all decomposition functions in reversed order.
            - Determinism:
                Deterministic given the input and fixed function chain.

        Parameters
            abstract_graph : AbstractGraph
                The input graph to transform through the composition chain.

        Returns
            AbstractGraph
                The final graph after applying all functions in reverse order.

        Algorithm
            1. Initialise with the input `abstract_graph`.
            2. For each function f in `reversed(decomposition_functions)`:
                abstract_graph = f(abstract_graph)
            3. Return the final `abstract_graph`.

        Complexity
            Let m = number of functions, and T_f = cost of each:
              - Time: Σ T_f
              - Memory: governed by the largest intermediate AbstractGraph.

        Interactions
            - Pairs with decomposition primitives (`cycle`, `clique`, `filter_by_*`).
            - Often wrapped in higher-level pipelines for symbolic XML composition.
            - Useful for operators that require preprocessing by another function
              before they run (e.g. filtering before merging).

        Examples
            # Compose cycle detection after connected components
            cc_then_cycle = compose(cycle_decomposition,
                                    connected_components_decomposition)
            qg_out = cc_then_cycle(qg_in)
            # Equivalent to cycle_decomposition(connected_components_decomposition(qg_in))

        Domain Analogies
            - Mathematics: function composition f∘g, evaluated right-to-left.
            - Image processing: apply a blur after resizing.
            - Social networks: detect cliques inside already-partitioned communities.

        Failure Modes
            - Empty composition chain returns the input graph unchanged.
            - Any exception in a function aborts the chain.
            - Ordering mistakes: easy to confuse with `forward_compose`
              since semantics differ only by evaluation order.
        """
        xml_chain = _get_source_chain_xml() or _operator_xml_string(composed)
        with _source_chain_context(xml_chain):
            for func in reversed(decomposition_functions):
                abstract_graph = _call_decomposition(func, abstract_graph)
                if dedup:
                    abstract_graph = deduplicate(abstract_graph)
            return abstract_graph

    composed.__name__ = "compose"
    composed.chain = decomposition_functions
    composed.operator_type = "compose"  # Mark as a compose operator
    composed.params = {"dedup": dedup}
    return composed

def forward_compose(*decomposition_functions, dedup: bool = True):
    def composed(abstract_graph: 'AbstractGraph'):
        """Forward-order composition of decomposition functions on a AbstractGraph.
        Summary
            Applies a chain of decomposition functions to a quotient graph,
            evaluating them from left to right (first provided function runs first).

        Semantics
            - Input AG state:
                Takes one AbstractGraph as the starting point.
            - Output AG state:
                Returns the transformed AbstractGraph after applying each function
                in forward order.
            - Determinism:
                Deterministic given the input and fixed function chain.

        Parameters
            abstract_graph : AbstractGraph
                The input graph to transform through the chain.

        Returns
            AbstractGraph
                The final graph after applying all functions in left-to-right order.

        Algorithm
            1. Initialise with the input `abstract_graph`.
            2. For each function f in `decomposition_functions`:
                abstract_graph = f(abstract_graph)
            3. Return the final `abstract_graph`.

        Complexity
            Let m = number of functions, and T_f = cost of each:
              - Time: Σ T_f
              - Memory: governed by the largest intermediate AbstractGraph.

        Interactions
            - Useful in pipelines where function order mirrors natural workflow.
            - Combines well with `filter_by_*` then `merge` style operations.
            - More intuitive than `compose` when read left-to-right.

        Examples
            # Apply connected components first, then cycle detection
            cc_then_cycle = forward_compose(connected_components_decomposition,
                                            cycle_decomposition)
            qg_out = cc_then_cycle(qg_in)
            # Equivalent to cycle_decomposition(connected_components_decomposition(qg_in))

        Domain Analogies
            - Functional programming: pipeline operator |> in F# or Elixir.
            - Image processing: resize → blur → sharpen (in order).
            - Social networks: partition into communities, then analyse cliques.

        Failure Modes
            - Empty composition chain returns the input graph unchanged.
            - Exceptions propagate immediately from any function in the chain.
            - Can be confused with `compose` if users forget directionality.
        """
        xml_chain = _get_source_chain_xml() or _operator_xml_string(composed)
        with _source_chain_context(xml_chain):
            for func in decomposition_functions:
                abstract_graph = _call_decomposition(func, abstract_graph)
                if dedup:
                    abstract_graph = deduplicate(abstract_graph)
            return abstract_graph

    composed.__name__ = "forward_compose"
    composed.chain = decomposition_functions
    composed.operator_type = "forward_compose"  # Mark as a forward_compose operator
    composed.params = {"dedup": dedup}
    return composed

#--------------------------------------------------------------------------------
def compose_product(combiner, *decomposition_functions, dedup: bool = True):
    def composed(abstract_graph: 'AbstractGraph'):
        """Parallel product composition of decomposition functions with a combiner.
        Summary
            Applies multiple decomposition functions independently to the same
            input AbstractGraph, then fuses their outputs using a user-supplied
            `combiner` function.

        Semantics
            - Input AG state:
                One AbstractGraph, used as input to each decomposition function.
            - Output AG state:
                A new AbstractGraph returned by `combiner(*results)` where each
                result is the output of one decomposition function.
            - Determinism:
                Deterministic given deterministic decomposition functions and combiner.

        Parameters
            combiner : Callable[[AbstractGraph, ...], AbstractGraph]
                A function that takes all decomposition outputs and produces a
                single AbstractGraph (e.g. via addition, merge, intersection).
            decomposition_functions : tuple[Callable[[AbstractGraph], AbstractGraph], ...]
                The decomposition functions to apply in parallel.

        Returns
            AbstractGraph
                The combined result of applying all functions to the input
                AbstractGraph and reducing them with `combiner`.

        Algorithm
            1. For each f in `decomposition_functions`, compute f(abstract_graph).
            2. Collect all results into a list.
            3. Return combiner(*results).

        Complexity
            Let m = number of functions, T_f = cost of each, and C = cost of combiner:
              - Time: Σ T_f + C
              - Memory: sum of sizes of intermediate AbstractGraphs + size of combined output.

        Interactions
            - Natural generalisation of `add`: use `combiner = operator.add`.
            - Can encode intersections, unions, or custom fusions depending on
              combiner.
            - Useful for multi-view decomposition (e.g. apply different structural
              detectors and then fuse).

        Examples
            # Product with addition as combiner (union of outputs)
            union_op = compose_product(lambda a, b: a + b,
                                       cycle_decomposition,
                                       connected_components_decomposition)
            qg_out = union_op(qg_in)

            # Product with custom intersection combiner
            def intersect(a, b): return a & b
            intersect_op = compose_product(intersect,
                                           clique_decomposition,
                                           filter_by_label("Person"))
            qg_out = intersect_op(qg_in)

        Domain Analogies
            - Chemistry: detect rings and functional groups separately, then fuse.
            - Social networks: compute communities by geography and by activity,
              then combine.
            - Vision: apply edge detector and texture detector, then overlay results.

        Failure Modes
            - Empty function list: `compose_product` returns `combiner()` with no
              args → usually raises a TypeError.
            - Incompatible combiner: if outputs are not compatible with the combiner
              (e.g. types mismatch), runtime error.
            - Expensive combiner: cost may dominate total runtime if reduction is heavy.
        """
        xml_chain = _get_source_chain_xml() or _operator_xml_string(composed)
        with _source_chain_context(xml_chain):
            results = []
            for func in decomposition_functions:
                out = _call_decomposition(func, abstract_graph)
                if dedup:
                    out = deduplicate(out)
                results.append(out)
            return combiner(*results)

    composed.__name__ = "compose_product"
    composed.decomposition_functions = decomposition_functions
    composed.operator_type = "product"
    composed.combiner = combiner
    composed.params = {"dedup": dedup}
    return composed

#====================================================================================================
# CONDITIONAL OPERATORS
#====================================================================================================

@curry
def if_then_else(
    abstract_graph: 'AbstractGraph',
    predicate: Callable[['AbstractGraph'], bool],
    then_function: Callable[['AbstractGraph'], 'AbstractGraph'],
    else_function: Callable[['AbstractGraph'], 'AbstractGraph']
) -> 'AbstractGraph':
    """Conditional branching operator for AbstractGraph transformations.
    Summary
        Evaluates a predicate on the input AbstractGraph and applies either
        `then_function` (if True) or `else_function` (if False).

    Semantics
        - Input AG state:
            The given `abstract_graph` is passed unchanged to both predicate
            and whichever branch function is chosen.
        - Output AG state:
            Result of applying either `then_function` or `else_function` to
            the input graph.
        - Determinism:
            Deterministic if predicate and branch functions are deterministic.

    Parameters
        abstract_graph : AbstractGraph
            Input graph on which the predicate is evaluated.
        predicate : Callable[[AbstractGraph], bool]
            Function deciding which branch to execute.
        then_function : Callable[[AbstractGraph], AbstractGraph]
            Transformation applied if predicate returns True.
        else_function : Callable[[AbstractGraph], AbstractGraph]
            Transformation applied if predicate returns False.

    Returns
        AbstractGraph
            Output of the selected transformation function.

    Algorithm
        1. Evaluate `predicate(abstract_graph)`.
        2. If result is True, return `then_function(abstract_graph)`.
        3. Otherwise, return `else_function(abstract_graph)`.

    Complexity
        - Time: cost(predicate) + cost(branch) for whichever branch is chosen.
        - Memory: size of branch output.
        - No extra overhead beyond predicate and chosen function.

    Interactions
        - Integrates into pipelines built with `forward_compose` or `compose`
          to enable conditional flows.
        - Combines naturally with decomposition selectors like
          `filter_by_*` or `merge`.
        - Can emulate “switch”-style logic when nested.

    Examples
        # Branch by number of interpretation nodes
        workflow = forward_compose(
            connected_component(),
            if_then_else(
                predicate=lambda ag: ag.interpretation_graph.number_of_nodes() > 10,
                then_function=merge(),
                else_function=cycle()
            ),
            filter_by_number_of_nodes(number_of_nodes=(4, 10))
        )
        qg_out = workflow(qg_in)

    Domain Analogies
        - Programming: the classic `if ... then ... else ...` control structure.
        - Chemistry: apply a different reaction depending on whether a molecule
          exceeds a threshold property.
        - Social networks: choose community detection algorithm based on graph size.

    Failure Modes
        - Predicate exceptions: if predicate raises an error, execution halts.
        - Branch mismatch: if `then_function` or `else_function` produce outputs
          incompatible with downstream operators, pipeline may fail.
        - Non-deterministic predicates lead to unpredictable branching.
    """
    if predicate(abstract_graph):
        return _call_decomposition(then_function, abstract_graph)
    else:
        return _call_decomposition(else_function, abstract_graph)


@curry
def if_then_elif_else(
    abstract_graph: 'AbstractGraph',
    conditions_functions: List[Tuple[Callable[['AbstractGraph'], bool], Callable[['AbstractGraph'], 'AbstractGraph']]],
    else_function: Callable[['AbstractGraph'], 'AbstractGraph']
) -> 'AbstractGraph':
    """Multi-branch conditional operator for AbstractGraph transformations.
    Summary
        Evaluates a sequence of (predicate, function) pairs on the input
        AbstractGraph. The first predicate that evaluates True determines the
        branch function to apply. If none match, `else_function` is applied.

    Semantics
        - Input AG state:
            Input AbstractGraph is passed unchanged to all predicates and to
            the selected transformation function.
        - Output AG state:
            Result of applying the first matching branch function or else_function.
        - Determinism:
            Deterministic if predicates and branch functions are deterministic.

    Parameters
        abstract_graph : AbstractGraph
            Input graph to evaluate conditions against.
        conditions_functions : list[tuple[Callable[[AbstractGraph], bool], Callable[[AbstractGraph], AbstractGraph]]]
            Ordered list of (predicate, function) pairs. Each predicate decides
            whether its paired function should run.
        else_function : Callable[[AbstractGraph], AbstractGraph]
            Transformation to apply if none of the predicates evaluate True.

    Returns
        AbstractGraph
            Output of the first matching branch function, or else_function if
            no predicates match.

    Algorithm
        1. For each (predicate, func) in conditions_functions:
              if predicate(abstract_graph) is True:
                  return func(abstract_graph)
        2. If no predicate matched, return else_function(abstract_graph).

    Complexity
        - Time: sum of costs of predicates until first True + cost of one branch.
        - Memory: size of branch output (only one branch executed).
        - Worst case: evaluate all predicates if none match.

    Interactions
        - Extends `if_then_else` with multiple “elif” clauses.
        - Works inside pipelines with `forward_compose` or `compose` for
          conditional multi-path logic.
        - Often combined with threshold-based or structural tests on graphs.

    Examples
        # Branch by number of interpretation nodes with multiple conditions
        workflow = forward_compose(
            connected_component(),
            if_then_elif_else(
                conditions_functions=[
                    (lambda ag: ag.interpretation_graph.number_of_nodes() > 20, merge()),
                    (lambda ag: ag.interpretation_graph.number_of_nodes() > 10, cycle()),
                    (lambda ag: ag.interpretation_graph.number_of_nodes() > 5, clique())
                ],
                else_function=path()
            ),
            filter_by_number_of_nodes(number_of_nodes=(4, 10))
        )
        qg_out = workflow(qg_in)

    Domain Analogies
        - Programming: an `if … elif … elif … else …` chain.
        - Chemistry: apply different analysis depending on molecule size ranges.
        - Social networks: use different detection algorithms depending on group size.

    Failure Modes
        - Empty conditions list: always falls through to else_function.
        - Predicate exceptions: if any predicate raises, evaluation stops with error.
        - Branch incompatibility: if selected function produces outputs not
          compatible with downstream operators.
        - Ordering pitfalls: only the first True predicate is used, later matches ignored.
    """
    for predicate, func in conditions_functions:
        if predicate(abstract_graph):
            return _call_decomposition(func, abstract_graph)
    return _call_decomposition(else_function, abstract_graph)


#====================================================================================================
# ITERATION OPERATORS
#====================================================================================================

@curry
def for_loop(
    abstract_graph: 'AbstractGraph',
    function: Callable[['AbstractGraph'], 'AbstractGraph'],
    n_iterations: int = 1
) -> 'AbstractGraph':
    """Fixed-iteration loop operator for AbstractGraph transformations.
    Summary
        Repeatedly applies a decomposition function to the input graph a fixed
        number of times.

    Semantics
        - Input AG state:
            Takes the input AbstractGraph and repeatedly transforms it.
        - Output AG state:
            Result of applying `function` exactly `n_iterations` times in sequence.
        - Determinism:
            Deterministic if the function is deterministic.

    Parameters
        abstract_graph : AbstractGraph
            The starting graph.
        function : Callable[[AbstractGraph], AbstractGraph]
            Transformation to apply in each iteration.
        n_iterations : int, optional (default=1)
            Number of times to apply the function.

    Returns
        AbstractGraph
            The graph obtained after `n_iterations` applications of the function.

    Algorithm
        1. Initialise current = abstract_graph.
        2. Repeat `n_iterations` times:
              current = function(current).
        3. Return current.

    Complexity
        - Time: n_iterations × cost(function).
        - Memory: dominated by the largest intermediate AbstractGraph.

    Interactions
        - Useful when functions converge toward a fixed point within a bounded number of steps.
        - Can approximate iterative refinement (e.g. repeated filtering).
        - Often paired with `merge`, `cycle`, or `filter_by_*`.

    Examples
        workflow = forward_compose(
            connected_component(),
            for_loop(cycle(), n_iterations=3),
            filter_by_number_of_nodes(number_of_nodes=(4, 10))
        )
        qg_out = workflow(qg_in)

    Domain Analogies
        - Programming: a standard `for` loop with fixed iteration count.
        - Chemistry: apply the same reaction step repeatedly (e.g., washing cycles).
        - Social networks: re-apply clustering to stabilise group boundaries.

    Failure Modes
        - n_iterations <= 0 → function is never applied (returns input unchanged).
        - Non-idempotent or divergent function → unstable or meaningless result.
    """
    for _ in range(n_iterations):
        abstract_graph = _call_decomposition(function, abstract_graph)
    return abstract_graph

@curry
def while_loop(
    abstract_graph: 'AbstractGraph',
    function: Callable[['AbstractGraph'], 'AbstractGraph'],
    predicate: Callable[['AbstractGraph'], bool],
    max_iterations: int = 100
) -> 'AbstractGraph':
    """Predicate-controlled loop operator for AbstractGraph transformations.
    Summary
        Repeatedly applies a decomposition function as long as a predicate on
        the current graph is True, or until `max_iterations` is reached.

    Semantics
        - Input AG state:
            Starts with the provided AbstractGraph, repeatedly checks predicate.
        - Output AG state:
            Final state after zero or more iterations of `function`, stopped when
            predicate fails or iteration cap reached.
        - Determinism:
            Deterministic if function and predicate are deterministic.

    Parameters
        abstract_graph : AbstractGraph
            The starting graph.
        function : Callable[[AbstractGraph], AbstractGraph]
            Transformation to apply on each iteration.
        predicate : Callable[[AbstractGraph], bool]
            Loop continues while this condition evaluates True.
        max_iterations : int, optional (default=100)
            Upper bound on iterations to avoid infinite loops.

    Returns
        AbstractGraph
            The graph obtained after applying the function repeatedly until the
            predicate is False or the iteration limit is reached.

    Algorithm
        1. Initialise current = abstract_graph, iteration = 0.
        2. While predicate(current) is True and iteration < max_iterations:
              current = function(current)
              iteration += 1
        3. Return current.

    Complexity
        - Time: O(k × cost(function) + k × cost(predicate)), where k is the number of iterations executed.
        - Memory: governed by the largest intermediate AbstractGraph.

    Interactions
        - Enables fixed-point iteration until convergence criteria are met.
        - Works well with `merge` or `deduplicate` to shrink until stable.
        - Natural complement to `for_loop`.

    Examples
        workflow = forward_compose(
            connected_component(),
            while_loop(
                cycle(),
                predicate=lambda ag: ag.interpretation_graph.number_of_nodes() > 5,
                max_iterations=10
            ),
            merge()
        )
        qg_out = workflow(qg_in)

    Domain Analogies
        - Programming: a `while` loop with condition and safety cap.
        - Chemistry: repeat titration steps until pH threshold reached.
        - Social networks: reapply clustering until no further group changes occur.

    Failure Modes
        - Predicate always False → input returned unchanged.
        - Predicate never False and function non-convergent → forced stop at max_iterations.
        - Predicate exceptions halt execution.
        - Function divergence may produce runaway growth in intermediate graphs.
    """
    iteration = 0
    while predicate(abstract_graph) and iteration < max_iterations:
        abstract_graph = _call_decomposition(function, abstract_graph)
        iteration += 1
    return abstract_graph

#====================================================================================================
# UNARY OPERATORS
#====================================================================================================

@curry
def identity(
    abstract_graph: 'AbstractGraph',
    param=None
) -> 'AbstractGraph':
    """Identity operator for AbstractGraph transformations.
    Summary
        Returns the input AbstractGraph unchanged, serving as a no-op in pipelines.

    Semantics
        - Input AG state:
            Accepts a AbstractGraph and creates a new wrapper that references it.
        - Output AG state:
            Equivalent to the input graph; no modification to the base or interpretation graph.
        - Determinism:
            Deterministic (output always mirrors input).

    Parameters
        abstract_graph : AbstractGraph
            The input graph to be returned as output.
        param : Any, optional (default=None)
            Placeholder argument for consistency with operator signatures; unused.

    Returns
        AbstractGraph
            The same graph as the input, wrapped as a new AbstractGraph instance.

    Algorithm
        1. Construct a new AbstractGraph referencing the input.
        2. Return it directly.

    Complexity
        - Time: O(1).
        - Memory: O(1), except for wrapper instantiation overhead.

    Interactions
        - Useful as a placeholder in dynamically generated pipelines.
        - Can act as a neutral element in composition operators (`compose`, `add`).
        - Helpful for debugging: insert identity to check intermediate graph state.

    Examples
        # Use identity in a pipeline
        workflow = forward_compose(
            connected_component(),
            identity(),
            merge()
        )
        qg_out = workflow(qg_in)

    Domain Analogies
        - Mathematics: the identity function f(x) = x.
        - Chemistry: a reagent that leaves the molecule unchanged.
        - Social networks: “observe but do not intervene.”

    Failure Modes
        - None: safe in all contexts.
        - Potential confusion: some users may expect `identity` to clone deeply,
          but here it only returns an equivalent AbstractGraph wrapper.
    """
    out_abstract_graph = AbstractGraph(abstract_graph=abstract_graph)
    return out_abstract_graph

#--------------------------------------------------------------------------------
@curry
def random_part(
    abstract_graph: 'AbstractGraph',
    n_samples=1
    ) -> 'AbstractGraph':
    """Emit bootstrapped random subgraphs from each mapped subgraph.
    Summary
        For each mapped subgraph, draw bootstrap samples of its nodes (with replacement),
        induce subgraphs on the sampled node sets, and keep only the largest connected component.
        Repeat this process n_samples times per interpretation node.

    Semantics
        - Input AG state: Reads abstract_graph.base_graph and current interpretation-node mapped subgraphs.
        - Output AG state: Returns a new AbstractGraph with one interpretation node per bootstrap sample,
          each mapped to the largest connected component of the sampled-induced subgraph.
        - Determinism: Non-deterministic due to random sampling.

    Parameters
        n_samples : int, default 1
            Number of bootstrap samples to generate per associated subgraph.

    Algorithm
        - For each associated subgraph:
            * Sample |V| nodes with replacement.
            * Take the unique set of sampled nodes.
            * Induce the subgraph on that set and keep its largest connected component.
            * Emit one interpretation node for each sample.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )
    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        nodes = list(subgraph.nodes())
        if not nodes:
            continue
        sample_size = len(nodes)
        for _ in range(int(n_samples)):
            sampled = random.choices(nodes, k=sample_size)
            node_set = set(sampled)
            if not node_set:
                continue
            induced = subgraph.subgraph(node_set)
            if induced.number_of_nodes() == 0:
                continue
            if induced.is_directed():
                components = nx.connected_components(induced.to_undirected())
            else:
                components = nx.connected_components(induced)
            try:
                largest = max(components, key=len)
            except ValueError:
                continue
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                largest,
                meta=build_meta_from_function_context()
            )
    return out_abstract_graph

#--------------------------------------------------------------------------------
@curry
def node(
    abstract_graph: 'AbstractGraph',
    param=None
    ) -> 'AbstractGraph':
    """Emit one interpretation node for each singleton vertex in the current mapped subgraphs.
    Summary
        For every mapped subgraph, decompose it into its constituent single vertices and
        create one new interpretation node per vertex. Each new interpretation node maps to an induced subgraph consisting
        of exactly that one vertex.

    Semantics
        - Input AG state: Reads abstract_graph.base_graph and all current interpretation-node mapped subgraphs.
        - Output AG state: Returns a new AbstractGraph with the same base graph and an interpretation graph in which
          each node corresponds to a singleton subgraph {v}. Provenance metadata is stored for traceability.
        - Determinism: Deterministic given the input graph; order of singleton emission is not semantically significant.

    Parameters
        param : Any, optional
            Placeholder argument for interface consistency. Ignored.

    Algorithm
        - Initialize a fresh AbstractGraph with the same base graph.
        - For each mapped subgraph:
            * Iterate over all its nodes.
            * For each node, create a singleton subgraph [{node}].
            * Call create_interpretation_node_with_subgraph_from_nodes(singleton, meta=build_meta_from_function_context()).

    Complexity
        Let S be the number of mapped subgraphs, and N_i their node counts.
        Time: Σ_i O(N_i) to iterate and create singleton interpretation nodes.
        Memory: O(total number of nodes) for storing singletons.

    Side Effects & Metadata
        - Each created interpretation node stores:
            * 'mapped_subgraph': a subgraph containing a single vertex from the base graph.
            * 'meta': {'source_function': 'node', 'params': {...}} from build_meta_from_function_context().
        - Labels/attributes are not computed here; call update() to populate them.

    Interactions
        - Often used as the "finest granularity" seed for subsequent operators (e.g., neighborhoods, degree filters).
        - Useful in `add` to combine node-level views with larger motifs.
        - Composes naturally with `neighborhood` to generate ego-graphs around individual nodes.

    Constraints & Invariants
        - Works with any undirected or directed base graph.
        - Emits exactly one singleton per node in each input mapped subgraph.
        - If a mapped subgraph is empty, no interpretation nodes are created.

    Examples
        # Break graph into singleton interpretation nodes
        workflow = forward_compose(node())
        Q2 = workflow(Q).update()

        # Use node() + neighborhood to generate radius-1 ego graphs around every vertex
        workflow = forward_compose(node(), neighborhood(radius=(1,1)))
        Q2 = workflow(Q).update()

    Domain Analogies
        - Social networks: one interpretation node per individual user.
        - Computer networks: one interpretation node per device.
        - Chemistry: one interpretation node per atom in a molecule.

    Failure Modes & Diagnostics
        - Potential explosion in interpretation-node count for very large graphs; mitigate with sampling or filters.
        - Ensure downstream operators handle large numbers of singletons efficiently.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = [[node] for node in subgraph.nodes()]
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )

    return out_abstract_graph

#--------------------------------------------------------------------------------
@curry
def edge(
    abstract_graph: 'AbstractGraph',
    param=None
    ) -> 'AbstractGraph':
    """Emit one interpretation node for each edge in the current mapped subgraphs.
    Summary
        For every mapped subgraph, decompose it into its constituent edges and
        create one new interpretation node per edge. Each new interpretation node maps to the induced subgraph
        consisting of exactly the two incident vertices and their connecting edge.

    Semantics
        - Input AG state: Reads abstract_graph.base_graph and all current interpretation-node mapped subgraphs.
        - Output AG state: Returns a new AbstractGraph with the same base graph and an interpretation graph in which
          each node corresponds to a single edge subgraph. Provenance metadata is attached for traceability.
        - Determinism: Deterministic given the input graph; order of edge emission is not semantically significant.

    Parameters
        param : Any, optional
            Placeholder argument for interface consistency. Ignored.

    Algorithm
        - Initialize a fresh AbstractGraph with the same base graph.
        - For each mapped subgraph:
            * Iterate over its edge list.
            * For each edge (u, v), build a 2-node induced subgraph {u, v} with the connecting edge.
            * Call create_interpretation_node_with_subgraph_from_nodes(edge, meta=build_meta_from_function_context()).

    Complexity
        Let S be the number of mapped subgraphs, and E_i their edge counts.
        Time: Σ_i O(E_i) to iterate and create edge-based interpretation nodes.
        Memory: O(total number of edges) for storing induced 2-node subgraphs.

    Side Effects & Metadata
        - Each created interpretation node stores:
            * 'mapped_subgraph': a subgraph with exactly 2 vertices and 1 edge.
            * 'meta': {'source_function': 'edge', 'params': {...}} from build_meta_from_function_context().
        - Labels/attributes are not computed here; call update() to populate them.

    Interactions
        - Often paired with `node()` to produce both vertex-level and edge-level features.
        - Useful in `add` to mix edge-based subgraphs with higher-order motifs (cycles, cliques).
        - Can precede `neighborhood` to grow paths or ego-graphs from edges.

    Constraints & Invariants
        - Works with undirected and directed graphs (edges will reflect graph type).
        - Emits exactly one 2-node subgraph per edge in each input mapped subgraph.
        - If a mapped subgraph has no edges, no interpretation nodes are created.

    Examples
        # Break graph into edge subgraphs
        workflow = forward_compose(edge())
        Q2 = workflow(Q).update()

        # Combine edges and cycles
        workflow = forward_compose(add(edge(), cycle()))
        Q2 = workflow(Q).update()

    Domain Analogies
        - Social networks: one feature node per friendship/connection.
        - Computer networks: one feature node per physical or logical link.
        - Chemistry: one feature node per bond between two atoms.

    Failure Modes & Diagnostics
        - Explosion in interpretation-node count for dense graphs (O(n^2) edges). Use filters or degree constraints upstream.
        - Directed graphs yield ordered edge pairs; ensure downstream operators handle orientation if relevant.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )
    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = list(subgraph.edges())
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_abstract_graph

#--------------------------------------------------------------------------------
def connected_component_decomposition_function(subgraph):
    """Find connected components of a subgraph.

    Parameters
    ----------
    subgraph : networkx.Graph
        Input undirected graph to partition into connected components.

    Returns
    -------
    components : list[set]
        List of node sets, one per connected component.
    """
    components = list(nx.connected_components(subgraph))
    return components


@curry
def connected_component(
    abstract_graph: 'AbstractGraph',
    param=None
    ) -> 'AbstractGraph':
    """Emit one interpretation node per connected component in each mapped subgraph.
    Summary
        For every mapped subgraph, compute its connected components and create one new interpretation node
        per component, preserving the original base graph and adding provenance metadata.

    Semantics
        - Input AG state: Reads abstract_graph.base_graph and all current interpretation-node mapped subgraphs.
        - Output AG state: Returns a new AbstractGraph with the same base graph and an interpretation graph whose nodes
          each map to a connected component (node-induced subgraph) of the original mapped subgraphs.
          Invariants: base graph unchanged; newly created interpretation nodes have `mapped_subgraph` set and `meta` populated.
        - Determinism: Deterministic given inputs; the order of emitted components is not semantically significant.

    Parameters
        param : Any, optional
            Unused placeholder to keep a uniform operator signature. Ignored.

    Algorithm
        - Initialize `out_abstract_graph` with the same base graph.
        - For each mapped subgraph:
            * Compute components = connected_component_decomposition_function(subgraph).
            * For each component (set of nodes), call
              `create_interpretation_node_with_subgraph_from_nodes(component, meta=build_meta_from_function_context())`.

    Complexity
        Let S be the number of input mapped subgraphs and (V_i, E_i) their sizes.
        Time: Σ_i O(|V_i| + |E_i|) for components + overhead to create interpretation nodes.
        Memory: O(total emitted nodes + edges) across all component subgraphs.

    Side Effects & Metadata
        - Each created interpretation node stores:
            * `mapped_subgraph` : induced subgraph on the component’s node set.
            * 'meta' : {'source_function': 'connected_component', 'params': {...}} via build_meta_from_function_context().
        - Labels/attributes are not computed here; call `update()` later if needed.

    Interactions
        - Often followed by filters (e.g., `filter_by_number_of_nodes`) to bound instance counts.
        - Composes well with `add`, `product`, and distance-based combinators after reducing to components.

    Constraints & Invariants
        - Assumes mapped subgraphs are undirected or that connectedness is well-defined for them.
        - Empty mapped subgraphs emit no components.

    Examples
        # Minimal: break current mapped subgraphs into components
        workflow = forward_compose(connected_component())
        Q2 = workflow(Q).update()

        # With size filter to limit instances
        workflow = forward_compose(
            connected_component(),
            filter_by_number_of_nodes(number_of_nodes=(4, 50))
        )

    Domain Analogies
        - Social networks: split groups into disconnected communities before further analysis.
        - Computer networks: decompose a selected topology region into subnets.
        - Chemistry: separate disconnected fragments in a selected molecular region.

    Failure Modes & Diagnostics
        - If mapped subgraphs are directed graphs, nx.connected_components may fail; ensure they are undirected
          or convert appropriately upstream.
        - Excessive instance counts if the input mapped subgraphs are highly fragmented; mitigate with size filters.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = connected_component_decomposition_function(subgraph)
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_abstract_graph


#--------------------------------------------------------------------------------
def degree_decomposition_function(subgraph, min_degree=0, max_degree=2):
    """Select nodes in a subgraph whose degree lies within [min_degree, max_degree].

    Parameters
    ----------
    subgraph : networkx.Graph
        Input graph on which node degrees are computed.
    min_degree : int, default 0
        Inclusive lower bound for node degree.
    max_degree : int, default 2
        Inclusive upper bound for node degree.

    Returns
    -------
    components : list[set]
        A one-element list containing the set of nodes that satisfy the degree constraint.
    """
    deg = dict(nx.degree(subgraph))
    component = set([u for u in deg if max_degree >= deg[u] and deg[u] >= min_degree])
    components = [component]
    return components

@curry
def degree(
    abstract_graph: 'AbstractGraph',
    value = (0,2)
    ) -> 'AbstractGraph':
    """Emit one interpretation node per degree-filtered mapped subgraph.
    Summary
        For every mapped subgraph, select the subset of its vertices whose degree is between the
        specified bounds, and create a new interpretation node whose mapped subgraph is induced on that node set.

    Semantics
        - Input AG state: Reads abstract_graph.base_graph and current interpretation-node mapped subgraphs.
        - Output AG state: Returns a new AbstractGraph with the same base graph and one interpretation node
          per mapped subgraph, representing the degree-filtered set of vertices (possibly empty).
        - Determinism: Deterministic given input graph and degree bounds.

    Parameters
        value : int | tuple[int,int], default (0,2)
            Inclusive lower and upper bounds for degree selection.
            If a single int is given, treated as (value, value).

    Algorithm
        - Normalize value into (min_degree, max_degree).
        - For each mapped subgraph:
            * Call degree_decomposition_function(subgraph, min_degree, max_degree).
            * For the returned node set, create one interpretation node via
              create_interpretation_node_with_subgraph_from_nodes(component, meta=...).

    Complexity
        Let S be number of mapped subgraphs, and V_i their vertex counts.
        Time: Σ_i O(|V_i| + |E_i|) for degree calculation.
        Memory: O(total nodes across mapped subgraphs).

    Side Effects & Metadata
        - Each created interpretation node stores:
            * `mapped_subgraph`: induced subgraph on nodes satisfying the degree constraint.
            * 'meta': {'source_function': 'degree', 'params': {...}}.
        - Labels/attributes not computed here; call update() afterwards.

    Interactions
        - Commonly used to isolate hubs or periphery before applying further operators (neighborhood, cycle).
        - Can be combined with `filter_by_number_of_nodes` to suppress empty or tiny degree-selected sets.
        - Works naturally inside `add` to parallelize degree-based and structural decompositions.

    Constraints & Invariants
        - Works for both directed and undirected graphs, but degree is total degree for directed.
        - Produces exactly one interpretation node per mapped subgraph (possibly empty subgraph).

    Examples
        # Extract all degree-1 vertices across mapped subgraphs
        workflow = forward_compose(degree(value=1))
        Q2 = workflow(Q).update()

        # Select low-degree (1–2) and then apply neighborhood expansion
        workflow = forward_compose(
            degree(value=(1,2)),
            neighborhood(radius=(1,1))
        )

    Domain Analogies
        - Social networks: isolate leaves (followers only) vs. hubs (many friends).
        - Computer networks: edge devices vs. backbone routers.
        - Chemistry: hydrogens (degree=1) vs. branching carbons.

    Failure Modes & Diagnostics
        - Produces empty mapped subgraphs when no nodes match; can accumulate many empty interpretation nodes.
        - Explosion risk is low, but many singleton sets can be produced if graph is large.
    """
    value = value_to_2tuple(value)
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = degree_decomposition_function(
            subgraph,
            min_degree=min(value),
            max_degree=max(value)
        )
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_abstract_graph


#--------------------------------------------------------------------------------
def split_decomposition_function(subgraph, seed=0):
    """Bipartition a connected subgraph using Kernighan–Lin; else return the whole node set.

    Args:
        subgraph: Input (undirected) subgraph. If disconnected or if bisection
            fails, no split is performed.
        seed: Seed passed to Kernighan-Lin initialisation. Use a fixed integer
            for reproducibility or None for stochastic behaviour.

    Returns:
        list[set]: [set(part1), set(part2)] on success; otherwise [set(all_nodes)].
    """
    # If subgraph is not connected, we simply return the whole node set.
    if not nx.is_connected(subgraph):
        return [set(subgraph.nodes())]
    
    try:
        part1, part2 = kernighan_lin_bisection(subgraph, seed=seed)
        return [set(part1), set(part2)]
    except Exception:
        # In case of any error, fall back to not splitting.
        return [set(subgraph.nodes())]

@curry
def split(
    abstract_graph: 'AbstractGraph',
    n_parts=2,
    seed=0
    ) -> 'AbstractGraph':
    """Emit interpretation nodes by recursively bipartitioning each mapped subgraph via Kernighan–Lin.
    Summary
        For every mapped subgraph, repeatedly split the largest current part until
        `n_parts` are reached (or no further split is possible). Each part is emitted
        as an interpretation node.

    Semantics
        - Input AG state: Reads `abstract_graph.base_graph` and current interpretation-node mapped subgraphs.
        - Output AG state: Returns a new `AbstractGraph` with the same base graph; its interpretation graph contains
          one or more nodes per input mapped subgraph, each mapped to the induced subgraph on the part.
          Determinism: Deterministic when `seed` is fixed; stochastic when `seed=None`.

    Parameters
        n_parts : int, default 2
            Desired number of output parts per input mapped subgraph. The operator
            recursively splits the largest current part to approach this target.
        seed : int | None, default 0
            Random seed passed to Kernighan-Lin bisection. Keep fixed for
            deterministic partitions; set to None for stochastic partitions.

    Algorithm
        - Initialize `out_abstract_graph` with the same base graph.
        - For each mapped subgraph:
            * Start with one part containing all its nodes.
            * While number of parts is below `n_parts`:
                - Select the largest current part.
                - Attempt a KL bisection on that part's induced subgraph.
                - If split succeeds into two non-empty proper parts, replace the
                  selected part with the two new parts; otherwise stop.
            * Emit one interpretation node per resulting part.

    Complexity
        Let S be the number of mapped subgraphs, with sizes (V_i, E_i).
        - KL bisection per subgraph is roughly O(|E_i|) to O(|V_i|^2) depending on implementation and graph density.
        - Overall time: Σ_i KL_cost(subgraph_i) + interpretation-node creation overhead.
        - Memory: proportional to emitted induced subgraphs.

    Side Effects & Metadata
        - For split parts, each created interpretation node includes:
            * `mapped_subgraph` : induced subgraph on the part’s node set.
            * 'meta' : {'source_function': 'split', 'params': {...}} via build_meta_from_function_context().
        - Degenerate/non-splittable cases still emit one interpretation node with metadata.

    Interactions
        - Common precursor to `filter_by_number_of_nodes/edges` to prune tiny or huge parts.
        - Works well before `clique`, `cycle`, or distance-based combinators to limit combinatorics.
        - Can be iterated via `for_loop(split(), n)` or `while_loop(...)` for coarse-to-fine partitioning.

    Constraints & Invariants
        - Assumes undirected graphs for KL; on disconnected subgraphs or KL failures, falls back to a single part.
        - Does not modify the base graph.

    Examples
        # One-shot bisection of current mapped subgraphs
        workflow = forward_compose(split())
        Q2 = workflow(Q).update()

        # Iterative partition then keep medium-sized parts
        workflow = forward_compose(
            split(), split(),
            filter_by_number_of_nodes(number_of_nodes=(10, 200))
        )

    Domain Analogies
        - Social networks: partition a community into two cohorts (e.g., interest-based halves).
        - Computer networks: split a subnet into two clusters.
        - Chemistry: divide a large fragment into two sub-fragments before motif extraction.

    Failure Modes & Diagnostics
        - Highly irregular or tiny subgraphs may not split meaningfully; expect single-part output.
        - For very dense graphs, KL can be expensive—consider bounding subgraph size upstream.
    """
    if not isinstance(n_parts, int) or n_parts < 1:
        raise ValueError("n_parts must be an integer >= 1.")

    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )
    
    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        def _connected_parts_from_nodes(nodes):
            induced = subgraph.subgraph(nodes)
            if nx.is_directed(subgraph):
                return [set(c) for c in nx.weakly_connected_components(induced)]
            return [set(c) for c in nx.connected_components(induced)]

        if nx.is_directed(subgraph):
            parts = [set(c) for c in nx.weakly_connected_components(subgraph)]
        else:
            parts = [set(c) for c in nx.connected_components(subgraph)]
        if not parts:
            parts = [set()]

        while len(parts) < n_parts:
            split_done = False
            for largest_part_idx in sorted(range(len(parts)), key=lambda i: len(parts[i]), reverse=True):
                largest_part = parts[largest_part_idx]
                if len(largest_part) < 2:
                    continue

                largest_subgraph = subgraph.subgraph(largest_part).copy()
                split_parts = split_decomposition_function(largest_subgraph, seed=seed)
                split_parts = [set(p) for p in split_parts if len(p) > 0]

                # Accept only meaningful bisections that partition the selected part.
                if len(split_parts) < 2:
                    continue
                if any(p == largest_part for p in split_parts):
                    continue
                if len(set.union(*split_parts)) != len(largest_part):
                    continue

                # After splitting, continue recursion on connected components.
                connected_split_parts = []
                for p in split_parts:
                    connected_split_parts.extend(_connected_parts_from_nodes(p))
                connected_split_parts = [p for p in connected_split_parts if len(p) > 0]
                if len(connected_split_parts) < 2:
                    continue

                parts.pop(largest_part_idx)
                parts.extend(connected_split_parts)
                split_done = True
                break

            if not split_done:
                break

        for part in parts:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                list(part),
                meta=build_meta_from_function_context()
            )
    
    return out_abstract_graph


#--------------------------------------------------------------------------------
def get_reachable_nodes_bfs(
    graph: nx.Graph,
    source: Any,
    cutoff: int
    ) -> List[Any]:
    """Return nodes within a given BFS radius from a source node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph in which BFS is performed.
    source : Any
        Node ID to start the BFS from.
    cutoff : int
        Maximum hop distance to include (must be ≥ 0).

    Returns
    -------
    list[Any]
        Nodes reachable from `source` within the cutoff radius.

    Raises
    ------
    nx.NetworkXError
        If `source` is not present in the graph.
    ValueError
        If `cutoff` is negative.
    """
    if not isinstance(cutoff, int) or cutoff < 0:
        raise ValueError("Cutoff must be a non-negative integer.")
    
    if source not in graph:
        raise nx.NetworkXError(f"Source node {source} not present in graph.")

    if cutoff == 0:
        return [source]
    
    path_lengths = nx.single_source_shortest_path_length(graph, source, cutoff=cutoff)
    return list(path_lengths.keys())


@curry
def neighborhood(
    abstract_graph: 'AbstractGraph',
    radius=(0,1)
) -> 'AbstractGraph':
    """Emit interpretation nodes for BFS neighborhoods of each node in current mapped subgraphs.
    Summary
        For each node in each input subgraph, generate BFS balls of all radii r in [min_radius, max_radius].
        Each resulting ball is represented as a new interpretation node in the output AbstractGraph.

    Semantics
        - Input AG state: Reads abstract_graph.base_graph and current interpretation-node mapped subgraphs.
        - Output AG state: New AbstractGraph whose interpretation graph contains one node per BFS neighborhood.
          Provenance metadata records operator and parameters.
        - Determinism: Fully deterministic given graph and radius.

    Parameters
        radius : int | tuple[int,int], default (0,1)
            Inclusive radius bounds. If a single int is given, interpreted as (r,r).

    Algorithm
        - Normalize radius bounds using value_to_2tuple().
        - For each mapped subgraph:
            * For r from min_radius to max_radius:
                - For each node in the subgraph:
                    - Call get_reachable_nodes_bfs(subgraph, source=node, cutoff=r).
                    - Create an interpretation node with the induced subgraph on reachable nodes.

    Complexity
        Let N = total nodes, E = total edges, R = number of radius values.
        - BFS cost per node is O(E) worst-case; repeated for N × R nodes.
        - Output size: O(N × R) interpretation nodes per mapped subgraph.

    Side Effects & Metadata
        - Each created interpretation node stores:
            * `mapped_subgraph`: induced subgraph on reachable set.
            * 'meta': {'source_function': 'neighborhood', 'params': {'radius': (rmin,rmax)}}.
        - Labels/attributes not computed; call update() downstream.

    Interactions
        - Naturally follows `node()` to expand singletons into ego-graphs.
        - Can be combined with `filter_by_number_of_nodes` to control explosion in large neighborhoods.
        - Works well with `add` to mix neighborhoods with other motifs.

    Constraints & Invariants
        - Works with any connected or disconnected graph.
        - Emits one interpretation node per (node, r) pair.
        - Empty results only possible for r=0 (singleton neighborhoods).

    Examples
        # Expand nodes into radius-1 ego-graphs
        workflow = forward_compose(node(), neighborhood(radius=1))

        # Multi-scale neighborhoods up to radius 3
        workflow = forward_compose(neighborhood(radius=(1,3)))

    Domain Analogies
        - Social networks: “friends-of-friends” circles at increasing hop distance.
        - Computer networks: subnets at hop distance r from a given device.
        - Chemistry: atom-centered fragments at increasing bond distance.

    Failure Modes & Diagnostics
        - Potential combinatorial blow-up in dense graphs; mitigate with radius limits or filters.
        - For very large radius ranges, output size can exceed memory quickly.
    """
    radius = value_to_2tuple(radius)
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        for r in range(min(radius), max(radius) + 1):
            for source in subgraph.nodes():
                component = get_reachable_nodes_bfs(subgraph, source, cutoff=r)
                out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                    component,
                    meta=build_meta_from_function_context()
                )
    
    return out_abstract_graph

#--------------------------------------------------------------------------------
def get_edges_from_cycle(cycle):
    """Yield edges (u, v) along a cycle, ensuring u < v for consistency."""
    for i, c in enumerate(cycle):
        j = (i + 1) % len(cycle)
        u, v = cycle[i], cycle[j]
        if u < v:
            yield u, v
        else:
            yield v, u

def get_cycle_basis_edges(g):
    """Return the list of edges belonging to all cycles in the graph."""
    ebunch = []
    cs = nx.cycle_basis(g)
    for c in cs:
        ebunch += list(get_edges_from_cycle(c))
    return ebunch

def edge_list_complement(g, ebunch):
    """Return edges of g that are not in ebunch."""
    edge_set = set(ebunch)
    other_ebunch = [e for e in g.edges() if e not in edge_set]
    return other_ebunch

def edge_subgraph(g, ebunch):
    """Induce subgraph of g using only the given edge list ebunch."""
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
        g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for u, v in ebunch:
        g2.add_edge(u, v)
        g2.edges[u, v].update(g.edges[u, v])
    return g2

def edge_complement_subgraph(g, ebunch):
    """Induce subgraph from edges of g that are not in ebunch."""
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
        g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for e in g.edges():
        if e not in ebunch:
            u, v = e
            g2.add_edge(u, v)
            g2.edges[u, v].update(g.edges[u, v])
    return g2


def local_graph_complement_subgraph(subgraph):
    """Return the graph complement on the current mapped subgraph node set."""
    if nx.is_directed(subgraph):
        g2 = nx.DiGraph()
        g2.add_nodes_from((node, data.copy()) for node, data in subgraph.nodes(data=True))
        nodes = list(subgraph.nodes())
        for u in nodes:
            for v in nodes:
                if u == v or subgraph.has_edge(u, v):
                    continue
                g2.add_edge(u, v)
    else:
        g2 = nx.Graph()
        g2.add_nodes_from((node, data.copy()) for node, data in subgraph.nodes(data=True))
        nodes = list(subgraph.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if subgraph.has_edge(u, v):
                    continue
                g2.add_edge(u, v)
    return g2


def _graph_edge_key(edge, is_directed):
    u, v = edge
    return (u, v) if is_directed else frozenset((u, v))


def _mapped_subgraph_group_key(subgraph):
    is_directed = nx.is_directed(subgraph)
    node_key = tuple(sorted(subgraph.nodes()))
    edge_key = tuple(sorted(_graph_edge_key(edge, is_directed) for edge in subgraph.edges()))
    return is_directed, node_key, edge_key

def cycle_decomposition_function(subgraph):
    """Return node sets corresponding to all simple cycles in the subgraph."""
    cs = nx.cycle_basis(subgraph)
    cycle_components = list(map(set, cs))
    return cycle_components

def non_cycle_decomposition_function(subgraph):
    """Return node sets of acyclic connected components after removing cycle edges."""
    cs = nx.cycle_basis(subgraph)
    cycle_ebunch = get_cycle_basis_edges(subgraph)
    g2 = edge_complement_subgraph(subgraph, cycle_ebunch)
    non_cycle_components = nx.connected_components(g2)
    non_cycle_components = [c for c in non_cycle_components if len(c) >= 2]
    non_cycle_components = list(map(set, non_cycle_components))
    return non_cycle_components

@curry
def cycle(
    abstract_graph: 'AbstractGraph',
    param=None
    ) -> 'AbstractGraph':
    """Emit one interpretation node per cycle in each mapped subgraph.
    Summary
        For each input mapped subgraph, compute its simple cycle basis and create one interpretation node per cycle,
        with the mapped subgraph set to the induced subgraph on that cycle’s nodes.

    Semantics
        - Input AG state: Uses abstract_graph.base_graph and interpretation-node mapped subgraphs.
        - Output AG state: New AbstractGraph where each interpretation node corresponds to a simple cycle.
        - Determinism: Deterministic given the input graph; cycle_basis order is consistent per run.

    Parameters
        param : Any, optional
            Placeholder for operator interface; ignored.

    Algorithm
        - For each mapped subgraph:
            * Call cycle_decomposition_function(subgraph).
            * For each cycle (node set), create an interpretation node via create_interpretation_node_with_subgraph_from_nodes().

    Complexity
        - cycle_basis: O(|V| + |E|) per subgraph.
        - Total cost: sum over all mapped subgraphs.

    Metadata
        - Each emitted interpretation node stores `mapped_subgraph` (the cycle-induced subgraph) and `meta`
          with source_function='cycle'.

    Interactions
        - Complements `tree()` operator to split graph into cyclic vs. acyclic parts.
        - Useful before `combination` to link cycles with functional groups in chemistry,
          or with `neighborhood` to explore cycle context.

    Examples
        # Decompose into cycles
        workflow = forward_compose(connected_component(), cycle())

    Domain Analogies
        - Social networks: closed friendship circles.
        - Computer networks: routing loops.
        - Chemistry: aromatic rings or other cyclic motifs.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = cycle_decomposition_function(subgraph)
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_abstract_graph

@curry
def tree(
    abstract_graph: 'AbstractGraph',
    param=None
    ) -> 'AbstractGraph':
    """Emit one interpretation node per acyclic connected component in each mapped subgraph.
    Summary
        For each mapped subgraph, remove all cycle edges and compute the remaining connected components.
        For each nontrivial acyclic component, create an interpretation node mapped to that node set.

    Semantics
        - Input AG state: Uses abstract_graph.base_graph and interpretation-node mapped subgraphs.
        - Output AG state: New AbstractGraph where each interpretation node is an acyclic component (tree).
        - Determinism: Deterministic given input graph.

    Parameters
        param : Any, optional
            Placeholder argument; ignored.

    Algorithm
        - For each mapped subgraph:
            * Call non_cycle_decomposition_function(subgraph).
            * For each returned component (node set), create a new interpretation node.

    Complexity
        - Cycle detection + complement graph construction: O(|V| + |E|).
        - Connected components: O(|V| + |E|).

    Metadata
        - Each interpretation node stores `mapped_subgraph` (acyclic subgraph) and `meta` with source_function='tree'.

    Interactions
        - Complements `cycle()` to partition graph into cyclic and acyclic parts.
        - Useful for chemistry (chains vs. rings), social nets (tree-like follower structures), or
          computer networks (tree-like spanning subnetworks).

    Examples
        # Extract tree-like parts
        workflow = forward_compose(connected_component(), tree())

    Domain Analogies
        - Social networks: hierarchical tree structures (e.g., org charts).
        - Computer networks: spanning trees, acyclic subnetworks.
        - Chemistry: chain structures vs. rings.

    Failure Modes
        - Tiny graphs (<2 nodes) may not yield meaningful components.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = non_cycle_decomposition_function(subgraph)
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_abstract_graph

#--------------------------------------------------------------------------------
def path_decomposition_function(subgraph, min_number_of_edges=1, max_number_of_edges=None):
    """Return node sets corresponding to simple paths within a length range.

    Parameters
    ----------
    subgraph : networkx.Graph
        Input graph in which to search for paths.
    min_number_of_edges : int, default 1
        Minimum path length (in edges) to include.
    max_number_of_edges : int, optional
        Maximum path length (in edges). Defaults to number of nodes.

    Returns
    -------
    list[tuple]
        Unique tuples of node IDs representing paths within the length range.
    """
    if max_number_of_edges is None:
        max_number_of_edges = subgraph.number_of_nodes()
    edge_components = []
    for n in subgraph.nodes():
        ego_graph = nx.ego_graph(subgraph, n, radius=max_number_of_edges+1)
        for v in ego_graph.nodes():
            try:
                for path in nx.all_shortest_paths(ego_graph, source=n, target=v):
                    edge_component = set()
                    if len(path) >= min_number_of_edges + 1 and len(path) <= max_number_of_edges + 1:
                        for i, u in enumerate(path[:-1]):
                            w = path[i + 1]
                            edge_component.add(u)
                            edge_component.add(w)
                    if edge_component:
                        edge_component = tuple(sorted(edge_component))
                        edge_components.append(edge_component)
            except Exception:
                pass
    components = list(set(edge_components))
    return components

@curry
def path(
    abstract_graph: 'AbstractGraph',
    number_of_edges=(1,3)
    ) -> 'AbstractGraph':
    """Emit one interpretation node per path within given edge-length bounds.
    Summary
        For each subgraph, enumerate simple paths whose length in edges lies between
        `min_number_of_edges` and `max_number_of_edges`. Each distinct path’s nodes
        form a new interpretation node in the output AbstractGraph.

    Semantics
        - Input AG state: Reads base_graph and current interpretation-node mapped subgraphs.
        - Output AG state: New AbstractGraph with additional interpretation nodes, one per qualifying path.
        - Determinism: Deterministic given input graph and parameters.

    Parameters
        number_of_edges : int | tuple[int,int], default (1,3)
            Inclusive range of path lengths in edges. If a single int is given, it is treated as (n,n).

    Algorithm
        - Normalize number_of_edges with value_to_2tuple().
        - For each mapped subgraph:
            * Call path_decomposition_function(subgraph, min, max).
            * For each resulting node set, create a new interpretation node with the induced mapped subgraph.

    Complexity
        Path enumeration can grow exponentially with graph size.
        - all_shortest_paths dominates cost: O(#paths × path_length).
        - Filtering bounds keeps only paths within min/max.

    Metadata
        Each interpretation node stores `mapped_subgraph` (induced path subgraph) and `meta`
        with source_function='path' and parameters.

    Interactions
        - Useful in chemistry to capture chains of bonds of fixed lengths.
        - In social networks, can represent "friend-of-friend-of-friend" chains of specific depth.
        - Often paired with `combination` or `filter_by_number_of_nodes` to avoid blow-up.

    Examples
        # Extract paths of length 2–4
        workflow = forward_compose(
            connected_component(),
            path(number_of_edges=(2,4))
        )

    Domain Analogies
        - Chemistry: carbon chains of length n.
        - Social networks: paths of introductions (degree of separation).
        - Computer networks: routing paths up to certain hops.

    Failure Modes & Diagnostics
        - For large or dense graphs, path enumeration may explode in size.
        - Paths shorter than `min_number_of_edges` or longer than `max_number_of_edges` are ignored.
    """
    number_of_edges = value_to_2tuple(number_of_edges)
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = path_decomposition_function(
            subgraph,
            min_number_of_edges=min(number_of_edges),
            max_number_of_edges=max(number_of_edges)
        )
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_abstract_graph

#--------------------------------------------------------------------------------
def spine_decomposition_function(subgraph, radius=0, seed=42):
    """Return one spine node set built from a reproducible diameter path.

    For each source node, run BFS over the subgraph, collect all shortest paths
    from that source to the farthest reachable nodes, and keep every path whose
    edge length matches the graph-wide maximum. One of those longest paths is
    chosen uniformly at random using ``seed``. The final node set is the union
    of the chosen path and all nodes within ``radius`` hops of any path node.
    """
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    if subgraph.number_of_nodes() == 0:
        return []
    rng = random.Random(seed)

    longest_paths = []
    max_length = -1
    for source in subgraph.nodes():
        shortest_paths = nx.single_source_shortest_path(subgraph, source)
        if not shortest_paths:
            continue
        local_max_length = max(len(path) - 1 for path in shortest_paths.values())
        for path_nodes in shortest_paths.values():
            path_length = len(path_nodes) - 1
            if path_length != local_max_length:
                continue
            canonical_path = tuple(path_nodes)
            if path_length > max_length:
                max_length = path_length
                longest_paths = [canonical_path]
            elif path_length == max_length:
                longest_paths.append(canonical_path)

    if not longest_paths:
        chosen_path = (next(iter(subgraph.nodes())),)
    else:
        unique_paths = list(dict.fromkeys(longest_paths))
        chosen_path = rng.choice(unique_paths)

    spine_nodes = set(chosen_path)
    if radius > 0:
        for node_id in chosen_path:
            spine_nodes.update(get_reachable_nodes_bfs(subgraph, node_id, cutoff=radius))
    return [tuple(sorted(spine_nodes))]


@curry
def spine(
    abstract_graph: 'AbstractGraph',
    radius=0,
    seed=42
    ) -> 'AbstractGraph':
    """Emit one interpretation node per mapped subgraph using a diameter spine.
    Summary
        For each mapped subgraph, find one random path among the longest BFS
        paths (graph-diameter candidates), then optionally thicken that path by
        including all nodes within ``radius`` hops of the path nodes. The
        induced subgraph on that final node set becomes one interpretation
        node.

    Parameters
        radius : int, default 0
            Hop radius added around every node on the selected spine path.
            ``0`` returns only the path nodes themselves.
        seed : int | None, default 42
            Seed used to choose among equally long candidate spine paths.
            Use a fixed value for reproducible decompositions.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = spine_decomposition_function(subgraph, radius=radius, seed=seed)
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )

    return out_abstract_graph

#--------------------------------------------------------------------------------
def graphlet_decomposition_function(subgraph, radius=1, min_number_of_nodes=1, max_number_of_nodes=3):
    """Enumerate connected ego-subgraphs of bounded size.

    Parameters
    ----------
    subgraph : networkx.Graph
        Input graph to search for subgraphs.
    radius : int, default 1
        Ego radius around each node to consider when forming graphlets.
    min_number_of_nodes : int, default 1
        Minimum number of nodes in each subgraph.
    max_number_of_nodes : int, default 3
        Maximum number of nodes in each subgraph.

    Returns
    -------
    list[tuple]
        Unique tuples of node IDs, each defining a connected graphlet.
    """
    components = []
    for size in range(min_number_of_nodes, max_number_of_nodes + 1):
        for u in subgraph.nodes():
            ego_graph = nx.ego_graph(subgraph, u, radius=radius)
            for sub_nodes in itertools.combinations(ego_graph.nodes(), size):
                sub_subgraph = ego_graph.subgraph(sub_nodes)
                if nx.is_connected(sub_subgraph):
                    components.append(tuple(sorted(set(sub_nodes))))
    components = list(set(components))
    return components

@curry
def graphlet(
    abstract_graph: 'AbstractGraph',
    radius=1,
    number_of_nodes=(1,3)
    ) -> 'AbstractGraph':
    """Emit interpretation nodes for connected graphlets within ego neighborhoods.
    Summary
        For each input subgraph, enumerate all connected induced subgraphs
        (“graphlets”) of size between `min_number_of_nodes` and `max_number_of_nodes`
        inside ego neighborhoods of radius `r`. Each graphlet becomes a new interpretation node.

    Semantics
        - Input AG state: Reads base_graph and interpretation-node mapped subgraphs.
        - Output AG state: New AbstractGraph with one interpretation node per connected graphlet.
        - Determinism: Deterministic enumeration, though order of graphlets is not guaranteed.

    Parameters
        radius : int, default 1
            Ego radius around each node to expand before sampling graphlets.
        number_of_nodes : int | tuple[int,int], default (1,3)
            Inclusive bounds on graphlet size (number of nodes).

    Algorithm
        - Normalize number_of_nodes with value_to_2tuple().
        - For each mapped subgraph:
            * Call graphlet_decomposition_function(subgraph, radius, min, max).
            * For each node tuple, create an interpretation node representing the induced subgraph.

    Complexity
        - Exponential in subgraph size due to combinations.
        - Mitigated by limiting radius and max_number_of_nodes.

    Metadata
        - Each interpretation node stores `mapped_subgraph` (graphlet subgraph) and `meta`
          with source_function='graphlet' and params.

    Interactions
        - More fine-grained than `clique()` or `path()`.
        - Can be combined with `filter_by_number_of_nodes` to limit explosion.
        - Useful precursor to motif-based classification tasks.

    Examples
        # Extract all connected graphlets up to size 4
        workflow = forward_compose(
            connected_component(),
            graphlet(radius=2, number_of_nodes=(2,4))
        )

    Domain Analogies
        - Chemistry: small functional fragments around atoms.
        - Social networks: micro-groups (triads, quads).
        - Computer networks: small local subnetworks.

    Failure Modes & Diagnostics
        - Large radius or high max_number_of_nodes leads to combinatorial blow-up.
        - Disconnected candidate subgraphs are discarded.
    """
    number_of_nodes = value_to_2tuple(number_of_nodes) 
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = graphlet_decomposition_function(
            subgraph,
            radius=radius,
            min_number_of_nodes=min(number_of_nodes),
            max_number_of_nodes=max(number_of_nodes)
        )
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_abstract_graph

#--------------------------------------------------------------------------------    
def clique_decomposition_function(subgraph, min_number_of_nodes=1, max_number_of_nodes=None):
    """Enumerate cliques within size bounds in a subgraph.

    Parameters
    ----------
    subgraph : networkx.Graph
        Input graph to search for cliques.
    min_number_of_nodes : int, default 1
        Minimum clique size to include.
    max_number_of_nodes : int, optional
        Maximum clique size to include. Defaults to size of subgraph.

    Returns
    -------
    list[list]
        List of cliques (as lists of node IDs) whose size lies in [min, max].
    """
    if max_number_of_nodes is None:
        max_number_of_nodes = subgraph.number_of_nodes()
    cliques = nx.enumerate_all_cliques(subgraph)
    components = list(filter(lambda x: min_number_of_nodes <= len(x) <= max_number_of_nodes, cliques))
    return components

@curry
def clique(
    abstract_graph: 'AbstractGraph',
    number_of_nodes=(1,3)
    ) -> 'AbstractGraph':
    """Emit one interpretation node per clique of bounded size.
    Summary
        For each subgraph, enumerate all cliques (fully connected subgraphs) whose
        number of nodes lies between given bounds. Each clique becomes a new interpretation node.

    Semantics
        - Input AG state: Uses abstract_graph.base_graph and interpretation-node mapped subgraphs.
        - Output AG state: New AbstractGraph with interpretation nodes corresponding to cliques.
        - Determinism: Deterministic given NetworkX’s clique enumeration.

    Parameters
        number_of_nodes : int | tuple[int,int], default (1,3)
            Inclusive range of clique sizes (number of nodes).

    Algorithm
        - Normalize number_of_nodes with value_to_2tuple().
        - For each mapped subgraph:
            * Call clique_decomposition_function(subgraph, min, max).
            * For each clique, create an interpretation node with the induced mapped subgraph.

    Complexity
        Clique enumeration can be exponential in graph density and size.
        - Worst case: O(3^(n/3)) cliques.
        - Bounded by min/max size to limit blow-up.

    Metadata
        - Each interpretation node stores `mapped_subgraph` (clique subgraph) and `meta`
          with source_function='clique' and params.

    Interactions
        - Complements path(), cycle(), and graphlet() as structural motif extractors.
        - Useful in chemistry for aromatic rings or fully bonded functional groups.
        - In social networks, corresponds to tightly-knit groups.

    Examples
        # Extract cliques of size 3–5
        workflow = forward_compose(
            connected_component(),
            clique(number_of_nodes=(3,5))
        )

    Domain Analogies
        - Social networks: close friendship groups where everyone knows each other.
        - Computer networks: fully interconnected subnetworks.
        - Chemistry: ring systems where all atoms are bonded to each other.

    Failure Modes & Diagnostics
        - Large dense graphs may yield many cliques (combinatorial explosion).
        - Single-node cliques are included unless filtered by number_of_nodes.
    """
    number_of_nodes = value_to_2tuple(number_of_nodes)
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = clique_decomposition_function(
            subgraph,
            min_number_of_nodes=min(number_of_nodes),
            max_number_of_nodes=max(number_of_nodes)
        )
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_abstract_graph

#--------------------------------------------------------------------------------    
@curry
def complement(
    abstract_graph: 'AbstractGraph',
    param=None
    ) -> 'AbstractGraph':
    """Emit interpretation nodes representing the complement of each mapped subgraph.
    Summary
        For every mapped subgraph, create a new interpretation node
        whose mapped subgraph consists of all base-graph nodes *not* in the original subgraph.

    Semantics
        - Input AG state: Reads the base-graph node set and current interpretation-node mapped subgraphs.
        - Output AG state: New AbstractGraph where each interpretation node corresponds to the complement
          node set of its input mapped subgraph.
        - Determinism: Deterministic given the input graph and mapped subgraphs.

    Parameters
        param : ignored
            Present only for consistency with curried operator signatures.

    Algorithm
        - For each mapped subgraph:
            * Collect its node set.
            * Compute set difference with all nodes in base_graph.
            * Create a new interpretation node with that complementary node set.

    Complexity
        - Time: O(N) per subgraph, where N is number of nodes in base_graph.
        - Memory: proportional to output size (number of complement sets).

    Metadata
        - Each output interpretation node stores `mapped_subgraph` (induced subgraph of complement nodes)
          and 'meta' with source_function='complement'.

    Interactions
        - Often paired with cycle(), clique(), or path() to model inside–outside relationships.
        - Can be chained to build families like (subgraph, complement) pairs for contrastive features.
        - Useful for “negative space” reasoning: what is *not* included in a motif.

   

    Examples
        # Generate complements of connected components
        workflow = forward_compose(
            connected_component(),
            complement()
        )

    Domain Analogies
        - Social networks: people not in a given community.
        - Computer networks: devices outside a given subnet.
        - Chemistry: atoms outside a functional group.

    Failure Modes & Diagnostics
        - Empty complements arise if subgraph = entire base_graph.
        - Full complements (all nodes) arise if subgraph = ∅ (rare).
        - Complement size may dwarf the original subgraph; consider filtering.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        component = list(subgraph.nodes())
        component = set(abstract_graph.base_graph.nodes()).difference(set(component))
        out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
            component,
            meta=build_meta_from_function_context()
        )
    
    return out_abstract_graph

#--------------------------------------------------------------------------------    
@curry
def local_complement(
    abstract_graph: 'AbstractGraph',
    param=None
    ) -> 'AbstractGraph':
    """Emit node complements within each parent mapped subgraph.

    Interpretation nodes are grouped by the parent mapped subgraph they were
    decomposed from. For each parent, this operator collects the union of child
    component nodes and returns the remaining parent nodes as one node-induced
    subgraph.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    grouped_children = defaultdict(list)
    for _, data in abstract_graph.interpretation_graph.nodes(data=True):
        subgraph = get_mapped_subgraph(data)
        if subgraph is None:
            continue
        meta = data.get("meta", {})
        parent_subgraph = meta.get("parent_mapped_subgraph", subgraph)
        grouped_children[_mapped_subgraph_group_key(parent_subgraph)].append((parent_subgraph, subgraph))

    for grouped_subgraphs in grouped_children.values():
        parent_subgraph = grouped_subgraphs[0][0]
        covered_nodes = set()
        for _, subgraph in grouped_subgraphs:
            covered_nodes.update(subgraph.nodes())
        remaining_nodes = set(parent_subgraph.nodes()).difference(covered_nodes)
        if not remaining_nodes:
            continue
        complement_subgraph = parent_subgraph.subgraph(remaining_nodes).copy()
        meta = build_meta_from_function_context()
        meta["parent_mapped_subgraph"] = parent_subgraph.copy()
        out_abstract_graph.create_interpretation_node_with_subgraph_from_subgraph(
            complement_subgraph,
            meta=meta
        )

    return out_abstract_graph

#--------------------------------------------------------------------------------    
@curry
def edge_complement(
    abstract_graph: 'AbstractGraph',
    param=None
    ) -> 'AbstractGraph':
    """Emit interpretation nodes from the edge complement of each mapped subgraph.

    For each mapped subgraph, this operator takes all base-graph edges that are
    not present in the subgraph and emits the edge-induced subgraph from those edges.

    Args:
        abstract_graph: Input AbstractGraph.
        param: Ignored; kept for signature consistency with other curried operators.

    Returns:
        AbstractGraph: A new AbstractGraph with one interpretation node per input mapped subgraph,
        where each mapped subgraph is the edge-induced complement subgraph.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    is_directed = nx.is_directed(abstract_graph.base_graph)

    def edge_key(edge):
        u, v = edge
        return (u, v) if is_directed else frozenset((u, v))

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        subgraph_edge_keys = {edge_key(e) for e in subgraph.edges()}
        complement_edges = [
            e for e in abstract_graph.base_graph.edges()
            if edge_key(e) not in subgraph_edge_keys
        ]
        out_abstract_graph.create_interpretation_node_with_subgraph_from_edges(
            complement_edges,
            meta=build_meta_from_function_context()
        )

    return out_abstract_graph

#--------------------------------------------------------------------------------    
@curry
def local_edge_complement(
    abstract_graph: 'AbstractGraph',
    param=None
    ) -> 'AbstractGraph':
    """Emit edge complements within each parent mapped subgraph.

    Interpretation nodes are grouped by the parent mapped subgraph they were
    decomposed from. For each parent, this operator collects the union of child
    component edges and returns the remaining parent edges as one edge-induced
    subgraph.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    grouped_children = defaultdict(list)
    for _, data in abstract_graph.interpretation_graph.nodes(data=True):
        subgraph = get_mapped_subgraph(data)
        if subgraph is None:
            continue
        meta = data.get("meta", {})
        parent_subgraph = meta.get("parent_mapped_subgraph", subgraph)
        grouped_children[_mapped_subgraph_group_key(parent_subgraph)].append((parent_subgraph, subgraph))

    for grouped_subgraphs in grouped_children.values():
        parent_subgraph = grouped_subgraphs[0][0]
        is_directed = nx.is_directed(parent_subgraph)
        covered_edge_keys = set()
        for _, subgraph in grouped_subgraphs:
            covered_edge_keys.update(_graph_edge_key(edge, is_directed) for edge in subgraph.edges())
        remaining_edges = [
            edge for edge in parent_subgraph.edges()
            if _graph_edge_key(edge, is_directed) not in covered_edge_keys
        ]
        if not remaining_edges:
            continue
        complement_subgraph = parent_subgraph.edge_subgraph(remaining_edges).copy()
        meta = build_meta_from_function_context()
        meta["parent_mapped_subgraph"] = parent_subgraph.copy()
        out_abstract_graph.create_interpretation_node_with_subgraph_from_subgraph(
            complement_subgraph,
            meta=meta
        )

    return out_abstract_graph

#--------------------------------------------------------------------------------    
def betweenness_centrality_decomposition_function(subgraph, number_of_nodes=1, use_perifery=False):
    """Select nodes by betweenness centrality score.

    Parameters
    ----------
    subgraph : networkx.Graph
        Input graph to analyse.
    number_of_nodes : int, default 1
        Number of nodes to return.
    use_perifery : bool, default False
        If False, return the most central nodes; if True, return the least central nodes.

    Returns
    -------
    list[list]
        Single-element list containing a list of selected node IDs.
    """
    n_dict = nx.betweenness_centrality(subgraph)
    reverse = not use_perifery
    selected_ids = sorted(n_dict, key=lambda x: n_dict[x], reverse=reverse)[:number_of_nodes]
    components = [selected_ids] 
    return components

@curry
def betweenness_centrality(
    abstract_graph: 'AbstractGraph',
    number_of_nodes=1,
    use_perifery=False
    ) -> 'AbstractGraph':
    """Emit interpretation nodes for nodes ranked by betweenness centrality.
    Summary
        For each subgraph, compute betweenness centrality scores and select either
        the top-k most central nodes or the bottom-k least central nodes. Each selected
        set is emitted as a new interpretation node.

    Semantics
        - Input AG state: Reads base_graph and interpretation-node mapped subgraphs.
        - Output AG state: New AbstractGraph with interpretation nodes corresponding to
          sets of central or peripheral nodes.
        - Determinism: Deterministic given the input graph and parameters.

    Parameters
        number_of_nodes : int, default 1
            Number of nodes to select.
        use_perifery : bool, default False
            If False, select top central nodes. If True, select least central nodes.

    Algorithm
        - Compute betweenness centrality on each subgraph with NetworkX.
        - Sort nodes by centrality score (descending unless use_perifery=True).
        - Take the first `number_of_nodes`.
        - Create a new interpretation node for that set.

    Complexity
        Betweenness centrality is O(|V||E|) on unweighted graphs.
        - Cost grows quickly with graph size; suitable mainly for small subgraphs.

    Metadata
        - Each interpretation node stores `mapped_subgraph` (selected node set) and `meta`
          with source_function='betweenness_centrality' and params.

    Interactions
        - Complements structural operators like cycle(), path(), or clique()
          by capturing centrality instead of topology alone.
        - Useful for filtering to “hub” or “periphery” roles in networks.

    Examples
        # Select 2 most central nodes in each connected component
        workflow = forward_compose(
            connected_component(),
            betweenness_centrality(number_of_nodes=2, use_perifery=False)
        )

    Domain Analogies
        - Social networks: most influential users (high centrality) vs. fringe users (low centrality).
        - Transportation: hub airports vs. peripheral stops.
        - Biology: bottleneck proteins in interaction networks.

    Failure Modes & Diagnostics
        - On large subgraphs, centrality computation may be expensive.
        - If number_of_nodes exceeds subgraph size, returns all nodes.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = betweenness_centrality_decomposition_function(
            subgraph,
            number_of_nodes=number_of_nodes,
            use_perifery=use_perifery
        )
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_abstract_graph

#--------------------------------------------------------------------------------    
def betweenness_centrality_split_decomposition_function(subgraph, number_of_nodes=5):
    """Split ranked nodes by betweenness centrality into fixed-size chunks.

    Args:
        subgraph: Input NetworkX graph to analyse.
        number_of_nodes: Chunk size for the ranked node list. Each output
            component contains up to this many node IDs.
    Returns:
        list[list]: Components where each component is a chunk of ranked node IDs.
    """
    if number_of_nodes <= 0:
        raise ValueError(f"number_of_nodes must be > 0, got {number_of_nodes}")

    n_dict = nx.betweenness_centrality(subgraph)
    ranked_ids = sorted(n_dict, key=lambda x: n_dict[x], reverse=True)
    components = [
        ranked_ids[i:i + number_of_nodes]
        for i in range(0, len(ranked_ids), number_of_nodes)
    ]
    return components


@curry
def betweenness_centrality_split(
    abstract_graph: 'AbstractGraph',
    number_of_nodes=5
    ) -> 'AbstractGraph':
    """Emit induced subgraphs from chunks of betweenness-ranked nodes.

    Args:
        abstract_graph: Input AbstractGraph.
        number_of_nodes: Chunk size for grouped ranked nodes.
    Returns:
        AbstractGraph: A new AbstractGraph where each interpretation node is the induced
        subgraph of one chunk from the betweenness-ranked node list.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = betweenness_centrality_split_decomposition_function(
            subgraph,
            number_of_nodes=number_of_nodes
        )
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )

    return out_abstract_graph

#--------------------------------------------------------------------------------    
def betweenness_centrality_hop_split_decomposition_function(subgraph, n_hops=1):
    """Split by BFS hop windows from high-betweenness anchors with overlap.

    The graph is explored starting from nodes sorted by descending betweenness
    centrality. For each unexplored anchor, BFS distances define hop windows:
    [0, n_hops], [n_hops, 2*n_hops], [2*n_hops, 3*n_hops], ... .
    Each window is node-induced and then split into connected components.
    Adjacent windows share the boundary hop (e.g., distance n_hops), creating
    overlap between consecutive outputs.

    Args:
        subgraph: Input NetworkX graph to analyse.
        n_hops: Hop-window width and stride (must be > 0).

    Returns:
        list[list]: Components as lists of node IDs.
    """
    if n_hops <= 0:
        raise ValueError(f"n_hops must be > 0, got {n_hops}")

    if subgraph.number_of_nodes() == 0:
        return []

    n_dict = nx.betweenness_centrality(subgraph)
    ranked_ids = sorted(n_dict, key=lambda x: n_dict[x], reverse=True)

    def iter_connected_components(g):
        if nx.is_directed(g):
            for c in nx.weakly_connected_components(g):
                yield c
        else:
            for c in nx.connected_components(g):
                yield c

    components = []
    explored = set()

    for anchor in ranked_ids:
        if anchor in explored:
            continue

        distances = nx.single_source_shortest_path_length(subgraph, anchor)
        explored.update(distances.keys())

        max_dist = max(distances.values(), default=0)
        start = 0
        while start <= max_dist:
            stop = start + n_hops
            window_nodes = [node_id for node_id, d in distances.items() if start <= d <= stop]
            if window_nodes:
                window_subgraph = subgraph.subgraph(window_nodes)
                for comp in iter_connected_components(window_subgraph):
                    if comp:
                        components.append(list(comp))
            start += n_hops

    return components


@curry
def betweenness_centrality_hop_split(
    abstract_graph: 'AbstractGraph',
    n_hops=1
    ) -> 'AbstractGraph':
    """Emit connected subgraphs from overlapping BFS hop windows.

    Args:
        abstract_graph: Input AbstractGraph.
        n_hops: Hop-window width/stride used during BFS windowing.

    Returns:
        AbstractGraph: A new AbstractGraph with one interpretation node per emitted
        connected component from the hop-window decomposition.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = betweenness_centrality_hop_split_decomposition_function(
            subgraph,
            n_hops=n_hops
        )
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )

    return out_abstract_graph

#--------------------------------------------------------------------------------
def _count_boundary_nodes(subgraph: nx.Graph, part_nodes: set) -> int:
    """
    Count nodes in a part that touch at least one node outside the part.

    Args:
        subgraph: Parent subgraph containing the part.
        part_nodes: Node set representing one partition part.

    Returns:
        int: Number of boundary nodes in the part.
    """
    boundary = 0
    for node_id in part_nodes:
        if any(neigh not in part_nodes for neigh in subgraph.neighbors(node_id)):
            boundary += 1
    return int(boundary)


def _count_cut_edges(subgraph: nx.Graph, part_a: set, part_b: set) -> int:
    """
    Count edges crossing between two partition parts.

    Args:
        subgraph: Parent subgraph.
        part_a: First partition part.
        part_b: Second partition part.

    Returns:
        int: Number of crossing edges.
    """
    count = 0
    for u, v in subgraph.edges():
        if (u in part_a and v in part_b) or (u in part_b and v in part_a):
            count += 1
    return int(count)


def _best_kernighan_split(
    subgraph: nx.Graph,
    *,
    max_split_trials: int = 8,
    balance_tolerance: float = 0.2,
    seed: Optional[int] = None,
):
    """
    Compute a best-effort bisection using multiple Kernighan-Lin restarts.

    Args:
        subgraph: Connected subgraph to split.
        max_split_trials: Number of randomized restarts.
        balance_tolerance: Minimum min(|A|,|B|)/max(|A|,|B|) ratio.
        seed: Optional RNG seed.

    Returns:
        Optional[tuple[set, set, int, int]]:
            (part_a, part_b, max_boundary_nodes, cut_edges) or None.
    """
    if subgraph.number_of_nodes() < 2:
        return None
    rng = random.Random(seed)
    nodes = list(subgraph.nodes())
    trials = max(1, int(max_split_trials))
    best = None
    best_score = None

    for _ in range(trials):
        shuffled = list(nodes)
        rng.shuffle(shuffled)
        mid = max(1, len(shuffled) // 2)
        init_a = set(shuffled[:mid])
        init_b = set(shuffled[mid:])
        if not init_b:
            continue
        try:
            part_a, part_b = kernighan_lin_bisection(
                subgraph,
                partition=(init_a, init_b),
                seed=rng.randint(0, 2**31 - 1),
            )
        except Exception:
            continue

        part_a = set(part_a)
        part_b = set(part_b)
        if not part_a or not part_b:
            continue
        lo = min(len(part_a), len(part_b))
        hi = max(len(part_a), len(part_b))
        ratio = (float(lo) / float(hi)) if hi > 0 else 0.0
        if ratio < float(balance_tolerance):
            continue

        max_boundary = max(
            _count_boundary_nodes(subgraph, part_a),
            _count_boundary_nodes(subgraph, part_b),
        )
        cut_edges = _count_cut_edges(subgraph, part_a, part_b)
        score = (int(max_boundary), int(cut_edges), int(abs(len(part_a) - len(part_b))))
        if best_score is None or score < best_score:
            best = (part_a, part_b, int(max_boundary), int(cut_edges))
            best_score = score

    return best


def _connected_fallback_bisect(
    subgraph: nx.Graph,
    *,
    min_part_size: int = 1,
):
    """
    Build a connected 2-way split with a BFS growth fallback.

    Args:
        subgraph: Connected subgraph to split.
        min_part_size: Minimum size per part.

    Returns:
        Optional[tuple[set, set, int, int]]:
            (part_a, part_b, max_boundary_nodes, cut_edges) or None.
    """
    n = subgraph.number_of_nodes()
    if n < 2:
        return None
    min_sz = max(1, int(min_part_size))
    if n < 2 * min_sz:
        return None

    # Start from a high-degree seed to grow one connected side.
    seed = max(subgraph.nodes(), key=lambda u: subgraph.degree(u))
    target = n // 2
    visited = {seed}
    queue = [seed]
    head = 0
    while head < len(queue) and len(visited) < target:
        u = queue[head]
        head += 1
        for v in subgraph.neighbors(u):
            if v in visited:
                continue
            visited.add(v)
            queue.append(v)
            if len(visited) >= target:
                break

    part_a = set(visited)
    part_b = set(subgraph.nodes()) - part_a
    if len(part_a) < min_sz or len(part_b) < min_sz:
        return None

    # Keep each side connected by taking largest CC if needed.
    if not nx.is_connected(subgraph.subgraph(part_a)):
        ccs = list(nx.connected_components(subgraph.subgraph(part_a)))
        ccs.sort(key=len, reverse=True)
        part_a = set(ccs[0])
        part_b = set(subgraph.nodes()) - part_a
    if not part_b or not nx.is_connected(subgraph.subgraph(part_b)):
        ccs = list(nx.connected_components(subgraph.subgraph(part_b)))
        if not ccs:
            return None
        ccs.sort(key=len, reverse=True)
        part_b = set(ccs[0])
        part_a = set(subgraph.nodes()) - part_b
    if len(part_a) < min_sz or len(part_b) < min_sz:
        return None

    max_boundary = max(
        _count_boundary_nodes(subgraph, part_a),
        _count_boundary_nodes(subgraph, part_b),
    )
    cut_edges = _count_cut_edges(subgraph, part_a, part_b)
    return part_a, part_b, int(max_boundary), int(cut_edges)


def _inject_overlap_nodes(
    subgraph: nx.Graph,
    *,
    part_a: set,
    part_b: set,
    min_overlap_nodes: int = 1,
):
    """
    Ensure the two parts share at least ``min_overlap_nodes`` base-graph nodes.

    Args:
        subgraph: Parent subgraph.
        part_a: First part node set.
        part_b: Second part node set.
        min_overlap_nodes: Minimum required overlap size.

    Returns:
        Optional[tuple[set, set]]: Updated parts, or None if overlap is impossible.
    """
    need = max(0, int(min_overlap_nodes))
    if need == 0:
        return set(part_a), set(part_b)

    a = set(part_a)
    b = set(part_b)
    overlap = set(a & b)
    if len(overlap) >= need:
        return a, b

    # Nodes on cut boundary that can be duplicated across parts.
    a_boundary = [u for u in a if any(v in b for v in subgraph.neighbors(u))]
    b_boundary = [v for v in b if any(u in a for u in subgraph.neighbors(v))]
    # Prefer stronger bridge nodes first.
    a_boundary.sort(key=lambda u: sum(1 for v in subgraph.neighbors(u) if v in b), reverse=True)
    b_boundary.sort(key=lambda v: sum(1 for u in subgraph.neighbors(v) if u in a), reverse=True)

    for u in a_boundary:
        if len(overlap) >= need:
            break
        if u in overlap:
            continue
        b.add(u)
        overlap.add(u)
    for v in b_boundary:
        if len(overlap) >= need:
            break
        if v in overlap:
            continue
        a.add(v)
        overlap.add(v)

    if len(overlap) < need:
        return None
    return a, b


def _edge_cover_components(subgraph: nx.Graph):
    """
    Cover a subgraph with connected edge-components (plus isolated singletons).

    Args:
        subgraph: Input subgraph to cover.

    Returns:
        list[list]: Components represented as node lists.
    """
    comps = []
    touched = set()
    for u, v in subgraph.edges():
        comps.append([u, v])
        touched.add(u)
        touched.add(v)
    for node_id in subgraph.nodes():
        if node_id not in touched:
            comps.append([node_id])
    return comps


def _count_attachment_edges(subgraph: nx.Graph, part_nodes: set) -> int:
    """
    Count edges from a node set to the rest of the subgraph.

    Args:
        subgraph: Parent subgraph.
        part_nodes: Candidate detachable node set.

    Returns:
        int: Number of attachment edges.
    """
    inside = set(part_nodes)
    attachment = 0
    for u in inside:
        for v in subgraph.neighbors(u):
            if v not in inside:
                attachment += 1
    return int(attachment)


def _best_low_attachment_split(
    subgraph: nx.Graph,
    *,
    max_attachment_edges: int = 2,
    min_part_size: int = 3,
    min_detach_size: int = 6,
    max_pair_trials: int = 3000,
    seed: Optional[int] = None,
):
    """
    Prefer splitting off a large component attached by very few edges (1-2).

    Args:
        subgraph: Connected subgraph to split.
        max_attachment_edges: Maximum allowed attachment edges for detachable side.
        min_part_size: Minimum size for both split sides.
        min_detach_size: Minimum size for detached side.
        max_pair_trials: Cap on 2-edge removal trials.
        seed: Optional RNG seed.

    Returns:
        Optional[tuple[set, set, int, int]]:
            (part_a, part_b, max_boundary_nodes, cut_edges) or None.
    """
    n = subgraph.number_of_nodes()
    if n < 2:
        return None
    min_sz = max(1, int(min_part_size))
    min_detach = max(min_sz, int(min_detach_size))
    max_attach = max(1, int(max_attachment_edges))
    rng = random.Random(seed)

    candidates = []
    seen = set()

    def _consider_component(comp_nodes: set):
        comp = set(comp_nodes)
        if not comp:
            return
        rem = set(subgraph.nodes()) - comp
        if len(comp) < min_detach or len(rem) < min_sz:
            return
        attach = _count_attachment_edges(subgraph, comp)
        if attach > max_attach:
            return
        key = frozenset(comp if len(comp) <= len(rem) else rem)
        if key in seen:
            return
        seen.add(key)
        a, b = comp, rem
        max_boundary = max(
            _count_boundary_nodes(subgraph, a),
            _count_boundary_nodes(subgraph, b),
        )
        cut_edges = _count_cut_edges(subgraph, a, b)
        # Prefer larger detachable side; break ties by cleaner cuts.
        score = (-min(len(a), len(b)), int(cut_edges), int(max_boundary))
        candidates.append((score, a, b, int(max_boundary), int(cut_edges)))

    # 1-edge attachments via bridges.
    for u, v in nx.bridges(subgraph):
        g = subgraph.copy()
        g.remove_edge(u, v)
        for cc in nx.connected_components(g):
            _consider_component(set(cc))

    # 2-edge attachments: remove edge pairs (bounded).
    if max_attach >= 2:
        edges = list(subgraph.edges())
        pair_indices = [(i, j) for i in range(len(edges)) for j in range(i + 1, len(edges))]
        if len(pair_indices) > int(max_pair_trials):
            pair_indices = rng.sample(pair_indices, int(max_pair_trials))
        for i, j in pair_indices:
            e1 = edges[i]
            e2 = edges[j]
            g = subgraph.copy()
            if g.has_edge(*e1):
                g.remove_edge(*e1)
            if g.has_edge(*e2):
                g.remove_edge(*e2)
            if nx.is_connected(g):
                continue
            for cc in nx.connected_components(g):
                _consider_component(set(cc))

    if not candidates:
        return None
    _, part_a, part_b, max_boundary, cut_edges = min(candidates, key=lambda x: x[0])
    return part_a, part_b, int(max_boundary), int(cut_edges)


def low_cut_partition_decomposition_function(
    subgraph: nx.Graph,
    *,
    target_max_boundary_nodes: int = 2,
    target_max_cut_edges: int = 2,
    max_part_size: int = 24,
    min_part_size: int = 3,
    max_split_trials: int = 8,
    balance_tolerance: float = 0.2,
    max_depth: int = 6,
    force_split_oversized: bool = True,
    allow_small_parts: bool = False,
    min_overlap_nodes: int = 1,
    strict_max_boundary: bool = False,
    prefer_low_attachment_split: bool = True,
    low_attachment_max_edges: int = 2,
    low_attachment_min_component_size: int = 6,
    low_attachment_max_pair_trials: int = 3000,
    seed: Optional[int] = None,
):
    """
    Partition a graph into connected parts while preferring low-complexity cuts.

    Args:
        subgraph: Graph to partition.
        target_max_boundary_nodes: Target max boundary-node count per part.
        target_max_cut_edges: Target max crossing-edge count for accepted splits.
        max_part_size: Preferred upper bound on part size.
        min_part_size: Minimum size for accepted child parts.
        max_split_trials: Number of randomized KL bisection restarts.
        balance_tolerance: Minimum split-balance ratio.
        max_depth: Recursion depth cap.
        force_split_oversized: Allow best-effort split when part is oversized.
        allow_small_parts: If False, reject splits creating too-small parts.
        min_overlap_nodes: Minimum node overlap injected between sibling parts.
        strict_max_boundary: If True, enforce boundary cap with guaranteed fallback
            decomposition (edge cover / singleton cover).
        prefer_low_attachment_split: If True, first try splitting off large parts
            attached by few edges (1-2) to avoid tiny fragmented components.
        low_attachment_max_edges: Max attachment edges allowed for detachable side.
        low_attachment_min_component_size: Minimum detachable component size.
        low_attachment_max_pair_trials: Cap on 2-edge-cut pair trials.
        seed: Optional RNG seed.

    Returns:
        list[list]: List of node lists representing partition parts.
    """
    if subgraph.number_of_nodes() == 0:
        return []

    tgt_boundary = max(0, int(target_max_boundary_nodes))
    tgt_cut_edges = max(0, int(target_max_cut_edges))
    max_size = max(1, int(max_part_size))
    min_size = max(1, int(min_part_size))
    depth_cap = max(0, int(max_depth))
    min_overlap = max(0, int(min_overlap_nodes))
    rng = random.Random(seed)

    components = []
    worklist = []
    for comp_nodes in nx.connected_components(subgraph):
        worklist.append((set(comp_nodes), 0))

    while worklist:
        nodes_set, depth = worklist.pop()
        if not nodes_set:
            continue
        part_n = len(nodes_set)
        if part_n <= max_size or depth >= depth_cap:
            components.append(list(nodes_set))
            continue

        local = subgraph.subgraph(nodes_set).copy()
        split = None
        if bool(prefer_low_attachment_split):
            split = _best_low_attachment_split(
                local,
                max_attachment_edges=low_attachment_max_edges,
                min_part_size=min_size,
                min_detach_size=low_attachment_min_component_size,
                max_pair_trials=low_attachment_max_pair_trials,
                seed=rng.randint(0, 2**31 - 1),
            )
        if split is None:
            split = _best_kernighan_split(
                local,
                max_split_trials=max_split_trials,
                balance_tolerance=balance_tolerance,
                seed=rng.randint(0, 2**31 - 1),
            )
        if split is None and force_split_oversized and part_n > max_size:
            split = _connected_fallback_bisect(
                local,
                min_part_size=min_size,
            )
        if split is None:
            components.append(list(nodes_set))
            continue

        part_a, part_b, max_boundary, cut_edges = split
        too_small = (len(part_a) < min_size) or (len(part_b) < min_size)
        if too_small and (not allow_small_parts):
            if force_split_oversized and part_n > max_size:
                fallback_split = _connected_fallback_bisect(
                    local,
                    min_part_size=min_size,
                )
                if fallback_split is None:
                    components.append(list(nodes_set))
                    continue
                part_a, part_b, max_boundary, cut_edges = fallback_split
            else:
                components.append(list(nodes_set))
                continue

        split_ok = (max_boundary <= tgt_boundary) and (cut_edges <= tgt_cut_edges)
        oversized = part_n > max_size
        if not split_ok and not (force_split_oversized and oversized):
            components.append(list(nodes_set))
            continue

        injected = _inject_overlap_nodes(
            local,
            part_a=part_a,
            part_b=part_b,
            min_overlap_nodes=min_overlap,
        )
        if injected is None:
            components.append(list(nodes_set))
            continue
        part_a, part_b = injected

        for child in (part_a, part_b):
            child_graph = local.subgraph(child)
            for cc in nx.connected_components(child_graph):
                worklist.append((set(cc), depth + 1))

    # Final safety pass: keep connected chunks only.
    final_components = []
    for nodes in components:
        local = subgraph.subgraph(nodes)
        for cc in nx.connected_components(local):
            final_components.append(list(cc))

    if not bool(strict_max_boundary):
        return final_components

    enforced = []
    for comp_nodes in final_components:
        part = set(comp_nodes)
        boundary = _count_boundary_nodes(subgraph, part)
        if boundary <= tgt_boundary:
            enforced.append(list(part))
            continue
        local = subgraph.subgraph(part).copy()
        if tgt_boundary <= 1:
            # Singleton cover guarantees boundary <= 1 in node terms.
            for node_id in local.nodes():
                enforced.append([node_id])
        else:
            # Edge cover guarantees boundary <= 2 per emitted component.
            enforced.extend(_edge_cover_components(local))
    return enforced


@curry
def low_cut_partition(
    abstract_graph: 'AbstractGraph',
    target_max_boundary_nodes: int = 2,
    target_max_cut_edges: int = 2,
    max_part_size: int = 24,
    min_part_size: int = 3,
    max_split_trials: int = 8,
    balance_tolerance: float = 0.2,
    max_depth: int = 6,
    force_split_oversized: bool = True,
    allow_small_parts: bool = False,
    min_overlap_nodes: int = 1,
    strict_max_boundary: bool = False,
    prefer_low_attachment_split: bool = True,
    low_attachment_max_edges: int = 2,
    low_attachment_min_component_size: int = 6,
    low_attachment_max_pair_trials: int = 3000,
    seed: Optional[int] = None,
) -> 'AbstractGraph':
    """
    Partition mapped subgraphs into connected pieces with low cut-interface complexity.

    Args:
        abstract_graph: Input AbstractGraph.
        target_max_boundary_nodes: Target max boundary-node count per part.
        target_max_cut_edges: Target max crossing-edge count for accepted splits.
        max_part_size: Preferred upper bound on part size.
        min_part_size: Minimum size for accepted child parts.
        max_split_trials: Number of randomized KL bisection restarts.
        balance_tolerance: Minimum split-balance ratio in [0, 1].
        max_depth: Recursion depth cap.
        force_split_oversized: Allow best-effort split when part is oversized.
        allow_small_parts: If False, reject splits creating too-small parts.
        min_overlap_nodes: Minimum node overlap injected between sibling parts.
        strict_max_boundary: If True, strictly enforce boundary cap via fallback.
        prefer_low_attachment_split: If True, prioritize splitting large low-attachment
            components (1-2 attachment edges) before generic bisection.
        low_attachment_max_edges: Max attachment edges allowed for detachable side.
        low_attachment_min_component_size: Minimum detachable component size.
        low_attachment_max_pair_trials: Cap on 2-edge-cut pair trials.
        seed: Optional RNG seed.

    Returns:
        AbstractGraph: New AbstractGraph with one interpretation node per partition part.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = low_cut_partition_decomposition_function(
            subgraph,
            target_max_boundary_nodes=target_max_boundary_nodes,
            target_max_cut_edges=target_max_cut_edges,
            max_part_size=max_part_size,
            min_part_size=min_part_size,
            max_split_trials=max_split_trials,
            balance_tolerance=balance_tolerance,
            max_depth=max_depth,
            force_split_oversized=force_split_oversized,
            allow_small_parts=allow_small_parts,
            min_overlap_nodes=min_overlap_nodes,
            strict_max_boundary=strict_max_boundary,
            prefer_low_attachment_split=prefer_low_attachment_split,
            low_attachment_max_edges=low_attachment_max_edges,
            low_attachment_min_component_size=low_attachment_min_component_size,
            low_attachment_max_pair_trials=low_attachment_max_pair_trials,
            seed=seed,
        )
        for component in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )

    return out_abstract_graph

#--------------------------------------------------------------------------------    
@curry
def merge(
    abstract_graph: 'AbstractGraph',
    use_edges=False
    ) -> 'AbstractGraph':
    """Merge all mapped subgraphs into a single interpretation node.
    Summary
        Combine the contents of all mapped subgraphs into one new interpretation node.
        By default, collects all nodes; if `use_edges=True`, collects all edges instead.

    Semantics
        - Input AG state: Reads mapped subgraphs from all current interpretation nodes.
        - Output AG state: New AbstractGraph with a single interpretation node containing
          either the union of all nodes or the union of all edges.
        - Determinism: Deterministic union, order of accumulation does not matter.

    Parameters
        use_edges : bool, default False
            If False, merge all node sets into one.  
            If True, merge all edge sets into one.

    Algorithm
        - Initialize an empty component.
        - For each mapped subgraph:
            * Collect nodes (default) or edges (`use_edges=True`).
            * Extend the component list.
        - Create a single interpretation node with this combined component.

    Complexity
        - Time: O(sum of sizes of all subgraphs).  
        - Memory: proportional to merged node or edge set.

    Metadata
        - Each output interpretation node stores `mapped_subgraph` (merged subgraph) and `meta`
          with source_function='merge' and params.

    Interactions
        - Useful as a “collapsing” step after decomposition, producing a single
          representative subgraph.
        - Can be paired with complement() or filters for aggregate reasoning.

    Examples
        # Merge all connected components into one node set
        workflow = forward_compose(
            connected_component(),
            merge()
        )

        # Merge all edges from cycle and tree decomposition
        workflow = add(cycle(), tree())
        workflow = forward_compose(workflow, merge(use_edges=True))

    Domain Analogies
        - Social networks: treat multiple communities as one collective group.
        - Computer networks: aggregate all subnetworks into one backbone.
        - Chemistry: union of functional groups into a single motif.

    Failure Modes & Diagnostics
        - If mapped subgraphs are empty, creates a single empty interpretation node.
        - `use_edges=True` produces edges but may result in disconnected node sets.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    if use_edges:
        component = []
        for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
            component.extend(subgraph.edges())
        out_abstract_graph.create_interpretation_node_with_subgraph_from_edges(
            component,
            meta=build_meta_from_function_context()
        )
    else:
        component = []
        for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
            component.extend(subgraph.nodes())
        out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
            component,
            meta=build_meta_from_function_context()
        )
    
    return out_abstract_graph

#--------------------------------------------------------------------------------
@curry
def deduplicate(
    abstract_graph: 'AbstractGraph',
) -> 'AbstractGraph':
    """Drop duplicate interpretation nodes based on their underlying base-graph node sets.
    Summary
        Keep the first occurrence of each mapped subgraph; duplicates are detected
        via a deterministic hash of the sorted base-graph node IDs.

    Semantics
        - Input: reads interpretation_graph nodes/edges and their mapped subgraphs.
        - Output: new AbstractGraph with deduped interpretation nodes; preserves
          base_graph and reattaches interpretation-graph edges between retained nodes.

    Algorithm
        - For each interpretation node, hash the sorted node IDs of its mapped subgraph.
        - If the hash has not been seen, keep the node; otherwise skip.
        - Carry over meta/label/attribute for kept nodes; remap edges accordingly.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    def key_fn(subgraph: nx.Graph) -> int:
        return hash_set(tuple(sorted(subgraph.nodes())))

    seen = set()
    old_to_new = {}

    for old_id, data in abstract_graph.interpretation_graph.nodes(data=True):
        mapped_subgraph = get_mapped_subgraph(data)
        if mapped_subgraph is None:
            continue
        k = key_fn(mapped_subgraph)
        if k in seen:
            continue
        seen.add(k)
        meta = dict(data.get("meta", {}))
        new_id = out_abstract_graph._add_interpretation_node(mapped_subgraph=mapped_subgraph.copy(), meta=meta)
        for attr_name in ("label", "attribute"):
            if attr_name in data:
                out_abstract_graph.interpretation_graph.nodes[new_id][attr_name] = data[attr_name]
        old_to_new[old_id] = new_id

    for u, v, edata in abstract_graph.interpretation_graph.edges(data=True):
        if u in old_to_new and v in old_to_new:
            out_abstract_graph.interpretation_graph.add_edge(old_to_new[u], old_to_new[v], **edata)

    return out_abstract_graph

# Legacy alias for backward compatibility; prefer deduplicate.
@curry
def unique(
    abstract_graph: 'AbstractGraph',
) -> 'AbstractGraph':
    """Deprecated alias for deduplicate."""
    return deduplicate(abstract_graph=abstract_graph)

#--------------------------------------------------------------------------------
@curry
def remove_redundant_associations(
    abstract_graph: 'AbstractGraph',
) -> 'AbstractGraph':
    """Remove interpretation nodes whose mapped subgraphs are strictly covered by larger ones.

    Args:
        abstract_graph: Input AbstractGraph.

    Returns:
        AbstractGraph: A new AbstractGraph with covered smaller mapped subgraphs removed.
    """
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    is_directed = nx.is_directed(abstract_graph.base_graph)

    def edge_key(edge):
        u, v = edge
        return (u, v) if is_directed else frozenset((u, v))

    image_nodes_data = list(abstract_graph.interpretation_graph.nodes(data=True))
    assoc_cache = {}
    for old_id, data in image_nodes_data:
        mapped_subgraph = get_mapped_subgraph(data)
        if not isinstance(mapped_subgraph, nx.Graph):
            mapped_subgraph = nx.Graph()
        nodes = set(mapped_subgraph.nodes())
        edges = {edge_key(e) for e in mapped_subgraph.edges()}
        assoc_cache[old_id] = (mapped_subgraph, nodes, edges, data)

    def is_strictly_covered(smaller_id, larger_id):
        _a_assoc, a_nodes, a_edges, _a_data = assoc_cache[smaller_id]
        _b_assoc, b_nodes, b_edges, _b_data = assoc_cache[larger_id]
        if not a_nodes.issubset(b_nodes):
            return False
        if not a_edges.issubset(b_edges):
            return False
        return (a_nodes != b_nodes) or (a_edges != b_edges)

    redundant_ids = set()
    old_ids = [old_id for old_id, _data in image_nodes_data]
    for i, old_id in enumerate(old_ids):
        for j, other_id in enumerate(old_ids):
            if i == j:
                continue
            if is_strictly_covered(old_id, other_id):
                redundant_ids.add(old_id)
                break

    kept_old_ids = [old_id for old_id in old_ids if old_id not in redundant_ids]

    old_to_new = {}
    for old_id in kept_old_ids:
        mapped_subgraph, _nodes, _edges, data = assoc_cache[old_id]
        meta = dict(data.get("meta", {}))
        new_id = out_abstract_graph._add_interpretation_node(mapped_subgraph=mapped_subgraph.copy(), meta=meta)
        for attr_name in ("label", "attribute"):
            if attr_name in data:
                out_abstract_graph.interpretation_graph.nodes[new_id][attr_name] = data[attr_name]
        old_to_new[old_id] = new_id

    for u, v, edata in abstract_graph.interpretation_graph.edges(data=True):
        if u in old_to_new and v in old_to_new:
            out_abstract_graph.interpretation_graph.add_edge(old_to_new[u], old_to_new[v], **edata)

    return out_abstract_graph


@curry
def remove_redundant_mapped_subgraphs(
    abstract_graph: 'AbstractGraph',
) -> 'AbstractGraph':
    """Canonical alias for remove_redundant_associations."""
    return remove_redundant_associations(abstract_graph=abstract_graph)

#--------------------------------------------------------------------------------    
@curry
def intersection(
    abstract_graph: 'AbstractGraph',
    node_size=None,
    must_be_connected: bool = True
) -> 'AbstractGraph':
    """Emit interpretation nodes for intersections of every pair of mapped subgraphs.
    Summary
        For each unordered pair of interpretation nodes in the input AbstractGraph, compute the
        intersection of their mapped subgraphs' node sets. If the intersection size
        is within the inclusive range `node_size`, create a new interpretation node whose
        mapped subgraph is the induced subgraph on those intersecting nodes.

    Semantics
        - Input AG state: Reads mapped subgraphs from current interpretation nodes and the base_graph.
        - Output AG state: New AbstractGraph with one interpretation node per qualifying intersection.
        - Determinism: Deterministic given the input graph and `node_size`.

    Parameters
        node_size : None | int | tuple[int,int], default None
            When None, no size filtering is applied to the intersection.
            If an int k is given, it is treated as (k,k). If a tuple (min,max)
            is given, the intersection size must satisfy min ≤ |I| ≤ max.
        must_be_connected : bool, default True
            If True, accept the intersection only when its induced subgraph on the
            base graph forms exactly one connected component.

    Algorithm
        - Iterate over all unordered pairs of interpretation nodes (u, v), u < v.
        - Compute intersection I = nodes(assoc[u]) ∩ nodes(assoc[v]).
        - If min(node_size) ≤ |I| ≤ max(node_size), create an interpretation node with the induced mapped subgraph on I.

    Complexity
        - Time: O(M^2 · d) where M = number of interpretation nodes, d = average subgraph size.
        - Memory: proportional to number and size of emitted intersections.

    Interactions
        - Complements `intersection_edges`, but creates new interpretation nodes instead of edges.
        - Often followed by `filter_by_number_of_nodes` or connectivity filters.

    Examples
        # Intersections among neighborhoods of radius 1 with size between 2 and 5
        workflow = forward_compose(
            neighborhood(radius=1),
            intersection(node_size=(2,5))
        )
    """
    # Normalise size bounds if provided
    if node_size is not None:
        node_size = value_to_2tuple(node_size)

    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    # Work on unordered pairs to avoid duplicates (u,v) and (v,u)
    # Additionally, deduplicate identical intersections across different pairs.
    seen = set()  # set[frozenset]
    img_nodes = list(abstract_graph.interpretation_graph.nodes())
    for u, v in combinations(img_nodes, 2):
        sub_u = get_mapped_subgraph(abstract_graph.interpretation_graph.nodes[u])
        sub_v = get_mapped_subgraph(abstract_graph.interpretation_graph.nodes[v])
        if sub_u is None or sub_v is None:
            continue
        inter_nodes = set(sub_u.nodes()).intersection(sub_v.nodes())
        inter_len = len(inter_nodes)

        # Apply optional size filtering
        size_ok = True
        if node_size is not None:
            size_ok = (min(node_size) <= inter_len <= max(node_size))
        if not size_ok:
            continue

        # Apply optional connectivity constraint
        if must_be_connected:
            if inter_len == 0:
                continue
            induced = abstract_graph.base_graph.subgraph(inter_nodes)
            try:
                cc_count = len(list(nx.connected_components(induced)))
            except Exception:
                cc_count = 0
            if cc_count != 1:
                continue

        key = frozenset(inter_nodes)
        if key in seen:
            continue
        seen.add(key)

        out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
            inter_nodes,
            meta=build_meta_from_function_context()
        )

    return out_abstract_graph

#--------------------------------------------------------------------------------   
def get_distance(graph1, graph2, basegraph):
    """Compute shortest-path distance between two subgraphs.

    Parameters
    ----------
    graph1, graph2 : networkx.Graph
        Subgraphs to compare (nodes only are used).
    basegraph : networkx.Graph
        Full graph in which distances are computed.

    Returns
    -------
    int
        Length of shortest path between any node in graph1 and any node in graph2.
    """
    return min(nx.shortest_path_length(basegraph, source=u, target=v)
               for u in graph1.nodes() for v in graph2.nodes())


def get_distance_matrix(subgraphs1, subgraphs2, basegraph, min_distance, max_distance):
    """Compute pairwise distance matrix between two sets of subgraphs."""
    distance_matrix = np.full((len(subgraphs1), len(subgraphs2)), np.nan)
    for i, subgraph_i in enumerate(subgraphs1):
        for j, subgraph_j in enumerate(subgraphs2):
            try:
                dist = get_distance(subgraph_i, subgraph_j, basegraph)
                if min_distance <= dist <= max_distance:
                    distance_matrix[i, j] = dist
            except Exception:
                pass  # Keep as NaN
    return distance_matrix


def all_distances_are_feasible(combination_idxs, distance_matrix):
    """Check if all pairwise distances within a combination are valid.

    Parameters
    ----------
    combination_idxs : iterable[int]
        Indices of subgraphs in the combination.
    distance_matrix : np.ndarray
        Pairwise distance matrix.

    Returns
    -------
    bool
        True if all pairwise distances are finite (non-NaN), else False.
    """
    pairs = combinations(combination_idxs, 2)
    for i, j in pairs:
        distance = distance_matrix[i, j]
        if np.isnan(distance):
            return False
    return True


def combination_decomposition_function(subgraphs, graph,
                                       number_of_elements=(2, 2),
                                       distance=(0, 1)):
    """Combine subgraphs into larger components based on distance constraints.

    Parameters
    ----------
    subgraphs : list[networkx.Graph]
        Input subgraphs to combine.
    graph : networkx.Graph
        Full graph used for distance computation.
    number_of_elements : tuple(int, int), default (2,2)
        Min and max number of subgraphs to combine.
    distance : tuple(int, int), default (0,1)
        Acceptable range for pairwise distances between combined subgraphs.

    Returns
    -------
    list[set]
        List of combined node sets formed from feasible subgraph combinations.
    """
    # NOTE: get_distance_matrix expects (min_distance, max_distance). Passing them
    # in the wrong order would filter out all valid pairs. Ensure correct ordering.
    distance_matrix = get_distance_matrix(
        subgraphs,
        subgraphs,
        graph,
        min(distance),
        max(distance)
    )
    components = []
    component_combinations = [list(subgraph.nodes()) for subgraph in subgraphs]
    for order in range(min(number_of_elements), max(number_of_elements) + 1):
        combination_idxs_list = combinations(range(len(component_combinations)), order)
        for combination_idxs in combination_idxs_list:
            if distance_matrix is not None and not all_distances_are_feasible(combination_idxs, distance_matrix):
                continue
            component_combination = [component_combinations[idx] for idx in combination_idxs]
            component = set(node for nodes in component_combination for node in nodes)
            components.append(component)
    return components

@curry
def combination(
    abstract_graph: 'AbstractGraph',
    number_of_elements=(2,2),
    distance=(0,1)
    ) -> 'AbstractGraph':
    """Emit interpretation nodes formed by combining multiple mapped subgraphs subject to distance constraints.
    Summary
        For each feasible combination of mapped subgraphs, create a new interpretation node whose
        mapped subgraph is the union of their node sets. A combination is feasible if:
        - The number of subgraphs lies within `number_of_elements`.
        - All pairwise distances between them (measured in base_graph) fall
          within the specified `distance` range.

    Semantics
        - Input AG state: Consumes the current set of interpretation-node mapped subgraphs.
        - Output AG state: New AbstractGraph with interpretation nodes representing unions
          of feasible subgraph combinations.
        - Determinism: Deterministic given input graph and parameters.

    Parameters
        number_of_elements : tuple(int,int), default (2,2)
            Minimum and maximum number of subgraphs to combine.
        distance : tuple(int,int), default (0,1)
            Inclusive range of allowed shortest-path distances between any two
            subgraphs in the combination.

    Algorithm
        - Build distance matrix for all pairs of subgraphs.
        - Enumerate all subgraph combinations of size within bounds.
        - Keep only those with feasible distances.
        - Form union of node sets for each valid combination.
        - Emit one interpretation node per union.

    Complexity
        - Distance matrix: O(k² * |V| + |E|), where k = number of subgraphs.
        - Combinations: exponential in k, practical only for small subgraph sets.

    Metadata
        - Each output interpretation node stores `mapped_subgraph` (unioned node set) and `meta`
          with source_function='combination' and params.

    Interactions
        - Generalises cycle()+complement() patterns by considering multiple subgraphs together.
        - Can express multi-part motifs: e.g., “two cycles within distance 3”.

    Examples
        # Combine cycles (from AG1) with degree-1 nodes (from AG2) within 2 hops
        qg1 = forward_compose(connected_component(), cycle())(Q)
        qg2 = forward_compose(node(), degree(value=1))(Q)
        workflow = forward_compose(
            lambda _: qg1,  # assuming wrappers that return fixed AGs
            lambda _: qg2,
        )
        # Combine them:
        out = binary_combination(qg1, qg2, distance=(0,2))

    Domain Analogies
        - Social networks: groups formed by nearby communities.
        - Computer networks: subnetworks within bounded latency.
        - Chemistry: functional groups within certain bond distance.

    Failure Modes & Diagnostics
        - Infeasible parameters may yield zero combinations.
        - Large number_of_elements can explode runtime.
    """
    number_of_elements = value_to_2tuple(number_of_elements)
    distance = value_to_2tuple(distance)
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    components = combination_decomposition_function(
        abstract_graph.get_interpretation_nodes_mapped_subgraphs(),
        abstract_graph.base_graph,
        number_of_elements=number_of_elements, 
        distance=distance
    )
    for component in components:
        out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
            component,
            meta=build_meta_from_function_context()
        )

    return out_abstract_graph

#--------------------------------------------------------------------------------
def union_of_shortest_paths_decomposition_function(
    subgraph: nx.Graph,
    *,
    min_len: int,
    max_len: int,
) -> List[List[Any]]:
    """Return node sets covering unions of shortest paths for qualifying node pairs."""
    nodes = list(subgraph.nodes())
    if len(nodes) < 2:
        return []

    distance_lookup = {
        node: dict(nx.single_source_shortest_path_length(subgraph, node, cutoff=max_len))
        for node in nodes
    }

    components: List[List[Any]] = []
    for u, v in combinations(nodes, 2):
        dist_uv = distance_lookup[u].get(v)
        if dist_uv is None or dist_uv < min_len or dist_uv > max_len:
            continue

        component_nodes = [
            w
            for w, dist_u_w in distance_lookup[u].items()
            if dist_u_w <= dist_uv
            and w in distance_lookup[v]
            and dist_u_w + distance_lookup[v][w] == dist_uv
        ]
        if len(component_nodes) < 2:
            continue
        components.append(component_nodes)
    return components

@curry
def union_of_shortest_paths(
    abstract_graph: 'AbstractGraph',
    length=(1, 3)
) -> 'AbstractGraph':
    """Emit one interpretation node per node pair whose shortest paths fall within bounds."""
    length = value_to_2tuple(length)
    min_len = max(0, int(min(length)))
    max_len = int(max(length))

    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    if max_len < 0:
        return out_abstract_graph

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        components = union_of_shortest_paths_decomposition_function(
            subgraph,
            min_len=min_len,
            max_len=max_len,
        )
        for component_nodes in components:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                component_nodes,
                meta=build_meta_from_function_context(),
            )

    return out_abstract_graph


#====================================================================================================
# EDGE OPERATORS
#====================================================================================================
@curry
def intersection_edges(
    abstract_graph: 'AbstractGraph',
    size_threshold=1,
    accept_connection_by_edge=False
) -> 'AbstractGraph':
    """Add edges between interpretation nodes whose mapped subgraphs overlap or are adjacent.
    Summary
        For each pair of interpretation nodes, add an edge in the interpretation graph if:
        - Their associated subgraphs share at least `size_threshold` nodes, OR
        - (optional) any node in one subgraph is directly connected by an edge
          in the base graph to a node in the other subgraph
          (`accept_connection_by_edge=True`).

    Semantics
        - Input AG state: Reads interpretation-node mapped subgraphs and base graph.
        - Output AG state: Returns a new AbstractGraph with extra edges
          added between interpretation nodes satisfying the criteria.
        - Determinism: Deterministic given the input graph and parameters.

    Parameters
        size_threshold : int, default 1
            Minimum number of shared nodes between subgraphs required to add an edge.
        accept_connection_by_edge : bool, default False
            If True, also connect interpretation nodes if any pair of their base-graph nodes
            share an edge in the base graph.

    Algorithm
        - Iterate over all ordered pairs of interpretation nodes (u,v).
        - Extract node sets from each mapped subgraph.
        - If |intersection| ≥ size_threshold, mark as connected.
        - If `accept_connection_by_edge=True`, also scan base-graph edges
          between the node sets.
        - Add edge (u,v) if condition holds.

    Complexity
        - Time: O(M² * d) where M = number of interpretation nodes and d is average subgraph size.
        - Memory: O(1) beyond input graphs and new interpretation-graph edges.

    Metadata
        - Output interpretation-graph edges are unlabeled unless edge_function later annotates them.

    Interactions
        - Complements decomposition operators: can create higher-level
          adjacency among substructures (e.g., overlapping cycles).
        - Useful for building “graph of motifs” where motifs are linked
          if they overlap or touch.

    Examples
        # Connect overlapping neighborhoods
        workflow = forward_compose(
            neighborhood(radius=(1,2)),
            intersection_edges(size_threshold=2)
        )

        # Connect cycles if they are adjacent in the molecule
        workflow = forward_compose(
            cycle(),
            intersection_edges(size_threshold=0, accept_connection_by_edge=True)
        )

    Domain Analogies
        - Social networks: overlap between friend groups.
        - Computer networks: subnetworks sharing routers.
        - Chemistry: functional groups sharing atoms or bonds.

    Failure Modes & Diagnostics
        - Large interpretation graphs (many nodes) may make pairwise checks costly.
        - If no overlaps and `accept_connection_by_edge=False`, output graph
          may be entirely disconnected.
    """
    out_abstract_graph = AbstractGraph(abstract_graph=abstract_graph)
    
    # Determine the graph to use for edge queries.
    if isinstance(abstract_graph.base_graph, AbstractGraph):
        pre_img = abstract_graph.base_graph.interpretation_graph
    else:
        pre_img = abstract_graph.base_graph

    for u in abstract_graph.interpretation_graph.nodes():
        subgraph_u = get_mapped_subgraph(abstract_graph.interpretation_graph.nodes[u])
        if subgraph_u is None:
            continue
        nodes_u = list(subgraph_u.nodes())
        set_u = set(nodes_u)
        for v in abstract_graph.interpretation_graph.nodes():
            if u != v:
                subgraph_v = get_mapped_subgraph(abstract_graph.interpretation_graph.nodes[v])
                if subgraph_v is None:
                    continue
                nodes_v = list(subgraph_v.nodes())
                set_v = set(nodes_v)

                shared_count = len(set_u.intersection(set_v))

                # Flag to decide whether to add an edge from u to v.
                add_edge = shared_count >= size_threshold
                connected_by_preimage_edge = False

                if accept_connection_by_edge and not add_edge:
                    # Check for any pre_image edge between nodes_u and nodes_v.
                    for node_u in nodes_u:
                        for node_v in nodes_v:
                            if pre_img.has_edge(node_u, node_v):
                                add_edge = True
                                connected_by_preimage_edge = True
                                break
                        if add_edge:
                            break

                if add_edge:
                    out_abstract_graph.interpretation_graph.add_edge(
                        u,
                        v,
                        shared_preimage_nodes=shared_count,
                        connected_by_preimage_edge=connected_by_preimage_edge,
                    )
    
    return out_abstract_graph

#--------------------------------------------------------------------------------    
# FILTER OPERATORS
#--------------------------------------------------------------------------------    
@curry
def filter_by_number_of_connected_components(
    abstract_graph: 'AbstractGraph',
    number_of_components=(1,1)
    ) -> 'AbstractGraph':
    """Filter subgraphs by their number of connected components.
    Summary
        Retain only those subgraphs whose number of connected components
        falls within the specified interval.

    Semantics
        - Input AG state: Reads mapped subgraphs of current interpretation nodes.
        - Output AG state: Returns a new AbstractGraph containing only
          interpretation nodes whose mapped subgraphs satisfy the component-count filter.
        - Determinism: Deterministic given input and parameter range.

    Parameters
        number_of_components : tuple(int,int), default (1,1)
            Inclusive range [min,max] for the number of connected components
            allowed. E.g. (1,1) keeps only connected subgraphs.

    Algorithm
        - For each interpretation-node mapped subgraph:
            * Compute connected components with `nx.connected_components`.
            * Count how many components exist.
            * Keep the subgraph if count ∈ [min,max].
        - Discard otherwise.

    Complexity
        - Time: O(|V|+|E|) per subgraph (connected-components computation).
        - Memory: proportional to node count of the kept subgraphs.

    Metadata
        - Each surviving interpretation node keeps original meta with added provenance
          (source_function='filter_by_number_of_connected_components').

    Interactions
        - Often used to enforce connectedness after decomposition
          (e.g., keeping only connected cliques).
        - Can prune trivial decompositions with too many disconnected pieces.

    Examples
        # Keep only connected neighborhoods
        workflow = forward_compose(
            neighborhood(radius=(1,2)),
            filter_by_number_of_connected_components((1,1))
        )

        # Allow up to 3 components
        workflow = forward_compose(
            complement(),
            filter_by_number_of_connected_components((1,3))
        )

    Domain Analogies
        - Social networks: require groups to be internally connected.
        - Computer networks: retain subnets with limited fragmentation.
        - Chemistry: filter fragments to ensure they form a connected molecule.

    Failure Modes & Diagnostics
        - Subgraphs with zero nodes yield 0 components and will be discarded
          unless 0 lies in the range.
        - Overly narrow ranges may filter out all subgraphs.
    """
    number_of_components = value_to_2tuple(number_of_components)
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        cc = list(nx.connected_components(subgraph))
        if min(number_of_components) <= len(cc) <= max(number_of_components):
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(subgraph.nodes())
    
    return out_abstract_graph

#--------------------------------------------------------------------------------    
@curry
def filter_by_number_of_nodes(
    abstract_graph: 'AbstractGraph',
    number_of_nodes=(1,10)
    ) -> 'AbstractGraph':
    """Filter subgraphs by their node count.
    Summary
        Retain only those subgraphs whose number of nodes lies within
        the specified inclusive range.

    Semantics
        - Input AG state: Reads mapped subgraphs of current interpretation nodes.
        - Output AG state: Returns a new AbstractGraph containing only
          interpretation nodes whose mapped subgraphs satisfy the node-count constraint.
        - Determinism: Deterministic given input and parameter range.

    Parameters
        number_of_nodes : tuple(int,int), default (1,10)
            Inclusive range [min,max] for the number of nodes allowed
            in each subgraph.

    Algorithm
        - For each interpretation-node mapped subgraph:
            * Compute its node count.
            * Keep the subgraph if count ∈ [min,max].
        - Discard otherwise.

    Complexity
        - Time: O(1) per subgraph (node count lookup).
        - Memory: proportional to number of retained subgraphs.

    Metadata
        - Each surviving interpretation node keeps its original meta,
          with provenance marking the filter application.

    Interactions
        - Often paired with decomposition operators to constrain
          the granularity of extracted substructures (e.g., paths,
          cliques, neighborhoods).
        - Helps control combinatorial explosion by filtering overly
          large or trivial subgraphs.

    Examples
        # Keep only small cliques (≤ 5 nodes)
        workflow = forward_compose(
            clique(),
            filter_by_number_of_nodes((1,5))
        )

        # Keep only subgraphs of medium size (10–20 nodes)
        workflow = forward_compose(
            connected_component(),
            filter_by_number_of_nodes((10,20))
        )

    Domain Analogies
        - Social networks: keep groups within a desired size band.
        - Computer networks: select subnetworks of bounded size.
        - Chemistry: retain fragments with an atom count in a range.

    Failure Modes & Diagnostics
        - Empty subgraphs (0 nodes) are discarded unless 0 is in range.
        - Narrow ranges may filter out all subgraphs.
    """
    number_of_nodes = value_to_2tuple(number_of_nodes)
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        if subgraph.number_of_nodes() >= min(number_of_nodes):
            if subgraph.number_of_nodes() <= max(number_of_nodes): 
                out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(subgraph.nodes())
    
    return out_abstract_graph

#--------------------------------------------------------------------------------    
@curry
def filter_by_number_of_edges(
    abstract_graph: 'AbstractGraph',
    number_of_edges=(1,10)
    ) -> 'AbstractGraph':
    """Filter subgraphs by their edge count.
    Summary
        Retain only those subgraphs whose number of edges lies within
        the specified inclusive range.

    Semantics
        - Input AG state: Reads edge sets of current interpretation-node mapped subgraphs.
        - Output AG state: Returns a new AbstractGraph containing only
          interpretation nodes whose mapped subgraphs satisfy the edge-count constraint.
        - Determinism: Deterministic given input and parameter range.

    Parameters
        number_of_edges : tuple(int,int), default (1,10)
            Inclusive range [min,max] for the number of edges allowed
            in each subgraph.

    Algorithm
        - For each interpretation-node mapped subgraph:
            * Compute its edge count.
            * Keep the subgraph if count ∈ [min,max].
        - Discard otherwise.

    Complexity
        - Time: O(1) per subgraph (edge count lookup).
        - Memory: proportional to number of retained subgraphs.

    Metadata
        - Each surviving interpretation node keeps its original meta,
          with provenance marking the filter application.

    Interactions
        - Complements filter_by_number_of_nodes by constraining edge
          density explicitly.
        - Helps discard trivial or overly dense subgraphs depending on task.

    Examples
        # Keep only sparse neighborhoods with ≤ 5 edges
        workflow = forward_compose(
            neighborhood(radius=(1,2)),
            filter_by_number_of_edges((1,5))
        )

        # Retain only large dense cliques (≥ 20 edges)
        workflow = forward_compose(
            clique(),
            filter_by_number_of_edges((20,100))
        )

    Domain Analogies
        - Social networks: filter groups by number of relationships.
        - Computer networks: retain subnetworks with bounded link counts.
        - Chemistry: restrict molecular fragments by bond count.

    Failure Modes & Diagnostics
        - Subgraphs with zero edges are discarded unless 0 is in range.
        - Narrow ranges may filter out all subgraphs.
    """
    number_of_edges = value_to_2tuple(number_of_edges)
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        if subgraph.number_of_edges() >= min(number_of_edges):
            if subgraph.number_of_edges() <= max(number_of_edges): 
                out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(subgraph.nodes())
    
    return out_abstract_graph

#--------------------------------------------------------------------------------    
@curry
def filter_by_sampling(
    abstract_graph: 'AbstractGraph',
    n_samples=1,
    seed: Optional[int] = None
    ) -> 'AbstractGraph':
    """Randomly subsample interpretation nodes.
    Summary
        Keep a random subset of interpretation nodes and their edges. Supports absolute
        counts (int) or fractions (float in (0,1)); sampling is without replacement.

    Semantics
        - Input AG state: reads current interpretation nodes/edges and mapped subgraphs.
        - Output AG state: new AbstractGraph with only sampled interpretation nodes; preserves
          base_graph and remaps interpretation-graph edges between retained nodes.
        - Special case: if there is a single interpretation node covering the full base graph,
          delegate to random_part(n_samples=...) to bootstrap subgraphs.
        - Determinism: deterministic if `seed` is provided and the special case does not trigger.

    Parameters
        n_samples : int | float, default 1
            If int ≥ 1: number of interpretation nodes to sample (capped at available count).
            If 0 < float < 1: fraction of interpretation nodes to sample (rounded with min 1).
        seed : int | None, default None
            Seed for the RNG; None leaves it non-deterministic.

    Examples
        # Keep 10 random interpretation nodes
        filter_by_sampling(n_samples=10)
        # Keep ~20% of interpretation nodes, deterministic
        filter_by_sampling(n_samples=0.2, seed=42)

    Failure Modes & Diagnostics
        - If n_samples ≤ 0, returns an empty interpretation graph.
        - Fractions round down; min 1 to avoid empty result unless n_samples ≤ 0.
        - Special-case delegation ignores `seed` (random_part is non-deterministic).
    """
    mapped_subgraphs = abstract_graph.get_interpretation_nodes_mapped_subgraphs()
    if (
        len(mapped_subgraphs) == 1
        and mapped_subgraphs[0].number_of_nodes() == abstract_graph.base_graph.number_of_nodes()
        and mapped_subgraphs[0].number_of_edges() == abstract_graph.base_graph.number_of_edges()
    ):
        return random_part(abstract_graph, n_samples=n_samples)

    rng = random.Random(seed)
    nodes = list(abstract_graph.interpretation_graph.nodes())

    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    if len(nodes) == 0:
        return out_abstract_graph

    # Resolve sample size
    k = 0
    if isinstance(n_samples, float) and 0 < n_samples < 1:
        k = max(1, int(n_samples * len(nodes)))
    else:
        try:
            k = int(n_samples)
        except Exception:
            k = 0
        k = max(0, min(k, len(nodes)))

    if k == 0:
        return out_abstract_graph

    sampled = set(rng.sample(nodes, k))
    old_to_new = {}

    for old_id in sampled:
        data = abstract_graph.interpretation_graph.nodes[old_id]
        mapped_subgraph = get_mapped_subgraph(data)
        if mapped_subgraph is None:
            continue
        meta = dict(data.get("meta", {}))
        new_id = out_abstract_graph._add_interpretation_node(mapped_subgraph=mapped_subgraph.copy(), meta=meta)
        for attr_name in ("label", "attribute"):
            if attr_name in data:
                out_abstract_graph.interpretation_graph.nodes[new_id][attr_name] = data[attr_name]
        old_to_new[old_id] = new_id

    for u, v, edata in abstract_graph.interpretation_graph.edges(data=True):
        if u in old_to_new and v in old_to_new:
            out_abstract_graph.interpretation_graph.add_edge(old_to_new[u], old_to_new[v], **edata)

    return out_abstract_graph

#--------------------------------------------------------------------------------    
@curry
def filter_by_node_label(
    abstract_graph: 'AbstractGraph',
    key='label',
    must_have_one_of=None,
    cannot_have_any_in=None
    ) -> 'AbstractGraph':
    """Filter subgraphs by the labels of their constituent nodes.
    Summary
        Retain only those subgraphs that (optionally) contain at least one node
        with a label in `must_have_one_of` and contain no nodes with labels
        in `cannot_have_any_in`.

    Semantics
        - Input AG state: Reads node attributes of current interpretation-node mapped subgraphs.
        - Output AG state: Returns a new AbstractGraph containing only
          interpretation nodes that satisfy both inclusion and exclusion criteria.
        - Determinism: Deterministic given input graph and label sets.

    Parameters
        key : str, default "label"
            The attribute key to inspect on each node.
        must_have_one_of : list, default []
            If non-empty, subgraphs must contain at least one node whose
            `key` value is in this list.
        cannot_have_any_in : list, default []
            If non-empty, subgraphs are discarded if they contain any node
            whose `key` value is in this list.

    Algorithm
        - For each interpretation-node mapped subgraph:
            * Check if at least one node’s label ∈ must_have_one_of
              (if constraint provided).
            * Check that no node’s label ∈ cannot_have_any_in
              (if constraint provided).
            * Keep subgraph only if both conditions are met.
        - Discard otherwise.

    Complexity
        - Time: O(|V_sub|) per subgraph (node scan).
        - Memory: O(1) additional per subgraph.

    Metadata
        - Retained subgraphs inherit original metadata,
          with provenance of the filter operation.

    Interactions
        - Complements structural filters (by size or connectivity) with
          semantic filtering.
        - Enables domain-specific constraints (e.g., presence/absence of
          atom types in chemistry, role labels in social networks).

    Examples
        # Keep only subgraphs with at least one oxygen atom
        workflow = forward_compose(
            neighborhood(radius=(1,2)),
            filter_by_node_label(key='atom', must_have_one_of=['O'])
        )

        # Keep only user groups containing "admin" but no "banned"
        workflow = forward_compose(
            connected_component(),
            filter_by_node_label(key='role', must_have_one_of=['admin'], cannot_have_any_in=['banned'])
        )

    Domain Analogies
        - Social networks: require at least one influencer; exclude any group with bots.
        - Computer networks: keep subnets with a router, exclude those with deprecated nodes.
        - Chemistry: retain fragments with a functional atom, exclude toxic groups.

    Failure Modes & Diagnostics
        - If all constraints are empty, all subgraphs are passed through unchanged.
        - Overly strict filters may yield zero surviving subgraphs.
        - If the `key` attribute is missing on nodes, default `0` is used
          and constraints may fail unexpectedly.
    """
    must_have_one_of = [] if must_have_one_of is None else must_have_one_of
    cannot_have_any_in = [] if cannot_have_any_in is None else cannot_have_any_in
    
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        if len(must_have_one_of) > 0:
            must_conditions_are_met = False
            for u in subgraph.nodes(): 
                if subgraph.nodes[u].get(key,0) in must_have_one_of:
                    must_conditions_are_met = True
                    break
        else:
            must_conditions_are_met = True

        if len(cannot_have_any_in) > 0:
            cannot_conditions_are_met = True
            for u in subgraph.nodes(): 
                if subgraph.nodes[u].get(key,0) in cannot_have_any_in:
                    cannot_conditions_are_met = False
                    break
        else:
            cannot_conditions_are_met = True

        if must_conditions_are_met and cannot_conditions_are_met:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(subgraph.nodes())
    
    return out_abstract_graph

#--------------------------------------------------------------------------------    
@curry
def filter_by_edge_label(
    abstract_graph: 'AbstractGraph',
    key: str = 'label',
    must_have_one_of=None,
    cannot_have_any_in=None,
) -> 'AbstractGraph':
    """Filter subgraphs by the labels of their constituent edges.
    Summary
        Retain only those subgraphs that (optionally) contain at least one edge
        with a label in `must_have_one_of` and contain no edges with labels
        in `cannot_have_any_in`.

    Semantics
        - Input AG state: Reads edge attributes of current interpretation-node mapped subgraphs.
        - Output AG state: Returns a new AbstractGraph containing only
          interpretation nodes that satisfy both inclusion and exclusion criteria on edges.
        - Determinism: Deterministic given input graph and label sets.

    Parameters
        key : str, default "label"
            The attribute key to inspect on each edge.
        must_have_one_of : list, default []
            If non-empty, subgraphs must contain at least one edge whose
            `key` value is in this list.
        cannot_have_any_in : list, default []
            If non-empty, subgraphs are discarded if they contain any edge
            whose `key` value is in this list.

    Algorithm
        - For each interpretation-node mapped subgraph:
            * Check if at least one edge’s label ∈ must_have_one_of
              (if constraint provided).
            * Check that no edge’s label ∈ cannot_have_any_in
              (if constraint provided).
            * Keep subgraph only if both conditions are met.
        - Discard otherwise.

    Complexity
        - Time: O(|E_sub|) per subgraph (edge scan).
        - Memory: O(1) additional per subgraph.

    Metadata
        - Retained subgraphs inherit original metadata,
          with provenance of the filter operation.

    Interactions
        - Complements node-label filters and structural operators.
        - In chemoinformatics, enables constraints like aromatic bonds or
          double bonds when edge labels encode bond types (e.g.,
          "AROMATIC", "1", "2", "3").

    Examples
        # Keep only subgraphs containing at least one aromatic bond
        workflow = forward_compose(
            cycle(),
            filter_by_edge_label(key='label', must_have_one_of=['AROMATIC'])
        )

        # Keep only subgraphs that contain double bonds and no aromatic bonds
        workflow = forward_compose(
            connected_component(),
            filter_by_edge_label(key='label', must_have_one_of=['2'], cannot_have_any_in=['AROMATIC'])
        )

    Failure Modes & Diagnostics
        - If all constraints are empty, all subgraphs are passed through unchanged.
        - Overly strict filters may yield zero surviving subgraphs.
        - If the `key` attribute is missing on edges, default `0` is used and
          constraints may fail unexpectedly.
    """
    must_have_one_of = [] if must_have_one_of is None else must_have_one_of
    cannot_have_any_in = [] if cannot_have_any_in is None else cannot_have_any_in

    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs():
        # must-have check
        if len(must_have_one_of) > 0:
            must_ok = False
            for u, v in subgraph.edges():
                if subgraph.edges[u, v].get(key, 0) in must_have_one_of:
                    must_ok = True
                    break
        else:
            must_ok = True

        # cannot-have check
        if len(cannot_have_any_in) > 0:
            cannot_ok = True
            for u, v in subgraph.edges():
                if subgraph.edges[u, v].get(key, 0) in cannot_have_any_in:
                    cannot_ok = False
                    break
        else:
            cannot_ok = True

        if must_ok and cannot_ok:
            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(subgraph.nodes())

    return out_abstract_graph
#--------------------------------------------------------------------------------    
@curry
def select_top_by_feature_ranking(
    abstract_graph: 'AbstractGraph',
    ranked_features,
    max_num: int = 1,
) -> 'AbstractGraph':
    """Select top-K interpretation nodes based on an external feature-importance ranking.
    Summary
        Given an ordered list of feature IDs (most important first), rank all
        current interpretation nodes by their label and retain only the top `max_num`.

    Semantics
        - Input AG state: Reads interpretation-node labels (computes them on the fly
          for nodes missing a 'label' attribute using the AG's label_function).
        - Output AG state: Returns a new AbstractGraph containing only the
          selected interpretation nodes (their mapped subgraphs are copied). Operator
          settings (label/attribute/edge functions) are preserved.

    Parameters
        ranked_labels : Sequence[int] or Mapping[int, float]
            Label IDs in descending importance order, or a mapping from label
            to importance score. Labels not present receive lowest priority.
        max_num : int, default 1
            Number of top-ranked interpretation nodes to keep (globally across the interpretation graph).

    Notes
        - If labels are provided as an ordering, nodes are ranked by the index
          position (lower index = higher priority).
        - If a mapping is provided, nodes are ranked by score (higher is better).
        - Interpretation-node labels are treated as integers consistent with hashing-based
          label functions. Labels not found in the ranking are assigned worst rank.
    """
    # Build scoring function from input ranking.
    score_map = None
    is_mapping = hasattr(ranked_features, 'items')
    if is_mapping:
        score_map = dict(ranked_features)
    else:
        # Higher priority → larger score; use reverse index so index 0 gets max score.
        n = len(ranked_features)
        score_map = {lbl: (n - i) for i, lbl in enumerate(ranked_features)}

    # Gather candidates with scores.
    candidates = []  # (score, node_id, mapped_subgraph, meta)
    for node_id, data in abstract_graph.interpretation_graph.nodes(data=True):
        label = data.get('label', None)
        if label is None and getattr(abstract_graph, 'label_function', None) is not None:
            try:
                label = abstract_graph.label_function(data)
            except Exception:
                label = None
        if label is None:
            continue  # cannot rank without a label
        score = score_map.get(label, float('-inf'))
        if score == float('-inf'):
            continue  # skip labels not present in the ranking
        assoc = get_mapped_subgraph(data)
        meta = data.get('meta')
        candidates.append((score, node_id, assoc, meta))

    # Sort and select top-K.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    selected = candidates[: max(0, int(max_num)+1)]

    # Construct the output AbstractGraph with selected mapped subgraphs.
    out_abstract_graph = AbstractGraph(
        graph=abstract_graph.base_graph,
        label_function=abstract_graph.label_function,
        attribute_function=abstract_graph.attribute_function,
        edge_function=abstract_graph.edge_function,
    )

    for _, _, assoc, meta in selected:
        if assoc is not None:
            # Preserve metadata if available
            out_abstract_graph.create_interpretation_node_with_subgraph_from_subgraph(assoc.copy(), meta=meta)

    return out_abstract_graph

#====================================================================================================
# BINARY OPERATORS
#====================================================================================================
def binary_combination_decomposition_function(subgraphs1, subgraphs2, graph, distance=(0,1)):
    """Combine one subgraph from set1 with one from set2 if their pairwise distance is within bounds.

    Parameters
    ----------
    subgraphs1, subgraphs2 : list[networkx.Graph]
        Two collections of subgraphs to pairwise combine (one from each set).
    graph : networkx.Graph
        Base graph used to compute shortest-path distances between subgraphs.
    distance : tuple(int, int), default (0,1)
        Inclusive [min, max] bounds on shortest-path distance between any node
        of a subgraph from set1 and any node of a subgraph from set2.

    Returns
    -------
    list[set]
        List of combined node sets (union of nodes from each valid pair).
    """
    # NOTE: get_distance_matrix signature is (min_distance, max_distance). Ensure
    # we pass the bounds in the correct order to avoid discarding valid matches.
    distance_matrix = get_distance_matrix(
        subgraphs1,
        subgraphs2,
        graph,
        min(distance),
        max(distance)
    )
    components = []
    component_combinations1 = [list(subgraph.nodes()) for subgraph in subgraphs1]
    component_combinations2 = [list(subgraph.nodes()) for subgraph in subgraphs2]
    combination_idxs_list = product(range(len(component_combinations1)), range(len(component_combinations2)))
    for combination_idxs in combination_idxs_list:
        if distance_matrix is not None and all_distances_are_feasible(combination_idxs, distance_matrix) is False:
            continue
        nodes1_list = [node for node in component_combinations1[combination_idxs[0]]]
        nodes2_list = [node for node in component_combinations2[combination_idxs[1]]]
        component = set(nodes1_list + nodes2_list)
        components.append(component)
    return components

@curry
def binary_combination(
    first_abstract_graph: 'AbstractGraph',
    second_abstract_graph: 'AbstractGraph',
    distance=(0,1)
    ) -> 'AbstractGraph':
    """Emit interpretation nodes by pairing subgraphs from two AGs when their inter-distance is within bounds.
    Summary
        Take one mapped subgraph from the first AbstractGraph and one from the second; if the
        shortest-path distance between them (in the shared base graph) lies within `distance`,
        emit a new interpretation node whose mapped subgraph is the union of both subgraphs’ node sets.

    Semantics
        - Input AG state: Reads interpretation-node mapped subgraphs from two input AGs and uses the first AG’s
          base_graph to compute distances.
        - Output AG state: Returns a new AbstractGraph (with the first AG’s base_graph) whose
          interpretation nodes each represent a valid pairwise combination (union) of subgraphs.
        - Determinism: Deterministic given inputs and parameters.

    Parameters
        distance : int | tuple[int,int], default (0,1)
            Inclusive [min, max] bounds for allowed shortest-path distances between any node in a
            subgraph from the first set and any node in a subgraph from the second set. A scalar `d`
            is treated as (d, d).

    Algorithm
        - Normalize `distance` via value_to_2tuple().
        - Compute pairwise distances via `get_distance_matrix(subgraphs1, subgraphs2, basegraph, ...)`.
        - Enumerate all pairs (i, j); keep only those whose distance is finite and within bounds.
        - For each valid pair, form the union of nodes and create a new interpretation node with that induced subgraph.
        - Attach provenance via `build_meta_from_function_context()`.

    Complexity
        - Distance matrix computation: O(k1 * k2 * D) where k1, k2 are counts of subgraphs and D is
          the cost of shortest paths between node pairs (depends on basegraph size/structure).
        - Pair enumeration: O(k1 * k2).
        - Practical usage suggests applying upstream filters to keep k1, k2 small.

    Side Effects & Metadata
        - Each emitted interpretation node stores:
            * `mapped_subgraph` : induced subgraph on the unioned node set.
            * 'meta'        : {'source_function': 'binary_combination', 'params': {...}}.
        - Labels/attributes are not computed here; call `update()` to populate them.

    Interactions
        - Expresses cross-family motifs, e.g., “a cycle near a high-betweenness node” by pairing
          outputs of `cycle()` (first AG) and `betweenness_centrality()` (second AG).
        - Can be followed by size/connectivity filters to control combinatorial growth.

    Constraints & Invariants
        - Assumes both AGs refer to the same underlying base-graph node ID space.
        - If either AG has zero mapped subgraphs, the output will be empty.
        - If `distance` is too strict, no pairs may be produced.

    Examples
        # Pair cycles (from AG1) with degree-1 nodes (from AG2) within 2 hops
        qg1 = forward_compose(connected_component(), cycle())(Q)
        qg2 = forward_compose(node(), degree(value=1))(Q)
        workflow = forward_compose(
            lambda _: qg1,  # assuming wrappers that return fixed AGs
            lambda _: qg2,
        )
        # Combine them:
        out = binary_combination(qg1, qg2, distance=(0,2))

    Domain Analogies
        - Social networks: pair a community with a nearby influencer set.
        - Computer networks: pair a subnet with a nearby gateway/router group.
        - Chemistry: pair a ring system with a nearby heteroatom set.

    Failure Modes & Diagnostics
        - Large k1/k2 or loose distance bounds can cause quadratic blow-up in pairs.
        - Tight bounds (e.g., distance=(0,0)) may yield no combinations if subgraphs do not touch.
        - Ensure both AGs share the same base graph; otherwise distances are undefined.
    """
    distance = value_to_2tuple(distance)
    out_abstract_graph = AbstractGraph(
        graph=first_abstract_graph.base_graph,
        label_function=first_abstract_graph.label_function,
        attribute_function=first_abstract_graph.attribute_function,
        edge_function=first_abstract_graph.edge_function,
    )

    components = binary_combination_decomposition_function(
        first_abstract_graph.get_interpretation_nodes_mapped_subgraphs(),
        second_abstract_graph.get_interpretation_nodes_mapped_subgraphs(),
        first_abstract_graph.base_graph,
        distance=distance
    )
    for component in components:
        out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
            component,
            meta=build_meta_from_function_context()
        )

    return out_abstract_graph

#--------------------------------------------------------------------------------    
@curry
def binary_intersection(
    first_abstract_graph: 'AbstractGraph',
    second_abstract_graph: 'AbstractGraph',
    node_size=None,
    must_be_connected: bool = True
) -> 'AbstractGraph':
    """Emit interpretation nodes for intersections between subgraphs from two AbstractGraphs.
    Summary
        For each pair consisting of one associated subgraph from the first AbstractGraph
        and one from the second, compute the intersection of their node sets. If the
        intersection satisfies the optional size bounds `node_size` and the connectivity
        constraint `must_be_connected`, emit a new interpretation node whose mapped subgraph is the
        induced subgraph on the intersecting nodes.

    Semantics
        - Input AG state: Reads interpretation-node mapped subgraphs from both input AGs and uses the
          first AG’s base_graph to induce intersection subgraphs.
        - Output AG state: Returns a new AbstractGraph (with the first AG’s base_graph)
          containing one interpretation node per qualifying intersection.
        - Determinism: Deterministic given inputs and parameters.

    Parameters
        node_size : None | int | tuple[int,int], default None
            If None, no size filter is applied. If int k, treated as (k,k). If a tuple (min,max)
            is provided, require min ≤ |I| ≤ max for the intersection I.
        must_be_connected : bool, default True
            If True, accept only intersections whose induced subgraph forms exactly one
            connected component (empty intersections are rejected).

    Algorithm
        - Optionally normalise node_size via value_to_2tuple when not None.
        - For each subgraph a in first AG and each subgraph b in second AG:
            * Compute I = nodes(a) ∩ nodes(b).
            * Apply size filter (if any).
            * If must_be_connected, require induced subgraph on I to have exactly 1 component.
            * If accepted, create an interpretation node for I.

    Complexity
        - Time: O(k1 · k2 · d) where k1, k2 are counts of subgraphs in the two AGs,
          and d is average subgraph size.
        - Memory: proportional to number and size of emitted intersections.

    Metadata
        - Each output interpretation node stores `mapped_subgraph` and `meta` with source_function='binary_intersection'.

    Interactions
        - Complements `binary_combination` by intersecting instead of unioning pairs.
        - Often followed by `filter_by_number_of_nodes` to bound sizes explicitly.

    Examples
        # Intersect cycles (from AG1) with neighborhoods (from AG2), require connected intersections
        qg1 = forward_compose(connected_component(), cycle())(Q)
        qg2 = forward_compose(node(), neighborhood(radius=2))(Q)
        out = binary_intersection(qg1, qg2, node_size=None, must_be_connected=True)
    """
    # Normalise size bounds if provided
    if node_size is not None:
        node_size = value_to_2tuple(node_size)

    out_abstract_graph = AbstractGraph(
        graph=first_abstract_graph.base_graph,
        label_function=first_abstract_graph.label_function,
        attribute_function=first_abstract_graph.attribute_function,
        edge_function=first_abstract_graph.edge_function,
    )

    subgraphs1 = first_abstract_graph.get_interpretation_nodes_mapped_subgraphs()
    subgraphs2 = second_abstract_graph.get_interpretation_nodes_mapped_subgraphs()

    # Deduplicate identical intersections across different pairs
    seen = set()  # set[frozenset]

    for sg1 in subgraphs1:
        nodes1 = set(sg1.nodes())
        for sg2 in subgraphs2:
            nodes2 = set(sg2.nodes())
            inter_nodes = nodes1.intersection(nodes2)
            inter_len = len(inter_nodes)

            # Size filter
            size_ok = True
            if node_size is not None:
                size_ok = (min(node_size) <= inter_len <= max(node_size))
            if not size_ok:
                continue

            # Connectivity filter
            if must_be_connected:
                if inter_len == 0:
                    continue
                induced = first_abstract_graph.base_graph.subgraph(inter_nodes)
                try:
                    cc_count = len(list(nx.connected_components(induced)))
                except Exception:
                    cc_count = 0
                if cc_count != 1:
                    continue

            key = frozenset(inter_nodes)
            if key in seen:
                continue
            seen.add(key)

            out_abstract_graph.create_interpretation_node_with_subgraph_from_nodes(
                inter_nodes,
                meta=build_meta_from_function_context()
            )

    return out_abstract_graph

#====================================================================================================
# BASE GRAPH OPERATORS
#====================================================================================================
@curry
def unlabel(
    abstract_graph: 'AbstractGraph', 
    label='-'
    ) -> 'AbstractGraph':
    """Replace all node and edge labels in the base graph with a constant.
    Summary
        Before overwriting, copy the current 'label' into 'original_label' on
        both nodes and edges (if not already present). Then reset the 'label'
        attribute of every node and edge to the same constant value.

    Semantics
        - Input AG state: Reads base_graph of the given AbstractGraph.
        - Output AG state: Returns a new AbstractGraph with identical structure
          but with all 'label' fields overwritten to `label` and an
          'original_label' field preserving the previous value when absent.
        - Determinism: Deterministic given `label`.

    Args:
        label: Constant value to assign to every node and edge 'label'.

    Algorithm
        - Copy the input AbstractGraph.
        - For each node and edge: if 'original_label' is not present, set it to
          the current 'label' (or None if missing).
        - Overwrite 'label' with the provided constant.

    Complexity
        - Time: O(|V| + |E|).
        - Memory: O(1) extra beyond the graph copy.

    Interactions
        - Useful to erase semantic bias before structural decomposition.
        - Preserving 'original_label' allows downstream operators to recover
          prior semantics if needed.

    Examples
        # Strip all labels to a neutral "-" while preserving originals
        qg2 = unlabel(qg1, label='-')

    Failure Modes
        - If base_graph is empty, nothing is modified.
        - Only affects 'label' and introduces/preserves 'original_label'.
    """
    out_abstract_graph = AbstractGraph(abstract_graph=abstract_graph)
    g = out_abstract_graph.base_graph
    # Nodes: preserve original label if not already present, then overwrite label
    for n, data in g.nodes(data=True):
        if 'original_label' not in data:
            data['original_label'] = data.get('label')
        data['label'] = label
    # Edges: same preservation and overwrite
    for u, v, data in g.edges(data=True):
        if 'original_label' not in data:
            data['original_label'] = data.get('label')
        data['label'] = label
    return out_abstract_graph

@curry
def prepend_label(
    abstract_graph: 'AbstractGraph', 
    label: Union[str, int] = '-'
) -> 'AbstractGraph':
    """Prepend a string prefix to every node and edge label in the base graph.
    Summary
        Before modification, copy the current 'label' into 'original_label' on
        both nodes and edges (if not already present). Then modify the 'label'
        attribute of each node and edge by concatenating the provided prefix in
        front of the existing label value.

    Semantics
        - Input AG state: Reads base_graph of the given AbstractGraph.
        - Output AG state: Returns a new AbstractGraph with labels modified by
          prepending the chosen prefix.
        - Determinism: Deterministic given `label`.

    Parameters
        label : str | int, default '-'
            Prefix to prepend. Converted to string if not already.

    Algorithm
        - Copy the input AbstractGraph.
        - For each node and edge:
            * Read existing 'label' (default to empty string).
            * Set new label = f"{prefix}{old_label}".

    Complexity
        - Time: O(|V| + |E|).
        - Memory: O(1) extra beyond the graph copy.

    Interactions
        - Can namespace multiple graphs before composition.
        - Useful to distinguish contributions from different workflows.

    Examples
        # Prefix all labels with "chem_"
        qg2 = prepend_label(qg1, label="chem_")

    Domain Analogies
        - Chemistry: prefix functional groups with context tags.
        - Social networks: add organizational prefixes to role labels.

    Failure Modes
        - If 'label' key is missing, treated as empty string.
        - Repeated application prepends multiple prefixes (idempotence not guaranteed).
    """
    label_str: str = str(label)
    out_abstract_graph = AbstractGraph(abstract_graph=abstract_graph)

    # Prepend the label to each node's existing label, preserving original
    for node, data in out_abstract_graph.base_graph.nodes(data=True):
        current_label = data.get('label', '')
        if 'original_label' not in data:
            data['original_label'] = current_label
        new_label = f"{label_str}{str(current_label)}"
        data['label'] = new_label

    # Prepend the label to each edge's existing label, preserving original
    for u, v, data in out_abstract_graph.base_graph.edges(data=True):
        current_label = data.get('label', '')
        if 'original_label' not in data:
            data['original_label'] = current_label
        new_label = f"{label_str}{str(current_label)}"
        data['label'] = new_label
        
    return out_abstract_graph

@curry
def restore_label(
    abstract_graph: 'AbstractGraph',
    *,
    fallback: Optional[str] = None,
    drop_original: bool = False,
) -> 'AbstractGraph':
    """Restore labels from 'original_label' for nodes and edges.
    Summary
        For every node and edge in the base graph, if an 'original_label'
        attribute exists, copy it back into 'label'. Optionally drop the
        'original_label' attribute after restoration.

    Semantics
        - Input AG state: Reads and writes base_graph attributes.
        - Output AG state: Returns a new AbstractGraph with labels restored
          where possible.
        - Determinism: Deterministic given inputs.

    Args:
        fallback: If provided and 'original_label' is missing, set 'label' to
            this value; if None, leave the current label unchanged.
        drop_original: If True, remove 'original_label' after restoring.

    Complexity
        - Time: O(|V| + |E|).
        - Memory: O(1) extra beyond the graph copy.
    """
    out_abstract_graph = AbstractGraph(abstract_graph=abstract_graph)
    g = out_abstract_graph.base_graph
    # Nodes
    for n, data in g.nodes(data=True):
        if 'original_label' in data:
            data['label'] = data['original_label']
            if drop_original:
                try:
                    del data['original_label']
                except Exception:
                    pass
        elif fallback is not None:
            data['label'] = fallback
    # Edges
    for u, v, data in g.edges(data=True):
        if 'original_label' in data:
            data['label'] = data['original_label']
            if drop_original:
                try:
                    del data['original_label']
                except Exception:
                    pass
        elif fallback is not None:
            data['label'] = fallback
    return out_abstract_graph

#====================================================================================================
# INTERPRETATION GRAPH META OPERATORS
#====================================================================================================
def name(text: str = "default"):
    """
    Build an operator that attaches a user-defined name to each interpretation node.

    Args:
        text: Name string to store in meta["user_name"] for each interpretation node.

    Returns:
        Callable[[AbstractGraph], AbstractGraph]: Operator tagging interpretation-node metadata.
    """
    def composed(abstract_graph: 'AbstractGraph') -> 'AbstractGraph':
        out_abstract_graph = AbstractGraph(abstract_graph=abstract_graph)
        for _, data in out_abstract_graph.interpretation_graph.nodes(data=True):
            meta = data.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            meta["user_name"] = text
            data["meta"] = meta
        return out_abstract_graph

    composed.__name__ = "name"
    composed.operator_type = "name"
    composed.params = {"text": text}
    return composed

#====================================================================================================
# SCALAR OPERATORS
#====================================================================================================
@curry
def number_of_image_graph_nodes(
    abstract_graph: 'AbstractGraph',
    param=None
) -> int:
    """Deprecated alias for counting interpretation nodes.
    Summary
        Return the total number of nodes in the interpretation graph of the AbstractGraph.

    Parameters
        abstract_graph : AbstractGraph
            The input graph.

    Returns
        int : number of interpretation nodes.
    """
    return abstract_graph.interpretation_graph.number_of_nodes()


@curry
def number_of_interpretation_graph_nodes(
    abstract_graph: 'AbstractGraph',
    param=None
) -> int:
    """Count the number of interpretation nodes."""
    return abstract_graph.interpretation_graph.number_of_nodes()


def number_of_image_graph_edges(
    abstract_graph: 'AbstractGraph',
    param=None
) -> int:
    """Deprecated alias for counting interpretation edges.
    Summary
        Return the total number of edges in the interpretation graph of the AbstractGraph.

    Parameters
        abstract_graph : AbstractGraph
            The input graph.

    Returns
        int : number of interpretation edges.
    """
    return abstract_graph.interpretation_graph.number_of_edges()


def number_of_interpretation_graph_edges(
    abstract_graph: 'AbstractGraph',
    param=None
) -> int:
    """Count the number of interpretation edges."""
    return abstract_graph.interpretation_graph.number_of_edges()


def quantile_number_of_subgraph_nodes(
    abstract_graph: 'AbstractGraph',
    q=0.5
) -> int:
    """Quantile of subgraph sizes by nodes.
    Summary
        Compute the q-quantile of the distribution of node counts
        across all image-node subgraphs.

    Parameters
        q : float, default 0.5
            Quantile in [0,1].

    Returns
        float : q-th quantile of subgraph node counts.
    """
    return np.quantile(
        [subgraph.number_of_nodes() for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs()],
        q,
    )


def quantile_number_of_subgraph_edges(
    abstract_graph: 'AbstractGraph',
    q=0.5
) -> int:
    """Quantile of subgraph sizes by edges.
    Summary
        Compute the q-quantile of the distribution of edge counts
        across all image-node subgraphs.

    Parameters
        q : float, default 0.5
            Quantile in [0,1].

    Returns
        float : q-th quantile of subgraph edge counts.
    """
    return np.quantile(
        [subgraph.number_of_edges() for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs()],
        q,
    )


def max_number_of_subgraph_nodes(
    abstract_graph: 'AbstractGraph',
    param=None
) -> int:
    """Maximum subgraph size by nodes.
    Summary
        Return the maximum number of nodes among all image-node subgraphs.

    Returns
        int : maximum node count across subgraphs.
    """
    return max([subgraph.number_of_nodes() for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs()])


def min_number_of_subgraph_nodes(
    abstract_graph: 'AbstractGraph',
    param=None
) -> int:
    """Minimum subgraph size by nodes.
    Summary
        Return the minimum number of nodes among all image-node subgraphs.

    Returns
        int : minimum node count across subgraphs.
    """
    return min([subgraph.number_of_nodes() for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs()])


def max_number_of_subgraph_edges(
    abstract_graph: 'AbstractGraph',
    param=None
) -> int:
    """Maximum subgraph size by edges.
    Summary
        Return the maximum number of edges among all image-node subgraphs.

    Returns
        int : maximum edge count across subgraphs.
    """
    return max([subgraph.number_of_edges() for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs()])


def min_number_of_subgraph_edges(
    abstract_graph: 'AbstractGraph',
    param=None
) -> int:
    """Minimum subgraph size by edges.
    Summary
        Return the minimum number of edges among all image-node subgraphs.

    Returns
        int : minimum edge count across subgraphs.
    """
    return min([subgraph.number_of_edges() for subgraph in abstract_graph.get_interpretation_nodes_mapped_subgraphs()])

#====================================================================================================
# XML REGISTRATION
#====================================================================================================
# Explicitly register all AbstractGraph operators with the XML serializer/deserializer.
# This avoids relying on implicit discovery and ensures stable round-trips by name.
try:
    from abstractgraph.xml import register_operator

    # List only operators that operate on and/or return AbstractGraph instances or pipelines.
    # Scalar reducers (for example, number_of_interpretation_graph_nodes) are intentionally excluded from XML pipelines.
    _AG_OPERATORS = [
        # Higher-order composition
        add,
        compose,
        forward_compose,
        compose_product,

        # Conditionals and loops
        if_then_else,
        if_then_elif_else,
        for_loop,
        while_loop,

        # Unary / decomposition operators
        identity,
        random_part,
        node,
        edge,
        connected_component,
        degree,
        split,
        neighborhood,
        cycle,
        tree,
        path,
        spine,
        graphlet,
        clique,

        # Unary graph transforms
        complement,
        local_complement,
        edge_complement,
        local_edge_complement,
        betweenness_centrality,
        betweenness_centrality_split,
        betweenness_centrality_hop_split,
        low_cut_partition,
        merge,
        deduplicate,
        remove_redundant_associations,
        unique,  # legacy alias
        combination,
        union_of_shortest_paths,
        intersection,
        intersection_edges,

        # Filters
        filter_by_number_of_connected_components,
        filter_by_number_of_nodes,
        filter_by_number_of_edges,
        filter_by_node_label,
        filter_by_edge_label,
        filter_by_sampling,

        # Binary composition & relabelling
        binary_combination,
        binary_intersection,
        unlabel,
        prepend_label,
        restore_label,
        name,
    ]

    for _op in _AG_OPERATORS:
        try:
            register_operator(getattr(_op, "__name__", None))(_op)
        except Exception:
            # Best-effort registration; ignore individual failures to avoid import-time crashes.
            pass
    # Backward-compatibility alias: support legacy XML referring to 'pairwise_intersection'
    try:
        register_operator("pairwise_intersection")(intersection)
    except Exception:
        pass
except Exception:
    # If XML module is unavailable, skip registration without failing import.
    pass
