from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import abstractgraph.operators as ops
from abstractgraph.graphs import (
    AbstractGraph,
    get_interpretation_label_to_mapped_subgraphs,
    get_mapped_subgraph,
    graph_to_abstract_graph,
    is_simple_graph,
)
from abstractgraph.hashing import hash_graph
from abstractgraph.vectorize import vectorize


def _make_graph() -> nx.Graph:
    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)
        graph.nodes[node]["attribute"] = np.array([1.0])
    return graph


def test_canonical_graph_attributes() -> None:
    ag = AbstractGraph(graph=_make_graph())
    assert set(ag.base_graph.nodes()) == {0, 1, 2}
    assert not hasattr(ag, "preimage_graph")
    assert not hasattr(ag, "image_graph")


def test_interpretation_node_helpers_store_mapped_subgraph() -> None:
    ag = AbstractGraph(graph=_make_graph())
    ag.create_default_interpretation_node()

    node_data = next(iter(ag.interpretation_graph.nodes(data=True)))[1]
    assert get_mapped_subgraph(node_data) is not None
    assert node_data["mapped_subgraph"].number_of_nodes() == ag.base_graph.number_of_nodes()
    assert "association" not in node_data
    assert not hasattr(ag, "create_default_image_node")


def test_interpretation_label_to_mapped_subgraphs_groups_by_label() -> None:
    ag = AbstractGraph(graph=_make_graph())
    ag.create_interpretation_node_with_subgraph_from_nodes([0, 1])
    ag.create_interpretation_node_with_subgraph_from_nodes([1, 2])
    ag.create_interpretation_node_with_subgraph_from_nodes([0, 1])
    ag.apply_label_function()

    label_map = get_interpretation_label_to_mapped_subgraphs(ag)
    assert sum(len(grouped) for grouped in label_map.values()) == 3

    unique_label_map = ag.get_interpretation_label_to_mapped_subgraphs(unique=True)
    unique_grouped = [subgraph for grouped in unique_label_map.values() for subgraph in grouped]
    assert len(unique_grouped) == 2
    assert all(is_simple_graph(subgraph) for subgraph in unique_grouped)


def test_vectorize_and_to_graph_work_with_canonical_names() -> None:
    ag = AbstractGraph(graph=_make_graph())
    ag.create_default_interpretation_node()
    ag = ops.node()(ag)
    ag.update()

    matrix = vectorize(ag, nbits=6, return_dense=True)
    assert matrix.shape[0] == ag.base_graph.number_of_nodes()
    assert matrix.shape[1] == 2**6
    assert np.all(matrix[:, 0] == 1)

    materialized = ag.to_graph()
    base_kind_count = sum(1 for _, data in materialized.nodes(data=True) if data.get("kind") == "base")
    interpretation_kind_count = sum(
        1 for _, data in materialized.nodes(data=True) if data.get("kind") == "interpretation"
    )
    assert base_kind_count == ag.base_graph.number_of_nodes()
    assert interpretation_kind_count == ag.interpretation_graph.number_of_nodes()


def test_connected_components_from_feature_ranking_projects_to_base_nodes() -> None:
    graph = nx.path_graph(5)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)
        graph.nodes[node]["attribute"] = np.array([1.0])

    ag = AbstractGraph(graph=graph)
    ag.create_interpretation_node_with_subgraph_from_nodes([0, 1])
    ag.create_interpretation_node_with_subgraph_from_nodes([1, 2])
    ag.create_interpretation_node_with_subgraph_from_nodes([2, 3])
    ag.create_interpretation_node_with_subgraph_from_nodes([3, 4])

    labels = [10, 20, 30, 40]
    for node_id, label in zip(ag.interpretation_graph.nodes(), labels):
        ag.interpretation_graph.nodes[node_id]["label"] = label

    out = ops.connected_components_from_feature_ranking(
        ranked_features=[40, 30, 20, 10],
        max_num_base_nodes=3,
        node_agg="sum",
    )(ag)

    mapped = out.get_interpretation_nodes_mapped_subgraphs()
    assert len(mapped) == 1
    assert set(mapped[0].nodes()) == {2, 3, 4}


def test_directed_base_graph_survives_core_pipeline_and_export() -> None:
    graph = nx.DiGraph()
    graph.add_node(0, label="a", attribute=np.array([1.0]))
    graph.add_node(1, label="b", attribute=np.array([2.0]))
    graph.add_node(2, label="c", attribute=np.array([3.0]))
    graph.add_edge(0, 1, label="x")
    graph.add_edge(1, 2, label="y")

    ag = AbstractGraph(graph=graph)
    ag.create_default_interpretation_node()
    ag = ops.connected_component()(ag)
    ag.update()

    assert ag.base_graph.is_directed()
    assert not ag.interpretation_graph.is_directed()
    assert ag.copy().base_graph.is_directed()

    materialized = ag.to_graph()
    assert materialized.is_directed()
    assert materialized.has_edge(0, 1)
    assert not materialized.has_edge(1, 0)


def test_hash_graph_distinguishes_edge_orientation() -> None:
    undirected = nx.Graph()
    undirected.add_node(0, label="a")
    undirected.add_node(1, label="b")
    undirected.add_edge(0, 1, label="x")

    directed_forward = nx.DiGraph()
    directed_forward.add_node(0, label="a")
    directed_forward.add_node(1, label="b")
    directed_forward.add_edge(0, 1, label="x")

    directed_reverse = nx.DiGraph()
    directed_reverse.add_node(0, label="a")
    directed_reverse.add_node(1, label="b")
    directed_reverse.add_edge(1, 0, label="x")

    assert hash_graph(undirected, nbits=18) != hash_graph(directed_forward, nbits=18)
    assert hash_graph(directed_forward, nbits=18) != hash_graph(directed_reverse, nbits=18)


def test_hash_graph_distinguishes_same_label_same_degree_cross_edge_structure() -> None:
    labels = {node: "A" for node in range(1, 7)}
    graph_a_edges = [
        (3, 4),
        (4, 5),
        (3, 5),
        (3, 2),
        (5, 6),
        (2, 6),
        (2, 1),
        (1, 6),
    ]
    graph_b_edges = [
        (3, 4),
        (4, 5),
        (3, 2),
        (5, 6),
        (2, 1),
        (1, 6),
        (3, 6),
        (2, 5),
    ]

    graph_a = nx.Graph()
    graph_b = nx.Graph()
    for graph, edges in ((graph_a, graph_a_edges), (graph_b, graph_b_edges)):
        graph.add_nodes_from((node, {"label": label}) for node, label in labels.items())
        graph.add_edges_from((u, v, {"label": ""}) for u, v in edges)

    assert sorted(dict(graph_a.degree()).values()) == sorted(dict(graph_b.degree()).values())
    assert hash_graph(graph_a, nbits=31) != hash_graph(graph_b, nbits=31)

    relabeled = nx.relabel_nodes(graph_a, {node: f"node-{node}" for node in graph_a.nodes()})
    assert hash_graph(graph_a, nbits=31) == hash_graph(relabeled, nbits=31)


def test_directed_graph_to_abstract_graph_uses_weak_connectivity() -> None:
    graph = nx.DiGraph()
    graph.add_node(0, label="a", attribute=np.array([1.0]))
    graph.add_node(1, label="b", attribute=np.array([1.0]))
    graph.add_node(2, label="c", attribute=np.array([1.0]))
    graph.add_edge(0, 1, label="x")
    graph.add_edge(2, 1, label="y")

    ag = graph_to_abstract_graph(graph, decomposition_function=ops.connected_component(), nbits=6)

    mapped = ag.get_interpretation_nodes_mapped_subgraphs()
    assert len(mapped) == 1
    assert mapped[0].is_directed()
    assert set(mapped[0].nodes()) == {0, 1, 2}


def test_local_edge_complement_preserves_directed_orientation() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            (0, {"label": "a", "attribute": np.array([1.0])}),
            (1, {"label": "b", "attribute": np.array([1.0])}),
            (2, {"label": "c", "attribute": np.array([1.0])}),
        ]
    )
    graph.add_edge(0, 1, label="x")
    graph.add_edge(1, 2, label="y")
    graph.add_edge(2, 0, label="z")

    ag = AbstractGraph(graph=graph)
    ag.create_interpretation_node_with_subgraph_from_edges([(0, 1), (1, 2)])
    ag.interpretation_graph.nodes[0]["meta"]["parent_mapped_subgraph"] = graph.copy()

    out = ops.local_edge_complement()(ag)
    mapped = out.get_interpretation_nodes_mapped_subgraphs()

    assert len(mapped) == 1
    assert mapped[0].is_directed()
    assert set(mapped[0].edges()) == {(2, 0)}


def test_operator_registry_declares_directed_support() -> None:
    registry = ops.get_operator_registry()
    assert registry
    assert all(getattr(operator, "directed_support", None) for operator in registry)
    assert ops.get_directed_support(ops.clique(number_of_nodes=3)) == "undirected_only"


def test_undirected_only_operator_rejects_directed_base_graph() -> None:
    graph = nx.DiGraph()
    graph.add_node(0, label="a", attribute=np.array([1.0]))
    graph.add_node(1, label="b", attribute=np.array([1.0]))
    graph.add_edge(0, 1, label="x")

    ag = AbstractGraph(graph=graph).create_default_interpretation_node()

    with pytest.raises(ValueError, match="undirected base graphs"):
        ops.clique()(ag)


def test_apply_local_node_decomposition_materializes_node_induced_subgraphs() -> None:
    graph = nx.path_graph(4)
    ag = AbstractGraph(graph=graph).create_default_interpretation_node()

    def custom_operator(abstract_graph: AbstractGraph) -> AbstractGraph:
        return ops.apply_local_node_decomposition(
            abstract_graph,
            lambda subgraph: [{1, 2}],
            source_operator=custom_operator,
            params={"kind": "middle"},
        )

    custom_operator.directed_support = "preserve"
    out = custom_operator(ag)
    mapped = out.get_interpretation_nodes_mapped_subgraphs()
    meta = next(iter(out.interpretation_graph.nodes(data=True)))[1]["meta"]

    assert len(mapped) == 1
    assert set(mapped[0].nodes()) == {1, 2}
    assert set(mapped[0].edges()) == {(1, 2)}
    assert meta["source_function"] == "custom_operator"
    assert meta["params"] == {"kind": "middle"}
    assert meta["source_chain"] == "custom_operator"
    assert set(meta["parent_mapped_subgraph"].nodes()) == {0, 1, 2, 3}


def test_apply_local_edge_decomposition_materializes_edge_induced_subgraphs() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
    ag = AbstractGraph(graph=graph).create_default_interpretation_node()

    def custom_operator(abstract_graph: AbstractGraph) -> AbstractGraph:
        return ops.apply_local_edge_decomposition(
            abstract_graph,
            lambda subgraph: [[(0, 1), (1, 2)]],
            source_operator=custom_operator,
        )

    custom_operator.directed_support = "preserve"
    out = custom_operator(ag)
    mapped = out.get_interpretation_nodes_mapped_subgraphs()

    assert len(mapped) == 1
    assert set(mapped[0].nodes()) == {0, 1, 2}
    assert set(mapped[0].edges()) == {(0, 1), (1, 2)}
    assert not mapped[0].has_edge(0, 2)


def test_apply_global_node_decomposition_receives_all_mapped_subgraphs() -> None:
    graph = nx.path_graph(5)
    ag = AbstractGraph(graph=graph)
    ag.create_interpretation_node_with_subgraph_from_nodes([0, 1])
    ag.create_interpretation_node_with_subgraph_from_nodes([3, 4])
    seen_counts = []

    def custom_operator(abstract_graph: AbstractGraph) -> AbstractGraph:
        def decompose(subgraphs, base_graph):
            seen_counts.append(len(subgraphs))
            return [set().union(*(set(subgraph.nodes()) for subgraph in subgraphs))]

        return ops.apply_global_node_decomposition(
            abstract_graph,
            decompose,
            source_operator=custom_operator,
        )

    custom_operator.directed_support = "preserve"
    out = custom_operator(ag)
    mapped = out.get_interpretation_nodes_mapped_subgraphs()

    assert seen_counts == [2]
    assert len(mapped) == 1
    assert set(mapped[0].nodes()) == {0, 1, 3, 4}


def test_apply_global_edge_decomposition_receives_all_mapped_subgraphs() -> None:
    graph = nx.path_graph(4)
    ag = AbstractGraph(graph=graph)
    ag.create_interpretation_node_with_subgraph_from_edges([(0, 1)])
    ag.create_interpretation_node_with_subgraph_from_edges([(2, 3)])
    seen_counts = []

    def custom_operator(abstract_graph: AbstractGraph) -> AbstractGraph:
        def decompose(subgraphs, base_graph):
            seen_counts.append(len(subgraphs))
            return [[edge for subgraph in subgraphs for edge in subgraph.edges()]]

        return ops.apply_global_edge_decomposition(
            abstract_graph,
            decompose,
            source_operator=custom_operator,
        )

    custom_operator.directed_support = "preserve"
    out = custom_operator(ag)
    mapped = out.get_interpretation_nodes_mapped_subgraphs()

    assert seen_counts == [2]
    assert len(mapped) == 1
    assert set(mapped[0].edges()) == {(0, 1), (2, 3)}


def test_apply_decomposition_rejects_generator_outputs() -> None:
    graph = nx.path_graph(3)
    ag = AbstractGraph(graph=graph).create_default_interpretation_node()

    def custom_operator(abstract_graph: AbstractGraph) -> AbstractGraph:
        def decompose(subgraph):
            yield {0, 1}

        return ops.apply_local_node_decomposition(
            abstract_graph,
            decompose,
            source_operator=custom_operator,
        )

    custom_operator.directed_support = "preserve"

    with pytest.raises(TypeError, match="must return a list of components"):
        custom_operator(ag)


def test_apply_decomposition_skip_empty_suppresses_empty_components() -> None:
    graph = nx.path_graph(3)
    ag = AbstractGraph(graph=graph).create_default_interpretation_node()

    def custom_operator(abstract_graph: AbstractGraph) -> AbstractGraph:
        return ops.apply_local_node_decomposition(
            abstract_graph,
            lambda subgraph: [set(), {0, 1}],
            source_operator=custom_operator,
            skip_empty=True,
        )

    custom_operator.directed_support = "preserve"
    out = custom_operator(ag)
    mapped = out.get_interpretation_nodes_mapped_subgraphs()

    assert len(mapped) == 1
    assert set(mapped[0].nodes()) == {0, 1}


def test_apply_decomposition_rejects_unsupported_directed_graphs() -> None:
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    ag = AbstractGraph(graph=graph).create_default_interpretation_node()

    def custom_operator(abstract_graph: AbstractGraph) -> AbstractGraph:
        return ops.apply_local_node_decomposition(
            abstract_graph,
            lambda subgraph: [{0, 1}],
            source_operator=custom_operator,
        )

    custom_operator.directed_support = "undirected_only"

    with pytest.raises(ValueError, match="undirected base graphs"):
        custom_operator(ag)


def test_refactored_degree_and_merge_use_scaffold_metadata() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3)])
    ag = AbstractGraph(graph=graph).create_default_interpretation_node()

    degree_out = ops.degree(value=2)(ag)
    degree_data = next(iter(degree_out.interpretation_graph.nodes(data=True)))[1]
    degree_meta = degree_data["meta"]

    assert degree_meta["source_function"] == "degree"
    assert degree_meta["params"] == {"value": (2, 2)}
    assert set(get_mapped_subgraph(degree_data).nodes()) == {0, 1}

    edge_ag = AbstractGraph(graph=graph)
    edge_ag.create_interpretation_node_with_subgraph_from_edges([(0, 1)])
    edge_ag.create_interpretation_node_with_subgraph_from_edges([(1, 2)])

    merge_out = ops.merge(use_edges=True)(edge_ag)
    merge_data = next(iter(merge_out.interpretation_graph.nodes(data=True)))[1]
    merge_meta = merge_data["meta"]
    mapped = get_mapped_subgraph(merge_data)

    assert merge_meta["source_function"] == "merge"
    assert merge_meta["params"] == {"use_edges": True}
    assert set(mapped.edges()) == {(0, 1), (1, 2)}
    assert not mapped.has_edge(0, 2)
