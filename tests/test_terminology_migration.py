from __future__ import annotations

import warnings

import networkx as nx
import numpy as np

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


def test_canonical_graph_attributes_and_deprecated_aliases() -> None:
    ag = AbstractGraph(graph=_make_graph())
    assert set(ag.base_graph.nodes()) == {0, 1, 2}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert set(ag.preimage_graph.nodes()) == {0, 1, 2}
        assert any("preimage_graph" in str(w.message) for w in caught)


def test_interpretation_node_helpers_store_mapped_subgraph_and_legacy_alias() -> None:
    ag = AbstractGraph(graph=_make_graph())
    ag.create_default_interpretation_node()

    node_data = next(iter(ag.interpretation_graph.nodes(data=True)))[1]
    assert get_mapped_subgraph(node_data) is not None
    assert node_data["mapped_subgraph"].number_of_nodes() == ag.base_graph.number_of_nodes()
    assert node_data["association"].number_of_nodes() == ag.base_graph.number_of_nodes()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ag.create_default_image_node()
        assert any("create_default_image_node" in str(w.message) for w in caught)


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
