from __future__ import annotations

import matplotlib
import networkx as nx

from abstractgraph.artificial import (
    artificial_node_label_colors,
    generate_artificial_dataset,
    generate_cycle_path_star_graph,
)


matplotlib.use("Agg")


def _attributed_isomorphic(left: nx.Graph, right: nx.Graph) -> bool:
    return nx.is_isomorphic(
        left,
        right,
        node_match=lambda a, b: a == b,
        edge_match=lambda a, b: a == b,
    ) and left.graph == right.graph


def test_cycle_path_star_graph_records_ground_truth_components() -> None:
    graph = generate_cycle_path_star_graph(
        cycle_length=4,
        num_cycles=2,
        path_length=2,
        num_rays=3,
        ray_length=2,
        node_alphabet_size=2,
        edge_alphabet_size=2,
        seed=7,
    )

    assert nx.is_connected(graph)
    assert len(nx.cycle_basis(graph)) == 2
    assert graph.graph["metadata"]["num_cycles"] == 2
    assert graph.graph["metadata"]["iteration_parameters"] == [
        {
            "cycle_length": 4,
            "num_cycles": 2,
            "path_length": 2,
            "num_rays": 3,
            "ray_length": 2,
        }
    ]
    assert {data["label_component"] for _, data in graph.nodes(data=True)} == {
        "cycle",
        "path",
        "star",
    }


def test_artificial_dataset_is_seed_reproducible() -> None:
    kwargs = dict(
        num_graphs=4,
        cycle_length=(3, 5),
        num_cycles=(0, 2),
        n_iterations=2,
        path_length=(0, 2),
        num_rays=(1, 2),
        ray_length=(0, 2),
        node_alphabet_size=(1, 3),
        edge_alphabet_size=2,
        seed=13,
    )

    first, _ = generate_artificial_dataset(**kwargs)
    second, _ = generate_artificial_dataset(**kwargs)

    assert len(first) == len(second) == 4
    assert all(_attributed_isomorphic(left, right) for left, right in zip(first, second))


def test_artificial_dataset_samples_ranges_per_materialized_unit() -> None:
    graphs, _ = generate_artificial_dataset(
        2,
        cycle_length=(3, 5),
        num_cycles=(1, 2),
        n_iterations=2,
        path_length=(0, 2),
        num_rays=2,
        ray_length=(1, 2),
        seed=5,
    )

    for graph in graphs:
        parameters = graph.graph["metadata"]["iteration_parameters"]
        assert len(parameters) == 3
        assert all(3 <= item["cycle_length"] <= 5 for item in parameters)
        assert all(1 <= item["num_cycles"] <= 2 for item in parameters)
        assert all(0 <= item["path_length"] <= 2 for item in parameters)


def test_artificial_plotter_uses_stable_component_palette() -> None:
    graphs, plotter = generate_artificial_dataset(
        2,
        cycle_length=4,
        path_length=1,
        num_rays=1,
        ray_length=1,
        node_alphabet_size=2,
        seed=7,
    )

    assert plotter.node_label_colors == artificial_node_label_colors(2)
    figure = plotter(graphs, n_cols=2, size=3)
    assert tuple(figure.get_size_inches()) == (6.0, 3.0)
