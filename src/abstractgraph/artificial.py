"""Deterministic synthetic graph constructors and visualization helpers.

The cycle-path-star family is the default controlled benchmark for studying
the discriminative capacity of graph representations and operator programs.
Every generated graph records the sampled ground-truth component parameters in
``graph.graph["metadata"]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable

import networkx as nx
import numpy as np


_COMPONENTS = ("cycle", "path", "star")
_ARTIFICIAL_SECTION_COLOR_RAMPS = {
    "cycle": ("#fee2e2", "#fca5a5", "#dc2626"),
    "path": ("#dbeafe", "#93c5fd", "#2563eb"),
    "star": ("#dcfce7", "#86efac", "#16a34a"),
}


def _sample_int_parameter(
    value: int | tuple[int, int] | list[int],
    rng: random.Random,
    name: str,
    *,
    minimum: int | None = None,
    valid_values: Callable[[int], bool] | None = None,
) -> int:
    """Resolve a fixed integer or an inclusive integer range."""
    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, tuple):
        if len(value) != 2 or not all(isinstance(bound, int) for bound in value):
            raise TypeError(f"{name} range must contain exactly two integers.")
        low, high = value
        if high < low:
            raise ValueError(f"{name} range maximum must be >= its minimum.")
        candidates = list(range(low, high + 1))
        if minimum is not None:
            candidates = [candidate for candidate in candidates if candidate >= minimum]
        if valid_values is not None:
            candidates = [candidate for candidate in candidates if valid_values(candidate)]
        if not candidates:
            raise ValueError(f"{name} range {value!r} contains no valid values.")
        return rng.choice(candidates)
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or inclusive two-integer range.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if valid_values is not None and not valid_values(value):
        raise ValueError(f"{name} has invalid value {value!r}.")
    return value


def _max_int_parameter(value: int | tuple[int, int] | list[int]) -> int:
    if isinstance(value, (tuple, list)):
        return int(value[1])
    return int(value)


def _make_alphabet(size: int, kind: str = "int", offset: int = 0) -> list[int | str]:
    if size < 1:
        raise ValueError("Alphabet size must be >= 1.")
    if kind == "int":
        return list(range(offset, offset + size))
    if kind == "letter":
        if offset + size > 26:
            raise ValueError("Letter alphabets support at most 26 symbols in total.")
        return [chr(ord("A") + index) for index in range(offset, offset + size)]
    raise ValueError("Alphabet kind must be 'int' or 'letter'.")


def _make_component_alphabets(
    size: int,
    kind: str = "int",
    *,
    component_specific_alphabets: bool = True,
) -> dict[str, list[int | str]]:
    if not component_specific_alphabets:
        shared = _make_alphabet(size, kind)
        return {component: shared for component in _COMPONENTS}
    return {
        component: _make_alphabet(size, kind, offset=index * size)
        for index, component in enumerate(_COMPONENTS)
    }


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))


def _rgb_to_hex(rgb: Iterable[float]) -> str:
    return "#" + "".join(
        f"{max(0, min(255, int(round(channel)))):02x}" for channel in rgb
    )


def _interpolate_color(left: str, right: str, fraction: float) -> str:
    return _rgb_to_hex(
        (1.0 - fraction) * left_channel + fraction * right_channel
        for left_channel, right_channel in zip(_hex_to_rgb(left), _hex_to_rgb(right))
    )


def _section_palette(section: str, size: int) -> list[str]:
    if size < 1:
        raise ValueError("Alphabet size must be >= 1.")
    ramp = _ARTIFICIAL_SECTION_COLOR_RAMPS[section]
    if size == 1:
        return [ramp[-1]]
    if size == 2:
        return [ramp[0], ramp[-1]]
    if size == 3:
        return list(ramp)
    colors = []
    for index in range(size):
        fraction = index / float(size - 1)
        if fraction <= 0.5:
            colors.append(_interpolate_color(ramp[0], ramp[1], fraction * 2.0))
        else:
            colors.append(_interpolate_color(ramp[1], ramp[2], (fraction - 0.5) * 2.0))
    return colors


def artificial_node_label_colors(
    node_alphabet_size: int,
    *,
    node_alphabet_kind: str = "int",
    component_specific_alphabets: bool = True,
) -> dict[int | str, str]:
    """Return stable red, blue, and green palettes for component labels."""
    alphabets = _make_component_alphabets(
        int(node_alphabet_size),
        node_alphabet_kind,
        component_specific_alphabets=component_specific_alphabets,
    )
    return {
        label: color
        for component in _COMPONENTS
        for label, color in zip(
            alphabets[component],
            _section_palette(component, int(node_alphabet_size)),
        )
    }


def make_artificial_graph_plotter(
    node_alphabet_size: int,
    *,
    node_alphabet_kind: str = "int",
    component_specific_alphabets: bool = True,
):
    """Build a plotter whose component-label colors remain fixed across graphs."""
    node_label_colors = artificial_node_label_colors(
        node_alphabet_size,
        node_alphabet_kind=node_alphabet_kind,
        component_specific_alphabets=component_specific_alphabets,
    )

    def _layout(graph: nx.Graph, layout, layout_seed: int):
        if callable(layout):
            return layout(graph)
        if layout == "spring":
            return nx.spring_layout(graph, seed=layout_seed)
        if layout == "circular":
            return nx.circular_layout(graph)
        if layout == "shell":
            return nx.shell_layout(graph)
        return nx.kamada_kawai_layout(graph)

    def _draw_single_graph(
        graph: nx.Graph | None,
        *,
        ax=None,
        size: float = 4,
        title: str | None = None,
        layout="spring",
        layout_seed: int = 0,
        show_label: bool = True,
        node_size: float = 300,
        node_edgecolors: str = "black",
        node_linewidths: float = 2,
        edge_width: float = 2,
        label_font_size: float = 8,
        title_font_size: float = 9,
    ):
        import matplotlib.pyplot as plt

        figure = None
        if ax is None:
            figure, ax = plt.subplots(1, 1, figsize=(float(size), float(size)))
        ax.axis("off")
        if title is not None:
            ax.set_title(str(title), fontsize=title_font_size)
        if graph is None or graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "None" if graph is None else "empty", ha="center", va="center")
            return figure
        positions = _layout(graph, layout, layout_seed)
        labels = {node: str(attrs.get("label", "")) for node, attrs in graph.nodes(data=True)}
        node_colors = [
            node_label_colors.get(attrs.get("label", ""), "#d1d5db")
            for _, attrs in graph.nodes(data=True)
        ]
        nx.draw_networkx_edges(graph, positions, width=edge_width, ax=ax)
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            node_color=node_colors,
            edgecolors=node_edgecolors,
            linewidths=node_linewidths,
            node_size=node_size,
        )
        if show_label:
            nx.draw_networkx_labels(graph, positions, labels=labels, font_size=label_font_size, ax=ax)
        return figure

    def plot_artificial_graphs(
        graph_or_graphs=None,
        *,
        n_cols: int | None = None,
        titles: list[str] | None = None,
        size: float = 4,
        **kwargs,
    ):
        """Plot one graph or a sequence of graphs using the benchmark palette."""
        import matplotlib.pyplot as plt

        if graph_or_graphs is None and "graph" in kwargs:
            graph_or_graphs = kwargs.pop("graph")
        if isinstance(graph_or_graphs, nx.Graph) or graph_or_graphs is None:
            return _draw_single_graph(graph_or_graphs, size=size, **kwargs)
        graphs = list(graph_or_graphs)
        if not graphs:
            return None
        if titles is not None and len(titles) != len(graphs):
            raise ValueError("titles must contain exactly one title per graph.")
        n_cols = len(graphs) if n_cols is None else max(1, int(n_cols))
        n_rows = int(math.ceil(len(graphs) / float(n_cols)))
        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(float(size) * n_cols, float(size) * n_rows),
        )
        axes = [axes] if n_rows == n_cols == 1 else list(np.asarray(axes).flatten())
        for index, graph in enumerate(graphs):
            title = None if titles is None else titles[index]
            _draw_single_graph(graph, ax=axes[index], title=title, **kwargs)
        for ax in axes[len(graphs):]:
            ax.axis("off")
        figure.tight_layout()
        return figure

    plot_artificial_graphs.node_label_colors = node_label_colors
    plot_artificial_graphs.plot_kwargs = {
        "size": 4,
        "show_label": True,
        "node_size": 300,
        "node_linewidths": 2,
        "edge_width": 2,
    }
    return plot_artificial_graphs


def generate_cycle_path_star_graph(
    cycle_length: int,
    path_length: int,
    num_rays: int,
    ray_length: int,
    *,
    num_cycles: int = 1,
    n_iterations: int = 1,
    node_alphabet_size: int = 1,
    edge_alphabet_size: int = 1,
    node_alphabet_kind: str = "int",
    edge_alphabet_kind: str = "int",
    component_specific_alphabets: bool = True,
    seed: int | None = None,
    _unit_parameter_sampler: Callable[[], dict[str, int]] | None = None,
) -> nx.Graph:
    """Generate a connected, annotated cycle -> path -> star-ray graph."""
    if not isinstance(n_iterations, int) or n_iterations < 1:
        raise ValueError("n_iterations must be an integer >= 1.")

    def validate_unit_parameters(parameters: dict[str, int]) -> None:
        if any(parameters[name] < 0 for name in ("cycle_length", "path_length", "num_rays", "ray_length")):
            raise ValueError("Structural parameters must be non-negative.")
        if parameters["num_cycles"] < 0:
            raise ValueError("num_cycles must be >= 0.")
        if parameters["num_cycles"] == parameters["path_length"] == parameters["num_rays"] == 0:
            raise ValueError("At least one component family must be present.")
        if parameters["num_cycles"] > 0 and parameters["cycle_length"] not in (0,) and parameters["cycle_length"] < 3:
            raise ValueError("cycle_length must be 0 or >= 3 when cycles are requested.")
        if parameters["cycle_length"] == 0 and parameters["num_cycles"] > 1:
            raise ValueError("num_cycles must be 1 when cycle_length is 0.")

    fixed_parameters = {
        "cycle_length": cycle_length,
        "num_cycles": num_cycles,
        "path_length": path_length,
        "num_rays": num_rays,
        "ray_length": ray_length,
    }
    if _unit_parameter_sampler is None:
        validate_unit_parameters(fixed_parameters)

    rng = random.Random(seed)
    node_alphabets = _make_component_alphabets(
        node_alphabet_size,
        node_alphabet_kind,
        component_specific_alphabets=component_specific_alphabets,
    )
    edge_alphabets = _make_component_alphabets(
        edge_alphabet_size,
        edge_alphabet_kind,
        component_specific_alphabets=component_specific_alphabets,
    )
    graph = nx.Graph()
    next_node = 0
    iteration_parameters: list[dict[str, int]] = []

    def add_node(role: str, component: str) -> int:
        nonlocal next_node
        node = next_node
        next_node += 1
        graph.add_node(
            node,
            role=role,
            label=rng.choice(node_alphabets[component]),
            label_component=component,
        )
        return node

    def add_edge(left: int, right: int, role: str, component: str) -> None:
        graph.add_edge(
            left,
            right,
            role=role,
            label=rng.choice(edge_alphabets[component]),
            label_component=component,
        )

    def add_unit(attach_to: int | None = None) -> list[int]:
        parameters = dict(
            _unit_parameter_sampler() if _unit_parameter_sampler is not None else fixed_parameters
        )
        validate_unit_parameters(parameters)
        iteration_parameters.append(parameters)
        current = None
        cycle_nodes: list[int] = []
        if parameters["num_cycles"] > 0:
            if parameters["cycle_length"] == 0:
                cycle_nodes = [add_node("cycle_anchor", "cycle")]
            else:
                cycle_nodes = [add_node("cycle", "cycle") for _ in range(parameters["cycle_length"])]
                for index, node in enumerate(cycle_nodes):
                    add_edge(node, cycle_nodes[(index + 1) % len(cycle_nodes)], "cycle", "cycle")
                previous_cycle = cycle_nodes
                for _ in range(1, parameters["num_cycles"]):
                    shared_index = rng.randrange(parameters["cycle_length"])
                    shared_left = previous_cycle[shared_index]
                    shared_right = previous_cycle[(shared_index + 1) % parameters["cycle_length"]]
                    new_cycle = [shared_left, shared_right]
                    new_cycle.extend(
                        add_node("cycle", "cycle")
                        for _ in range(parameters["cycle_length"] - 2)
                    )
                    for index in range(1, len(new_cycle)):
                        add_edge(new_cycle[index], new_cycle[(index + 1) % len(new_cycle)], "cycle", "cycle")
                    previous_cycle = new_cycle
            current = rng.choice(cycle_nodes)
            graph.nodes[current]["role"] = "cycle_anchor"
            if attach_to is not None:
                add_edge(attach_to, current, "iteration_link", "cycle")

        for _ in range(parameters["path_length"]):
            new_node = add_node("path", "path")
            if current is not None:
                add_edge(current, new_node, "path", "path")
            elif attach_to is not None:
                add_edge(attach_to, new_node, "iteration_link", "path")
            current = new_node

        endpoints: list[int] = []
        if parameters["num_rays"] > 0:
            hub = add_node("star_hub", "star")
            if current is not None:
                add_edge(current, hub, "star_ray", "star")
            elif attach_to is not None:
                add_edge(attach_to, hub, "iteration_link", "star")
            for ray_index in range(parameters["num_rays"]):
                current = hub
                for step in range(parameters["ray_length"]):
                    role = f"ray_{ray_index}_leaf" if step == parameters["ray_length"] - 1 else f"ray_{ray_index}_node"
                    new_node = add_node(role, "star")
                    add_edge(current, new_node, "star_ray", "star")
                    current = new_node
                endpoints.append(current)
        return endpoints

    frontier: list[int | None] = [None]
    for _ in range(n_iterations):
        next_frontier = []
        for attachment in frontier:
            next_frontier.extend(add_unit(attachment))
        frontier = next_frontier
        if not frontier:
            break

    graph.graph["metadata"] = {
        **fixed_parameters,
        "n_iterations": n_iterations,
        "iteration_parameters": iteration_parameters,
        "node_alphabet_size": node_alphabet_size,
        "edge_alphabet_size": edge_alphabet_size,
        "node_alphabet_kind": node_alphabet_kind,
        "edge_alphabet_kind": edge_alphabet_kind,
        "component_specific_alphabets": component_specific_alphabets,
        "node_alphabets_by_component": node_alphabets,
        "edge_alphabets_by_component": edge_alphabets,
        "seed": seed,
    }
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        raise RuntimeError("Cycle-path-star generation produced a disconnected graph.")
    return graph


def generate_artificial_dataset(
    num_graphs: int,
    cycle_length: int | tuple[int, int] | list[int],
    path_length: int | tuple[int, int] | list[int],
    num_rays: int | tuple[int, int] | list[int],
    ray_length: int | tuple[int, int] | list[int],
    *,
    num_cycles: int | tuple[int, int] | list[int] = 1,
    n_iterations: int = 1,
    node_alphabet_size: int | tuple[int, int] | list[int] = 1,
    edge_alphabet_size: int | tuple[int, int] | list[int] = 1,
    node_alphabet_kind: str = "int",
    edge_alphabet_kind: str = "int",
    component_specific_alphabets: bool = True,
    seed: int | None = 0,
):
    """Generate the default controlled benchmark and its matching plotter.

    Range-valued structural parameters are inclusive and sampled independently
    for every materialized unit. Reusing the same arguments and seed produces
    attributed-isomorphic graphs with identical metadata.
    """
    if not isinstance(num_graphs, int) or num_graphs < 0:
        raise ValueError("num_graphs must be a non-negative integer.")
    if not isinstance(n_iterations, int) or n_iterations < 1:
        raise ValueError("n_iterations must be an integer >= 1.")
    rng = random.Random(seed)
    graphs = []
    for graph_index in range(num_graphs):
        sampled_node_alphabet_size = _sample_int_parameter(
            node_alphabet_size, rng, "node_alphabet_size", minimum=1
        )
        sampled_edge_alphabet_size = _sample_int_parameter(
            edge_alphabet_size, rng, "edge_alphabet_size", minimum=1
        )

        def sample_unit_parameters() -> dict[str, int]:
            for _ in range(128):
                sampled_num_cycles = _sample_int_parameter(num_cycles, rng, "num_cycles", minimum=0)
                sampled_cycle_length = _sample_int_parameter(
                    cycle_length,
                    rng,
                    "cycle_length",
                    minimum=0,
                    valid_values=(
                        None
                        if sampled_num_cycles == 0
                        else lambda value: value == 0 or value >= 3
                    ),
                )
                parameters = {
                    "cycle_length": sampled_cycle_length,
                    "num_cycles": sampled_num_cycles,
                    "path_length": _sample_int_parameter(
                        path_length, rng, "path_length", minimum=0
                    ),
                    "num_rays": _sample_int_parameter(
                        num_rays, rng, "num_rays", minimum=0
                    ),
                    "ray_length": _sample_int_parameter(
                        ray_length, rng, "ray_length", minimum=0
                    ),
                }
                if (
                    parameters["num_cycles"] > 0
                    or parameters["path_length"] > 0
                    or parameters["num_rays"] > 0
                ) and not (
                    parameters["cycle_length"] == 0
                    and parameters["num_cycles"] > 1
                ):
                    return parameters
            raise ValueError("Parameter ranges did not produce a valid non-empty unit.")

        pending_parameters = [sample_unit_parameters()]

        def next_unit_parameters() -> dict[str, int]:
            return pending_parameters.pop(0) if pending_parameters else sample_unit_parameters()

        first_parameters = pending_parameters[0]
        graph_seed = rng.randint(0, 2**32 - 1)
        graph = generate_cycle_path_star_graph(
            **first_parameters,
            n_iterations=n_iterations,
            node_alphabet_size=sampled_node_alphabet_size,
            edge_alphabet_size=sampled_edge_alphabet_size,
            node_alphabet_kind=node_alphabet_kind,
            edge_alphabet_kind=edge_alphabet_kind,
            component_specific_alphabets=component_specific_alphabets,
            seed=graph_seed,
            _unit_parameter_sampler=next_unit_parameters,
        )
        graph.graph["metadata"].update(
            dataset_seed=seed,
            graph_index=graph_index,
            graph_seed=graph_seed,
        )
        graphs.append(graph)

    plotter = make_artificial_graph_plotter(
        _max_int_parameter(node_alphabet_size),
        node_alphabet_kind=node_alphabet_kind,
        component_specific_alphabets=component_specific_alphabets,
    )
    return graphs, plotter


def _label_pool(alphabet_size: int, integers_range: int) -> list[int]:
    return list(range(min(max(1, alphabet_size), max(1, integers_range))))


def _set_default_graph_labels(
    graph: nx.Graph,
    *,
    rng: random.Random,
    alphabet_size: int,
    integers_range: int,
    attribute_size: int = 0,
) -> nx.Graph:
    labels = _label_pool(alphabet_size, integers_range)
    for node in graph.nodes():
        graph.nodes[node]["label"] = int(rng.choice(labels))
        if attribute_size > 0:
            graph.nodes[node]["attribute"] = np.asarray(
                [rng.random() for _ in range(attribute_size)],
                dtype=float,
            )
    for u, v in graph.edges():
        graph.edges[u, v]["label"] = "-"
    return graph


def _graph_from_type(graph_type: str, size: int, rng: random.Random) -> nx.Graph:
    size = max(2, int(size))
    if graph_type == "path":
        return nx.path_graph(size)
    if graph_type == "cycle":
        return nx.cycle_graph(size)
    if graph_type == "tree":
        seed = rng.randint(0, 10**9)
        random_tree = getattr(nx, "random_labeled_tree", None)
        if random_tree is None:
            random_tree = nx.random_tree
        return random_tree(size, seed=seed)
    if graph_type == "dense":
        return nx.gnp_random_graph(size, 0.45, seed=rng.randint(0, 10**9))
    if graph_type == "regular":
        degree = min(3, max(2, size - 1))
        if degree * size % 2 == 1:
            degree -= 1
        degree = max(1, degree)
        return nx.random_regular_graph(degree, size, seed=rng.randint(0, 10**9))
    if graph_type == "degree":
        center_degree = min(size - 1, max(2, size // 2))
        graph = nx.star_graph(center_degree)
        next_node = graph.number_of_nodes()
        while graph.number_of_nodes() < size:
            leaf = rng.randrange(1, graph.number_of_nodes())
            graph.add_edge(leaf, next_node)
            next_node += 1
        return graph
    raise ValueError(f"Unsupported graph_type {graph_type!r}")


@dataclass
class RandomGraphConstructor:
    integers_range: int = 12
    instance_size: int = 40
    alphabet_size: int = 4
    attribute_size: int = 0
    graph_type: str = "dense"
    seed: int | None = 0

    def sample(self, n: int | None = None):
        rng = random.Random(self.seed)

        def _one() -> nx.Graph:
            graph = _graph_from_type(self.graph_type, self.instance_size, rng)
            return _set_default_graph_labels(
                graph,
                rng=rng,
                alphabet_size=self.alphabet_size,
                integers_range=self.integers_range,
                attribute_size=self.attribute_size,
            )

        if n is None:
            return _one()
        n = int(n)
        if n == 1:
            return _one()
        return [_one() for _ in range(n)]


@dataclass
class ArtificialGraphDatasetConstructor:
    graph_generator_target_type_pos: str = "cycle"
    graph_generator_context_type_pos: str = "cycle"
    graph_generator_target_type_neg: str = "path"
    graph_generator_context_type_neg: str = "path"
    target_size_pos: int = 10
    context_size_pos: int = 10
    n_link_edges_pos: int = 1
    alphabet_size_pos: int = 4
    target_size_neg: int = 10
    context_size_neg: int = 10
    n_link_edges_neg: int = 1
    alphabet_size_neg: int = 4
    integers_range: int = 32
    attribute_size: int = 0
    seed: int | None = 0

    def _compose_graph(
        self,
        *,
        target_type: str,
        context_type: str,
        target_size: int,
        context_size: int,
        n_link_edges: int,
        alphabet_size: int,
        rng: random.Random,
    ) -> nx.Graph:
        target = _graph_from_type(target_type, target_size, rng)
        context = _graph_from_type(context_type, context_size, rng)
        graph = nx.disjoint_union(target, context)
        target_nodes = list(range(target.number_of_nodes()))
        context_nodes = list(range(target.number_of_nodes(), graph.number_of_nodes()))
        for _ in range(max(1, int(n_link_edges))):
            graph.add_edge(rng.choice(target_nodes), rng.choice(context_nodes))
        return _set_default_graph_labels(
            graph,
            rng=rng,
            alphabet_size=alphabet_size,
            integers_range=self.integers_range,
            attribute_size=self.attribute_size,
        )

    def sample(self, n: int) -> tuple[list[nx.Graph], list[int]]:
        rng = random.Random(self.seed)
        n = int(n)
        graphs: list[nx.Graph] = []
        targets: list[int] = []
        for _ in range(n):
            graphs.append(
                self._compose_graph(
                    target_type=self.graph_generator_target_type_pos,
                    context_type=self.graph_generator_context_type_pos,
                    target_size=self.target_size_pos,
                    context_size=self.context_size_pos,
                    n_link_edges=self.n_link_edges_pos,
                    alphabet_size=self.alphabet_size_pos,
                    rng=rng,
                )
            )
            targets.append(1)
            graphs.append(
                self._compose_graph(
                    target_type=self.graph_generator_target_type_neg,
                    context_type=self.graph_generator_context_type_neg,
                    target_size=self.target_size_neg,
                    context_size=self.context_size_neg,
                    n_link_edges=self.n_link_edges_neg,
                    alphabet_size=self.alphabet_size_neg,
                    rng=rng,
                )
            )
            targets.append(0)
        return graphs, targets
