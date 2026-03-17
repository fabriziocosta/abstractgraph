"""Synthetic graph constructors used by example notebooks."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import networkx as nx
import numpy as np


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
