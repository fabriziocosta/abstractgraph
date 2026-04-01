from collections import defaultdict
from typing import Any, Callable, DefaultDict, Iterable, List, Tuple

import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from abstractgraph.display import get_color, stable_hash
from abstractgraph.graphs import (
    AbstractGraph,
    get_interpretation_label_to_mapped_subgraphs,
    graphs_to_abstract_graphs,
)
from abstractgraph.labels import graph_hash_label_function_factory


def feature_subgraphs(
    graphs: Iterable[nx.Graph],
    decomposition_function: Callable[[AbstractGraph], AbstractGraph],
    nbits: int = 10,
) -> DefaultDict[Any, List[nx.Graph]]:
    """Collect every unique mapped base subgraph grouped by interpretation-node label."""

    label_map: DefaultDict[Any, List[nx.Graph]] = defaultdict(list)

    for abstract_graph in graphs_to_abstract_graphs(
        list(graphs),
        decomposition_function=decomposition_function,
        nbits=nbits,
    ):
        current = get_interpretation_label_to_mapped_subgraphs(
            abstract_graph,
            unique=True,
            copy=True,
        )
        for label, subgraphs in current.items():
            label_map[label].extend(subgraphs)

    return label_map


def display_feature_subgraphs(
    graphs: Iterable[nx.Graph],
    decomposition_function: Callable[[AbstractGraph], AbstractGraph],
    nbits: int = 10,
    n_cols: int = 10,
    figsize_per_cell: Tuple[float, float] = (2.3, 2.3),
    color_map: str = "hsv",
) -> DefaultDict[Any, List[nx.Graph]]:
    """Render every unique mapped base subgraph grouped by interpretation-node label."""

    def _node_colors(subgraph: nx.Graph) -> List[Any]:
        colors: List[Any] = []
        for node, data in subgraph.nodes(data=True):
            node_label = data.get("label")
            if node_label is None:
                node_label = stable_hash(str(node))
            colors.append(get_color(node_label, cmap_name=color_map))
        return colors

    def _grid_axes(axs, rows: int, cols: int):
        if isinstance(axs, plt.Axes):
            return [[axs]]
        arr = np.array(axs, dtype=object)
        return arr.reshape(rows, cols).tolist()

    label_map = feature_subgraphs(graphs, decomposition_function, nbits=nbits)

    if not label_map:
        print("No labeled feature subgraphs were collected.")
        return label_map

    n_subgraphs = sum(len(label_map[label]) for label in label_map)
    print(f'{len(label_map)} unique labels for {n_subgraphs} subgraphs.')
    for label in sorted(label_map, key=lambda value: str(value)):
        subgraphs = label_map[label]
        num = len(subgraphs)
        n_rows = max(1, math.ceil(num / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * figsize_per_cell[0], n_rows * figsize_per_cell[1]),
        )
        axes_grid = _grid_axes(axes, n_rows, n_cols)

        for idx in range(n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            ax = axes_grid[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if idx < num:
                subgraph = subgraphs[idx]
                layout = nx.kamada_kawai_layout(subgraph) if subgraph.number_of_nodes() else {}
                nx.draw(
                    subgraph,
                    pos=layout,
                    ax=ax,
                    with_labels=False,
                    node_color=_node_colors(subgraph),
                    edge_color="gray",
                    edgecolors="black",
                    linewidths=1.0,
                    node_size=120,
                )
            else:
                ax.set_facecolor("white")
                for spine in ax.spines.values():
                    spine.set_visible(False)

        fig.suptitle(f"Interpretation label {label} ({num} unique subgraphs)", y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
