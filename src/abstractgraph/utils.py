"""
Shared plotting utilities for AbstractGraph experiments.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from collections import Counter
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Polygon
from matplotlib.path import Path
import numpy as np
try:  # Pandas is required for notebook utilities but keep imports lazy
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def plot_dataset_method_bars(
    results: Union[Sequence[Mapping[str, Any]], "pd.DataFrame"],
    *,
    dataset_col: str = "assay_id",
    dataset_label_col: Optional[str] = None,
    dataset_label_formatter: Optional[Callable[[Any, Mapping[str, Any]], str]] = None,
    dataset_order: Optional[Sequence[Any]] = None,
    method_col: str = "df",
    method_order: Optional[Sequence[Any]] = None,
    metric_col: str = "roc_auc",
    std_col: str = "roc_auc_std",
    palette: Optional[Any] = None,
    figsize: Tuple[float, float] = (10, 6),
    group_width: float = 0.8,
    label_rotation: float = 0.0,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    legend: bool = True,
    legend_title: Optional[str] = None,
    capsize: float = 4.0,
    force_zero_y: bool = False,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot grouped bars (one color per method) with std error bars per dataset.
    """

    if pd is not None and isinstance(results, pd.DataFrame):
        records: List[Mapping[str, Any]] = results.to_dict(orient="records")
    else:
        try:
            records = list(results)  # type: ignore[arg-type]
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError("results must be a DataFrame or an iterable of mappings") from exc

    if not records:
        raise ValueError("results is empty; run the evaluation loop before plotting.")

    def _ordered(values: Iterable[Any]) -> List[Any]:
        ordered: Dict[Any, None] = {}
        for v in values:
            ordered.setdefault(v, None)
        return list(ordered.keys())

    dataset_values = [
        row[dataset_col]
        for row in records
        if dataset_col in row
    ]
    if not dataset_values:
        raise KeyError(f"Column '{dataset_col}' not found in any result row.")
    if dataset_order is None:
        dataset_order = _ordered(dataset_values)

    method_values = [
        row[method_col]
        for row in records
        if method_col in row
    ]
    if not method_values:
        raise KeyError(f"Column '{method_col}' not found in any result row.")
    if method_order is None:
        method_order = _ordered(method_values)

    if not dataset_order:
        raise ValueError("dataset_order resolved to an empty list.")
    if not method_order:
        raise ValueError("method_order resolved to an empty list.")

    group_width = max(0.1, min(group_width, 0.95))
    n_methods = len(method_order)
    n_datasets = len(dataset_order)
    base_positions = np.arange(n_datasets, dtype=float)
    if n_methods == 1:
        offsets = np.array([0.0])
    else:
        bar_width = group_width / max(1, n_methods)
        offsets = np.linspace(
            start=-group_width / 2 + bar_width / 2,
            stop=group_width / 2 - bar_width / 2,
            num=n_methods,
        )
    bar_width = group_width / max(1, n_methods)

    sample_rows: Dict[Any, Mapping[str, Any]] = {}
    value_map: Dict[Tuple[Any, Any], Tuple[float, float]] = {}
    for row in records:
        if dataset_col not in row or method_col not in row or metric_col not in row:
            continue
        dataset_id = row[dataset_col]
        method_id = row[method_col]
        try:
            metric_val = float(row[metric_col])
        except (TypeError, ValueError):
            continue
        std_val = row.get(std_col, 0.0)
        try:
            std_val = abs(float(std_val))
        except (TypeError, ValueError):
            std_val = 0.0
        value_map[(dataset_id, method_id)] = (metric_val, std_val)
        sample_rows.setdefault(dataset_id, row)

    if not value_map:
        raise ValueError(
            f"No rows contained '{dataset_col}', '{method_col}' and '{metric_col}'."
        )

    legend_title = legend_title if legend_title is not None else method_col
    ylabel = ylabel if ylabel is not None else metric_col
    value_extrema = [
        (metric - std, metric + std)
        for metric, std in value_map.values()
    ]
    data_min = min((lo for lo, _ in value_extrema), default=0.0)
    data_max = max((hi for _, hi in value_extrema), default=1.0)
    color_cycle = plt.rcParams.get("axes.prop_cycle")
    base_colors = None
    if color_cycle is not None:
        try:
            base_colors = color_cycle.by_key().get("color", None)
        except Exception:  # pragma: no cover - very unlikely
            base_colors = None
    if not base_colors:
        base_colors = [plt.cm.tab10(i / max(1, n_methods - 1)) for i in range(n_methods)]

    palette_lookup: Dict[Any, Any] = {}
    if palette is not None:
        if hasattr(palette, "keys"):
            palette_lookup = {
                method: palette.get(method)  # type: ignore[attr-defined]
                for method in method_order
            }
        else:
            palette_list = list(palette)  # type: ignore[arg-type]
            if palette_list:
                palette_lookup = {
                    method: palette_list[i % len(palette_list)]
                    for i, method in enumerate(method_order)
                }
    for i, method in enumerate(method_order):
        palette_lookup.setdefault(method, base_colors[i % len(base_colors)])

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for method_index, method in enumerate(method_order):
        positions: List[float] = []
        heights: List[float] = []
        errors: List[float] = []
        for dataset_index, dataset_id in enumerate(dataset_order):
            entry = value_map.get((dataset_id, method))
            if entry is None:
                continue
            positions.append(base_positions[dataset_index] + offsets[method_index])
            heights.append(entry[0])
            errors.append(entry[1])
        if not positions:
            continue
        ax.bar(
            positions,
            heights,
            width=bar_width,
            yerr=errors,
            capsize=capsize,
            color=palette_lookup[method],
            edgecolor="black",
            linewidth=0.4,
            label=method,
        )

    dataset_labels: List[str] = []
    for dataset_id in dataset_order:
        row = sample_rows.get(dataset_id, {})
        if dataset_label_formatter is not None:
            dataset_labels.append(dataset_label_formatter(dataset_id, row))
        elif dataset_label_col and dataset_label_col in row:
            dataset_labels.append(str(row[dataset_label_col]))
        else:
            dataset_labels.append(str(dataset_id))

    ax.set_xticks(base_positions)
    ax.set_xticklabels(
        dataset_labels,
        rotation=label_rotation,
        ha="right" if label_rotation else "center",
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(dataset_col if dataset_label_col is None else dataset_label_col)
    ax.set_title(title or f"{metric_col} by dataset")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    pad = 0.02 * (data_max - data_min) if np.isfinite(data_max - data_min) else 0.02
    pad = max(pad, 1e-3)
    if force_zero_y:
        top = data_max + pad if np.isfinite(data_max) else 1.0
        if top <= 0.0:
            top = 1.0
        ax.set_ylim(bottom=0.0, top=top)
    else:
        bottom = data_min - pad if np.isfinite(data_min) else 0.0
        top = data_max + pad if np.isfinite(data_max) else 1.0
        if top <= bottom:
            top = bottom + 1.0
        ax.set_ylim(bottom=bottom, top=top)

    if legend:
        ax.legend(
            title=legend_title,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=max(1, len(method_order)),
            borderaxespad=0.0,
        )

    if show:
        plt.show()
    return ax


def plot_pareto(
    results: Union[Sequence[Mapping[str, Any]], "pd.DataFrame"],
    *,
    dataset_col: str = "assay_id",
    method_col: str = "df",
    axis_1_col: str = "elapsed",
    axis_2_col: str = "roc_auc",
    dataset_order: Optional[Sequence[Any]] = None,
    method_order: Optional[Sequence[Any]] = None,
    dataset_palette: Optional[Any] = None,
    marker_cycle: Optional[Sequence[str]] = None,
    pareto_tol: float = 1e-12,
    figsize: Tuple[float, float] = (9, 6),
    log_axis_1: bool = True,
    title: Optional[str] = None,
    axis_1_label: Optional[str] = None,
    axis_2_label: Optional[str] = None,
    axis_1_ascending: bool = False,
    axis_2_ascending: bool = True,
    legend: bool = True,
    legend_kwargs: Optional[Mapping[str, Any]] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Scatter plot of two arbitrary axes with Pareto-front highlighting per dataset.
    `axis_1_ascending` / `axis_2_ascending` control whether larger values are
    treated as better (True) or worse (False) for the Pareto calculation.
    """

    if pd is None:
        raise ImportError("pandas is required to plot Pareto fronts")
    if isinstance(results, pd.DataFrame):
        df = results.copy()
    else:
        df = pd.DataFrame(list(results))
    # Ensure we have a unique, position-based index so Pareto flags don't
    # accidentally mark every row that shares the same label (e.g. when the
    # caller passes a DataFrame with a non-unique index).
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("results is empty; run the evaluation loop before plotting.")

    required = [dataset_col, method_col, axis_1_col, axis_2_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if dataset_order is None:
        dataset_order = sorted(df[dataset_col].unique())
    else:
        dataset_order = list(dataset_order)
    if method_order is None:
        method_order = sorted(df[method_col].unique())
    else:
        method_order = list(method_order)

    if not dataset_order:
        raise ValueError("dataset_order resolved to an empty list.")
    if not method_order:
        raise ValueError("method_order resolved to an empty list.")

    marker_cycle = list(marker_cycle or ['o', 's', 'D', '^', 'v', 'P', 'X'])
    markers = {
        method: marker_cycle[i % len(marker_cycle)]
        for i, method in enumerate(method_order)
    }

    if dataset_palette is None:
        base_palette = [
            plt.cm.tab10(i / max(1, len(dataset_order) - 1))
            for i in range(len(dataset_order))
        ]
        palette_lookup = {
            dataset: base_palette[i % len(base_palette)]
            for i, dataset in enumerate(dataset_order)
        }
    else:
        if hasattr(dataset_palette, "keys"):
            palette_lookup = {
                dataset: dataset_palette.get(dataset)  # type: ignore[attr-defined]
                for dataset in dataset_order
            }
        else:
            palette_list = list(dataset_palette)
            palette_lookup = {
                dataset: palette_list[i % len(palette_list)]
                for i, dataset in enumerate(dataset_order)
            }

    fallback_cycle = plt.rcParams.get("axes.prop_cycle")
    fallback_colors = None
    if fallback_cycle is not None:
        try:
            fallback_colors = fallback_cycle.by_key().get("color", None)
        except Exception:  # pragma: no cover
            fallback_colors = None
    if not fallback_colors:
        fallback_colors = [plt.cm.tab10(i / max(1, len(dataset_order) - 1)) for i in range(max(1, len(dataset_order)))]
    for i, dataset in enumerate(dataset_order):
        if palette_lookup.get(dataset) is None:
            palette_lookup[dataset] = fallback_colors[i % len(fallback_colors)]

    df = df.copy()
    df["pareto_front"] = False
    pareto_points: Dict[Any, pd.DataFrame] = {}
    pareto_shell2_points: Dict[Any, pd.DataFrame] = {}
    axis1_norm_col = "__axis1_norm"
    axis2_norm_col = "__axis2_norm"
    df[axis_1_col] = pd.to_numeric(df[axis_1_col], errors="coerce")
    df[axis_2_col] = pd.to_numeric(df[axis_2_col], errors="coerce")
    axis1_min_value = float(df[axis_1_col].min(skipna=True))
    axis1_max_value = float(df[axis_1_col].max(skipna=True))
    axis2_min_value = float(df[axis_2_col].min(skipna=True))
    axis2_max_value = float(df[axis_2_col].max(skipna=True))
    df[axis1_norm_col] = df[axis_1_col].copy()
    df[axis2_norm_col] = df[axis_2_col].copy()
    if not axis_1_ascending:
        df[axis1_norm_col] = -df[axis1_norm_col]
    if not axis_2_ascending:
        df[axis2_norm_col] = -df[axis2_norm_col]

    def _compute_front(sorted_subset: "pd.DataFrame") -> List[int]:
        best_metric = -np.inf
        front_idx: List[int] = []
        for idx, row in sorted_subset.iterrows():
            metric_val = row[axis2_norm_col]
            if metric_val >= best_metric - pareto_tol:
                front_idx.append(idx)
                best_metric = max(best_metric, metric_val)
        return front_idx

    for dataset_id in dataset_order:
        subset = df[df[dataset_col] == dataset_id]
        if subset.empty:
            pareto_points[dataset_id] = subset.iloc[0:0]
            pareto_shell2_points[dataset_id] = subset.iloc[0:0]
            continue
        subset = subset[np.isfinite(subset[axis1_norm_col]) & np.isfinite(subset[axis2_norm_col])]
        if subset.empty:
            pareto_points[dataset_id] = subset.iloc[0:0]
            pareto_shell2_points[dataset_id] = subset.iloc[0:0]
            continue
        subset_sorted = subset.sort_values(axis1_norm_col, ascending=False)
        front_idx = _compute_front(subset_sorted)
        if front_idx:
            df.loc[front_idx, "pareto_front"] = True
            pareto_points[dataset_id] = df.loc[front_idx].sort_values(
                axis_1_col,
                ascending=True,
            )
        else:
            pareto_points[dataset_id] = subset.iloc[0:0]

        remainder = subset_sorted.drop(index=front_idx, errors="ignore")
        second_idx = _compute_front(remainder)
        if second_idx:
            pareto_shell2_points[dataset_id] = df.loc[second_idx].sort_values(
                axis_1_col,
                ascending=True,
            )
        else:
            pareto_shell2_points[dataset_id] = subset.iloc[0:0]
    df = df.drop(columns=[axis1_norm_col, axis2_norm_col])

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    non_front = df[~df["pareto_front"]]
    front = df[df["pareto_front"]]
    plot_df = df[np.isfinite(df[axis_1_col]) & np.isfinite(df[axis_2_col])]

    def _scatter_subset(
        subset_df: "pd.DataFrame",
        *,
        size: float,
        linewidth: float,
        zorder: float,
    ) -> None:
        if subset_df.empty:
            return
        for dataset_id in dataset_order:
            dataset_subset = subset_df[subset_df[dataset_col] == dataset_id]
            if dataset_subset.empty:
                continue
            for method in method_order:
                group = dataset_subset[dataset_subset[method_col] == method]
                if group.empty:
                    continue
                ax.scatter(
                    group[axis_1_col],
                    group[axis_2_col],
                    s=size,
                    marker=markers[method],
                    c=[palette_lookup.get(dataset_id, "black")],
                    edgecolors="black",
                    linewidths=linewidth,
                    zorder=zorder,
                )

    _scatter_subset(non_front, size=110, linewidth=0.4, zorder=2.6)
    _scatter_subset(front, size=120, linewidth=1.6, zorder=3.0)

    def _lighten_color(color: Any, amount: float = 0.4) -> Any:
        try:
            rgb = np.array(mcolors.to_rgb(color))
            white = np.ones_like(rgb)
            blended = np.clip(rgb + (white - rgb) * amount, 0, 1)
            return tuple(blended)
        except Exception:
            return color

    line_segments: List[Any] = []
    for dataset_id, points in pareto_points.items():
        if points is None or points.empty or len(points) < 2:
            continue
        points = points.sort_values(axis_1_col)
        line = ax.plot(
            points[axis_1_col],
            points[axis_2_col],
            color=palette_lookup.get(dataset_id, "black"),
            linewidth=2.0,
            alpha=0.9,
            zorder=2,
        )
        line_segments.extend(line)

    for dataset_id, points in pareto_shell2_points.items():
        if points is None or points.empty or len(points) < 2:
            continue
        color = _lighten_color(palette_lookup.get(dataset_id, "black"), amount=0.55)
        points = points.sort_values(axis_1_col)
        line = ax.plot(
            points[axis_1_col],
            points[axis_2_col],
            color=color,
            linewidth=1.6,
            alpha=0.8,
            zorder=1.5,
        )
        line_segments.extend(line)

    axis2_worse_target = axis2_min_value if axis_2_ascending else axis2_max_value
    axis1_worse_target = axis1_max_value if not axis_1_ascending else axis1_min_value
    for dataset_id, points in pareto_points.items():
        if points is None or points.empty:
            continue
        ordered_points = points.sort_values(axis_1_col, ascending=True)
        first_point = ordered_points.iloc[0]
        last_point = ordered_points.iloc[-1]
        if np.isfinite(axis2_worse_target) and np.isfinite(first_point[axis_2_col]):
            line = ax.plot(
                [first_point[axis_1_col], first_point[axis_1_col]],
                [first_point[axis_2_col], axis2_worse_target],
                color=palette_lookup.get(dataset_id, "black"),
                linestyle="--",
                linewidth=1.1,
                alpha=0.6,
                zorder=1.2,
            )
            line_segments.extend(line)
        if np.isfinite(axis1_worse_target) and np.isfinite(last_point[axis_1_col]):
            line = ax.plot(
                [last_point[axis_1_col], axis1_worse_target],
                [last_point[axis_2_col], last_point[axis_2_col]],
                color=palette_lookup.get(dataset_id, "black"),
                linestyle="--",
                linewidth=1.1,
                alpha=0.6,
                zorder=1.2,
            )
            line_segments.extend(line)

    # Ensure markers render above all lines
    for line in line_segments:
        line.set_zorder(1.4)

    ax.set_xlabel(axis_1_label or axis_1_col)
    ax.set_ylabel(axis_2_label or axis_2_col)
    ax.set_title(title or f"{axis_2_col} vs {axis_1_col}")
    ax.grid(True, alpha=0.3)
    if log_axis_1:
        try:
            ax.set_xscale("log")
        except Exception:  # pragma: no cover
            pass

    if legend:
        dataset_handles = [
            Line2D(
                [0],
                [0],
                marker='o',
                color='none',
                markerfacecolor=palette_lookup.get(dataset_id, "black"),
                markeredgecolor='black',
                markersize=8,
                linestyle='None',
                label=str(dataset_id),
            )
            for dataset_id in dataset_order
            if dataset_id in set(plot_df[dataset_col])
        ]
        method_handles = [
            Line2D(
                [0],
                [0],
                marker=markers[method],
                color='black',
                markerfacecolor='white',
                markeredgecolor='black',
                markersize=8,
                linestyle='None',
                label=str(method),
            )
            for method in method_order
            if method in set(plot_df[method_col])
        ]
        handles = dataset_handles + method_handles
        if handles:
            legend_args = dict(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                title="Legend",
            )
            if legend_kwargs:
                legend_args.update(legend_kwargs)
            ax.legend(handles=handles, **legend_args)
    else:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    if show:
        plt.show()
    return ax

# Plot utilities migrated from notebooks/plot_utils.py

def medoid_positive_graph(
    graph_estimator,
    graphs,
    targets,
    positive_label=1,
    *,
    n_clusters: int = 5,
    return_index: bool = False,
    return_embedding: bool = False,
    return_proba: bool = False,
    return_avg_distance: bool = False,
):
    """
    Return per-cluster Pareto fronts among positive-class graphs based on
    transform embeddings and predictive scores.

    If ``n_clusters == 1``, computes a single cluster over all positives and
    returns all non-dominated instances (Pareto front) under the two objectives:
      - maximize probability of being positive
      - minimize average Euclidean distance to other members of the cluster

    If ``n_clusters > 1``, performs hierarchical Ward clustering on the
    positive embeddings and returns, for each cluster, its Pareto front under
    the same two objectives.

    Parameters
        graph_estimator: object with a ``transform(X)`` method that returns embeddings
        graphs: sequence of input graphs
        targets: array-like of labels aligned with ``graphs``
        positive_label: label value considered positive (default 1)
        n_clusters: number of Ward clusters over positives (default 5)
        return_index: if True, also return lists of original indices per cluster
        return_embedding: if True, also return embeddings for front instances per cluster
        return_proba: if True, also return positive-class probabilities per cluster
        return_avg_distance: if True, also return per-instance average distances within cluster

    Returns
        A list (length = number of clusters) where each element is the list of
        graphs on that cluster's Pareto front.

        If return_* flags are used, returns a tuple where additional elements are:
          - list of lists of indices (per cluster)
          - list of arrays of embeddings (per cluster)
          - list of arrays of positive probabilities (per cluster)
          - list of arrays of average distances (per cluster)

    Notes
        - Uses Euclidean distances on the embeddings returned by ``transform``.
        - If there is only one positive sample, it forms a single cluster/front.
        - Cluster label order is the natural label order from AgglomerativeClustering.
    """
    import numpy as _np
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
    except Exception:
        AgglomerativeClustering = None  # type: ignore

    y = _np.asarray(list(targets))
    pos_mask = (y == positive_label)
    pos_idx = _np.nonzero(pos_mask)[0]
    if pos_idx.size == 0:
        raise ValueError("No positive samples found for the given positive_label")
    pos_graphs = [graphs[i] for i in pos_idx]
    embs = _np.asarray(graph_estimator.transform(pos_graphs))
    n_pos = embs.shape[0]
    k = int(max(1, min(int(n_clusters), n_pos)))

    # Positive-class probabilities/scores (higher is better)
    def _positive_scores(model, X):
        try:
            P = model.predict_proba(X)
            # locate positive column
            cls = getattr(getattr(model, 'estimator_', model), 'classes_', None)
            if cls is not None:
                try:
                    cls = _np.asarray(cls)
                    j = int(_np.where(cls == positive_label)[0][0])
                except Exception:
                    j = P.shape[1] - 1
            else:
                j = P.shape[1] - 1
            return _np.asarray(P)[:, j].astype(float)
        except Exception:
            # Fallback to decision_function -> min-max to [0,1]
            try:
                df = getattr(getattr(model, 'estimator_', model), 'decision_function')
                s = _np.asarray(df(model._transform_raw(X)))  # use raw features
                if s.ndim > 1:
                    # binary: convert to one-vs-rest for positive_label if possible
                    # use last column as positive score otherwise
                    s = s[:, -1]
                s = s.astype(float)
                s_min, s_max = _np.min(s), _np.max(s)
                if s_max > s_min:
                    return (s - s_min) / (s_max - s_min)
                return _np.full_like(s, 0.5, dtype=float)
            except Exception:
                # As a last resort, return 0.5 for all
                return _np.full(n_pos, 0.5, dtype=float)

    p_pos = _positive_scores(graph_estimator, pos_graphs)

    # Helper to compute medoid index within a subset of rows
    def _subset_medoid(sub_indices: _np.ndarray) -> int:
        if sub_indices.size == 1:
            return int(sub_indices[0])
        sub = embs[sub_indices]
        # squared Euclidean distances (argmin invariant to sqrt)
        diff = sub[:, None, :] - sub[None, :, :]
        D = _np.sum(diff * diff, axis=2)
        sums = _np.sum(D, axis=1)
        local = int(_np.argmin(sums))
        return int(sub_indices[local])

    # Pareto front helper: maximize p, minimize d
    def _pareto_front(scores: _np.ndarray, avg_d: _np.ndarray) -> _np.ndarray:
        n = scores.shape[0]
        keep = _np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if (scores[j] >= scores[i] and avg_d[j] <= avg_d[i]) and (
                    scores[j] > scores[i] or avg_d[j] < avg_d[i]
                ):
                    keep[i] = False
                    break
        return _np.nonzero(keep)[0]

    # Cluster positives (or create a single cluster if unavailable)
    if k == 1 or AgglomerativeClustering is None:
        clabels = _np.zeros(n_pos, dtype=int)
        uniq = [0]
    else:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        clabels = model.fit_predict(embs)
        uniq = list(sorted(_np.unique(clabels)))

    graphs_front_per_cluster = []
    idx_front_per_cluster = []
    embs_front_per_cluster = []
    prob_front_per_cluster = []
    avgd_front_per_cluster = []

    for cid in uniq:
        members = _np.nonzero(clabels == cid)[0]
        sub = embs[members]
        # Pairwise Euclidean distances among cluster members
        if members.size <= 1:
            avg_d = _np.array([0.0])
        else:
            diff = sub[:, None, :] - sub[None, :, :]
            D = _np.sqrt(_np.sum(diff * diff, axis=2))
            # exclude self when averaging
            avg_d = (_np.sum(D, axis=1) - 0.0) / _np.maximum(1, D.shape[1] - 1)
        scores = p_pos[members]
        front_local = _pareto_front(scores, avg_d)
        sel = members[front_local]

        graphs_front_per_cluster.append([pos_graphs[i] for i in sel])
        idx_front_per_cluster.append([int(pos_idx[i]) for i in sel])
        embs_front_per_cluster.append(embs[sel])
        prob_front_per_cluster.append(scores[front_local])
        avgd_front_per_cluster.append(avg_d[front_local])

    # Assemble returns
    out_any = [graphs_front_per_cluster]
    if return_index:
        out_any.append(idx_front_per_cluster)
    if return_embedding:
        out_any.append(embs_front_per_cluster)
    if return_proba:
        out_any.append(prob_front_per_cluster)
    if return_avg_distance:
        out_any.append(avgd_front_per_cluster)
    return out_any[0] if len(out_any) == 1 else tuple(out_any)


def _convex_hull(points):
    """Return hull vertex indices for a 2D point set.

    Tries scipy.spatial.ConvexHull, falls back to angle sort around centroid.
    points: (n,2) array.
    """
    if points.shape[0] < 3:
        return None
    try:
        from scipy.spatial import ConvexHull  # type: ignore
        hull = ConvexHull(points)
        return hull.vertices
    except Exception:
        c = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
        order = np.argsort(angles)
        return order


def _build_local_hulls(Zi, k, z, fallback=True):
    """Return list of local hull polygons for class points Zi.

    Filtering:
      - Compute class-wide threshold using the k-nearest distances per point:
        thr = mean(D_sorted[:, :k]) + z * std(D_sorted[:, :k])
      - For each anchor j, keep neighbors among its k nearest with d <= thr.
      - If fewer than 2 neighbors remain: if fallback=True keep the two closest; else skip anchor.
    """
    polys = []
    n = Zi.shape[0]
    if n < (k + 1):
        return polys
    # Pairwise distances within class in 2D
    D = np.linalg.norm(Zi[:, None, :] - Zi[None, :, :], axis=2)
    # Class-wide stats over the k nearest distances for each row (exclude self at col 0 after sort)
    D_sorted = np.sort(D, axis=1)[:, 1 : k + 1]
    mu_c = float(D_sorted.mean())
    sigma_c = float(D_sorted.std())
    thr = mu_c + z * sigma_c
    for j in range(n):
        order = np.argsort(D[j])
        base_idx = order[1 : k + 1]
        base_d = D[j, base_idx]
        keep = base_idx[base_d <= thr]
        if keep.size < 2:
            if not fallback:
                continue
            keep = base_idx[:2]
        pts = np.vstack([Zi[j], Zi[keep]])
        hull_idx = _convex_hull(pts)
        if hull_idx is None or len(hull_idx) < 3:
            continue
        polys.append(pts[hull_idx])
    return polys


def _union_polygons(polys):
    """Union a list of polygons (each as (m,2) array). Returns shapely geometry or None."""
    try:
        from shapely.geometry import Polygon as SPolygon  # type: ignore
        from shapely.ops import unary_union  # type: ignore
    except Exception:
        return None
    s_polys = []
    for p in polys:
        if p.shape[0] < 3:
            continue
        try:
            sp = SPolygon(p)
            if not sp.is_valid:
                sp = sp.buffer(0)
            if sp.is_empty:
                continue
            s_polys.append(sp)
        except Exception:
            continue
    if not s_polys:
        return None
    try:
        u = unary_union(s_polys)
        if u.is_empty:
            return None
        return u
    except Exception:
        # Second attempt after re-buffering, in case of GEOS topology errors
        try:
            cleaned = []
            for sp in s_polys:
                try:
                    buf = sp.buffer(0)
                except Exception:
                    continue
                if not buf.is_empty:
                    cleaned.append(buf)
            if not cleaned:
                return None
            u = unary_union(cleaned)
            return None if u.is_empty else u
        except Exception:
            return None


def _pathpatch_from_shapely(geom, facecolor, alpha=None):
    """Convert shapely Polygon/MultiPolygon to a PathPatch with holes (evenodd rule)."""
    from shapely.geometry import Polygon as SPolygon  # type: ignore

    def polygon_to_path(sp):
        verts = []
        codes = []
        ext = np.asarray(sp.exterior.coords)
        if ext.shape[0] > 1 and np.allclose(ext[0], ext[-1]):
            ext = ext[:-1]
        verts.extend(ext.tolist())
        codes += [Path.MOVETO] + [Path.LINETO] * (len(ext) - 1) + [Path.CLOSEPOLY]
        verts.append(ext[0].tolist())
        for ring in sp.interiors:
            r = np.asarray(ring.coords)
            if r.shape[0] > 1 and np.allclose(r[0], r[-1]):
                r = r[:-1]
            verts.extend(r.tolist())
            codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 1) + [Path.CLOSEPOLY]
            verts.append(r[0].tolist())
        return Path(np.asarray(verts, float), codes)

    paths = []
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            paths.append(polygon_to_path(g))
    else:
        paths.append(polygon_to_path(geom))
    if not paths:
        return None
    verts = np.concatenate([p.vertices for p in paths])
    codes = np.concatenate([p.codes for p in paths])
    if alpha is None:
        patch = PathPatch(Path(verts, codes), facecolor=facecolor, edgecolor="none")
    else:
        patch = PathPatch(Path(verts, codes), facecolor=facecolor, edgecolor="none", alpha=alpha)
    try:
        patch.set_fillrule("evenodd")
    except Exception:
        pass
    return patch


def _knn_predict(Z2d, y_labels, k):
    if k < 1:
        raise ValueError("k must be >= 1 for knn-based modes")
    y_labels = np.asarray(y_labels)
    n = Z2d.shape[0]
    classes = np.unique(y_labels)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    D = np.linalg.norm(Z2d[:, None, :] - Z2d[None, :, :], axis=2)
    pred_lbl_idx = np.empty(n, dtype=int)
    for i_pt in range(n):
        order = np.argsort(D[i_pt])
        neigh = order[1 : k + 1]
        votes = np.zeros(classes.size, dtype=int)
        for jj in neigh:
            votes[class_to_idx[y_labels[jj]]] += 1
        pred_lbl_idx[i_pt] = votes.argmax()
    return classes[pred_lbl_idx]


def plot_embedding_2d(
    clf,
    graphs,
    labels,
    Z=None,
    title_prefix="Embeddings",
    mode="scatter",
    alpha=0.5,
    k=3,
    z=1.0,
    ax=None,
    show=True,
    show_instances: bool = True,
    edges: bool = True,
    quantile: float = 1.0,
):
    """
    Reduce high-dimensional embeddings to 2D and visualize.

    Parameters
        clf: estimator with .transform(X)
        graphs: sequence of graphs
        labels: array-like of class indices
        title_prefix: str for plot title
        mode: 'scatter' (points), 'simplex' (local hulls), 'class_union' (single union polygon per class; requires shapely),
              'knn' (color by k-NN prediction), 'knn_simplex' (simplex over k-NN labels), 'knn_class_union' (union over k-NN labels; requires shapely)
        alpha: fill transparency for filled modes
        k: number of neighbors (>=2 for filled modes)
        z: neighbor distance filter (default 1.0); keeps neighbors with d <= mean+z*std among the k nearest.
    ax: optional matplotlib Axes to draw on
    show: if True and a new figure is created, calls plt.show()
    show_instances: if True, also plot the raw instance embeddings as hollow markers (useful when mode uses filled regions)
    edges: if True, draw filled regions with black borders and class-dependent line widths
    Z: optional precomputed 2D embedding; if provided, skips fitting a reducer
    quantile: central quantile used to set axis limits for a robust view.
        Must satisfy 0 < quantile <= 1.0. Use 1.0 to include all points
        (default), or e.g. 0.99 to reduce outlier influence.
    """
    reducer_name = None
    if Z is None:
        embs = clf.transform(graphs)
        try:
            import umap.umap_ as umap

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*n_jobs value .* overridden .* by setting random_state.*",
                    category=UserWarning,
                    module="umap.umap_",
                )
                Z = umap.UMAP(
                    n_components=2,
                    random_state=42,
                    init="random",
                ).fit_transform(embs)
            reducer_name = "UMAP"
        except Exception:
            try:
                from sklearn.manifold import TSNE

                Z = TSNE(n_components=2, random_state=42, init="pca").fit_transform(embs)
                reducer_name = "t-SNE (fallback)"
            except Exception:
                from sklearn.decomposition import PCA

                Z = PCA(n_components=2, random_state=42).fit_transform(embs)
                reducer_name = "PCA (fallback)"
    else:
        reducer_name = "precomputed"

    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f"Z must be a 2D array; got shape {Z.shape}")
    if Z.shape[1] != 2:
        warnings.warn(
            f"Provided Z has shape {Z.shape}; projecting to 2D with PCA for plotting.",
            RuntimeWarning,
        )
        try:
            from sklearn.decomposition import PCA

            Z = PCA(n_components=2, random_state=42).fit_transform(Z)
            reducer_name = (reducer_name or "precomputed") + "->PCA2"
        except Exception:
            # Last resort: take first two columns
            Z = Z[:, :2]
            reducer_name = (reducer_name or "precomputed") + "->first2"
    else:
        reducer_name = reducer_name or "precomputed"

    # Always apply a final 2D PCA as the last step before plotting
    try:
        from sklearn.decomposition import PCA

        Z = PCA(n_components=2, random_state=42).fit_transform(Z)
        if reducer_name is None:
            reducer_name = "precomputed"
        if "->PCA2" not in reducer_name:
            reducer_name = reducer_name + "->PCA2"
    except Exception:
        # If PCA is unavailable or fails, proceed with the existing 2D embedding
        pass

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        created_ax = True
    else:
        fig = ax.figure

    labels = np.asarray(labels)
    unique = np.unique(labels)
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    def _edge_linewidth(class_index: int) -> float:
        if class_index <= 0:
            return 0.5
        if class_index == 1:
            return 1.0
        return 2.0 + (class_index - 2)

    if show_instances and mode != "scatter":
        # Plot all instances as hollow circles behind overlays
        ax.scatter(Z[:, 0], Z[:, 1], s=14, facecolors="none", edgecolors="gray", linewidths=0.5, alpha=0.6)

    if mode == "scatter":
        for i, cls in enumerate(unique):
            idx = np.where(labels == cls)[0]
            color = palette[i % len(palette)]
            ax.scatter(Z[idx, 0], Z[idx, 1], s=18, c=color, label=f"Class {int(cls)}", alpha=0.8)
        ax.legend(frameon=False)
    elif mode in ("simplex", "class_union"):
        if k < 2:
            raise ValueError("k must be >= 2 for simplex/union modes")
        handles = []
        for i, cls in enumerate(unique):
            idx = np.where(labels == cls)[0]
            if idx.size < (k + 1):
                continue
            Zi = Z[idx]
            color = palette[i % len(palette)]
            polys = _build_local_hulls(Zi, k, z, fallback=(mode == "simplex"))
            if not polys:
                continue
            edgecolor = "black" if edges else "none"
            linewidth = _edge_linewidth(i) if edges else 0.0
            handle = Patch(
                facecolor=(*mcolors.to_rgb(color), alpha),
                edgecolor=edgecolor,
                linewidth=linewidth,
                label=f"Class {int(cls)}",
            )
            if mode == "simplex":
                handles.append(handle)
                for poly in polys:
                    ax.add_patch(
                        Polygon(
                            poly,
                            closed=True,
                            facecolor=(*mcolors.to_rgb(color), alpha),
                            edgecolor=edgecolor,
                            linewidth=linewidth,
                        )
                    )
            else:  # class_union
                try:
                    import shapely  # noqa: F401
                except Exception as e:
                    raise ImportError(
                        "class_union mode requires shapely. Install with `pip install shapely` and rerun."
                    ) from e
                geom = _union_polygons(polys)
                if geom is None:
                    warnings.warn(
                        "Failed to compute union geometry from local hulls; drawing individual hulls instead.",
                        RuntimeWarning,
                    )
                    drew = False
                    for poly in polys:
                        if poly.shape[0] < 3:
                            continue
                        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor="none", alpha=alpha))
                        drew = True
                    if drew:
                        handles.append(handle)
                    continue
                patch = _pathpatch_from_shapely(geom, facecolor=(*mcolors.to_rgb(color), alpha), alpha=None)
                if patch is None:
                    warnings.warn(
                        "Failed to convert union geometry to a patch; drawing individual hulls instead.",
                        RuntimeWarning,
                    )
                    drew = False
                    for poly in polys:
                        if poly.shape[0] < 3:
                            continue
                        ax.add_patch(
                            Polygon(
                                poly,
                                closed=True,
                                facecolor=(*mcolors.to_rgb(color), alpha),
                                edgecolor=edgecolor,
                                linewidth=linewidth,
                            )
                        )
                        drew = True
                    if drew:
                        handles.append(handle)
                else:
                    patch.set_edgecolor(edgecolor)
                    patch.set_linewidth(linewidth)
                    handles.append(handle)
                    ax.add_patch(patch)
        if handles:
            ax.legend(handles=handles, frameon=False)
    elif mode == "knn":
        pred_labels = _knn_predict(Z, labels, k)
        classes = np.unique(pred_labels)
        for i_c, cls in enumerate(classes):
            idx = np.where(pred_labels == cls)[0]
            color = palette[i_c % len(palette)]
            ax.scatter(Z[idx, 0], Z[idx, 1], s=18, c=color, label=f"Pred {int(cls)}", alpha=0.8)
        ax.legend(frameon=False)
    elif mode == "knn_simplex":
        pred_labels = _knn_predict(Z, labels, k)
        handles = []
        classes = np.unique(pred_labels)
        for i_c, cls in enumerate(classes):
            idx = np.where(pred_labels == cls)[0]
            if idx.size < (k + 1):
                continue
            Zi = Z[idx]
            color = palette[i_c % len(palette)]
            edgecolor = "black" if edges else "none"
            linewidth = _edge_linewidth(i_c) if edges else 0.0
            handles.append(
                Patch(
                    facecolor=(*mcolors.to_rgb(color), alpha),
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    label=f"Pred {int(cls)}",
                )
            )
            polys = _build_local_hulls(Zi, k, z, fallback=True)
            for poly in polys:
                ax.add_patch(
                    Polygon(
                        poly,
                        closed=True,
                        facecolor=(*mcolors.to_rgb(color), alpha),
                        edgecolor=edgecolor,
                        linewidth=linewidth,
                    )
                )
        if handles:
            ax.legend(handles=handles, frameon=False)
    elif mode == "knn_class_union":
        pred_labels = _knn_predict(Z, labels, k)
        try:
            import shapely  # noqa: F401
        except Exception as e:
            raise ImportError("knn_class_union requires shapely. Install with `pip install shapely`.") from e
        handles = []
        classes = np.unique(pred_labels)
        for i_c, cls in enumerate(classes):
            idx = np.where(pred_labels == cls)[0]
            if idx.size < (k + 1):
                continue
            Zi = Z[idx]
            color = palette[i_c % len(palette)]
            edgecolor = "black" if edges else "none"
            linewidth = _edge_linewidth(i_c) if edges else 0.0
            polys = _build_local_hulls(Zi, k, z, fallback=False)
            if not polys:
                polys = _build_local_hulls(Zi, k, z, fallback=True)
            if not polys:
                warnings.warn(
                    f"No valid hulls for class {int(cls)} with k={k}; skipping patch for this class.",
                    RuntimeWarning,
                )
                continue
            handle = Patch(
                facecolor=(*mcolors.to_rgb(color), alpha),
                edgecolor=edgecolor,
                linewidth=linewidth,
                label=f"Pred {int(cls)}",
            )
            geom = _union_polygons(polys)
            if geom is None:
                warnings.warn(
                    "Failed to compute union geometry from local hulls (knn); drawing individual hulls instead.",
                    RuntimeWarning,
                )
                drew = False
                for poly in polys:
                    if poly.shape[0] < 3:
                        continue
                    ax.add_patch(
                        Polygon(
                            poly,
                            closed=True,
                            facecolor=(*mcolors.to_rgb(color), alpha),
                            edgecolor=edgecolor,
                            linewidth=linewidth,
                        )
                    )
                    drew = True
                if drew:
                    handles.append(handle)
                continue
            patch = _pathpatch_from_shapely(geom, facecolor=(*mcolors.to_rgb(color), alpha), alpha=None)
            if patch is None:
                warnings.warn(
                    "Failed to convert union geometry to a patch (knn); drawing individual hulls instead.",
                    RuntimeWarning,
                )
                drew = False
                for poly in polys:
                    if poly.shape[0] < 3:
                        continue
                    ax.add_patch(
                        Polygon(
                            poly,
                            closed=True,
                            facecolor=(*mcolors.to_rgb(color), alpha),
                            edgecolor=edgecolor,
                            linewidth=linewidth,
                        )
                    )
                    drew = True
                if drew:
                    handles.append(handle)
            else:
                patch.set_edgecolor(edgecolor)
                patch.set_linewidth(linewidth)
                handles.append(handle)
                ax.add_patch(patch)
        if handles:
            ax.legend(handles=handles, frameon=False)
    else:
        raise ValueError(
            "mode must be 'scatter', 'simplex', 'class_union', or 'knn', 'knn_simplex', 'knn_class_union'"
        )

    # Ensure axes cover a robust portion of points (or all points if quantile=1).
    q = float(quantile)
    if not (0.0 < q <= 1.0):
        raise ValueError(f"quantile must satisfy 0 < quantile <= 1.0; got {quantile}")
    if q < 1.0:
        tail = (1.0 - q) / 2.0
        lo_q = tail
        hi_q = 1.0 - tail
        xmin, xmax = np.quantile(Z[:, 0], [lo_q, hi_q]).astype(float)
        ymin, ymax = np.quantile(Z[:, 1], [lo_q, hi_q]).astype(float)
    else:
        xmin, xmax = float(Z[:, 0].min()), float(Z[:, 0].max())
        ymin, ymax = float(Z[:, 1].min()), float(Z[:, 1].max())
    dx, dy = xmax - xmin, ymax - ymin
    pad_x, pad_y = (0.05 * dx if dx > 0 else 1.0), (0.05 * dy if dy > 0 else 1.0)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.set_title(f"{title_prefix} - {reducer_name}")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")

    if show:
        if created_ax:
            fig.tight_layout()
            plt.show()
        else:
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass

    return ax


def plot_graph_label_counts(
    graphs,
    node_key: str = "label",
    edge_key: str = "label",
    top: int | None = None,
    normalize: bool = False,
    figsize: tuple[int, int] = (16, 4),
    title: str | None = None,
    log_scale: bool = False,
    annotate: bool = True,
):
    """
    Plot node/edge label frequencies and a node-size histogram across graphs.

    Parameters
    - graphs: iterable of NetworkX graphs with node/edge attributes containing `node_key`/`edge_key`.
    - node_key: node attribute name to read labels from (default 'label').
    - edge_key: edge attribute name to read labels from (default 'label').
    - top: if set, show only the top-N labels by count.
    - normalize: if True, plot proportions; otherwise raw counts.
    - figsize: figure size.
    - title: optional suptitle.

    Returns: (fig, axes, node_counts, edge_counts)
    """
    import matplotlib.pyplot as plt

    node_counts: Counter = Counter()
    edge_counts: Counter = Counter()
    node_sizes = []

    for g in graphs:
        node_sizes.append(g.number_of_nodes())
        # Nodes
        node_counts.update(
            d.get(node_key) for _, d in g.nodes(data=True) if d.get(node_key) is not None
        )
        # Edges
        edge_counts.update(
            d.get(edge_key) for _, _, d in g.edges(data=True) if d.get(edge_key) is not None
        )

    def _prep(counter: Counter):
        items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
        if top is not None:
            items = items[:top]
        labels = [str(k) for k, _ in items]
        vals = [v for _, v in items]
        if normalize:
            total = sum(counter.values()) or 1
            vals = [v / total for v in vals]
        return labels, vals

    node_labels, node_vals = _prep(node_counts)
    edge_labels, edge_vals = _prep(edge_counts)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # Nodes
    bars0 = axes[0].bar(range(len(node_labels)), node_vals, color="#4C78A8")
    axes[0].set_xticks(range(len(node_labels)))
    axes[0].set_xticklabels(node_labels, rotation=45, ha="right")
    axes[0].set_ylabel("proportion" if normalize else "count")
    axes[0].set_title(f"Node label counts{'' if top is None else f' (top {top})'}")
    if log_scale:
        axes[0].set_yscale('log')
    if annotate:
        for rect in bars0:
            h = rect.get_height()
            if h <= 0:
                continue
            txt = f"{h:.3g}" if normalize else f"{int(h)}"
            axes[0].annotate(
                txt,
                xy=(rect.get_x() + rect.get_width() / 2.0, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    # Edges
    bars1 = axes[1].bar(range(len(edge_labels)), edge_vals, color="#F58518")
    axes[1].set_xticks(range(len(edge_labels)))
    axes[1].set_xticklabels(edge_labels, rotation=45, ha="right")
    axes[1].set_ylabel("proportion" if normalize else "count")
    axes[1].set_title(f"Edge label counts{'' if top is None else f' (top {top})'}")
    if log_scale:
        axes[1].set_yscale('log')
    if annotate:
        for rect in bars1:
            h = rect.get_height()
            if h <= 0:
                continue
            txt = f"{h:.3g}" if normalize else f"{int(h)}"
            axes[1].annotate(
                txt,
                xy=(rect.get_x() + rect.get_width() / 2.0, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    # Node sizes
    hist_kwargs = {"bins": "auto"}
    if normalize:
        hist_kwargs["density"] = True
    axes[2].hist(node_sizes, color="#54A24B", **hist_kwargs)
    axes[2].set_ylabel("proportion" if normalize else "count")
    axes[2].set_xlabel("nodes per graph")
    axes[2].set_title("Node count histogram")
    if log_scale:
        axes[2].set_yscale('log')

    if title:
        fig.suptitle(title)

    return fig, axes, node_counts, edge_counts

__all__ = [
    "plot_dataset_method_bars",
    "plot_pareto",
    "medoid_positive_graph",
    "plot_embedding_2d",
    "plot_graph_label_counts"
]
