"""Visualization utilities for AbstractGraph and operator pipelines."""

import os
import inspect
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from toolz.functoolz import curry
from networkx.drawing.nx_agraph import to_agraph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import hashlib
from typing import Optional, Dict, Any, Tuple, List, Callable, Iterable, Union
import math
from abstractgraph.graphs import AbstractGraph
from abstractgraph.hashing import hash_graph

_NETWORKX_GRAPH_TYPES = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)


def _is_networkx_graph(obj: Any) -> bool:
    """
    Check if an object is a NetworkX graph instance.

    Args:
        obj: Object to test.

    Returns:
        True if obj is a NetworkX graph, otherwise False.
    """
    return isinstance(obj, _NETWORKX_GRAPH_TYPES)


def _infer_graph_list_kind(graphs: List[Any]) -> str:
    """
    Infer the graph type for a list of graphs.

    Args:
        graphs: List of graph-like objects.

    Returns:
        "abstract" if all are AbstractGraph,
        "networkx" if all are NetworkX graphs,
        "mixed" otherwise.
    """
    if all(isinstance(g, AbstractGraph) for g in graphs):
        return "abstract"
    if all(_is_networkx_graph(g) for g in graphs):
        return "networkx"
    return "mixed"


def _packed_kamada_kawai_layout(
    graph: nx.Graph,
    *,
    padding: float = 1.0,
) -> Dict[Any, Tuple[float, float]]:
    """
    Compute a Kamada-Kawai layout with non-overlapping disconnected components.

    Args:
        graph: Graph to lay out.
        padding: Extra spacing inserted between packed components.

    Returns:
        Dict mapping node ids to (x, y) positions.
    """
    if graph.number_of_nodes() == 0:
        return {}
    undirected = graph.to_undirected()
    components = [list(c) for c in nx.connected_components(undirected)]
    if len(components) <= 1:
        return nx.kamada_kawai_layout(graph)

    component_positions = []
    for comp_nodes in components:
        sub = graph.subgraph(comp_nodes)
        pos = nx.kamada_kawai_layout(sub)
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        if width == 0:
            width = 1.0
        if height == 0:
            height = 1.0
        normalized = {n: (p[0] - min_x, p[1] - min_y) for n, p in pos.items()}
        component_positions.append((normalized, width, height, len(comp_nodes)))

    component_positions.sort(key=lambda item: item[3], reverse=True)
    cols = max(1, int(math.ceil(math.sqrt(len(component_positions)))))
    packed: Dict[Any, Tuple[float, float]] = {}
    x_cursor = 0.0
    y_cursor = 0.0
    row_height = 0.0
    col_idx = 0
    for pos, width, height, _size in component_positions:
        if col_idx >= cols:
            col_idx = 0
            x_cursor = 0.0
            y_cursor += row_height + padding
            row_height = 0.0
        for node, (x, y) in pos.items():
            packed[node] = (x + x_cursor, y + y_cursor)
        x_cursor += width + padding
        row_height = max(row_height, height)
        col_idx += 1
    return packed


def _layout_bounds(pos: Dict[Any, Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    """Return (min_x, max_x, min_y, max_y) for a layout dict."""
    if not pos:
        return None
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    return min(xs), max(xs), min(ys), max(ys)


def _scale_layout_to_height(
    pos: Dict[Any, Tuple[float, float]],
    target_height: float,
) -> Dict[Any, Tuple[float, float]]:
    """Scale a layout's y-span to match target_height, keeping its center."""
    bounds = _layout_bounds(pos)
    if bounds is None:
        return pos
    _min_x, _max_x, min_y, max_y = bounds
    height = max_y - min_y
    if height <= 0 or target_height <= 0:
        return pos
    scale = target_height / height
    cy = (min_y + max_y) * 0.5
    scaled = {n: (x, (y - cy) * scale + cy) for n, (x, y) in pos.items()}
    return scaled


def _normalize_layout_to_box(
    pos: Dict[Any, Tuple[float, float]],
    *,
    target_width: float,
    target_height: float,
) -> Dict[Any, Tuple[float, float]]:
    """Scale a layout uniformly to fit within a target box, preserving aspect."""
    bounds = _layout_bounds(pos)
    if bounds is None:
        return pos
    min_x, max_x, min_y, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return pos
    scale = min(target_width / width, target_height / height)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    return {n: ((x - cx) * scale, (y - cy) * scale) for n, (x, y) in pos.items()}


def _add_axes_frame(
    fig: Figure,
    ax: plt.Axes,
    *,
    edge_color: str,
    line_width: float,
    inset_x: float = 0.0,
    inset_y: float = 0.0,
    fixed_inset_px: Optional[float] = None,
) -> None:
    """
    Draw a figure-level frame around one subplot axis.

    Args:
        fig: Figure that owns the axis.
        ax: Axis whose bounds are used for the frame.
        edge_color: Border color.
        line_width: Border line width.
        inset_x: Horizontal inset in figure coordinates.
        inset_y: Vertical inset in figure coordinates.
        fixed_inset_px: Optional fixed inset in pixels.

    Returns:
        None.
    """
    bbox = ax.get_position()
    fig_width_px, fig_height_px = fig.get_size_inches() * fig.dpi
    effective_inset_px = _effective_axes_inset_px(
        fig,
        ax,
        inset_x=inset_x,
        inset_y=inset_y,
        fixed_inset_px=fixed_inset_px,
    )
    effective_inset_x = min(effective_inset_px / fig_width_px, bbox.width * 0.45)
    effective_inset_y = min(effective_inset_px / fig_height_px, bbox.height * 0.45)
    frame = Rectangle(
        (bbox.x0 + effective_inset_x, bbox.y0 + effective_inset_y),
        max(0.0, bbox.width - 2 * effective_inset_x),
        max(0.0, bbox.height - 2 * effective_inset_y),
        transform=fig.transFigure,
        fill=False,
        edgecolor=edge_color,
        linewidth=line_width,
        zorder=10,
    )
    fig.add_artist(frame)


def _effective_axes_inset_px(
    fig: Figure,
    ax: plt.Axes,
    *,
    inset_x: float,
    inset_y: float,
    fixed_inset_px: Optional[float] = None,
) -> float:
    """
    Compute the effective inset used for subplot frames in pixels.

    Args:
        fig: Figure that owns the axis.
        ax: Axis whose bounds are used for the inset.
        inset_x: Requested horizontal inset in figure coordinates.
        inset_y: Requested vertical inset in figure coordinates.
        fixed_inset_px: Optional fixed inset in pixels.

    Returns:
        Effective inset in pixels.
    """
    bbox = ax.get_position()
    fig_width_px, fig_height_px = fig.get_size_inches() * fig.dpi
    bbox_width_px = bbox.width * fig_width_px
    bbox_height_px = bbox.height * fig_height_px
    max_inset_px = min(bbox_width_px, bbox_height_px) * 0.45
    if fixed_inset_px is not None:
        return min(max(0.0, float(fixed_inset_px)), max_inset_px)
    min_inset_px = min(bbox_width_px, bbox_height_px) * 0.06
    requested_inset_px = max(inset_x * fig_width_px, inset_y * fig_height_px)
    return min(max(min_inset_px, requested_inset_px), max_inset_px)

def _format_param_value_for_label(value: Any) -> str:
    """
    Produce a compact string for parameter values in the decomposition graph.

    - If value is a list and longer than 3 elements, show the first two, then '...', then the last.
      Example: [a, b, ..., z]
    - Otherwise, fall back to str(value).

    Note: Tuples (including length-2 tuples) are left as-is via str(value).

    Args:
        value: Parameter value to format.

    Returns:
        str: Compact string representation.
    """
    try:
        if isinstance(value, list) and len(value) > 3:
            # Convert elements to strings safely
            head = ", ".join(str(x) for x in value[:2])
            tail = str(value[-1])
            return f"[{head}, ..., {tail}]"
    except Exception:
        # Fallback to default string conversion if anything goes wrong
        pass
    return str(value)

def stable_hash(x: str) -> int:
    """
    Computes a stable hash from a string using MD5.

    Args:
        x: The string to hash.
    
    Returns:
        int: Stable integer hash value.
    """
    return int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)

def get_color(label: Any, cmap_name: str = 'hsv') -> Any:
    """
    Maps a label deterministically to a color via a continuous colormap.

    Previously we reduced the hash modulo 20 and used the 'tab20' palette,
    which can cause collisions (different labels producing the same color).
    Using a continuous colormap like 'hsv' and a normalized hash in [0,1)
    greatly reduces collisions while remaining stable across runs.

    Args:
        label: Any value; converted to an integer via int(...) or stable hash.
        cmap_name: Matplotlib colormap name (default 'hsv').
    
    Returns:
        Any: Color (RGBA tuple) as returned by the colormap.
    """
    cmap = cm.get_cmap(cmap_name)
    # Always hash the label (even if it's an int) to spread values across [0,1)
    # This avoids tiny normalized values (e.g., 9708/2^32) that cluster near hue 0 (red) in 'hsv'.
    num = stable_hash(str(label))
    # Normalize the integer into [0, 1) using 32-bit space to keep it stable.
    norm = (num % (2**32)) / float(2**32)
    return cmap(norm)

def display_graph(
    graph: nx.Graph,
    ax: Optional[plt.Axes] = None,
    style: Optional[Dict[str, Any]] = None,
    pos: Optional[Dict[Any, Tuple[float, float]]] = None,
    offset: Tuple[float, float] = (0, 0),
    size: Tuple[int, int] = (5, 4),
    *,
    display_nodes: bool = True,
    pack_disconnected: bool = True,
    pack_padding: float = 1.0,
    node_labels: bool = False,
    node_label_attr: str = 'label',
    node_label_font_size: int = 8,
    edge_labels: bool = False,
    edge_label_attr: str = 'label',
    edge_label_font_size: int = 7,
    fit_viewport: bool = True,
) -> Optional[plt.Axes]:
    """
    Draws a single NetworkX graph onto a Matplotlib axis with specified styling and offset.

    Node colors are assigned based on their 'label' attribute or a stable hash.

    Args:
        graph: The NetworkX graph to display.
        ax: The Matplotlib axis to draw on. If None, a new figure and axis are created.
        style: A dict of style parameters for drawing (node_size, edge_width, etc.).
        pos: Optional pre-calculated node positions. If None, Kamada-Kawai layout is used.
        offset: A tuple (x_offset, y_offset) to shift the graph's position.
        size: The figure size if `ax` is None.
        display_nodes: If False, do not draw node markers (circles). Labels can still be drawn.
        pack_disconnected: If True, pack disconnected components to avoid overlap.
        pack_padding: Spacing between packed components.
        node_labels: If True, draw node labels.
        node_label_attr: Node attribute name to use for labels (fallback to node id).
        node_label_font_size: Font size for node labels.
        edge_labels: If True, draw edge labels.
        edge_label_attr: Edge attribute name to use for labels (edges missing it are skipped).
        edge_label_font_size: Font size for edge labels.
        fit_viewport: If True, set axis limits to fit this graph with padding.

    Returns:
        The Matplotlib axis containing the visualization.
    """
    # Create figure and axis if not provided.
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        created_ax = True

    # Set default style if not provided.
    if style is None:
        style = {
            'node_size': 70, 'edge_width': 1.0, 'edge_style': 'solid',
            'node_border_width': 0.5, 'node_alpha': 0.8, 'edge_color': 'grey',
            'cmap': 'hsv'
        }
    # Ensure all expected keys have defaults if partially provided
    style.setdefault('node_size', 70)
    style.setdefault('edge_width', 1.0)
    style.setdefault('edge_style', 'solid')
    style.setdefault('node_border_width', 0.5)
    style.setdefault('node_alpha', 0.8)
    style.setdefault('edge_color', 'grey')
    style.setdefault('cmap', 'hsv')
    style.setdefault('node_edgecolors', 'black')
    style.setdefault('viewport_padding', 0.12)
    style.setdefault('viewport_padding_px', None)
    style.setdefault('viewport_padding_axes_fraction', None)
    style.setdefault('content_inset_px', 0.0)

    # Calculate positions if not provided. Handle empty graph.
    if pos is None:
        if graph.number_of_nodes() > 0:
            if pack_disconnected:
                pos = _packed_kamada_kawai_layout(graph, padding=pack_padding)
            else:
                pos = nx.kamada_kawai_layout(graph)
        else:
            pos = {} # Empty position dict for empty graph

    # Apply offset to positions for drawing
    final_pos = {node: (x + offset[0], y + offset[1]) for node, (x, y) in pos.items()}

    # Determine node colors.
    node_colors: List[Any] = []
    cmap_name = style.get('cmap', 'hsv')
    for node, data in graph.nodes(data=True):
        label = data.get('label', None)
        # Treat None as missing label to avoid hashing the string "None"
        if label is not None:
            node_colors.append(get_color(label, cmap_name=cmap_name))
        else:
            node_colors.append(get_color(stable_hash(str(node)), cmap_name=cmap_name))

    # Draw the graph nodes (optional).
    if display_nodes:
        nx.draw_networkx_nodes(
            graph, final_pos,
            node_size=style['node_size'],
            alpha=style['node_alpha'],
            linewidths=style['node_border_width'],
            node_color=node_colors,
            edgecolors=style['node_edgecolors'],
            ax=ax
        )

    # Draw the graph edges.
    nx.draw_networkx_edges(
        graph, final_pos,
        width=style['edge_width'],
        style=style['edge_style'],
        edge_color=style['edge_color'],
        ax=ax
    )

    # Optional node labels
    if node_labels:
        try:
            node_lbls = {n: str(data.get(node_label_attr, n)) for n, data in graph.nodes(data=True)}
        except Exception:
            node_lbls = {n: str(n) for n in graph.nodes()}
        nx.draw_networkx_labels(graph, final_pos, labels=node_lbls, font_size=node_label_font_size, ax=ax)

    # Optional edge labels (only for edges that have the attribute)
    if edge_labels:
        try:
            edge_lbls = {(u, v): str(d[edge_label_attr]) for u, v, d in graph.edges(data=True) if edge_label_attr in d}
        except Exception:
            edge_lbls = {}
        if edge_lbls:
            nx.draw_networkx_edge_labels(graph, final_pos, edge_labels=edge_lbls, font_size=edge_label_font_size, ax=ax)

    # Keep rendered content away from subplot borders so enclosing frames do not
    # touch nodes or labels, especially in the mapping grid.
    bounds = _layout_bounds(final_pos)
    if fit_viewport and bounds is not None:
        min_x, max_x, min_y, max_y = bounds
        span_x = max_x - min_x
        span_y = max_y - min_y
        if span_x <= 0:
            span_x = 1.0
        if span_y <= 0:
            span_y = 1.0
        viewport_padding_px = style.get('viewport_padding_px')
        viewport_padding_axes_fraction = style.get('viewport_padding_axes_fraction')
        if viewport_padding_px is not None or viewport_padding_axes_fraction is not None:
            fig = ax.figure
            fig_width_px, fig_height_px = fig.get_size_inches() * fig.dpi
            bbox = ax.get_position()
            ax_width_px = max(bbox.width * fig_width_px, 1.0)
            ax_height_px = max(bbox.height * fig_height_px, 1.0)
            explicit_pad_px = max(0.0, float(viewport_padding_px or 0.0))
            min_pad_px = 0.0
            if viewport_padding_axes_fraction is not None:
                min_pad_px = max(
                    0.0,
                    min(ax_width_px, ax_height_px) * float(viewport_padding_axes_fraction),
                )
            content_inset_px = max(0.0, float(style.get('content_inset_px', 0.0)))
            pad_px = max(explicit_pad_px, min_pad_px) + content_inset_px
            pad_x = span_x * pad_px / max(ax_width_px - 2 * pad_px, 1.0)
            pad_y = span_y * pad_px / max(ax_height_px - 2 * pad_px, 1.0)
        else:
            pad_fraction = max(0.0, float(style.get('viewport_padding', 0.12)))
            pad_x = span_x * pad_fraction
            pad_y = span_y * pad_fraction

        # Reserve room for the actual rendered marker radius so nodes stay
        # inside enclosing frames even in very small subplot cells.
        fig = ax.figure
        fig_width_px, fig_height_px = fig.get_size_inches() * fig.dpi
        bbox = ax.get_position()
        ax_width_px = max(bbox.width * fig_width_px, 1.0)
        ax_height_px = max(bbox.height * fig_height_px, 1.0)
        node_size_pts2 = max(0.0, float(style.get('node_size', 70)))
        marker_radius_px = math.sqrt(node_size_pts2 / math.pi) * (fig.dpi / 72.0)
        marker_pad_x = span_x * marker_radius_px / max(ax_width_px - 2 * marker_radius_px, 1.0)
        marker_pad_y = span_y * marker_radius_px / max(ax_height_px - 2 * marker_radius_px, 1.0)
        pad_x = max(pad_x, marker_pad_x)
        pad_y = max(pad_y, marker_pad_y)
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

    if created_ax:
        ax.axis('off')
        plt.show()
        return None
    else:
        return ax

# --- Refactored display Function ---
def display(
    abstract_graph: Union["AbstractGraph", Iterable[Any]],
    base_style: Optional[Dict[str, Any]] = None,
    abstract_style: Optional[Dict[str, Any]] = None,
    connection_style: Optional[Dict[str, Any]] = None,
    size: Tuple[int, int] = (5, 4),
    ax: Optional[plt.Axes] = None, # Use plt.Axes for type hint
    show_legend: bool = False
) -> Optional[Union[plt.Axes, Figure]]:
    """
    Visualizes the full nested structure of a AbstractGraph using display_graph.

    The leftmost level is the base graph and the abstract level
    (interpretation_graph) is drawn in an additional column to the right.
    Connection lines are drawn between an interpretation node and every base node
    that appears in its mapped subgraph.

    Node colors are assigned consistently based on their numerical labels or stable hash.

    Args:
        abstract_graph: The AbstractGraph instance to visualize, or an iterable of graphs.
        base_style: A dict of style parameters for drawing the base graph.
        abstract_style: A dict of style parameters for drawing the abstract graph.
        connection_style: A dict of style parameters for drawing connection lines.
        size: The figure size as a tuple (width, height) if `ax` is None.
        ax: A Matplotlib axis to draw on. If None, a new figure and axis are created.
        show_legend: If True, displays a legend.

    Returns:
        The Matplotlib axis containing the visualization (or a Figure if multiple graphs are provided).
    """
    if not isinstance(abstract_graph, AbstractGraph):
        if isinstance(abstract_graph, (list, tuple)):
            return display_graphs(
                abstract_graph,
                base_style=base_style,
                abstract_style=abstract_style,
                connection_style=connection_style,
                size=size,
                show=False,
                show_legend=show_legend,
            )
        raise TypeError("display expects an AbstractGraph or a list of graphs.")
    # Set default styles (can be simplified if defaults are handled in display_graph)
    if base_style is None:
        base_style = {
            'node_size': 70, 'edge_width': 1.0, 'edge_style': 'solid',
            'node_border_width': 0.5, 'node_alpha': 0.8, 'edge_color': 'grey', 'cmap': 'hsv'
        }
    if abstract_style is None:
        abstract_style = {
            'node_size': 100, 'edge_width': 2.0, 'edge_style': 'solid',
            'node_border_width': 2.0, 'node_alpha': 0.9, 'edge_color': 'black', 'cmap': 'hsv'
        }
    if connection_style is None:
        connection_style = {
            'edge_width': 0.5, 'edge_style': 'dashed',
            'edge_color': 'grey', 'edge_alpha': 0.3
        }
    # Ensure connection style defaults
    connection_style.setdefault('edge_width', 0.5)
    connection_style.setdefault('edge_style', 'dashed')
    connection_style.setdefault('edge_color', 'grey')
    connection_style.setdefault('edge_alpha', 0.3)

    # Create figure and axis if not provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # --- Calculate Layouts ---
    # Calculate base positions. Handle empty graph.
    if abstract_graph.base_graph.number_of_nodes() > 0:
        pos_base = _packed_kamada_kawai_layout(abstract_graph.base_graph)
    else:
        pos_base = {}

    # Calculate abstract positions. Handle empty graph.
    if abstract_graph.interpretation_graph.number_of_nodes() > 0:
        pos_abstract = _packed_kamada_kawai_layout(abstract_graph.interpretation_graph)
    else:
        pos_abstract = {}

    # Normalize both layouts to the same box so they occupy equal area.
    base_bounds = _layout_bounds(pos_base)
    abstract_bounds = _layout_bounds(pos_abstract)
    if base_bounds is not None:
        base_height = base_bounds[3] - base_bounds[2]
        base_width = base_bounds[1] - base_bounds[0]
    else:
        base_height = 1.0
        base_width = 1.0
    if abstract_bounds is not None:
        abstract_height = abstract_bounds[3] - abstract_bounds[2]
        abstract_width = abstract_bounds[1] - abstract_bounds[0]
    else:
        abstract_height = 1.0
        abstract_width = 1.0
    target_height = max(base_height, abstract_height, 1.0)
    target_width = max(base_width, abstract_width, 1.0)
    if pos_base:
        pos_base = _normalize_layout_to_box(pos_base, target_width=target_width, target_height=target_height)
    if pos_abstract:
        pos_abstract = _normalize_layout_to_box(pos_abstract, target_width=target_width, target_height=target_height)

    # --- Determine Offset ---
    # Shift the abstract graph positions to the right.
    if pos_base:
        # Find max x-coordinate, handle potential empty pos_base values if layout failed partially
        valid_x_coords = [x for x, _ in pos_base.values() if isinstance(x, (int, float))]
        max_x = max(valid_x_coords) if valid_x_coords else 0
        min_x = min([x for x, _ in pos_base.values()]) if valid_x_coords else 0
        panel_width = max_x - min_x
    else:
        panel_width = target_width
    x_offset = panel_width + 1.5  # leave some space between graphs

    # --- Draw Graphs using Helper ---
    # Draw the base graph (no offset)
    ax = display_graph(
        abstract_graph.base_graph,
        ax=ax,
        style=base_style,
        pos=pos_base,
        offset=(0, 0), # Explicitly no offset
        fit_viewport=False,
    )

    # Draw the abstract graph (with x_offset)
    abstract_edgecolors = []
    for _node, data in abstract_graph.interpretation_graph.nodes(data=True):
        mapped_subgraph = data.get("mapped_subgraph", data.get("association"))
        if isinstance(mapped_subgraph, nx.Graph) and mapped_subgraph.number_of_nodes() > 0:
            abstract_edgecolors.append("black")
        else:
            abstract_edgecolors.append("white")
    ax = display_graph(
        abstract_graph.interpretation_graph,
        ax=ax,
        style={**abstract_style, "node_edgecolors": abstract_edgecolors},
        pos=pos_abstract, # Pass original positions
        offset=(x_offset, 0), # Apply offset
        fit_viewport=False,
    )

    # --- Draw Connection Lines ---
    # Need the final, offsetted positions for the abstract graph for connections
    final_pos_abstract = {node: (x + x_offset, y) for node, (x, y) in pos_abstract.items()}

    for anode, adata in abstract_graph.interpretation_graph.nodes(data=True):
        subg = adata.get("mapped_subgraph", adata.get("association"))
        if subg is None:
            continue
        # Ensure qnode exists in the final positions (handles empty abstract graph case)
        if anode not in final_pos_abstract:
            continue
        qnode_pos = final_pos_abstract[anode]

        for base_node in subg.nodes:
            # Ensure base_node exists in base positions
            if base_node in pos_base:
                base_pos = pos_base[base_node]
                ax.plot(
                    [qnode_pos[0], base_pos[0]], # x coordinates
                    [qnode_pos[1], base_pos[1]], # y coordinates
                    linewidth=connection_style['edge_width'],
                    linestyle=connection_style['edge_style'],
                    color=connection_style['edge_color'],
                    alpha=connection_style['edge_alpha'],
                    zorder=-1 # Draw connections behind nodes
                )

    # Fit one combined viewport so both base and interpretation graphs remain visible.
    combined_pos = {
        ("base", node): coords for node, coords in pos_base.items()
    }
    combined_pos.update(
        {
            ("abstract", node): coords
            for node, coords in final_pos_abstract.items()
        }
    )
    combined_bounds = _layout_bounds(combined_pos)
    if combined_bounds is not None:
        min_x, max_x, min_y, max_y = combined_bounds
        span_x = max_x - min_x
        span_y = max_y - min_y
        if span_x <= 0:
            span_x = 1.0
        if span_y <= 0:
            span_y = 1.0
        viewport_padding = max(
            float(base_style.get("viewport_padding", 0.12)),
            float(abstract_style.get("viewport_padding", 0.12)),
        )
        pad_x = span_x * viewport_padding
        pad_y = span_y * viewport_padding
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

    # Optionally add a legend.
    if show_legend:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Base Node',
                   markerfacecolor='grey', markersize=8, linestyle='None'), # Added linestyle='None'
            Line2D([0], [0], marker='o', color='w', label='abstract Node',
                   markerfacecolor='grey', markersize=10, linestyle='None'), # Added linestyle='None'
            Line2D([0], [0], color=connection_style['edge_color'], lw=connection_style['edge_width'],
                   linestyle=connection_style['edge_style'], label='Mapping')
        ]
        ax.legend(handles=legend_elements, loc='best') # Added loc='best'

    ax.axis('off')
    return ax

# --- Small grid utility to draw multiple graphs ---
def display_graphs(
    graphs,
    *,
    titles=None,
    n_graphs_per_line: int = 7,
    style: Optional[Dict[str, Any]] = None,
    base_style: Optional[Dict[str, Any]] = None,
    abstract_style: Optional[Dict[str, Any]] = None,
    connection_style: Optional[Dict[str, Any]] = None,
    size: Tuple[int, int] = (4, 3),
    show: bool = True,
    show_legend: bool = False,
    display_nodes: bool = True,
    node_labels: bool = False,
    node_label_attr: str = 'label',
    node_label_font_size: int = 8,
    edge_labels: bool = False,
    edge_label_attr: str = 'label',
    edge_label_font_size: int = 7,
):
    """
    Draw multiple graphs arranged in a grid.

    Args:
        graphs: Iterable of AbstractGraph or NetworkX graphs to render.
        titles: Optional list of per-graph titles (len >= number of graphs).
        n_graphs_per_line: Number of subplots per row.
        style: Optional style dict passed to display_graph (NetworkX graphs only).
        base_style: Optional style dict for base graph (AbstractGraph only).
        abstract_style: Optional style dict for abstract graph (AbstractGraph only).
        connection_style: Optional style dict for mapping lines (AbstractGraph only).
        size: Size of each subplot (width, height) in inches.
        show: If True, call plt.show() at the end.
        show_legend: If True, add a legend for AbstractGraph displays.
        display_nodes: If False, do not draw node markers (circles). Labels can still be drawn.
        node_labels: If True, draw node labels for each graph.
        node_label_attr: Node attribute name to use for labels (fallback to node id).
        node_label_font_size: Font size for node labels.
        edge_labels: If True, draw edge labels for each graph.
        edge_label_attr: Edge attribute name to use for labels (edges missing it are skipped).
        edge_label_font_size: Font size for edge labels.

    Returns:
        Matplotlib Figure with a grid of subplots.
    """
    graphs = list(graphs)
    n = len(graphs)
    if n == 0:
        fig, ax = plt.subplots(figsize=(size[0], size[1]))
        ax.axis('off')
        if show:
            plt.show()
        return fig

    kind = _infer_graph_list_kind(graphs)
    if kind == "mixed":
        raise TypeError("display_graphs expects all AbstractGraph or all NetworkX graphs.")

    cols = max(1, int(n_graphs_per_line))
    rows = (n + cols - 1) // cols
    figsize = (size[0] * cols, size[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # Normalize axes to a flat list
    if hasattr(axes, 'ravel'):
        axes_list = list(axes.ravel())
    else:
        axes_list = [axes]

    for i in range(rows * cols):
        ax = axes_list[i]
        if i < n:
            G = graphs[i]
            if kind == "abstract":
                display(
                    G,
                    base_style=base_style,
                    abstract_style=abstract_style,
                    connection_style=connection_style,
                    ax=ax,
                    show_legend=show_legend,
                )
            else:
                display_graph(
                    G,
                    ax=ax,
                    style=style,
                    display_nodes=display_nodes,
                    node_labels=node_labels,
                    node_label_attr=node_label_attr,
                    node_label_font_size=node_label_font_size,
                    edge_labels=edge_labels,
                    edge_label_attr=edge_label_attr,
                    edge_label_font_size=edge_label_font_size,
                )
            if titles is not None and i < len(titles):
                ax.set_title(str(titles[i]))
            ax.axis('off')
        else:
            ax.axis('off')

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _draw_group_frames(
    fig: Figure,
    axes: List[plt.Axes],
    group_to_cells: Dict[Any, List[int]],
    *,
    n_cols: int,
    inner_gap: float = 0.008,
    border_lw: float = 0.8,
    border_color: str = "black",
    inner_border_lw: float = 0.5,
    inner_border_color: str = "0.7",
    title_font_size: int = 8,
    title_pad_px: float = 2.0,
    title_formatter: Optional[Callable[[Any], str]] = None,
    footer_formatter: Optional[Callable[[Any], Optional[str]]] = None,
    footer_font_size: int = 8,
    footer_pad_px: float = 2.0,
    fixed_inset_px: Optional[float] = None,
) -> None:
    """Draw boxed group boundaries and one title per group across subplot cells."""
    if title_formatter is None:
        title_formatter = lambda group_key: f"Label: {group_key}"

    fig_height_px = fig.get_size_inches()[1] * fig.dpi
    title_pad = title_pad_px / fig_height_px
    footer_pad = footer_pad_px / fig_height_px

    for group_key, cell_ids in group_to_cells.items():
        if not cell_ids:
            continue

        for cid in cell_ids:
            _add_axes_frame(
                fig,
                axes[cid],
                edge_color=inner_border_color,
                line_width=inner_border_lw,
                inset_x=inner_gap,
                inset_y=inner_gap,
                fixed_inset_px=fixed_inset_px,
            )

        row_segments: List[Tuple[int, int, int]] = []
        rows_map: Dict[int, List[int]] = {}
        for cid in cell_ids:
            row = cid // n_cols
            col = cid % n_cols
            rows_map.setdefault(row, []).append(col)
        for row in sorted(rows_map.keys()):
            cols = sorted(rows_map[row])
            row_segments.append((row, cols[0], cols[-1]))

        first_row = row_segments[0][0]
        last_row = row_segments[-1][0]
        first_seg_row, first_seg_col0, first_seg_col1 = row_segments[0]
        first_ax_left = axes[first_seg_row * n_cols + first_seg_col0]
        first_ax_right = axes[first_seg_row * n_cols + first_seg_col1]
        first_left = first_ax_left.get_position().x0
        first_right = first_ax_right.get_position().x1
        first_top = first_ax_left.get_position().y1

        fig.text(
            0.5 * (first_left + first_right),
            first_top - title_pad,
            title_formatter(group_key),
            ha="center",
            va="top",
            fontsize=title_font_size,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )

        if footer_formatter is not None:
            footer_text = footer_formatter(group_key)
            if footer_text:
                last_seg_row, last_seg_col0, last_seg_col1 = row_segments[-1]
                last_ax_left = axes[last_seg_row * n_cols + last_seg_col0]
                last_ax_right = axes[last_seg_row * n_cols + last_seg_col1]
                last_left = last_ax_left.get_position().x0
                last_right = last_ax_right.get_position().x1
                last_bottom = last_ax_left.get_position().y0
                fig.text(
                    0.5 * (last_left + last_right),
                    last_bottom + footer_pad,
                    footer_text,
                    ha="center",
                    va="bottom",
                    fontsize=footer_font_size,
                    color="0.25",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
                )

        for row, col0, col1 in row_segments:
            left_ax = axes[row * n_cols + col0]
            right_ax = axes[row * n_cols + col1]
            bbox_l = left_ax.get_position()
            bbox_r = right_ax.get_position()
            x0, x1 = bbox_l.x0, bbox_r.x1
            y0, y1 = bbox_l.y0, bbox_l.y1

            fig.add_artist(Line2D([x0, x1], [y1, y1], transform=fig.transFigure, color=border_color, linewidth=border_lw))
            fig.add_artist(Line2D([x0, x1], [y0, y0], transform=fig.transFigure, color=border_color, linewidth=border_lw))

            has_next_row = row < last_row
            open_right = has_next_row and (col1 == n_cols - 1)
            if not open_right:
                fig.add_artist(
                    Line2D([x1, x1], [y0, y1], transform=fig.transFigure, color=border_color, linewidth=border_lw)
                )

            has_prev_row = row > first_row
            open_left = has_prev_row and (col0 == 0)
            if not open_left:
                fig.add_artist(
                    Line2D([x0, x0], [y0, y1], transform=fig.transFigure, color=border_color, linewidth=border_lw)
                )


def display_grouped_graphs(
    grouped_graphs: Union[Dict[Any, List[nx.Graph]], Iterable[Tuple[Any, List[nx.Graph]]]],
    *,
    n_graphs_per_line: int = 7,
    size: Tuple[float, float] = (3.0, 3.0),
    style: Optional[Dict[str, Any]] = None,
    title_formatter: Optional[Callable[[Any], str]] = None,
    show: bool = True,
) -> Figure:
    """Draw grouped NetworkX graphs with one boxed region and title per group."""
    if isinstance(grouped_graphs, dict):
        grouped_items = list(grouped_graphs.items())
    else:
        grouped_items = list(grouped_graphs)

    flattened_cells: List[Tuple[Any, nx.Graph]] = [
        (group_key, graph)
        for group_key, graphs in grouped_items
        for graph in graphs
    ]
    if not flattened_cells:
        fig, ax = plt.subplots(figsize=size)
        ax.axis("off")
        if show:
            plt.show()
        return fig

    n_cols = max(1, int(n_graphs_per_line))
    n_rows = max(1, math.ceil(len(flattened_cells) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * size[0], n_rows * size[1]))

    if isinstance(axes, plt.Axes):
        axes_list = [axes]
    elif n_rows == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row in axes for ax in row]

    group_to_cells: Dict[Any, List[int]] = {}
    for idx, (group_key, graph) in enumerate(flattened_cells):
        ax = axes_list[idx]
        display_graph(graph, ax=ax, style=style)
        ax.axis("off")
        group_to_cells.setdefault(group_key, []).append(idx)

    for idx in range(len(flattened_cells), len(axes_list)):
        axes_list[idx].axis("off")

    plt.tight_layout()
    _draw_group_frames(
        fig,
        axes_list,
        group_to_cells,
        n_cols=n_cols,
        title_formatter=title_formatter,
    )
    if show:
        plt.show()
    return fig


def display_mappings(
    abstract_graph: "AbstractGraph",
    subgraph_style: Optional[Dict[str, Any]] = None,
    n_elements_per_row: int = 15,
    size: float = 2.0,
    n_nodes_for_larger_size: int = 20,
    larger_inner_plot_size: Tuple[float, float] = (5.0, 5.0),
    fixed_inner_inset_px: float = 14.0,
) -> None:
    """
    Visualize mapping instances grouped by image-node label.

    All mapped base subgraphs are rendered (one instance per cell), grouped by
    interpretation-node label and sorted by descending group frequency. Each label group receives
    a single title and a thin black frame around its full instance array. If a label
    group wraps to the next row, the frame remains open at the end of the previous row
    and the beginning of the continuation row.

    Args:
        abstract_graph: The AbstractGraph instance to visualize.
        subgraph_style: A dict of style parameters for drawing each subgraph.
        n_elements_per_row: Number of displayed instances per row.
        size: The size (in inches) for each individual subgraph display.
        n_nodes_for_larger_size: Node-count threshold above which a denser
            rendering scale is used for the inner graph drawing.
        larger_inner_plot_size: Virtual inner drawing size used to derive a
            smaller node/edge scale for dense subgraphs while keeping the cell
            size unchanged.
        fixed_inner_inset_px: Fixed inset in pixels used for the inner bbox and
            matching graph padding.

    Returns:
        None.
    """
    if abstract_graph is None or abstract_graph.interpretation_graph is None or len(abstract_graph.interpretation_graph.nodes) == 0:
        print("[display_mappings] Empty abstract graph — nothing to display.")
        return

    def _operator_text_from_meta(node_data: Dict[str, Any]) -> Optional[str]:
        meta = node_data.get("meta", {})
        operator_text = meta.get("user_name")
        if operator_text is None:
            operator_text = meta.get("source_chain")
        if operator_text is None:
            return None
        if not isinstance(operator_text, str):
            operator_text = str(operator_text)
        return operator_text

    # Set default subgraph style if not provided.
    if subgraph_style is None:
        subgraph_style = {
            'node_size': 70,
            'edge_width': 1.0,
            'edge_style': 'solid',
            'node_border_width': 0.5,
            'node_alpha': 0.8,
            'edge_color': 'grey',
            'cmap': 'hsv'
        }
    else:
        subgraph_style.setdefault('cmap', 'hsv')
    
    # Group interpretation nodes by their existing label. Within each label
    # group, use a 19-bit graph hash to distinguish true isomorphic copies.
    mapping_dict: Dict[Any, Dict[Tuple[int, Optional[str]], List[nx.Graph]]] = {}
    label_to_operator_text: Dict[Any, Optional[str]] = {}
    for node, data in abstract_graph.interpretation_graph.nodes(data=True):
        mapped_subgraph = data.get("mapped_subgraph", data.get("association"))
        if mapped_subgraph is None:
            continue
        label = data.get("label")
        if label is None:
            label = stable_hash(str(node))
        iso_hash = hash_graph(mapped_subgraph, nbits=19)
        operator_text = _operator_text_from_meta(data)
        if label not in label_to_operator_text:
            label_to_operator_text[label] = operator_text
        mapping_dict.setdefault(label, {}).setdefault((iso_hash, operator_text), []).append(mapped_subgraph)

    # Sort label groups by total frequency descending. Within each label group,
    # collapse exact copies into one displayed cell with a multiplicity marker.
    sorted_mappings = sorted(
        mapping_dict.items(),
        key=lambda item: sum(len(group) for group in item[1].values()),
        reverse=True,
    )
    n_cols = max(1, int(n_elements_per_row))
    expanded_cells: List[Tuple[Any, nx.Graph, int, Optional[str]]] = []
    for label, iso_groups in sorted_mappings:
        for (_iso_hash, operator_text), subgraph_list in sorted(
            iso_groups.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        ):
            expanded_cells.append((label, subgraph_list[0], len(subgraph_list), operator_text))

    n_cells = len(expanded_cells)
    n_rows = max(1, math.ceil(n_cells / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * size, n_rows * size))
    fig_width_px, fig_height_px = fig.get_size_inches() * fig.dpi
    inner_gap = 0.008
    subgraph_style = {
        **subgraph_style,
        "viewport_padding_px": fixed_inner_inset_px,
    }
    dense_subgraph_style = dict(subgraph_style)
    base_inner_extent = max(float(size), 1.0)
    dense_inner_extent = max(float(max(larger_inner_plot_size)), base_inner_extent)
    dense_scale = min(1.0, base_inner_extent / dense_inner_extent)
    dense_subgraph_style["node_size"] = max(8.0, float(subgraph_style.get("node_size", 70)) * (dense_scale ** 2))
    dense_subgraph_style["node_border_width"] = max(
        0.3, float(subgraph_style.get("node_border_width", 0.5)) * dense_scale
    )
    dense_subgraph_style["edge_width"] = max(0.4, float(subgraph_style.get("edge_width", 1.0)) * dense_scale)

    # Flatten axes array to a 1D list.
    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    # Draw each instance in its own cell.
    # Track where each label appears so we can draw one grouped frame and one title.
    label_to_cells: Dict[Any, List[int]] = {}
    cell_copy_counts: Dict[int, int] = {}
    cell_draw_specs: Dict[int, Tuple[nx.Graph, Dict[str, Any]]] = {}
    for i, (label, subgraph, copy_count, operator_text) in enumerate(expanded_cells):
        ax = axes[i]
        style_for_subgraph = dense_subgraph_style if subgraph.number_of_nodes() >= n_nodes_for_larger_size else subgraph_style
        display_graph(subgraph, ax=ax, style=style_for_subgraph)
        ax.axis("off")
        label_to_cells.setdefault(label, []).append(i)
        cell_copy_counts[i] = copy_count
        cell_draw_specs[i] = (subgraph, style_for_subgraph)

    # Hide any unused subplots.
    for j in range(n_cells, len(axes)):
        axes[j].axis("off")

    # Finalize axis positions before drawing figure-level frames and titles.
    plt.tight_layout()
    fig_width_px, fig_height_px = fig.get_size_inches() * fig.dpi

    # Recompute graph viewport padding using the final axes geometry after
    # tight_layout(), otherwise nodes can spill past the inner frame.
    for i in range(n_cells):
        ax = axes[i]
        subgraph, style_for_subgraph = cell_draw_specs[i]
        effective_inset_px = _effective_axes_inset_px(fig, ax, inset_x=inner_gap, inset_y=inner_gap)
        ax.clear()
        final_style = {
            **style_for_subgraph,
            "content_inset_px": fixed_inner_inset_px,
        }
        display_graph(subgraph, ax=ax, style=final_style)
        ax.axis("off")

    show_operator_footer = getattr(getattr(abstract_graph, "label_function", None), "label_mode", None) == "operator_hash"

    def _footer_formatter(label: Any) -> Optional[str]:
        if not show_operator_footer:
            return None
        operator_text = label_to_operator_text.get(label)
        if not operator_text:
            return None
        max_chars = 72
        footer_text = operator_text if len(operator_text) <= max_chars else operator_text[: max_chars - 3] + "..."
        return f"Operator: {footer_text}"

    for cid in range(n_cells):
        copy_count = cell_copy_counts.get(cid, 1)
        bbox = axes[cid].get_position()
        inner_inset_px = _effective_axes_inset_px(
            fig,
            axes[cid],
            inset_x=inner_gap,
            inset_y=inner_gap,
            fixed_inset_px=fixed_inner_inset_px,
        )
        if copy_count > 1:
            count_pad_x = 4.0 / fig_width_px
            count_pad_y = 2.0 / fig_height_px
            fig.text(
                bbox.x1 - (inner_inset_px / fig_width_px) - count_pad_x,
                bbox.y0 + count_pad_y,
                f"#{copy_count}",
                ha="right",
                va="bottom",
                fontsize=7,
                color="0.35",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
            )

    _draw_group_frames(
        fig,
        axes,
        label_to_cells,
        n_cols=n_cols,
        inner_gap=inner_gap,
        title_formatter=lambda label: f"Label: {label}",
        footer_formatter=_footer_formatter,
        fixed_inset_px=fixed_inner_inset_px,
    )

    plt.show()

# ===========================
# UTILITY FUNCTIONS
# ===========================
def get_underlying_function(f):
    """
    Unwraps a function to retrieve its underlying callable.

    Args:
        f: Possibly wrapped callable.

    Returns:
        Callable: Underlying function object.
    """
    if hasattr(f, 'func'):
        return f.func
    if hasattr(f, '__wrapped__'):
        return f.__wrapped__
    return f

_GENERIC_OPERATOR_LABELS = {
    "add",
    "compose",
    "forward_compose",
    "product",
    "compose_product",
}

def _named_decomposition_label(comp_func):
    """
    Extract an explicit decomposition name declared with name(text=...).

    Args:
        comp_func: Composite function or operator.

    Returns:
        Optional[str]: User-provided decomposition name, if present.
    """
    op_type = getattr(comp_func, "operator_type", None)
    if op_type == "name":
        params = getattr(comp_func, "params", None)
        if isinstance(params, dict):
            text = params.get("text")
            if isinstance(text, str) and text:
                return text

    chain = getattr(comp_func, "chain", None)
    if isinstance(chain, (list, tuple)):
        for item in chain:
            label = _named_decomposition_label(item)
            if isinstance(label, str) and label:
                return label
    return None

def _edge_label_for_decomposition(comp_func):
    """
    Determine a display label for a decomposition operator edge.

    Args:
        comp_func: Composite function or operator.

    Returns:
        Optional[str]: Label to attach to the edge, if any.
    """
    label = _named_decomposition_label(comp_func)
    if isinstance(label, str) and label:
        return label
    label = getattr(comp_func, "name", None)
    if isinstance(label, str) and label:
        return label
    label = getattr(comp_func, "operator_type", None)
    if isinstance(label, str) and label and label not in _GENERIC_OPERATOR_LABELS:
        return label
    underlying = get_underlying_function(comp_func)
    name = getattr(underlying, "__name__", None)
    if isinstance(name, str) and name and name not in _GENERIC_OPERATOR_LABELS:
        return name
    return None

# Global constant for the initial input node.
GLOBAL_INPUT_NODE = "AbstractGraph"

# ===========================
# PARAMETER SUBGRAPH BUILDING
# ===========================
def build_parameter_subgraph(param_func, G, global_input):
    """
    For a callable parameter value:
      - If it accepts only one argument, assume it uses the global input.
      - Otherwise, recursively build its full decomposition subgraph.

    Args:
        param_func: Callable parameter value.
        G: Decomposition graph being built.
        global_input: Key for the global input node.

    Returns:
        str: Key of the output node for the parameter subgraph.
    """
    if hasattr(param_func, '__code__') and param_func.__code__.co_argcount == 1:
        return global_input
    else:
        dummy_key = f"ParamInput_{id(param_func)}"
        # Ensure the global input node is present.
        if not G.has_node(GLOBAL_INPUT_NODE):
            G.add_node(GLOBAL_INPUT_NODE, data_type="global", label="AbstractGraph")
        else:
            dummy_key = GLOBAL_INPUT_NODE  # reuse global input
        output_key = build_decomposition_subgraph(param_func, dummy_key, G)
        return output_key

def add_parameters(G, parent_key, comp_func, global_input):
    """
    For each parameter of comp_func, creates two nodes:
      - A parameter value node (data_type "value").
      - A parameter name node (data_type "parameter").
    Then links the value node to the name node and connects the name node to parent_key.

    Args:
        G: Decomposition graph being built.
        parent_key: Key of the parent node.
        comp_func: Composite function with parameters.
        global_input: Key for the global input node.

    Returns:
        None.
    """
    # Process keyword parameters.
    if hasattr(comp_func, 'keywords') and comp_func.keywords:
        for key, value in comp_func.keywords.items():
            if callable(value):
                value_node_key = build_parameter_subgraph(value, G, global_input)
            else:
                # Use a composite key to ensure each occurrence is unique.
                value_node_key = f"param_value_{key}_{id(comp_func)}_{id(value)}"
                G.add_node(value_node_key, data_type="value", label=_format_param_value_for_label(value))
            param_name_key = f"param_name_{key}_{id(comp_func)}"
            G.add_node(param_name_key, data_type="parameter", label=key)
            G.add_edge(value_node_key, param_name_key)
            G.add_edge(param_name_key, parent_key)
    # Process positional parameters.
    if hasattr(comp_func, 'args') and comp_func.args:
        for i, arg in enumerate(comp_func.args):
            if callable(arg):
                value_node_key = build_parameter_subgraph(arg, G, global_input)
            else:
                value_node_key = f"param_value_arg{i}_{id(comp_func)}_{id(arg)}"
                G.add_node(value_node_key, data_type="value", label=_format_param_value_for_label(arg))
            param_name_key = f"param_name_arg{i}_{id(comp_func)}"
            G.add_node(param_name_key, data_type="parameter", label=f"arg{i}")
            G.add_edge(value_node_key, param_name_key)
            G.add_edge(param_name_key, parent_key)

# ===========================
# DECOMPOSITION GRAPH BUILDING
# ===========================
def build_decomposition_subgraph(comp_func, input_node, G):
    """
    Recursively builds a subgraph for a composite function (comp_func) starting at input_node.

    Operator handling:
      - For operators (with attribute operator_type "add" or "product"),
        create an operator node (data_type "operator") and recursively add its children.

    Composition flattening:
      - For compose/forward_compose calls (identified by their __name__ and chain attribute),
        flatten the chain (reversing for "compose" and preserving order for "forward_compose").

    Leaf nodes:
      - If comp_func is callable, create a node with data_type "function"; otherwise, "value".
      - Parameter nodes are added using add_parameters.
      
    Returns the key of the output node.

    Args:
        comp_func: Composite function or operator to expand.
        input_node: Key of the input node for this subgraph.
        G: Decomposition graph being built.

    Returns:
        str: Key of the output node.
    """
    # --- Operator handling ---
    if hasattr(comp_func, 'operator_type'):
        op_type = comp_func.operator_type
        if op_type == "add":
            operator_key = f"operator_add_{id(comp_func)}"
            G.add_node(operator_key, data_type="operator", label="+")
            for child in comp_func.decomposition_functions:
                child_output = build_decomposition_subgraph(child, input_node, G)
                edge_label = _edge_label_for_decomposition(child)
                if edge_label:
                    G.add_edge(child_output, operator_key, label=edge_label)
                else:
                    G.add_edge(child_output, operator_key)
            return operator_key
        elif op_type == "product":
            combiner = comp_func.combiner
            underlying_combiner = get_underlying_function(combiner)
            combiner_name = getattr(underlying_combiner, '__name__', repr(underlying_combiner))
            operator_key = f"operator_product_{id(comp_func)}"
            G.add_node(operator_key, data_type="operator", label=combiner_name)
            if hasattr(combiner, 'keywords') or hasattr(combiner, 'args'):
                add_parameters(G, operator_key, combiner, input_node)
            for child in comp_func.decomposition_functions:
                child_output = build_decomposition_subgraph(child, input_node, G)
                edge_label = _edge_label_for_decomposition(child)
                if edge_label:
                    G.add_edge(child_output, operator_key, style="bold", penwidth="3", label=edge_label)
                else:
                    G.add_edge(child_output, operator_key, style="bold", penwidth="3")
            return operator_key

    # --- Composition flattening ---
    underlying = get_underlying_function(comp_func)
    if getattr(underlying, '__name__', None) in ("compose", "forward_compose") and hasattr(comp_func, "chain"):
        funcs = list(comp_func.chain)
        chain = list(reversed(funcs)) if underlying.__name__ == "compose" else funcs
    else:
        # Attempt to extract a function chain from the closure if available.
        funcs = []
        if inspect.isfunction(underlying):
            try:
                closure = inspect.getclosurevars(underlying)
                for _, val in closure.nonlocals.items():
                    if isinstance(val, tuple) and all(callable(x) for x in val):
                        funcs = list(val)
                        break
            except Exception:
                funcs = []
        chain = funcs

    if chain:
        current_input = input_node
        for f in chain:
            # If the function is an operator, recursively build its subgraph.
            if hasattr(f, 'operator_type'):
                child_output = build_decomposition_subgraph(f, current_input, G)
            else:
                underlying_f = get_underlying_function(f)
                func_name = getattr(underlying_f, '__name__', repr(underlying_f))
                node_key = f"{func_name}_{id(f)}"
                G.add_node(node_key, data_type="function", label=func_name)
                G.add_edge(current_input, node_key)
                if hasattr(f, 'keywords') or hasattr(f, 'args'):
                    add_parameters(G, node_key, f, input_node)
                child_output = node_key
            current_input = child_output
        return current_input
    else:
        # --- Leaf node handling ---
        real_name = getattr(underlying, '__name__', repr(underlying))
        if real_name == "AbstractGraph":
            return GLOBAL_INPUT_NODE
        leaf_key = f"{real_name}_{id(comp_func)}"
        data_type = "function" if callable(comp_func) else "value"
        G.add_node(leaf_key, data_type=data_type, label=real_name)
        G.add_edge(input_node, leaf_key)
        if hasattr(comp_func, 'keywords') or hasattr(comp_func, 'args'):
            add_parameters(G, leaf_key, comp_func, input_node)
        return leaf_key

def decomposition_to_graph(comp_func) -> nx.DiGraph:
    """
    Builds a full decomposition graph for comp_func.
    The graph starts with a single global input node and builds the subgraph recursively.

    Args:
        comp_func: Composite function or operator.

    Returns:
        nx.DiGraph: Decomposition graph.
    """
    G = nx.DiGraph()
    G.add_node(GLOBAL_INPUT_NODE, data_type="global", label="AGraph")
    build_decomposition_subgraph(comp_func, GLOBAL_INPUT_NODE, G)
    return G

# ===========================
# GRAPH DISPLAY FUNCTION
# ===========================
def display_decomposition_graph(comp_func_or_graph, output_file: str = "decomposition_graph.png", figsize=(12, 8)) -> None:
    """
    Draws and displays the decomposition graph for a composite function.
    
    Node data_types and their corresponding visual styles:
      - "global": Global input node.
      - "value": Parameter value node.
      - "function": Function node.
      - "parameter": Parameter name node.
      - "operator": Operator node.
    
    The graph is laid out using Graphviz's dot (bottom-to-top orientation) and saved to output_file.

    Args:
        comp_func_or_graph: Composite function or prebuilt decomposition graph.
        output_file: Output image path for Graphviz rendering.
        figsize: Matplotlib figure size.

    Returns:
        None.
    """
    # Mapping from our data_type to Graphviz attributes.
    DATA_TYPE_STYLES = {
        "global": {"shape": "circle", "fillcolor": "#98FB98"},
        "value": {"shape": "circle", "fillcolor": "#FFFFF0"},
        "function": {"shape": "rectangle", "fillcolor": "#A4D3EE"},
        "parameter": {"shape": "oval", "fillcolor": "#FFC125"},
        "operator": {"shape": "hexagon", "fillcolor": "#B0C4DE"},
    }
    
    if isinstance(comp_func_or_graph, nx.DiGraph):
        G = comp_func_or_graph
    else:
        G = decomposition_to_graph(comp_func_or_graph)
    
    try:
        A = to_agraph(G)

        for n in A.nodes():
            node_name = n.get_name()
            display_label = G.nodes[node_name].get("label", node_name)
            # Use "data_type" instead of "shape"
            data_type = G.nodes[node_name].get("data_type", "parameter")
            style = DATA_TYPE_STYLES.get(data_type, {"shape": "oval", "fillcolor": "grey"})

            n.attr['shape'] = style["shape"]
            n.attr['style'] = 'filled'
            n.attr['fillcolor'] = style["fillcolor"]
            n.attr['label'] = display_label

            if data_type == "operator":
                n.attr['fontsize'] = '14'

        # Set edge thickness based on the data_type of the tail node.
        for edge in A.edges():
            tail = edge[0]
            head = edge[1]
            tail_data_type = G.nodes[tail].get("data_type", "parameter")
            if tail_data_type == "function":
                if not edge.attr.get("penwidth"):
                    edge.attr["penwidth"] = "3"
            elif tail_data_type in ("parameter", "value"):
                if not edge.attr.get("penwidth"):
                    edge.attr["penwidth"] = "1"
            else:
                if not edge.attr.get("penwidth"):
                    edge.attr["penwidth"] = "3"
            edge_label = G.edges[tail, head].get("label")
            if edge_label:
                edge.attr["label"] = edge_label
                if not edge.attr.get("fontsize"):
                    edge.attr["fontsize"] = "10"

        A.graph_attr.update(
            rankdir="BT",
            nodesep="0.8",
            ranksep="1.2",
            splines="true",
            overlap="false"
        )

        A.layout(prog="dot")
        A.draw(output_file)

        if os.path.exists(output_file):
            img = mpimg.imread(output_file)
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        else:
            print("Error: output file was not created.")
        return
    except ImportError:
        pass

    # Fallback when Graphviz bindings are unavailable: use a layered NetworkX layout.
    levels = {node: 0 for node in G.nodes}
    for generation_idx, generation in enumerate(nx.topological_generations(G)):
        for node in generation:
            levels[node] = generation_idx

    grouped: Dict[int, List[Any]] = {}
    for node, level in levels.items():
        grouped.setdefault(level, []).append(node)

    pos: Dict[Any, Tuple[float, float]] = {}
    max_width = max((len(nodes) for nodes in grouped.values()), default=1)
    for level, nodes in grouped.items():
        ordered_nodes = sorted(nodes, key=str)
        width = len(ordered_nodes)
        x_offset = (max_width - width) / 2.0
        for idx, node in enumerate(ordered_nodes):
            pos[node] = (x_offset + idx, -level)

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_axis_off()

    node_colors = [
        DATA_TYPE_STYLES.get(G.nodes[node].get("data_type", "parameter"), {"fillcolor": "grey"})["fillcolor"]
        for node in G.nodes
    ]
    edge_widths = [
        1.0 if G.nodes[tail].get("data_type", "parameter") in ("parameter", "value") else 2.5
        for tail, _head in G.edges
    ]
    labels = {node: G.nodes[node].get("label", node) for node in G.nodes}
    edge_labels = {
        (tail, head): G.edges[tail, head]["label"]
        for tail, head in G.edges
        if G.edges[tail, head].get("label")
    }

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        width=edge_widths,
        edge_color="#4a5568",
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=2200,
        edgecolors="#1f2937",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=10)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=9)
    plt.tight_layout()
    plt.show()
