import matplotlib

from abstractgraph.display import get_color, stable_hash


def test_get_color_uses_current_matplotlib_colormap_registry() -> None:
    color = get_color("node-label", cmap_name="hsv")

    assert len(color) == 4
    assert all(0.0 <= channel <= 1.0 for channel in color)
    assert color == matplotlib.colormaps.get_cmap("hsv")(
        (stable_hash("node-label") % (2**32)) / float(2**32)
    )
