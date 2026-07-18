"""
Reproduction of a direction-dependent antialiasing bug in Canvas.line(): rendering the
same physical 2-point line segment produces a different result depending on which
endpoint is listed first in the input (x0,y0)->(x1,y1) vs (x1,y1)->(x0,y0) -- even
though the line's geometry, the pixel grid, and the canvas are all identical.

Root cause (traced in datashader/glyphs/line.py, _build_full_antialias /
_full_antialias):

    flip_order = y1 < y0 or (y1 == y0 and x1 < x0)          # line ~863

`flip_order` decides which of two symmetric branches computes the 4 scan-boundary
corners (buffer[0:8], lines ~884-901). Reversing which point is passed in as
(x0,y0) vs (x1,y1) flips this flag, and the corner-buffer computation for the two
branches is not perfectly numerically symmetric -- causing a small (~0.292893, i.e.
1 - 1/sqrt(2), a round-cap-at-distance-1/sqrt(2) value) discrepancy in exactly one
pixel near a segment endpoint's round cap.

Key findings from a downstream investigation (ais-shader, tiled AIS-vessel-traffic
rendering, which surfaced this as a visible "seam" of inflated counts at tile
boundaries in transit_count/mean-speed heatmaps):

* It reproduces for EVERY tested angle (0-345 degrees in 15-degree steps) and both a
  pixel-grid-aligned and a sub-pixel-unaligned line origin -- not just diagonals or
  a particular alignment. See test_direction_invariance_angle_sweep.
* It reproduces for both raw x/y-column input (test_direction_invariance_*) and the
  GeoDataFrame `geometry=` LineString code path (test_direction_invariance_geometry_*).
* It is NOT specific to canvas clipping: the affected pixel is at the segment's own
  *unclipped, fully-interior* endpoint (its natural round cap), not at a
  canvas-boundary-clipped point. See test_mismatch_is_at_the_endpoint_not_the_clip.
* Consequently, widening a rendering border/padding (a common workaround for tiled
  rendering to keep clipping artifacts away from the kept output) does NOT help --
  tested with border sizes from 4 to 64 pixels and endpoint-overreach from 1 to 32
  pixels beyond a "tile" edge, all 35 combinations still show the identical
  discrepancy. See test_padding_does_not_fix_it.
* A practical workaround (not a real fix) is to canonicalize each segment's point
  order to match datashader's own `flip_order` convention before rendering, so
  every segment takes the same internal branch regardless of its original
  digitization order. See test_canonicalizing_endpoint_order_is_a_workaround.
"""
import numpy as np
import pandas as pd
import pytest

import datashader as ds

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

CANVAS_SIZE = 10
CENTER = (5.0, 5.0)
REACH = 15.0  # how far outward the far endpoint extends beyond the canvas

DIRECTIONS = {
    "E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1),
    "NE": (1, 1), "NW": (-1, 1), "SE": (1, -1), "SW": (-1, -1),
}
ANGLES_DEG = list(range(0, 360, 15))
CENTERS = {
    "aligned": (5.0, 5.0),      # exactly on a pixel-grid line
    "unaligned": (5.37, 5.62),  # sub-pixel offset from the grid
}


def _render_xy(x0, y0, x1, y1, plot_size, x_range, y_range, line_width=1):
    df = pd.DataFrame({"x0": [x0], "y0": [y0], "x1": [x1], "y1": [y1]})
    cvs = ds.Canvas(plot_width=plot_size, plot_height=plot_size, x_range=x_range, y_range=y_range)
    agg = cvs.line(df, x=["x0", "x1"], y=["y0", "y1"], axis=1, agg=ds.count(), line_width=line_width)
    return agg.fillna(0).values


def _render_geometry(x0, y0, x1, y1, plot_size, x_range, y_range, line_width=1):
    gdf = gpd.GeoDataFrame(geometry=[shapely.LineString([(x0, y0), (x1, y1)])])
    cvs = ds.Canvas(plot_width=plot_size, plot_height=plot_size, x_range=x_range, y_range=y_range)
    agg = cvs.line(gdf, geometry="geometry", agg=ds.count(), line_width=line_width)
    return agg.fillna(0).values


def _canonicalize(x0, y0, x1, y1):
    """Match datashader's own flip_order convention (line.py ~863)."""
    flip = y1 < y0 or (y1 == y0 and x1 < x0)
    return (x1, y1, x0, y0) if flip else (x0, y0, x1, y1)


@pytest.mark.parametrize("name,vec", DIRECTIONS.items())
@pytest.mark.parametrize("orientation", ["outward", "inward"])
def test_direction_invariance_xy_columns(name, vec, orientation):
    """A line crossing the canvas boundary (raw x/y-column input), in each of 8
    compass directions, both heading outward (inside->outside) and inward
    (outside->inside), must render identically regardless of which endpoint is
    listed first. This currently FAILS for every direction/orientation combination."""
    ux, uy = vec
    norm = (ux ** 2 + uy ** 2) ** 0.5
    ux, uy = ux / norm, uy / norm
    far = (CENTER[0] + ux * REACH, CENTER[1] + uy * REACH)
    p0, p1 = (CENTER, far) if orientation == "outward" else (far, CENTER)

    fwd = _render_xy(p0[0], p0[1], p1[0], p1[1], CANVAS_SIZE, (0, CANVAS_SIZE), (0, CANVAS_SIZE))
    rev = _render_xy(p1[0], p1[1], p0[0], p0[1], CANVAS_SIZE, (0, CANVAS_SIZE), (0, CANVAS_SIZE))
    np.testing.assert_allclose(fwd, rev, atol=1e-6,
                                err_msg=f"direction dependence for {orientation} {name}")


@pytest.mark.parametrize("center_name,center", CENTERS.items())
@pytest.mark.parametrize("angle_deg", ANGLES_DEG)
def test_direction_invariance_angle_sweep(angle_deg, center_name, center):
    """Direction invariance swept across a full 360-degree circle of angles, at both
    a pixel-grid-aligned and a sub-pixel-unaligned origin point. Currently FAILS for
    every single angle/alignment combination (48/48), always with the identical
    magnitude 0.292893 (= 1 - 1/sqrt(2)) at exactly one pixel."""
    theta = np.radians(angle_deg)
    ux, uy = np.cos(theta), np.sin(theta)
    far = (center[0] + ux * REACH, center[1] + uy * REACH)

    fwd = _render_xy(center[0], center[1], far[0], far[1], CANVAS_SIZE, (0, CANVAS_SIZE), (0, CANVAS_SIZE))
    rev = _render_xy(far[0], far[1], center[0], center[1], CANVAS_SIZE, (0, CANVAS_SIZE), (0, CANVAS_SIZE))
    mismatch = np.abs(fwd - rev)
    assert mismatch.max() <= 1e-6, (
        f"direction dependence at angle={angle_deg}deg center={center_name}: "
        f"max mismatch={mismatch.max():.6f} at {np.unravel_index(np.argmax(mismatch), mismatch.shape)}"
    )


@pytest.mark.parametrize("name,vec", DIRECTIONS.items())
@pytest.mark.parametrize("orientation", ["outward", "inward"])
def test_direction_invariance_geometry_column(name, vec, orientation):
    """Same bug, via the GeoDataFrame geometry= (shapely LineString) code path
    instead of raw x/y columns -- confirms it isn't specific to one input API."""
    ux, uy = vec
    norm = (ux ** 2 + uy ** 2) ** 0.5
    ux, uy = ux / norm, uy / norm
    far = (CENTER[0] + ux * REACH, CENTER[1] + uy * REACH)
    p0, p1 = (CENTER, far) if orientation == "outward" else (far, CENTER)

    fwd = _render_geometry(p0[0], p0[1], p1[0], p1[1], CANVAS_SIZE, (0, CANVAS_SIZE), (0, CANVAS_SIZE))
    rev = _render_geometry(p1[0], p1[1], p0[0], p0[1], CANVAS_SIZE, (0, CANVAS_SIZE), (0, CANVAS_SIZE))
    np.testing.assert_allclose(fwd, rev, atol=1e-6,
                                err_msg=f"direction dependence (geometry path) for {orientation} {name}")


def test_mismatch_is_at_the_endpoint_not_the_clip():
    """The mismatched pixel sits at the segment's own INTERIOR, unclipped endpoint's
    round cap -- not at the canvas-boundary clip point of the far endpoint. This
    holds regardless of how far past the canvas edge the far endpoint reaches."""
    near = (10.0, 10.0)  # well inside the canvas, never clipped
    for overreach in (1, 4, 16, 32):
        far = (20.0 + overreach, 10.0)  # beyond the canvas's right edge
        fwd = _render_xy(near[0], near[1], far[0], far[1], 20, (0, 20), (0, 20))
        rev = _render_xy(far[0], far[1], near[0], near[1], 20, (0, 20), (0, 20))
        diff = np.abs(fwd - rev)
        ys, xs = np.where(diff > 1e-6)
        locations = set(zip(ys.tolist(), xs.tolist()))
        # Mismatch should be adjacent to `near` (row,col = 10,10), never near the
        # clipped far end (which would scale with overreach / canvas edge at col 19).
        assert locations, f"expected a mismatch for overreach={overreach}, found none"
        assert all(abs(y - 10) <= 1 and abs(x - 10) <= 1 for y, x in locations), (
            f"overreach={overreach}: expected mismatch adjacent to the interior "
            f"endpoint (10,10), got {locations}"
        )


@pytest.mark.parametrize("overreach", [1, 2, 4, 8, 16, 24, 32])
@pytest.mark.parametrize("border", [4, 8, 16, 32, 64])
def test_padding_does_not_fix_it(border, overreach):
    """A common tiled-rendering workaround is to render on a canvas padded beyond the
    'true' output window and then crop the padding away, so that any
    clipping-related artifacts land in the discarded region. This does NOT help
    here, because the bug isn't a clipping artifact -- it's at the segment's own
    endpoint cap, which doesn't move when padding changes. Every (border, overreach)
    combination below still shows the identical 0.292893 discrepancy in the kept
    (cropped) region."""
    tile_size = 20
    near = (tile_size / 2, tile_size / 2)
    far = (tile_size + overreach, tile_size / 2)

    def render_padded(p0, p1):
        x_range = (-border, tile_size + border)
        y_range = (-border, tile_size + border)
        plot_size = tile_size + 2 * border
        gdf = gpd.GeoDataFrame(geometry=[shapely.LineString([p0, p1])])
        cvs = ds.Canvas(plot_width=plot_size, plot_height=plot_size, x_range=x_range, y_range=y_range)
        agg = cvs.line(gdf, geometry="geometry", agg=ds.count(), line_width=1).fillna(0)
        cropped = agg.isel(x=slice(border, border + tile_size), y=slice(border, border + tile_size))
        return cropped.values

    fwd = render_padded(near, far)
    rev = render_padded(far, near)
    np.testing.assert_allclose(
        fwd, rev, atol=1e-6,
        err_msg=f"border={border} overreach={overreach}: still direction-dependent after padding+crop"
    )


@pytest.mark.parametrize("name,vec", DIRECTIONS.items())
@pytest.mark.parametrize("orientation", ["outward", "inward"])
def test_canonicalizing_endpoint_order_is_a_workaround(name, vec, orientation):
    """Not a fix, but documents a practical caller-side workaround: reordering each
    segment's endpoints to match datashader's own flip_order convention before
    rendering makes forward/reversed input collapse to the same canonical order,
    trivially eliminating the direction dependence. This should keep passing even
    after an upstream fix (a correct fix removes the need for canonicalization, it
    shouldn't break it)."""
    ux, uy = vec
    norm = (ux ** 2 + uy ** 2) ** 0.5
    ux, uy = ux / norm, uy / norm
    far = (CENTER[0] + ux * REACH, CENTER[1] + uy * REACH)
    p0, p1 = (CENTER, far) if orientation == "outward" else (far, CENTER)

    fwd_c = _canonicalize(p0[0], p0[1], p1[0], p1[1])
    rev_c = _canonicalize(p1[0], p1[1], p0[0], p0[1])
    fwd = _render_xy(*fwd_c, CANVAS_SIZE, (0, CANVAS_SIZE), (0, CANVAS_SIZE))
    rev = _render_xy(*rev_c, CANVAS_SIZE, (0, CANVAS_SIZE), (0, CANVAS_SIZE))
    np.testing.assert_allclose(fwd, rev, atol=1e-6,
                                err_msg=f"canonicalization workaround failed for {orientation} {name}")
