"""Hammer-Aitoff projection plots (Plotly) for loudspeaker layouts.

The projection is the same equal-area map used by the IEM AllRADecoder.
Plots are drawn with a transparent background so they follow the Streamlit
theme like the other Plotly charts in the app.

Convention: longitude = -azimuth, so loudspeakers on the left (positive
azimuth) appear on the left side of the map and 0° azimuth is in the centre.
"""

import base64
import io

import numpy as np
import plotly.graph_objects as go
from PIL import Image

_SQRT2 = np.sqrt(2.0)
X_MAX = 2.0 * _SQRT2   # half width of the ellipse
Y_MAX = _SQRT2         # half height of the ellipse

GRID_COLOR = "rgba(128,128,128,0.35)"
OUTLINE_COLOR = "rgba(128,128,128,0.9)"
HORIZON_COLOR = "rgba(128,128,128,0.7)"


# --------------------------------------------------------------------------
# Projection maths
# --------------------------------------------------------------------------
def project(lon, lat):
    """Forward Hammer-Aitoff projection (radians in, map units out)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    d = np.sqrt(1.0 + np.cos(lat) * np.cos(lon / 2.0))
    x = X_MAX * np.cos(lat) * np.sin(lon / 2.0) / d
    y = Y_MAX * np.sin(lat) / d
    return x, y


def unproject(x, y):
    """Inverse projection. Returns (lon, lat, inside) with inside=False
    for map points that lie outside the ellipse."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Points inside the ellipse satisfy (x/X_MAX)² + (y/Y_MAX)² ≤ 1; there
    # z² lies in [½, 1] (z² < ½ would map to |lon| > π).
    inside = (x / X_MAX) ** 2 + (y / Y_MAX) ** 2 <= 1.0 + 1e-9
    q = 1.0 - (x / 4.0) ** 2 - (y / 2.0) ** 2
    z = np.sqrt(np.clip(q, 0.0, None))
    lon = 2.0 * np.arctan2(z * x, 2.0 * (2.0 * z ** 2 - 1.0))
    lat = np.arcsin(np.clip(z * y, -1.0, 1.0))
    return lon, lat, inside


def az_el_to_xy(az_deg, el_deg):
    """Project azimuth/elevation (degrees, left = positive azimuth)."""
    return project(np.radians(-np.asarray(az_deg, dtype=float)),
                   np.radians(np.asarray(el_deg, dtype=float)))


def lat_floor(elevations, step=15.0, margin=10.0):
    """Lowest latitude (degrees) worth drawing for a set of loudspeaker
    elevations.  The map is cut off below the lowest loudspeaker (with some
    room to spare) so that an empty lower hemisphere does not waste space.
    Always keeps a strip below the horizon and never cuts above -15°."""
    if len(elevations) == 0:
        return -15.0
    lowest = float(min(elevations))
    floor = step * np.floor((lowest - margin) / step)
    return float(max(-90.0, min(-15.0, floor)))


# --------------------------------------------------------------------------
# Figure scaffolding
# --------------------------------------------------------------------------
def _az_label(az_deg):
    a = int(round(az_deg))
    if a == 0:
        return "0°"
    if abs(a) == 180:
        return "180°"
    return f"{abs(a)}°{'L' if a > 0 else 'R'}"


def _lat_label(lat_deg):
    return f"{int(round(lat_deg))}°"


def base_figure(floor_deg=-90.0, height=None, assumed_width=960):
    """Create a figure with the graticule, outline and axis labels.

    ``floor_deg`` is the lowest latitude that is drawn; everything below is
    cut away and the parallel at that latitude becomes the bottom edge.
    """
    floor_deg = float(np.clip(floor_deg, -90.0, -15.0))
    floor = np.radians(floor_deg)
    full = floor_deg <= -89.999

    fig = go.Figure()

    xs, ys = [], []

    def add_line(x, y):
        xs.extend(list(x) + [None])
        ys.extend(list(y) + [None])

    # Parallels (every 15°) inside the drawn region
    lon_line = np.linspace(-np.pi, np.pi, 181)
    for lat_deg in np.arange(-75.0, 90.0, 15.0):
        if lat_deg <= floor_deg or lat_deg == 0.0:
            continue
        add_line(*project(lon_line, np.full_like(lon_line, np.radians(lat_deg))))

    # Meridians (every 30°) from the floor up to the pole
    lat_line = np.linspace(floor, np.pi / 2, 91)
    for lon_deg in np.arange(-150.0, 180.0, 30.0):
        add_line(*project(np.full_like(lat_line, np.radians(lon_deg)), lat_line))

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color=GRID_COLOR, width=1),
        hoverinfo="skip", showlegend=False))

    # Horizon, slightly stronger
    hx, hy = project(lon_line, np.zeros_like(lon_line))
    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="lines", line=dict(color=HORIZON_COLOR, width=1.2),
        hoverinfo="skip", showlegend=False))

    # Outline: left edge up to the pole, down the right edge and back along
    # the floor parallel (which collapses to the south pole for a full map).
    lat_edge = np.linspace(floor, np.pi / 2, 91)
    lx, ly = project(np.full_like(lat_edge, -np.pi), lat_edge)
    rx, ry = project(np.full_like(lat_edge, np.pi), lat_edge)
    bx, by = project(lon_line, np.full_like(lon_line, floor))
    ox = np.concatenate([lx, rx[::-1], bx[::-1], lx[:1]])
    oy = np.concatenate([ly, ry[::-1], by[::-1], ly[:1]])
    fig.add_trace(go.Scatter(
        x=ox, y=oy, mode="lines", line=dict(color=OUTLINE_COLOR, width=1.5),
        hoverinfo="skip", showlegend=False))

    annotations = []
    label_font = dict(size=11)

    # Azimuth labels: along the bottom edge when the map is truncated and
    # that edge is wide enough, otherwise just below the horizon (the bottom
    # edge narrows towards the south pole and finally collapses to a point).
    labels_on_horizon = floor_deg <= -60.0
    for lon_deg in np.arange(-150.0, 180.0, 30.0):
        az = -lon_deg
        if labels_on_horizon:
            x, y = project(np.radians(lon_deg), 0.0)
            annotations.append(dict(
                x=float(x), y=float(y), text=_az_label(az), showarrow=False,
                font=label_font, opacity=0.7, yanchor="top", yshift=-11))
        else:
            x, y = project(np.radians(lon_deg), floor)
            annotations.append(dict(
                x=float(x), y=float(y), text=_az_label(az), showarrow=False,
                font=label_font, opacity=0.7, yanchor="top", yshift=-5))

    # Elevation labels on the left edge
    for lat_deg in np.arange(-60.0, 90.0, 30.0):
        if lat_deg < floor_deg:
            continue
        x, y = project(-np.pi, np.radians(lat_deg))
        annotations.append(dict(
            x=float(x), y=float(y), text=_lat_label(lat_deg), showarrow=False,
            font=label_font, opacity=0.7, xanchor="right", xshift=-6))

    y_min = float(by.min())
    x_pad = 0.32                    # room for the elevation labels
    y_pad_top = 0.12
    y_pad_bottom = 0.30 if not full else 0.12
    x_range = [-X_MAX - x_pad, X_MAX + x_pad]
    y_range = [y_min - y_pad_bottom, Y_MAX + y_pad_top]

    if height is None:
        # Size the figure for the map's aspect ratio at a typical column
        # width; ``constrain="domain"`` keeps the shape if the column differs.
        px_per_unit = (assumed_width - 20) / (x_range[1] - x_range[0])
        height = int((y_range[1] - y_range[0]) * px_per_unit) + 20
        height = int(np.clip(height, 280, 500))

    fig.update_layout(
        annotations=annotations,
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="closest",
        dragmode=False,
    )
    fig.update_xaxes(visible=False, range=x_range, fixedrange=True,
                     constrain="domain")
    fig.update_yaxes(visible=False, range=y_range, fixedrange=True,
                     scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def _hover_text(labels, az_deg, el_deg):
    return [f"<b>Ch {lab}</b><br>Azimuth {a:.1f}°<br>Elevation {e:.1f}°"
            for lab, a, e in zip(labels, az_deg, el_deg)]


# --------------------------------------------------------------------------
# Public figures
# --------------------------------------------------------------------------
def layout_figure(az_deg, el_deg, labels, floor_deg=None):
    """Loudspeaker positions on a Hammer-Aitoff map."""
    az_deg = np.asarray(az_deg, dtype=float)
    el_deg = np.asarray(el_deg, dtype=float)
    if floor_deg is None:
        floor_deg = lat_floor(el_deg)
    fig = base_figure(floor_deg)
    x, y = az_el_to_xy(az_deg, el_deg)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text",
        marker=dict(size=19, color="#ffd60a",
                    line=dict(color="rgba(0,0,0,0.6)", width=1)),
        text=[str(lab) for lab in labels], textposition="middle center",
        textfont=dict(color="#111111", size=10, family="Arial Black"),
        hovertext=_hover_text(labels, az_deg, el_deg), hoverinfo="text",
        cliponaxis=False, showlegend=False))
    return fig


_ENERGY_STOPS = [(0.00, "#000000"), (0.25, "#4d0000"), (0.50, "#a30000"),
                 (0.75, "#ff2b2b"), (1.00, "#ff9d9d")]


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _energy_rgb(t):
    """Map values in [0, 1] to RGB (uint8) using the energy colour map."""
    pos = np.array([p for p, _ in _ENERGY_STOPS])
    cols = np.array([_hex_to_rgb(c) for _, c in _ENERGY_STOPS], dtype=float)
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    out = np.stack([np.interp(t, pos, cols[:, i]) for i in range(3)], axis=-1)
    return np.round(out).astype(np.uint8)


def _bilinear(grid, az_deg, el_deg):
    """Sample ``grid`` (n_el, n_az) laid out over az ∈ [-180, 180] and
    el ∈ [-90, 90] at the given points using bilinear interpolation."""
    n_el, n_az = grid.shape
    fa = (az_deg + 180.0) / 360.0 * (n_az - 1)
    fe = (el_deg + 90.0) / 180.0 * (n_el - 1)
    ia = np.clip(np.floor(fa).astype(int), 0, n_az - 2)
    ie = np.clip(np.floor(fe).astype(int), 0, n_el - 2)
    wa = np.clip(fa - ia, 0.0, 1.0)
    we = np.clip(fe - ie, 0.0, 1.0)
    g00 = grid[ie, ia]
    g01 = grid[ie, ia + 1]
    g10 = grid[ie + 1, ia]
    g11 = grid[ie + 1, ia + 1]
    return ((1 - we) * ((1 - wa) * g00 + wa * g01)
            + we * ((1 - wa) * g10 + wa * g11))


def _energy_image(level_db, vmin, vmax, floor_deg, width=900, height=450):
    """Render the energy map as a PNG data URI covering the ellipse's
    bounding box; pixels outside the drawn region are transparent."""
    px = (np.arange(width) + 0.5) / width * 2 * X_MAX - X_MAX
    py = Y_MAX - (np.arange(height) + 0.5) / height * 2 * Y_MAX
    PX, PY = np.meshgrid(px, py)
    lon, lat, inside = unproject(PX, PY)
    az = -np.degrees(lon)
    el = np.degrees(lat)
    vals = _bilinear(level_db, np.clip(az, -180, 180), np.clip(el, -90, 90))
    t = (vals - vmin) / max(vmax - vmin, 1e-9)
    rgb = _energy_rgb(t)
    # Feather the edges over roughly one pixel to avoid a jagged outline.
    e = np.sqrt((PX / X_MAX) ** 2 + (PY / Y_MAX) ** 2)
    edge = np.clip((1.0 - e) * width / 2.0, 0.0, 1.0)
    cut = np.clip((el - floor_deg) / (180.0 / height), 0.0, 1.0)
    alpha = np.where(inside, edge * cut, 0.0)
    rgba = np.dstack([rgb, np.round(alpha * 255).astype(np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def energy_figure(level_db, mean_db, az_deg, el_deg, labels, floor_deg=None,
                  span_db=1.5):
    """Decoded energy distribution with the loudspeakers overlaid.

    ``level_db`` must be sampled on a regular grid covering azimuth
    -180…180 (columns) and elevation -90…90 (rows), as returned by
    :func:`allrad.energy_distribution`.
    """
    az_deg = np.asarray(az_deg, dtype=float)
    el_deg = np.asarray(el_deg, dtype=float)
    if floor_deg is None:
        floor_deg = lat_floor(el_deg)
    vmin, vmax = mean_db - span_db, mean_db + span_db

    fig = base_figure(floor_deg)
    fig.add_layout_image(dict(
        source=_energy_image(np.asarray(level_db, dtype=float), vmin, vmax, floor_deg),
        xref="x", yref="y", x=-X_MAX, y=Y_MAX, sizex=2 * X_MAX, sizey=2 * Y_MAX,
        sizing="stretch", layer="below"))

    x, y = az_el_to_xy(az_deg, el_deg)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text",
        marker=dict(size=17, color="rgba(0,0,0,0.3)",
                    line=dict(color="#00e5ff", width=1.5)),
        text=[str(lab) for lab in labels], textposition="middle center",
        textfont=dict(color="#ffffff", size=9, family="Arial Black"),
        hovertext=_hover_text(labels, az_deg, el_deg), hoverinfo="text",
        cliponaxis=False, showlegend=False))

    # Invisible trace that only carries the colour bar
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            colorscale=[[p, c] for p, c in _ENERGY_STOPS],
            cmin=-span_db, cmax=span_db, color=[0], showscale=True,
            colorbar=dict(
                title=dict(text="dB re mean", side="right", font=dict(size=11)),
                thickness=12, len=0.8, outlinewidth=0,
                tickfont=dict(size=10), tickvals=[-span_db, 0, span_db],
                ticktext=[f"{-span_db:+.1f}", "0", f"{span_db:+.1f}"],
            )),
        hoverinfo="skip", showlegend=False))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig
