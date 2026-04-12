import streamlit as st
st.set_page_config(page_title="Dome Loudspeaker Layout Generator", page_icon="🌐", layout="wide")
import numpy as np
import plotly.graph_objects as go
import json
import base64
from datetime import datetime
import matplotlib.pyplot as plt

# --- Load config from URL (must run before any widgets) ---
_url_cfg = None
if "cfg" in st.query_params:
    try:
        _url_cfg = json.loads(base64.b64decode(st.query_params["cfg"]).decode())
    except Exception:
        st.warning("Invalid share link — using defaults.")

if _url_cfg is not None:
    _cfg_hash = st.query_params["cfg"]
    if st.session_state.get("_cfg_hash") != _cfg_hash:
        st.session_state["_cfg_hash"] = _cfg_hash
        st.session_state["w_n_points"]       = int(_url_cfg.get("n", 23))
        st.session_state["w_n_rings"]        = int(_url_cfg.get("rings", 4))
        st.session_state["w_vog"]            = bool(_url_cfg.get("vog", 1))
        st.session_state["w_rbh"]            = bool(_url_cfg.get("rbh", 0))
        st.session_state["w_dome_radius"]    = float(_url_cfg.get("r", 3.0))
        st.session_state["w_listener_height"]= float(_url_cfg.get("lh", 1.3))

        _n  = st.session_state["w_n_points"]
        _nr = st.session_state["w_n_rings"]
        _vog = st.session_state["w_vog"]
        _rbh = st.session_state["w_rbh"]
        _rad = st.session_state["w_dome_radius"]
        _ck = f"N{_n}_R{_nr}_VoG{int(_vog)}_RBH{int(_rbh)}_r{_rad}"

        for _i, _ring in enumerate(_url_cfg.get("rings_data", [])):
            st.session_state[f"elev_{_i}_{_ck}"]      = float(_ring.get("elev", 0))
            st.session_state[f"count_{_i}_{_ck}"]     = int(_ring.get("count", 1))
            st.session_state[f"az_offset_{_i}_{_ck}"] = float(_ring.get("az", 0))
            st.session_state[f"tw_{_i}_{_ck}"]        = float(_ring.get("tw", 0))
            st.session_state[f"td_{_i}_{_ck}"]        = float(_ring.get("td", 0))
            st.session_state[f"th_{_i}_{_ck}"]        = float(_ring.get("th", 0))

# --- Parameter controls at the top ---
st.title("🌐 Ambisonic Dome Loudspeaker Layout Generator")
st.markdown(
    "Generate optimised loudspeaker layouts for ambisonic dome systems. "
    "Configure elevation rings, visualise the 3D layout and Mollweide projection, "
    "plan speaker placement on a rectangular **truss**, or project positions onto **room walls and ceiling**. "
    "Export the layout as an IEM AllRADecoder-compatible JSON file or share the configuration via URL."
)

col1, col2, col3 = st.columns(3)
with col1:
    N_points = st.number_input("Target Speakers (for auto-distribution)", min_value=1, max_value=1000, value=23, step=1, key="w_n_points")
    _actual_total_placeholder = st.empty()
with col2:
    N_rings = st.number_input("Number of Elevation Rings", min_value=1, max_value=20, value=4, step=1, key="w_n_rings")
    Voice_of_God = st.checkbox("Include Voice of God", value=True, key="w_vog")
    Ring_below_horizon = st.checkbox("Add Ring Below Horizon", value=False, key="w_rbh")
with col3:
    dome_radius = st.number_input("Dome Radius (m)", min_value=0.1, max_value=500.0, value=3.0, step=0.5, key="w_dome_radius")
    listener_height = st.number_input("Listener Height (m)", min_value=0.0, max_value=50.0, value=1.3, step=0.1, key="w_listener_height")

# --- Ring configuration ---
st.subheader("🔧 Ring Configuration")

# Calculate default theta (elevation) values
default_theta_vals = np.linspace(np.pi/2, 0, N_rings)

if Ring_below_horizon:
    theta_below = np.pi - default_theta_vals[1]
    default_theta_vals = np.insert(default_theta_vals, 0, theta_below)

# Normalize weights by sin(theta)
ring_weights = np.abs(np.sin(default_theta_vals))
default_ring_counts = np.round(ring_weights / ring_weights.sum() * N_points).astype(int)

if Voice_of_God:
    default_ring_counts[-1] = 1

diff = N_points - default_ring_counts.sum()
default_ring_counts[0] += diff

ring_point_counts = []
theta_vals = []
azimuth_offsets = []

r = dome_radius
stagger_offset = 0 if Ring_below_horizon else 1
cfg_key = f"N{N_points}_R{N_rings}_VoG{int(Voice_of_God)}_RBH{int(Ring_below_horizon)}_r{dome_radius}"

_n_rings_total = len(default_theta_vals)
for _row_start in range(0, _n_rings_total, 5):
    _row_count = min(5, _n_rings_total - _row_start)
    _row_cols = st.columns(_row_count)
    for _ci, _col in enumerate(_row_cols):
        i = _row_start + _ci
        with _col:
            with st.container(border=True):
                st.markdown(f"**Ring {i+1}**")
                elev_deg = round(90 - np.degrees(default_theta_vals[i]), 2)
                elev_input = st.number_input(
                    "Elevation (°)", min_value=-90.0, max_value=90.0,
                    value=elev_deg, step=1.0, key=f"elev_{i}_{cfg_key}"
                )
                theta = np.radians(90 - elev_input)

                count_input = st.number_input(
                    "Speakers", min_value=0, value=int(default_ring_counts[i]),
                    step=1, key=f"count_{i}_{cfg_key}"
                )

                default_az_offset = round(180.0 / count_input, 4) if (i % 2 == stagger_offset and count_input > 0) else 0.0
                az_offset_input = st.number_input(
                    "Azimuth Offset (°)", min_value=-180.0, max_value=180.0,
                    value=default_az_offset, step=1.0, key=f"az_offset_{i}_{cfg_key}"
                )

                theta_vals.append(theta)
                ring_point_counts.append(count_input)
                azimuth_offsets.append(az_offset_input)

_actual_total = sum(ring_point_counts)
_actual_total_placeholder.metric("Actual Total Speakers", _actual_total,
    delta=int(_actual_total - N_points) if _actual_total != N_points else None)

# --- Core logic ---
spherical_coords = []
points = []

for i, (theta, M, az_offset) in enumerate(zip(theta_vals, ring_point_counts, azimuth_offsets)):
    if M == 0:
        continue

    phi_offset = np.radians(az_offset)
    phi_vals = np.linspace(0, 2 * np.pi, M, endpoint=False) + phi_offset
    for phi in phi_vals:
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        points.append((x, y, z))

    for phi in phi_vals:
        azimuth_deg = np.degrees(phi)
        if azimuth_deg > 180:
            azimuth_deg -= 360
        elevation_deg = 90 - np.degrees(theta)
        spherical_coords.append({
            "Azimuth": round(azimuth_deg, 2),
            "Elevation": round(elevation_deg, 2),
            "Radius": 1.0,
            "IsImaginary": False,
            "Channel": len(spherical_coords) + 1,
            "Gain": 1.0
        })

points = np.array(points)

# Add imaginary speaker
spherical_coords.append({
    "Azimuth": 0,
    "Elevation": -90,
    "Radius": 1.0,
    "IsImaginary": True,
    "Channel": len(spherical_coords) + 1,
    "Gain": 1.0
})

indices = [str(i+1) for i in range(len(points))]

# Sphere mesh
u = np.linspace(0, np.pi, 50)
v = np.linspace(0, 2 * np.pi, 50)
u, v = np.meshgrid(u, v)
xs = r * np.sin(u) * np.cos(v)
ys = r * np.sin(u) * np.sin(v)
zs = r * np.cos(u)

# --- Plotting ---
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers+text',
    marker=dict(size=4, color='red'),
    text=indices,
    textposition='top center',
    name="Distributed Points"
))

fig.add_trace(go.Surface(
    x=xs,
    y=ys,
    z=zs,
    showscale=False,
    opacity=0.3,
    colorscale='Greys',
    name="Sphere"
))

scale = r * 1.1
fig.update_layout(
    scene=dict(
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1),
        xaxis=dict(range=[-scale, scale], title='X (front)'),
        yaxis=dict(range=[-scale, scale], title='Y (left)'),
        zaxis=dict(range=[-scale, scale], title='Z (up)')
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    title="Points Distributed on Full Sphere with Indices",
    scene_camera=dict(eye=dict(x=1.8, y=0, z=0.5))
)

azimuths = [spk["Azimuth"] for spk in spherical_coords if not spk["IsImaginary"]]
elevations = [spk["Elevation"] for spk in spherical_coords if not spk["IsImaginary"]]
labels = [str(spk["Channel"]) for spk in spherical_coords if not spk["IsImaginary"]]

azimuths_rad = np.radians(azimuths)
elevations_rad = np.radians(elevations)

fig2, ax = plt.subplots(figsize=(7, 4), subplot_kw={'projection': 'mollweide'})
ax.grid(True, linestyle='--', linewidth=0.5)
# Negate azimuths so that left (+) appears on the left of the plot
ax.scatter(-azimuths_rad, elevations_rad, color='yellow', s=20)

for _az, _el, label in zip(-azimuths_rad, elevations_rad, labels):
    ax.text(_az, _el, label, fontsize=9, fontweight='bold', ha='center', va='center', color='black')

ax.set_xticklabels(['150°L','120°L','90°L','60°L','30°L','0°','30°R','60°R','90°R','120°R','150°R'])
ax.set_title("Mollweide Projection  (left speaker = left side)", fontsize=10, pad=15)

col_3d, col_moll = st.columns(2)
with col_3d:
    st.subheader("🌐 3D View")
    st.plotly_chart(fig, use_container_width=True)
with col_moll:
    st.subheader("🌍 Mollweide Projection")
    st.pyplot(fig2)

# --- Expandable section: Loudspeaker Coordinates ---
with st.expander("📍 Show Loudspeaker Coordinates (Channel, Azimuth, Elevation, x, y, z)"):
    st.write("All angles are in degrees. Coordinates are in a unit sphere (radius = 1.0).")

    import pandas as pd
    coord_table = pd.DataFrame([
        {
            "Channel": spk["Channel"],
            "Azimuth (°)": spk["Azimuth"],
            "Elevation (°)": spk["Elevation"],
            "x": round(np.cos(np.radians(spk["Azimuth"])) * np.cos(np.radians(spk["Elevation"])), 6),
            "y": round(np.sin(np.radians(spk["Azimuth"])) * np.cos(np.radians(spk["Elevation"])), 6),
            "z": round(np.sin(np.radians(spk["Elevation"])), 6),
            "Imaginary": spk["IsImaginary"]
        }
        for spk in spherical_coords
    ])

    st.dataframe(coord_table, use_container_width=True, hide_index=True)


with st.expander("🏗️ Truss Planner", expanded=False):
    # --- Truss Configuration ---
    st.subheader("🏗️ Truss Configuration")
    truss_widths = []
    truss_depths = []
    truss_heights = []

    for _row_start in range(0, len(theta_vals), 5):
        _row_count = min(5, len(theta_vals) - _row_start)
        _row_cols = st.columns(_row_count)
        for _ci, _col in enumerate(_row_cols):
            i = _row_start + _ci
            theta = theta_vals[i]
            with _col:
                with st.container(border=True):
                    st.markdown(f"**Ring {i+1}**")
                    default_w = round(float(2 * np.sin(theta) * r), 4)
                    default_h = round(float(np.cos(theta) * r), 4)
                    tw = st.number_input("Width (m)", min_value=0.0, max_value=float(r * 20),
                        value=default_w, step=float(r * 0.1), key=f"tw_{i}_{cfg_key}")
                    td = st.number_input("Depth (m)", min_value=0.0, max_value=float(r * 20),
                        value=default_w, step=float(r * 0.1), key=f"td_{i}_{cfg_key}")
                    th = st.number_input("Height (m)", min_value=float(-r * 2), max_value=float(r * 2),
                        value=default_h, step=float(r * 0.05), key=f"th_{i}_{cfg_key}")
                    truss_widths.append(tw)
                    truss_depths.append(td)
                    truss_heights.append(th)

    # Compute projected positions per ring
    ring_orig_pts = []
    ring_proj_pts = []
    ring_channels_list = []
    projected_elevations_all = []
    projected_heights_afl = []   # height above floor = pz + listener_height
    orig_elevations_for_table = []
    orig_azimuths_for_table = []
    orig_channels_for_table = []

    channel_cursor = 1
    for i, (theta, M, az_offset, tw, td, th) in enumerate(
            zip(theta_vals, ring_point_counts, azimuth_offsets, truss_widths, truss_depths, truss_heights)):
        if M == 0:
            ring_orig_pts.append([])
            ring_proj_pts.append([])
            ring_channels_list.append([])
            continue

        phi_offset = np.radians(az_offset)
        phi_vals = np.linspace(0, 2 * np.pi, M, endpoint=False) + phi_offset
        half_w = tw / 2
        half_d = td / 2
        eps = 1e-9

        orig_ring, proj_ring, channels_ring = [], [], []

        for j, phi in enumerate(phi_vals):
            orig_ring.append((
                float(r * np.sin(theta) * np.cos(phi)),
                float(r * np.sin(theta) * np.sin(phi)),
                float(r * np.cos(theta))
            ))

            az_deg = np.degrees(phi)
            if az_deg > 180:
                az_deg -= 360
            orig_azimuths_for_table.append(round(float(az_deg), 2))
            orig_elevations_for_table.append(round(float(90 - np.degrees(theta)), 2))
            orig_channels_for_table.append(channel_cursor + j)

            cp, sp = np.cos(phi), np.sin(phi)
            if half_w < eps and half_d < eps:
                px, py, pz = 0.0, 0.0, float(th)
                proj_elev = 90.0
            else:
                t = min(half_w / max(abs(cp), eps), half_d / max(abs(sp), eps))
                px, py, pz = float(t * cp), float(t * sp), float(th)
                proj_elev = float(np.degrees(np.arctan2(th, t)))

            proj_ring.append((px, py, pz))
            projected_elevations_all.append(proj_elev)
            projected_heights_afl.append(round(pz + listener_height, 3))
            channels_ring.append(channel_cursor + j)

        ring_orig_pts.append(orig_ring)
        ring_proj_pts.append(proj_ring)
        ring_channels_list.append(channels_ring)
        channel_cursor += M

    # Truss 3D visualisation
    ring_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                   '#42d4f4', '#f032e6', '#bfef45', '#469990', '#dcbeff']

    fig_truss = go.Figure()

    fig_truss.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        showscale=False, opacity=0.1,
        colorscale='Greys', name="Sphere", hoverinfo='skip'
    ))

    for i, (tw, td, th, orig_ring, proj_ring, ch_ring) in enumerate(
            zip(truss_widths, truss_depths, truss_heights, ring_orig_pts, ring_proj_pts, ring_channels_list)):
        if not orig_ring:
            continue
        color = ring_colors[i % len(ring_colors)]
        W2, D2 = tw / 2, td / 2

        # Original sphere positions — open circles, same ring color
        fig_truss.add_trace(go.Scatter3d(
            x=[p[0] for p in orig_ring], y=[p[1] for p in orig_ring], z=[p[2] for p in orig_ring],
            mode='markers',
            marker=dict(size=6, color='white', opacity=0.9,
                        line=dict(color=color, width=2), symbol='circle'),
            name=f"Ring {i+1} Original"
        ))

        fig_truss.add_trace(go.Scatter3d(
            x=[-W2,  W2,  W2, -W2, -W2],
            y=[-D2, -D2,  D2,  D2, -D2],
            z=[ th,  th,  th,  th,  th],
            mode='lines', line=dict(color=color, width=4),
            name=f"Ring {i+1} Truss"
        ))

        line_x, line_y, line_z = [], [], []
        for orig, proj in zip(orig_ring, proj_ring):
            line_x += [orig[0], proj[0], None]
            line_y += [orig[1], proj[1], None]
            line_z += [orig[2], proj[2], None]
        fig_truss.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z, mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))

        fig_truss.add_trace(go.Scatter3d(
            x=[p[0] for p in proj_ring], y=[p[1] for p in proj_ring], z=[p[2] for p in proj_ring],
            mode='markers+text', marker=dict(size=5, color=color),
            text=[str(c) for c in ch_ring], textposition='top center',
            name=f"Ring {i+1} Projected"
        ))

    max_half = max((max(tw, td) / 2 for tw, td in zip(truss_widths, truss_depths) if max(tw, td) > 0), default=1.0)
    scale_truss = max(r * 1.1, max_half + 0.1)
    fig_truss.update_layout(
        scene=dict(
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(range=[-scale_truss, scale_truss], title='X (front)'),
            yaxis=dict(range=[-scale_truss, scale_truss], title='Y (left)'),
            zaxis=dict(range=[-scale_truss, scale_truss], title='Z (up)')
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Speaker Projection: Sphere → Truss",
        scene_camera=dict(eye=dict(x=1.8, y=0, z=0.5)),
        legend=dict(yanchor="top", y=0.85, xanchor="right", x=1.0)
    )

    import pandas as pd
    elev_rows = []
    for ch, az, orig_el, proj_el, h_afl in zip(
            orig_channels_for_table, orig_azimuths_for_table,
            orig_elevations_for_table, projected_elevations_all, projected_heights_afl):
        delta = round(proj_el - orig_el, 2)
        elev_rows.append({
            "Ch": ch, "Az (°)": az,
            "Orig Elev (°)": orig_el,
            "Proj Elev (°)": round(proj_el, 2),
            "Δ Elev (°)": delta,
            "Height afl (m)": h_afl,
        })
    elev_df = pd.DataFrame(elev_rows)

    def highlight_large_delta(row):
        color = 'background-color: #ffcccc' if abs(row["Δ Elev (°)"]) > 5 else ''
        return [color] * len(row)

    st.subheader("🏗️ Truss Projection")
    col_tv, col_tt = st.columns([3, 2])
    with col_tv:
        st.plotly_chart(fig_truss, use_container_width=True)
    with col_tt:
        st.markdown("**Elevation Changes (Sphere → Truss)**")
        st.dataframe(
            elev_df.style.apply(highlight_large_delta, axis=1),
            use_container_width=True, hide_index=True
        )


with st.expander("🏠 Wall Mount Planner", expanded=False):
    # --- Wall Mount Planner ---
    st.subheader("🏠 Wall Mount Planner")

    col_rw, col_rl, col_rh = st.columns(3)
    with col_rw:
        room_width = st.number_input("Room Width (m)", min_value=0.1, max_value=500.0, value=10.0, step=0.5)
    with col_rl:
        room_length = st.number_input("Room Length (m)", min_value=0.1, max_value=500.0, value=10.0, step=0.5)
    with col_rh:
        room_height = st.number_input("Room Height (m)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)

    # Adjusted frame: dome center (listening position) at origin, z up
    half_rw  = room_width  / 2
    half_rl  = room_length / 2
    z_ceil   = room_height - listener_height
    z_floor  = -listener_height

    SURFACE_NAMES = {
        'x+': 'Front wall', 'x-': 'Back wall',
        'y+': 'Left wall',  'y-': 'Right wall',
        'z+': 'Ceiling',    'z-': 'Floor',
    }
    SURFACE_COLORS = {
        'x+': '#e6194b', 'x-': '#4363d8',
        'y+': '#3cb44b', 'y-': '#f58231',
        'z+': '#911eb4', 'z-': '#808080',
    }

    # Compute mount positions for each speaker
    wall_data = []
    _eps = 1e-9
    _ch_cursor = 1

    for i, (theta, M, az_offset) in enumerate(zip(theta_vals, ring_point_counts, azimuth_offsets)):
        if M == 0:
            _ch_cursor += M
            continue
        phi_offset = np.radians(az_offset)
        phi_vals = np.linspace(0, 2 * np.pi, M, endpoint=False) + phi_offset

        for j, phi in enumerate(phi_vals):
            dx = float(np.sin(theta) * np.cos(phi))
            dy = float(np.sin(theta) * np.sin(phi))
            dz = float(np.cos(theta))

            candidates = []
            if abs(dx) > _eps:
                candidates.append((half_rw / abs(dx), 'x+' if dx > 0 else 'x-'))
            if abs(dy) > _eps:
                candidates.append((half_rl / abs(dy), 'y+' if dy > 0 else 'y-'))
            if dz > _eps:
                t = z_ceil / dz
                if t > 0:
                    candidates.append((t, 'z+'))
            elif dz < -_eps:
                t = z_floor / dz
                if t > 0:
                    candidates.append((t, 'z-'))

            if not candidates:
                _ch_cursor += 1
                continue

            t_min, surface = min(candidates, key=lambda c: c[0])
            px, py, pz = t_min * dx, t_min * dy, t_min * dz

            az_deg = float(np.degrees(phi))
            if az_deg > 180:
                az_deg -= 360

            wall_data.append({
                "channel":        _ch_cursor + j,
                "surface":        surface,
                "surface_name":   SURFACE_NAMES[surface],
                "color":          SURFACE_COLORS[surface],
                "x":              round(px, 3),
                "y":              round(py, 3),
                "z":              round(pz, 3),
                "h_above_floor":  round(pz + listener_height, 3),
                "azimuth":        round(az_deg, 2),
                "elevation":      round(float(90 - np.degrees(theta)), 2),
            })

        _ch_cursor += M

    # Build wall mount table before columns
    import pandas as pd
    wall_df = pd.DataFrame([{
        "Ch":                     d["channel"],
        "Surface":                d["surface_name"],
        "x (m)":                  d["x"],
        "y (m)":                  d["y"],
        "Height (m)":             d["h_above_floor"],
        "Az (°)":                 d["azimuth"],
        "Elev (°)":               d["elevation"],
    } for d in wall_data])

    # 3D visualisation
    st.subheader("🏠 Wall Mount Projection")

    def _box_edges(x0, x1, y0, y1, z0, z1):
        bx, by, bz = [], [], []
        for ex, ey, ez in [
            ([x0,x1],[y0,y0],[z0,z0]), ([x1,x1],[y0,y1],[z0,z0]),
            ([x1,x0],[y1,y1],[z0,z0]), ([x0,x0],[y1,y0],[z0,z0]),
            ([x0,x1],[y0,y0],[z1,z1]), ([x1,x1],[y0,y1],[z1,z1]),
            ([x1,x0],[y1,y1],[z1,z1]), ([x0,x0],[y1,y0],[z1,z1]),
            ([x0,x0],[y0,y0],[z0,z1]), ([x1,x1],[y0,y0],[z0,z1]),
            ([x0,x0],[y1,y1],[z0,z1]), ([x1,x1],[y1,y1],[z0,z1]),
        ]:
            bx += ex + [None]; by += ey + [None]; bz += ez + [None]
        return bx, by, bz

    fig_wall = go.Figure()

    # Sphere mesh (reference)
    fig_wall.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        showscale=False, opacity=0.08,
        colorscale='Greys', hoverinfo='skip', showlegend=False
    ))

    # Room box
    bx, by, bz = _box_edges(-half_rw, half_rw, -half_rl, half_rl, z_floor, z_ceil)
    fig_wall.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz, mode='lines',
        line=dict(color='lightgrey', width=2),
        name='Room', showlegend=False, hoverinfo='skip'
    ))

    # Listening position marker
    fig_wall.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(size=6, color='black', symbol='cross'),
        name='Listening position'
    ))

    # Original sphere positions
    if len(points) > 0:
        fig_wall.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=5, color='white', opacity=0.9,
                        line=dict(color='grey', width=2), symbol='circle'),
            name='Original (sphere)'
        ))

    # Per-surface: projection lines + speaker dots
    for surf_key in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
        pts = [d for d in wall_data if d['surface'] == surf_key]
        if not pts:
            continue
        color = SURFACE_COLORS[surf_key]

        lx, ly, lz = [], [], []
        for d in pts:
            lx += [0, d['x'], None]
            ly += [0, d['y'], None]
            lz += [0, d['z'], None]
        fig_wall.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))

        fig_wall.add_trace(go.Scatter3d(
            x=[d['x'] for d in pts],
            y=[d['y'] for d in pts],
            z=[d['z'] for d in pts],
            mode='markers+text',
            marker=dict(size=6, color=color),
            text=[str(d['channel']) for d in pts],
            textposition='top center',
            name=SURFACE_NAMES[surf_key]
        ))

    fig_wall.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (m)  ← Back | Front →'),
            yaxis=dict(title='Y (m)  ← Right | Left →'),
            zaxis=dict(title='Z (m)'),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Speaker Projection: Sphere → Room Walls",
        scene_camera=dict(eye=dict(x=1.8, y=0, z=0.5)),
        legend=dict(yanchor="top", y=0.85, xanchor="right", x=1.0)
    )

    col_wv, col_wt = st.columns([3, 2])
    with col_wv:
        st.plotly_chart(fig_wall, use_container_width=True)
    with col_wt:
        st.markdown("**Mount Positions**")
        st.dataframe(wall_df, use_container_width=True, hide_index=True)


# --- JSON download ---
json_data = {
    "Name": "All-Round Ambisonic decoder loudspeaker layout, importable in IEM AllRAD Decoder Plugin.",
    "Description": f"This configuration file was created with the Automatic Speaker Layout Generator from Matthias Kronlachner. {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "LoudspeakerLayout": {
        "Name": "Generated loudspeaker layout",
        "Loudspeakers": spherical_coords
    }
}

json_str = json.dumps(json_data, indent=2)
json_bytes = json_str.encode("utf-8")
filename = f"AllRAD_speakerLayout_N{N_points}_R{N_rings}_VoG{int(Voice_of_God)}_RBH{int(Ring_below_horizon)}.json"

st.markdown("###")  # Adds vertical space before the download button

st.download_button(
    label="⬇️ Download IEM AllRAD Layout JSON",
    data=json_bytes,
    file_name=filename,
    mime="application/json"
)


st.markdown("* IMPORT the `.json` file in the [IEM AllRADecoder plugin](https://plugins.iem.at), and press `Calculate Decoder`.")

# --- Share link ---
st.markdown("---")
if st.button("🔗 Generate Share Link"):
    _rings_data = []
    for _i in range(len(theta_vals)):
        _rings_data.append({
            "elev":  round(float(90 - np.degrees(theta_vals[_i])), 4),
            "count": int(ring_point_counts[_i]),
            "az":    float(azimuth_offsets[_i]),
            "tw":    float(truss_widths[_i]),
            "td":    float(truss_depths[_i]),
            "th":    float(truss_heights[_i]),
        })
    _cfg = {
        "n":          N_points,
        "rings":      N_rings,
        "vog":        int(Voice_of_God),
        "rbh":        int(Ring_below_horizon),
        "r":          dome_radius,
        "lh":         listener_height,
        "rings_data": _rings_data,
    }
    _encoded = base64.b64encode(json.dumps(_cfg).encode()).decode()
    st.query_params["cfg"] = _encoded
    st.success("URL updated — copy it from your browser's address bar to share this configuration.")

st.markdown("---")
st.markdown("© 2025–2026 Matthias Kronlachner")
