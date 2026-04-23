import streamlit as st
st.set_page_config(page_title="Dome Loudspeaker Layout Generator", page_icon="🌐", layout="wide")
import numpy as np
import plotly.graph_objects as go
import json
import base64
import re
import hashlib
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
        st.session_state["w_dome_radius"]      = float(_url_cfg.get("r", 3.0))
        st.session_state["w_listener_height"]  = float(_url_cfg.get("lh", 1.3))
        st.session_state["w_layout_title"]     = str(_url_cfg.get("title", ""))
        st.session_state["w_layout_description"] = str(_url_cfg.get("desc", ""))

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

        # Store per-speaker offsets; applied to the editor on first render.
        st.session_state["_pending_spk_offsets"] = {
            int(o["ch"]): (float(o.get("daz", 0.0)), float(o.get("del", 0.0)))
            for o in _url_cfg.get("spk_offsets", [])
        }
        # One-shot flags: consumed by the expanded= parameter on first render.
        st.session_state["_truss_exp_init"] = bool(_url_cfg.get("truss_exp", False))
        st.session_state["_wall_exp_init"]  = bool(_url_cfg.get("wall_exp",  False))

# --- Parameter controls at the top ---
st.title("🌐 Ambisonic Dome Loudspeaker Layout Generator")
st.markdown(
    "Generate optimised loudspeaker layouts for ambisonic dome systems. "
    "Configure elevation rings, visualise the 3D layout and Mollweide projection, "
    "plan speaker placement on a rectangular **truss**, or project positions onto **room walls and ceiling**. "
    "Export the layout as an IEM AllRADecoder-compatible JSON file or share the configuration via URL."
)

layout_title = st.text_input(
    "Layout Title", value="", placeholder="e.g. Concert Hall Dome — 24 ch",
    key="w_layout_title"
)
layout_description = st.text_area(
    "Description", value="", placeholder="Optional notes about this configuration...",
    key="w_layout_description", height=80
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
# With VoG: last ring lands at 90° (zenith) and gets count=1 (the VoG speaker).
# Without VoG: use step = 90°/N_rings so rings space evenly below zenith
#   e.g. 3 rings → 0°, 30°, 60° instead of 0°, 45°, 90°
if Voice_of_God:
    default_theta_vals = np.linspace(np.pi/2, 0, N_rings)
else:
    step = (np.pi / 2) / N_rings
    default_theta_vals = np.linspace(np.pi/2, step, N_rings)

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

                if elev_input >= 90.0:
                    default_az_offset = 0.0
                elif i % 2 == stagger_offset and count_input > 0:
                    default_az_offset = round(180.0 / count_input, 4)
                else:
                    default_az_offset = 0.0
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

# --- Per-speaker position editor ---
# Build a config hash that changes whenever any ring parameter changes so that
# manual edits are automatically discarded when the ring layout is reconfigured.
import pandas as pd
_spk_config_str = cfg_key + "|" + "|".join(
    f"{round(float(t),5)},{c},{round(float(a),4)}"
    for t, c, a in zip(theta_vals, ring_point_counts, azimuth_offsets)
)
_editor_key = "spk_editor_" + hashlib.md5(_spk_config_str.encode()).hexdigest()[:12]

# Maintain our own offset state so edits survive any widget interaction on the page
# (st.data_editor's internal session state can be silently re-initialized by Streamlit).
_offsets_state_key = f"_spk_offsets_{_editor_key}"
if _offsets_state_key not in st.session_state:
    # Ring config changed or first load — consume pending URL offsets if present.
    st.session_state[_offsets_state_key] = st.session_state.pop("_pending_spk_offsets", {})
_stored_offsets = st.session_state[_offsets_state_key]  # {channel: (daz, del)}

_auto_spk_list = [s for s in spherical_coords if not s["IsImaginary"]]

with st.expander("✏️ Speaker Position Offsets", expanded=False):
    st.caption("ΔAz and ΔEl are added to each speaker's auto-computed position. Resets when ring configuration changes.")
    if st.button("Reset all offsets to zero", key="reset_spk_offsets"):
        for _s in _auto_spk_list:
            _c = _s["Channel"]
            st.session_state[f"daz_{_c}_{_editor_key}"] = 0.0
            st.session_state[f"del_{_c}_{_editor_key}"] = 0.0
        st.session_state[_offsets_state_key] = {}
        st.rerun()

    # Group speakers by ring using ring_point_counts
    _delta_values = {}
    _daz_vals = {}
    _spk_idx = 0
    for _ring_i, (_ring_count, _ring_theta) in enumerate(zip(ring_point_counts, theta_vals)):
        if _ring_count == 0:
            continue
        _ring_spks = _auto_spk_list[_spk_idx: _spk_idx + _ring_count]
        _spk_idx += _ring_count
        _ring_el = round(90 - np.degrees(_ring_theta), 1)

        st.markdown(f"**Ring {_ring_i + 1} — El {_ring_el}°**")
        _label_w = 2
        _tcols = st.columns([_label_w] + [1] * _ring_count)

        # Ch header
        _tcols[0].markdown("**Ch**")
        for _i, _s in enumerate(_ring_spks):
            _tcols[_i + 1].markdown(f"**{_s['Channel']}**")

        # Az / El combined in one cell
        _tcols[0].write("Az / El (°)")
        for _i, _s in enumerate(_ring_spks):
            _tcols[_i + 1].write(f"{float(_s['Azimuth']):.1f} / {float(_s['Elevation']):.1f}")

        # ΔAz row
        _tcols[0].markdown("**ΔAz (°)**")
        for _i, _s in enumerate(_ring_spks):
            _c = _s["Channel"]
            _daz_vals[_c] = _tcols[_i + 1].number_input(
                "ΔAz", value=float(_stored_offsets.get(_c, (0.0, 0.0))[0]),
                min_value=-180.0, max_value=180.0,
                key=f"daz_{_c}_{_editor_key}", label_visibility="collapsed"
            )

        # ΔEl row
        _tcols[0].markdown("**ΔEl (°)**")
        for _i, _s in enumerate(_ring_spks):
            _c = _s["Channel"]
            _del = _tcols[_i + 1].number_input(
                "ΔEl", value=float(_stored_offsets.get(_c, (0.0, 0.0))[1]),
                min_value=-90.0, max_value=90.0,
                key=f"del_{_c}_{_editor_key}", label_visibility="collapsed"
            )
            _delta_values[_c] = (_daz_vals[_c], _del)

# Persist offsets back into our own state after every render.
st.session_state[_offsets_state_key] = _delta_values

# Rebuild spherical_coords and points applying the delta offsets.
# _edited_positions maps channel -> (final_az, final_el) for truss/wall planners.
_edited_positions = {}
_final_real = []
_final_pts = []
for _s in _auto_spk_list:
    _ch = _s["Channel"]
    _daz, _del = _delta_values.get(_ch, (0.0, 0.0))
    _az = max(-180.0, min(180.0, float(_s["Azimuth"]) + _daz))
    _el = max(-90.0,  min(90.0,  float(_s["Elevation"]) + _del))
    _edited_positions[_ch] = (_az, _el)
    _th = np.radians(90 - _el)
    _ph = np.radians(_az)
    _final_pts.append((r * np.sin(_th) * np.cos(_ph),
                       r * np.sin(_th) * np.sin(_ph),
                       r * np.cos(_th)))
    _final_real.append({"Azimuth": round(_az, 2), "Elevation": round(_el, 2),
                        "Radius": 1.0, "IsImaginary": False, "Channel": _ch, "Gain": 1.0})

spherical_coords = _final_real + [s for s in spherical_coords if s["IsImaginary"]]
points = np.array(_final_pts) if _final_pts else np.zeros((0, 3))

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
    textfont=dict(color='#111111', size=13, family='Arial Black'),
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
        xaxis=dict(range=[-scale, scale], title='X (m)  − back / + front'),
        yaxis=dict(range=[-scale, scale], title='Y (m)  − right / + left'),
        zaxis=dict(range=[-scale, scale], title='Z (m)')
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
    _m3d_cams = st.columns(6)
    for _vi, (_vn, _vx, _vy, _vz) in enumerate([
        ("3D", 1.8, 0, 0.5), ("Top", 0, 0, 2.5),
        ("Front", 2.5, 0, 0.3), ("Back", -2.5, 0, 0.3),
        ("Left", 0, 2.5, 0.3), ("Right", 0, -2.5, 0.3),
    ]):
        if _m3d_cams[_vi].button(_vn, key=f"main_cam_{_vn}", use_container_width=True):
            st.session_state["main_3d_camera"] = dict(eye=dict(x=_vx, y=_vy, z=_vz))
    fig.update_layout(
        scene_camera=st.session_state.get("main_3d_camera", dict(eye=dict(x=1.8, y=0, z=0.5)))
    )
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


_ORIGIN_OPTS = ["Center", "Front Left", "Front Right", "Rear Left", "Rear Right"]

with st.expander("🏗️ Truss Planner", key="truss_expander",
                  expanded=st.session_state.pop("_truss_exp_init", st.session_state.get("truss_expander", False))):
    # --- Truss Configuration ---
    st.subheader("🏗️ Truss Configuration")
    truss_widths = []
    truss_depths = []
    truss_heights = []
    truss_ring_visible = []

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
                    default_h = round(float(np.cos(theta) * r) + listener_height, 4)
                    tw = st.number_input("Width (m)", min_value=0.0, max_value=float(r * 20),
                        value=default_w, step=float(r * 0.1), key=f"tw_{i}_{cfg_key}")
                    td = st.number_input("Depth (m)", min_value=0.0, max_value=float(r * 20),
                        value=default_w, step=float(r * 0.1), key=f"td_{i}_{cfg_key}")
                    th = st.number_input("Height (m, above floor)", min_value=0.0, max_value=float(r * 4 + listener_height),
                        value=default_h, step=float(r * 0.05), key=f"th_{i}_{cfg_key}")
                    truss_widths.append(tw)
                    truss_depths.append(td)
                    truss_heights.append(th)
                    truss_ring_visible.append(st.checkbox(
                        "Show", value=True, key=f"truss_show_{i}_{cfg_key}"
                    ))

    # Compute projected positions per ring
    ring_orig_pts = []
    ring_proj_pts = []
    ring_channels_list = []
    projected_elevations_all = []
    projected_x_all = []
    projected_y_all = []
    projected_z_all = []   # absolute height above floor = pz + listener_height
    orig_elevations_for_table = []
    orig_azimuths_for_table = []
    orig_channels_for_table = []
    ring_idx_for_table = []

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
            _ch_j = channel_cursor + j
            # Apply per-speaker position override if present
            if _ch_j in _edited_positions:
                _az_ov, _el_ov = _edited_positions[_ch_j]
                _theta_ov = np.radians(90 - _el_ov)
                _phi_ov = np.radians(_az_ov)
            else:
                _theta_ov, _phi_ov = theta, phi

            orig_ring.append((
                float(r * np.sin(_theta_ov) * np.cos(_phi_ov)),
                float(r * np.sin(_theta_ov) * np.sin(_phi_ov)),
                float(r * np.cos(_theta_ov))
            ))

            az_deg = np.degrees(_phi_ov)
            if az_deg > 180:
                az_deg -= 360
            orig_azimuths_for_table.append(round(float(az_deg), 2))
            orig_elevations_for_table.append(round(float(90 - np.degrees(_theta_ov)), 2))
            orig_channels_for_table.append(_ch_j)
            ring_idx_for_table.append(i)

            cp, sp = np.cos(_phi_ov), np.sin(_phi_ov)
            # th is floor-relative; height above listener = th - listener_height
            _th_rel = th - listener_height
            if half_w < eps and half_d < eps:
                px, py, pz = 0.0, 0.0, float(th)
                proj_elev = 90.0
            else:
                t = min(half_d / max(abs(cp), eps), half_w / max(abs(sp), eps))
                px, py, pz = float(t * cp), float(t * sp), float(th)
                proj_elev = float(np.degrees(np.arctan2(_th_rel, t)))

            proj_ring.append((px, py, pz))
            projected_elevations_all.append(proj_elev)
            projected_x_all.append(round(px, 3))
            projected_y_all.append(round(py, 3))
            projected_z_all.append(round(pz, 3))
            channels_ring.append(channel_cursor + j)

        ring_orig_pts.append(orig_ring)
        ring_proj_pts.append(proj_ring)
        ring_channels_list.append(channels_ring)
        channel_cursor += M

    # Coordinate origin selector
    _vis_truss_w = [tw for tw, vis in zip(truss_widths, truss_ring_visible) if vis]
    _vis_truss_d = [td for td, vis in zip(truss_depths, truss_ring_visible) if vis]
    _t_max_W2 = max((_w / 2 for _w in _vis_truss_w), default=0.0)
    _t_max_D2 = max((_d / 2 for _d in _vis_truss_d), default=0.0)
    _t_origin = st.selectbox("Coordinate Origin", _ORIGIN_OPTS, key=f"truss_origin_{cfg_key}")
    _t_ox, _t_oy = {
        "Center":      (0.0,       0.0),
        "Front Left":  (_t_max_D2,  _t_max_W2),
        "Front Right": (_t_max_D2, -_t_max_W2),
        "Rear Left":   (-_t_max_D2,  _t_max_W2),
        "Rear Right":  (-_t_max_D2, -_t_max_W2),
    }[_t_origin]

    # Truss 3D visualisation
    ring_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                   '#42d4f4', '#f032e6', '#bfef45', '#469990', '#dcbeff']

    fig_truss = go.Figure()

    fig_truss.add_trace(go.Surface(
        x=xs - _t_ox, y=ys - _t_oy, z=zs + listener_height,
        showscale=False, opacity=0.1,
        colorscale='Greys', name="Sphere", hoverinfo='skip'
    ))

    for i, (tw, td, th, orig_ring, proj_ring, ch_ring) in enumerate(
            zip(truss_widths, truss_depths, truss_heights, ring_orig_pts, ring_proj_pts, ring_channels_list)):
        if not orig_ring:
            continue
        if not truss_ring_visible[i]:
            continue
        color = ring_colors[i % len(ring_colors)]
        W2, D2 = tw / 2, td / 2

        # Original sphere positions — open circles, same ring color
        fig_truss.add_trace(go.Scatter3d(
            x=[p[0] - _t_ox for p in orig_ring], y=[p[1] - _t_oy for p in orig_ring],
            z=[p[2] + listener_height for p in orig_ring],
            mode='markers',
            marker=dict(size=6, color='white', opacity=0.9,
                        line=dict(color=color, width=2), symbol='circle'),
            name=f"Ring {i+1} Original"
        ))

        fig_truss.add_trace(go.Scatter3d(
            x=[-D2 - _t_ox,  D2 - _t_ox,  D2 - _t_ox, -D2 - _t_ox, -D2 - _t_ox],
            y=[-W2 - _t_oy, -W2 - _t_oy,  W2 - _t_oy,  W2 - _t_oy, -W2 - _t_oy],
            z=[ th,  th,  th,  th,  th],
            mode='lines', line=dict(color=color, width=4),
            name=f"Ring {i+1} Truss"
        ))

        line_x, line_y, line_z = [], [], []
        for orig, proj in zip(orig_ring, proj_ring):
            line_x += [orig[0] - _t_ox, proj[0] - _t_ox, None]
            line_y += [orig[1] - _t_oy, proj[1] - _t_oy, None]
            line_z += [orig[2] + listener_height, proj[2], None]
        fig_truss.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z, mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))

        fig_truss.add_trace(go.Scatter3d(
            x=[p[0] - _t_ox for p in proj_ring], y=[p[1] - _t_oy for p in proj_ring],
            z=[p[2] for p in proj_ring],
            mode='markers+text', marker=dict(size=5, color=color),
            text=[str(c) for c in ch_ring], textposition='top center',
            textfont=dict(color='#111111', size=13, family='Arial Black'),
            name=f"Ring {i+1} Projected"
        ))

    max_half = max((max(tw, td) / 2 for tw, td in zip(truss_widths, truss_depths) if max(tw, td) > 0), default=1.0)
    scale_truss_xy = max(r * 1.1, max_half + 0.1)
    max_th_afl = max(truss_heights, default=listener_height)
    scale_truss_z = max(r * 1.1 + listener_height, max_th_afl + 0.1)
    fig_truss.update_layout(
        scene=dict(
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=scale_truss_z / (2 * scale_truss_xy)),
            xaxis=dict(range=[-scale_truss_xy - _t_ox, scale_truss_xy - _t_ox], title='X (m)  − back / + front'),
            yaxis=dict(range=[-scale_truss_xy - _t_oy, scale_truss_xy - _t_oy], title='Y (m)  − right / + left'),
            zaxis=dict(range=[0, scale_truss_z], title='Z (m, above floor)')
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Speaker Projection: Sphere → Truss",
        scene_camera=dict(eye=dict(x=1.8, y=0, z=0.5)),
        legend=dict(yanchor="top", y=0.85, xanchor="right", x=1.0)
    )

    import pandas as pd
    elev_rows = []
    for ch, az, orig_el, proj_el, px, py, pz, _ri in zip(
            orig_channels_for_table, orig_azimuths_for_table,
            orig_elevations_for_table, projected_elevations_all,
            projected_x_all, projected_y_all, projected_z_all, ring_idx_for_table):
        if not truss_ring_visible[_ri]:
            continue
        delta = round(proj_el - orig_el, 2)
        elev_rows.append({
            "Ch": ch, "Az (°)": az,
            "Orig Elev (°)": orig_el,
            "Proj Elev (°)": round(proj_el, 2),
            "Δ Elev (°)": delta,
            "x (m)": round(px - _t_ox, 3),
            "y (m)": round(py - _t_oy, 3),
            "z (m)": pz,
        })
    elev_df = pd.DataFrame(elev_rows)

    def highlight_large_delta(row):
        color = 'background-color: #c0392b; color: #ffffff' if abs(row["Δ Elev (°)"]) > 5 else ''
        return [color] * len(row)

    st.subheader("🏗️ Truss Projection")
    col_tv, col_2d_truss = st.columns(2)
    with col_tv:
        _tv_cams = st.columns(6)
        for _vi, (_vn, _vx, _vy, _vz) in enumerate([
            ("3D", 1.8, 0, 0.5), ("Top", 0, 0, 2.5),
            ("Front", 2.5, 0, 0.3), ("Back", -2.5, 0, 0.3),
            ("Left", 0, 2.5, 0.3), ("Right", 0, -2.5, 0.3),
        ]):
            if _tv_cams[_vi].button(_vn, key=f"truss_cam_{_vn}", use_container_width=True):
                st.session_state["truss_3d_camera"] = dict(eye=dict(x=_vx, y=_vy, z=_vz))
        fig_truss.update_layout(
            scene_camera=st.session_state.get("truss_3d_camera", dict(eye=dict(x=1.8, y=0, z=0.5)))
        )
        st.plotly_chart(fig_truss, use_container_width=True)
    with col_2d_truss:
        _t2d_view = st.selectbox("2D View", ["Front", "Side", "Top"], index=2,
                                 key=f"truss_2d_view_{cfg_key}")
        fig_truss_2d = go.Figure()
        _grid = dict(showgrid=True, gridcolor='rgba(128,128,128,0.25)', gridwidth=1, dtick=1)
        _tfont = dict(color='#111111', size=13, family='Arial Black')

        def _tpos2d(dx, dy):
            """Map screen-space direction from listener to a Plotly textposition."""
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                return 'top center'
            a = float(np.degrees(np.arctan2(dy, dx)))
            if   -22.5 <= a <=  22.5: return 'middle right'
            elif  22.5 <  a <=  67.5: return 'top right'
            elif  67.5 <  a <= 112.5: return 'top center'
            elif 112.5 <  a <= 157.5: return 'top left'
            elif a >  157.5 or a < -157.5: return 'middle left'
            elif -157.5 <= a < -112.5: return 'bottom left'
            elif -112.5 <= a <  -67.5: return 'bottom center'
            else:                      return 'bottom right'

        # Listener marker
        if _t2d_view == "Front":
            _t2d_lx, _t2d_ly = -_t_oy, listener_height
        elif _t2d_view == "Side":
            _t2d_lx, _t2d_ly = -_t_ox, listener_height
        else:
            _t2d_lx, _t2d_ly = -_t_oy, -_t_ox
        fig_truss_2d.add_trace(go.Scatter(
            x=[_t2d_lx], y=[_t2d_ly], mode='markers',
            marker=dict(size=10, color='white', symbol='cross',
                        line=dict(color='grey', width=2)),
            cliponaxis=False, name='Listener'))
        for _ri2, (tw2, td2, th2, proj2, ch2) in enumerate(
                zip(truss_widths, truss_depths, truss_heights,
                    ring_proj_pts, ring_channels_list)):
            if not proj2 or not truss_ring_visible[_ri2]:
                continue
            _c2 = ring_colors[_ri2 % len(ring_colors)]
            W22, D22 = tw2 / 2, td2 / 2
            _px2 = [p[0] - _t_ox for p in proj2]
            _py2 = [p[1] - _t_oy for p in proj2]
            _pz2 = [p[2] for p in proj2]
            _ct2 = [str(c) for c in ch2]
            def _altpos(horiz_vals, invert=False):
                """Alternate top/bottom per speaker; left/right from horizontal sign."""
                out = []
                for j, v in enumerate(horiz_vals):
                    vert = 'top' if j % 2 == 0 else 'bottom'
                    if invert:
                        side = ' left' if v > 0.05 else (' right' if v < -0.05 else ' center')
                    else:
                        side = ' right' if v > 0.05 else (' left' if v < -0.05 else ' center')
                    out.append(vert + side)
                return out

            if _t2d_view == "Front":
                # Y axis inverted: positive Y appears on LEFT of screen
                _tpos = _altpos(_py2, invert=True)
                fig_truss_2d.add_trace(go.Scatter(
                    x=[-W22 - _t_oy, W22 - _t_oy], y=[th2, th2],
                    mode='lines', line=dict(color=_c2, width=3),
                    name=f"Ring {_ri2+1}"))
                fig_truss_2d.add_trace(go.Scatter(
                    x=_py2, y=_pz2, mode='markers+text',
                    text=_ct2, textposition=_tpos, textfont=_tfont,
                    marker=dict(size=7, color=_c2), cliponaxis=False, showlegend=False))
            elif _t2d_view == "Side":
                # X axis not inverted: positive X appears on RIGHT of screen
                _tpos = _altpos(_px2, invert=False)
                fig_truss_2d.add_trace(go.Scatter(
                    x=[-D22 - _t_ox, D22 - _t_ox], y=[th2, th2],
                    mode='lines', line=dict(color=_c2, width=3),
                    name=f"Ring {_ri2+1}"))
                fig_truss_2d.add_trace(go.Scatter(
                    x=_px2, y=_pz2, mode='markers+text',
                    text=_ct2, textposition=_tpos, textfont=_tfont,
                    marker=dict(size=7, color=_c2), cliponaxis=False, showlegend=False))
            else:  # Top
                # screen: x = −py (Y inverted), y = px (front = up)
                _tpos = [_tpos2d(-py, px) for py, px in zip(_py2, _px2)]
                fig_truss_2d.add_trace(go.Scatter(
                    x=[-W22 - _t_oy, W22 - _t_oy, W22 - _t_oy, -W22 - _t_oy, -W22 - _t_oy],
                    y=[-D22 - _t_ox, -D22 - _t_ox, D22 - _t_ox, D22 - _t_ox, -D22 - _t_ox],
                    mode='lines', line=dict(color=_c2, width=3),
                    name=f"Ring {_ri2+1}"))
                fig_truss_2d.add_trace(go.Scatter(
                    x=_py2, y=_px2, mode='markers+text',
                    text=_ct2, textposition=_tpos, textfont=_tfont,
                    marker=dict(size=7, color=_c2), cliponaxis=False, showlegend=False))
        if _t2d_view == "Front":
            fig_truss_2d.update_xaxes(title_text="Y (m) + left / − right",
                autorange='reversed', **_grid)
            fig_truss_2d.update_yaxes(title_text="Z (m, above floor)",
                scaleanchor="x", scaleratio=1, **_grid)
        elif _t2d_view == "Side":
            fig_truss_2d.update_xaxes(title_text="X (m) − back / + front",
                **_grid)
            fig_truss_2d.update_yaxes(title_text="Z (m, above floor)",
                scaleanchor="x", scaleratio=1, **_grid)
        else:  # Top
            fig_truss_2d.update_xaxes(title_text="Y (m) + left / − right",
                autorange='reversed', **_grid)
            fig_truss_2d.update_yaxes(title_text="X (m) − back / + front",
                scaleanchor="x", scaleratio=1, **_grid)
        fig_truss_2d.update_layout(
            height=500, margin=dict(l=0, r=10, b=0, t=10),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=1.0),
            uirevision=_t2d_view,
        )
        st.plotly_chart(fig_truss_2d, use_container_width=True)
    st.markdown("**Elevation Changes (Sphere → Truss)**")
    st.dataframe(
        elev_df.style.apply(highlight_large_delta, axis=1),
        use_container_width=True, hide_index=True
    )


with st.expander("🏠 Wall Mount Planner", key="wall_expander",
                  expanded=st.session_state.pop("_wall_exp_init", st.session_state.get("wall_expander", False))):
    # --- Wall Mount Planner ---
    st.subheader("🏠 Wall Mount Planner")

    col_rw, col_rl, col_rh = st.columns(3)
    with col_rw:
        room_width = st.number_input("Room Width (m)", min_value=0.1, max_value=500.0, value=10.0, step=0.5)
    with col_rl:
        room_length = st.number_input("Room Length (m)", min_value=0.1, max_value=500.0, value=10.0, step=0.5)
    with col_rh:
        room_height = st.number_input("Room Height (m)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)

    _wall_vis_rings = [(i, c) for i, c in enumerate(ring_point_counts) if c > 0]
    wall_ring_visible = [True] * len(theta_vals)
    if _wall_vis_rings:
        st.markdown("**Ring Visibility**")
        _wvcols = st.columns(min(len(_wall_vis_rings), 5))
        for _wci, (i, _) in enumerate(_wall_vis_rings):
            wall_ring_visible[i] = _wvcols[_wci % 5].checkbox(
                f"Ring {i+1}", value=True, key=f"wall_show_{i}_{cfg_key}"
            )

    # Adjusted frame: dome center (listening position) at origin, z up
    half_rw  = room_width  / 2
    half_rl  = room_length / 2
    z_ceil   = room_height - listener_height
    z_floor  = -listener_height

    _w_origin = st.selectbox("Coordinate Origin", _ORIGIN_OPTS, key=f"wall_origin_{cfg_key}")
    _w_ox, _w_oy = {
        "Center":      (0.0,      0.0),
        "Front Left":  (half_rl,  half_rw),
        "Front Right": (half_rl, -half_rw),
        "Rear Left":   (-half_rl,  half_rw),
        "Rear Right":  (-half_rl, -half_rw),
    }[_w_origin]

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
            _ch_wj = _ch_cursor + j
            # Apply per-speaker position override if present
            if _ch_wj in _edited_positions:
                _az_ov, _el_ov = _edited_positions[_ch_wj]
                _theta_ov = np.radians(90 - _el_ov)
                _phi_ov = np.radians(_az_ov)
            else:
                _theta_ov, _phi_ov = theta, phi

            dx = float(np.sin(_theta_ov) * np.cos(_phi_ov))
            dy = float(np.sin(_theta_ov) * np.sin(_phi_ov))
            dz = float(np.cos(_theta_ov))

            candidates = []
            if abs(dx) > _eps:
                candidates.append((half_rl / abs(dx), 'x+' if dx > 0 else 'x-'))
            if abs(dy) > _eps:
                candidates.append((half_rw / abs(dy), 'y+' if dy > 0 else 'y-'))
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

            az_deg = float(np.degrees(_phi_ov))
            if az_deg > 180:
                az_deg -= 360

            wall_data.append({
                "channel":        _ch_wj,
                "ring_index":     i,
                "surface":        surface,
                "surface_name":   SURFACE_NAMES[surface],
                "color":          SURFACE_COLORS[surface],
                "x":              round(px, 3),
                "y":              round(py, 3),
                "z":              round(pz, 3),
                "h_above_floor":  round(pz + listener_height, 3),
                "azimuth":        round(az_deg, 2),
                "elevation":      round(float(90 - np.degrees(_theta_ov)), 2),
            })

        _ch_cursor += M

    # Build wall mount table before columns
    import pandas as pd
    wall_df = pd.DataFrame([{
        "Ch":                     d["channel"],
        "Surface":                d["surface_name"],
        "x (m)":                  round(d["x"] - _w_ox, 3),
        "y (m)":                  round(d["y"] - _w_oy, 3),
        "Height (m)":             d["h_above_floor"],
        "Az (°)":                 d["azimuth"],
        "Elev (°)":               d["elevation"],
    } for d in wall_data if wall_ring_visible[d["ring_index"]]])

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
        x=xs, y=ys, z=zs + listener_height,
        showscale=False, opacity=0.08,
        colorscale='Greys', hoverinfo='skip', showlegend=False
    ))

    # Room box — z_floor/z_ceil are listener-relative; add listener_height for absolute coords
    bx, by, bz = _box_edges(-half_rl - _w_ox, half_rl - _w_ox, -half_rw - _w_oy, half_rw - _w_oy,
                             z_floor + listener_height, z_ceil + listener_height)
    fig_wall.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz, mode='lines',
        line=dict(color='lightgrey', width=2),
        name='Room', showlegend=False, hoverinfo='skip'
    ))

    # Listening position marker
    fig_wall.add_trace(go.Scatter3d(
        x=[-_w_ox], y=[-_w_oy], z=[listener_height], mode='markers',
        marker=dict(size=6, color='black', symbol='cross'),
        name='Listening position'
    ))

    # Original sphere positions
    if len(points) > 0:
        fig_wall.add_trace(go.Scatter3d(
            x=points[:, 0] - _w_ox, y=points[:, 1] - _w_oy, z=points[:, 2] + listener_height,
            mode='markers',
            marker=dict(size=5, color='white', opacity=0.9,
                        line=dict(color='grey', width=2), symbol='circle'),
            name='Original (sphere)'
        ))

    # Per-surface: projection lines + speaker dots
    for surf_key in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
        pts = [d for d in wall_data if d['surface'] == surf_key and wall_ring_visible[d['ring_index']]]
        if not pts:
            continue
        color = SURFACE_COLORS[surf_key]

        lx, ly, lz = [], [], []
        for d in pts:
            lx += [-_w_ox, d['x'] - _w_ox, None]
            ly += [-_w_oy, d['y'] - _w_oy, None]
            lz += [listener_height, d['z'] + listener_height, None]
        fig_wall.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))

        fig_wall.add_trace(go.Scatter3d(
            x=[d['x'] - _w_ox for d in pts],
            y=[d['y'] - _w_oy for d in pts],
            z=[d['z'] + listener_height for d in pts],
            mode='markers+text',
            marker=dict(size=6, color=color),
            text=[str(d['channel']) for d in pts],
            textposition='top center',
            textfont=dict(color='#111111', size=13, family='Arial Black'),
            name=SURFACE_NAMES[surf_key]
        ))

    fig_wall.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (m)  − back / + front'),
            yaxis=dict(title='Y (m)  − right / + left'),
            zaxis=dict(title='Z (m)'),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Speaker Projection: Sphere → Room Walls",
        scene_camera=dict(eye=dict(x=1.8, y=0, z=0.5)),
        legend=dict(yanchor="top", y=0.85, xanchor="right", x=1.0)
    )

    col_wv, col_2d_wall = st.columns(2)
    with col_wv:
        _wv_cams = st.columns(6)
        for _vi, (_vn, _vx, _vy, _vz) in enumerate([
            ("3D", 1.8, 0, 0.5), ("Top", 0, 0, 2.5),
            ("Front", 2.5, 0, 0.3), ("Back", -2.5, 0, 0.3),
            ("Left", 0, 2.5, 0.3), ("Right", 0, -2.5, 0.3),
        ]):
            if _wv_cams[_vi].button(_vn, key=f"wall_cam_{_vn}", use_container_width=True):
                st.session_state["wall_3d_camera"] = dict(eye=dict(x=_vx, y=_vy, z=_vz))
        fig_wall.update_layout(
            scene_camera=st.session_state.get("wall_3d_camera", dict(eye=dict(x=1.8, y=0, z=0.5)))
        )
        st.plotly_chart(fig_wall, use_container_width=True)
    with col_2d_wall:
        _w2d_view = st.selectbox("2D View", ["Front", "Side", "Top"], index=2,
                                 key=f"wall_2d_view_{cfg_key}")
        fig_wall_2d = go.Figure()
        _wgrid = dict(showgrid=True, gridcolor='rgba(128,128,128,0.25)', gridwidth=1, dtick=1)

        def _wline(x0, x1, y0, y1, surf_key):
            c = SURFACE_COLORS[surf_key]
            return go.Scatter(
                x=[x0, x1], y=[y0, y1], mode='lines',
                line=dict(color=c, width=3),
                name=SURFACE_NAMES[surf_key], showlegend=False, hoverinfo='skip')

        # Room walls as coloured lines
        if _w2d_view == "Front":  # Y-Z plane
            fig_wall_2d.add_trace(_wline(half_rw - _w_oy, half_rw - _w_oy, 0, room_height, 'y+'))
            fig_wall_2d.add_trace(_wline(-half_rw - _w_oy, -half_rw - _w_oy, 0, room_height, 'y-'))
            fig_wall_2d.add_trace(_wline(-half_rw - _w_oy, half_rw - _w_oy, room_height, room_height, 'z+'))
            fig_wall_2d.add_trace(_wline(-half_rw - _w_oy, half_rw - _w_oy, 0, 0, 'z-'))
        elif _w2d_view == "Side":  # X-Z plane
            fig_wall_2d.add_trace(_wline(half_rl - _w_ox, half_rl - _w_ox, 0, room_height, 'x+'))
            fig_wall_2d.add_trace(_wline(-half_rl - _w_ox, -half_rl - _w_ox, 0, room_height, 'x-'))
            fig_wall_2d.add_trace(_wline(-half_rl - _w_ox, half_rl - _w_ox, room_height, room_height, 'z+'))
            fig_wall_2d.add_trace(_wline(-half_rl - _w_ox, half_rl - _w_ox, 0, 0, 'z-'))
        else:  # Top — Y-X plane
            fig_wall_2d.add_trace(_wline(half_rw - _w_oy, half_rw - _w_oy, -half_rl - _w_ox, half_rl - _w_ox, 'y+'))
            fig_wall_2d.add_trace(_wline(-half_rw - _w_oy, -half_rw - _w_oy, -half_rl - _w_ox, half_rl - _w_ox, 'y-'))
            fig_wall_2d.add_trace(_wline(-half_rw - _w_oy, half_rw - _w_oy, half_rl - _w_ox, half_rl - _w_ox, 'x+'))
            fig_wall_2d.add_trace(_wline(-half_rw - _w_oy, half_rw - _w_oy, -half_rl - _w_ox, -half_rl - _w_ox, 'x-'))
        # Listener marker
        if _w2d_view == "Front":
            _w2d_lx, _w2d_ly = -_w_oy, listener_height
        elif _w2d_view == "Side":
            _w2d_lx, _w2d_ly = -_w_ox, listener_height
        else:
            _w2d_lx, _w2d_ly = -_w_oy, -_w_ox
        fig_wall_2d.add_trace(go.Scatter(
            x=[_w2d_lx], y=[_w2d_ly], mode='markers',
            marker=dict(size=10, color='white', symbol='cross',
                        line=dict(color='grey', width=2)),
            cliponaxis=False, name='Listener'))
        # Speakers per surface
        for _skey in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
            _spts = [d for d in wall_data
                     if d['surface'] == _skey and wall_ring_visible[d['ring_index']]]
            if not _spts:
                continue
            _sc = SURFACE_COLORS[_skey]
            _sn = SURFACE_NAMES[_skey]
            _spx = [d['x'] - _w_ox for d in _spts]
            _spy = [d['y'] - _w_oy for d in _spts]
            _spz = [d['z'] + listener_height for d in _spts]
            _spt = [str(d['channel']) for d in _spts]
            if _w2d_view == "Front":
                _wtpos = _altpos(_spy, invert=True)
                fig_wall_2d.add_trace(go.Scatter(
                    x=_spy, y=_spz, mode='markers+text',
                    text=_spt, textposition=_wtpos, textfont=_tfont,
                    marker=dict(size=7, color=_sc), cliponaxis=False, name=_sn))
            elif _w2d_view == "Side":
                _wtpos = _altpos(_spx, invert=False)
                fig_wall_2d.add_trace(go.Scatter(
                    x=_spx, y=_spz, mode='markers+text',
                    text=_spt, textposition=_wtpos, textfont=_tfont,
                    marker=dict(size=7, color=_sc), cliponaxis=False, name=_sn))
            else:  # Top
                _wtpos = [_tpos2d(-sy, sx) for sy, sx in zip(_spy, _spx)]
                fig_wall_2d.add_trace(go.Scatter(
                    x=_spy, y=_spx, mode='markers+text',
                    text=_spt, textposition=_wtpos, textfont=_tfont,
                    marker=dict(size=7, color=_sc), cliponaxis=False, name=_sn))
        if _w2d_view == "Front":
            fig_wall_2d.update_xaxes(title_text="Y (m) + left / − right",
                autorange='reversed', **_wgrid)
            fig_wall_2d.update_yaxes(title_text="Z (m, above floor)",
                scaleanchor="x", scaleratio=1, **_wgrid)
        elif _w2d_view == "Side":
            fig_wall_2d.update_xaxes(title_text="X (m) − back / + front",
                **_wgrid)
            fig_wall_2d.update_yaxes(title_text="Z (m, above floor)",
                scaleanchor="x", scaleratio=1, **_wgrid)
        else:  # Top
            fig_wall_2d.update_xaxes(title_text="Y (m) + left / − right",
                autorange='reversed', **_wgrid)
            fig_wall_2d.update_yaxes(title_text="X (m) − back / + front",
                scaleanchor="x", scaleratio=1, **_wgrid)
        fig_wall_2d.update_layout(
            height=500, margin=dict(l=0, r=10, b=0, t=10),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=1.0),
            uirevision=_w2d_view,
        )
        st.plotly_chart(fig_wall_2d, use_container_width=True)
    st.markdown("**Mount Positions**")
    st.dataframe(wall_df, use_container_width=True, hide_index=True)


# --- JSON download ---
_safe_title = re.sub(r'[^\w\-]', '_', layout_title)[:40] if layout_title else ""
json_data = {
    "Name": layout_title or "All-Round Ambisonic decoder loudspeaker layout, importable in IEM AllRAD Decoder Plugin.",
    "Description": (layout_description + " — " if layout_description else "") +
                   f"Created with the Dome Loudspeaker Layout Generator by Matthias Kronlachner. {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "LoudspeakerLayout": {
        "Name": layout_title or "Generated loudspeaker layout",
        "Loudspeakers": spherical_coords
    }
}

json_str = json.dumps(json_data, indent=2)
json_bytes = json_str.encode("utf-8")
filename = f"{_safe_title + '_' if _safe_title else ''}AllRAD_speakerLayout_N{N_points}_R{N_rings}_VoG{int(Voice_of_God)}_RBH{int(Ring_below_horizon)}.json"

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
    _spk_offsets = [
        {"ch": ch, "daz": round(daz, 4), "del": round(del_, 4)}
        for ch, (daz, del_) in _delta_values.items()
        if abs(daz) > 1e-9 or abs(del_) > 1e-9
    ]
    _cfg = {
        "n":           N_points,
        "rings":       N_rings,
        "vog":         int(Voice_of_God),
        "rbh":         int(Ring_below_horizon),
        "r":           dome_radius,
        "lh":          listener_height,
        "title":       layout_title,
        "desc":        layout_description,
        "rings_data":  _rings_data,
        "spk_offsets": _spk_offsets,
        "truss_exp":   int(st.session_state.get("truss_expander", False)),
        "wall_exp":    int(st.session_state.get("wall_expander",  False)),
    }
    _encoded = base64.b64encode(json.dumps(_cfg).encode()).decode()
    st.query_params["cfg"] = _encoded
    st.success("URL updated — copy it from your browser's address bar to share this configuration.")

st.markdown("---")
st.markdown("© 2025–2026 Matthias Kronlachner")
