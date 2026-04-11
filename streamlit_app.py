import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime
import matplotlib.pyplot as plt

# --- Parameter controls at the top ---
st.title("🌐 Ambisonic Dome Loudspeaker Layout Generator")

col1, col2, col3 = st.columns(3)
with col1:
    N_points = st.number_input("Numer of total Speakers", min_value=1, max_value=1000, value=23, step=1)
with col2:
    N_rings = st.number_input("Number of Elevation Rings", min_value=1, max_value=20, value=4, step=1)
    Voice_of_God = st.checkbox("Include Voice of God", value=True)
    Ring_below_horizon = st.checkbox("Add Ring Below Horizon", value=False)
with col3:
    dome_radius = st.number_input("Dome Radius (m)", min_value=0.1, max_value=500.0, value=1.0, step=0.5)

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

stagger_offset = 0 if Ring_below_horizon else 1
cfg_key = f"N{N_points}_R{N_rings}_VoG{int(Voice_of_God)}_RBH{int(Ring_below_horizon)}_r{dome_radius}"

for i in range(len(default_theta_vals)):
    with st.expander(f"Ring {i+1} Settings", expanded=False):
        elev_deg = round(90 - np.degrees(default_theta_vals[i]), 2)
        elev_input = st.number_input(
            f"Elevation of Ring {i+1} (degrees)",
            min_value=-90.0, max_value=90.0,
            value=elev_deg,
            step=1.0, key=f"elev_{i}_{cfg_key}"
        )
        theta = np.radians(90 - elev_input)
        theta_vals.append(theta)

        count_input = st.number_input(
            f"Speakers in Ring {i+1}",
            min_value=0,
            value=int(default_ring_counts[i]),
            step=1, key=f"count_{i}_{cfg_key}"
        )
        ring_point_counts.append(count_input)

        default_az_offset = round(180.0 / count_input, 4) if (i % 2 == stagger_offset and count_input > 0) else 0.0
        az_offset_input = st.number_input(
            f"Azimuth Offset of Ring {i+1} (degrees)",
            min_value=-180.0, max_value=180.0,
            value=default_az_offset,
            step=1.0, key=f"az_offset_{i}_{cfg_key}"
        )
        azimuth_offsets.append(az_offset_input)

# --- Core logic ---
r = dome_radius
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
st.subheader("🌐 3D view")

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
        xaxis=dict(range=[-scale, scale], title='X'),
        yaxis=dict(range=[-scale, scale], title='Y'),
        zaxis=dict(range=[-scale, scale], title='Z')
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    title="Points Distributed on Full Sphere with Indices"
)

st.plotly_chart(fig)

# --- Mollweide projection plot ---
st.subheader("🌍 Mollweide Projection of Speaker Positions")

azimuths = [spk["Azimuth"] for spk in spherical_coords if not spk["IsImaginary"]]
elevations = [spk["Elevation"] for spk in spherical_coords if not spk["IsImaginary"]]
labels = [str(spk["Channel"]) for spk in spherical_coords if not spk["IsImaginary"]]

azimuths_rad = np.radians(azimuths)
elevations_rad = np.radians(elevations)

fig2, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
ax.grid(True, linestyle='--', linewidth=0.5)
ax.scatter(azimuths_rad, elevations_rad, color='yellow', s=20)

for x, y, label in zip(azimuths_rad, elevations_rad, labels):
    ax.text(x, y, label, fontsize=12, fontweight='bold', ha='center', va='center', color='black')

ax.set_xticklabels(['150°W','120°W','90°W','60°W','30°W','0°','30°E','60°E','90°E','120°E','150°E'])
ax.set_title("Mollweide Projection (Azimuth vs. Elevation)", fontsize=12, pad=20)

st.pyplot(fig2)

# --- Display speaker counts per ring ---
st.subheader("🔊 Speakers per Ring")

ring_info = []
for i, (theta, count) in enumerate(zip(theta_vals, ring_point_counts)):
    elevation_deg = round(90 - np.degrees(theta), 2)
    ring_info.append(f"Ring {i+1} (Elevation: {elevation_deg}°): {count} speakers")

for info in ring_info:
    st.markdown(f"- {info}")

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


# --- Truss Planner ---
st.subheader("🏗️ Truss Planner")

truss_widths = []
truss_depths = []
truss_heights = []

for i, (theta, M) in enumerate(zip(theta_vals, ring_point_counts)):
    with st.expander(f"Ring {i+1} Truss Settings", expanded=False):
        default_w = round(float(2 * np.sin(theta) * r), 4)
        default_h = round(float(np.cos(theta) * r), 4)

        tw = st.number_input(
            f"Truss Width (Ring {i+1})", min_value=0.0, max_value=float(r * 20),
            value=default_w, step=float(r * 0.1), key=f"tw_{i}_{cfg_key}"
        )
        td = st.number_input(
            f"Truss Depth (Ring {i+1})", min_value=0.0, max_value=float(r * 20),
            value=default_w, step=float(r * 0.1), key=f"td_{i}_{cfg_key}"
        )
        th = st.number_input(
            f"Truss Height (Ring {i+1})", min_value=float(-r * 2), max_value=float(r * 2),
            value=default_h, step=float(r * 0.05), key=f"th_{i}_{cfg_key}"
        )
        truss_widths.append(tw)
        truss_depths.append(td)
        truss_heights.append(th)

# Compute projected positions per ring
ring_orig_pts = []
ring_proj_pts = []
ring_channels_list = []
projected_elevations_all = []
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
        channels_ring.append(channel_cursor + j)

    ring_orig_pts.append(orig_ring)
    ring_proj_pts.append(proj_ring)
    ring_channels_list.append(channels_ring)
    channel_cursor += M

# Truss 3D visualisation
st.subheader("🏗️ Truss Projection 3D View")

ring_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
               '#42d4f4', '#f032e6', '#bfef45', '#469990', '#dcbeff']

fig_truss = go.Figure()

fig_truss.add_trace(go.Surface(
    x=xs, y=ys, z=zs,
    showscale=False, opacity=0.1,
    colorscale='Greys', name="Sphere", hoverinfo='skip'
))

if len(points) > 0:
    fig_truss.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=4, color='lightgrey', opacity=0.7),
        name="Original (sphere)"
    ))

for i, (tw, td, th, orig_ring, proj_ring, ch_ring) in enumerate(
        zip(truss_widths, truss_depths, truss_heights, ring_orig_pts, ring_proj_pts, ring_channels_list)):
    if not orig_ring:
        continue
    color = ring_colors[i % len(ring_colors)]
    W2, D2 = tw / 2, td / 2

    # Truss wireframe
    fig_truss.add_trace(go.Scatter3d(
        x=[-W2,  W2,  W2, -W2, -W2],
        y=[-D2, -D2,  D2,  D2, -D2],
        z=[ th,  th,  th,  th,  th],
        mode='lines',
        line=dict(color=color, width=4),
        name=f"Ring {i+1} Truss"
    ))

    # Connecting lines original → projected
    line_x, line_y, line_z = [], [], []
    for orig, proj in zip(orig_ring, proj_ring):
        line_x += [orig[0], proj[0], None]
        line_y += [orig[1], proj[1], None]
        line_z += [orig[2], proj[2], None]
    fig_truss.add_trace(go.Scatter3d(
        x=line_x, y=line_y, z=line_z,
        mode='lines',
        line=dict(color=color, width=1, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))

    # Projected speaker positions
    fig_truss.add_trace(go.Scatter3d(
        x=[p[0] for p in proj_ring],
        y=[p[1] for p in proj_ring],
        z=[p[2] for p in proj_ring],
        mode='markers+text',
        marker=dict(size=5, color=color),
        text=[str(c) for c in ch_ring],
        textposition='top center',
        name=f"Ring {i+1} Projected"
    ))

max_half = max((max(tw, td) / 2 for tw, td in zip(truss_widths, truss_depths) if max(tw, td) > 0), default=1.0)
scale_truss = max(1.1, max_half + 0.1)
fig_truss.update_layout(
    scene=dict(
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1),
        xaxis=dict(range=[-scale_truss, scale_truss], title='X'),
        yaxis=dict(range=[-scale_truss, scale_truss], title='Y'),
        zaxis=dict(range=[-scale_truss, scale_truss], title='Z')
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    title="Speaker Projection: Sphere → Truss"
)
st.plotly_chart(fig_truss)

# Elevation comparison table
with st.expander("📐 Show Elevation Changes (Sphere → Truss)"):
    import pandas as pd
    elev_rows = []
    for ch, az, orig_el, proj_el in zip(
            orig_channels_for_table, orig_azimuths_for_table,
            orig_elevations_for_table, projected_elevations_all):
        delta = round(proj_el - orig_el, 2)
        elev_rows.append({
            "Channel": ch,
            "Azimuth (°)": az,
            "Original Elev (°)": orig_el,
            "Projected Elev (°)": round(proj_el, 2),
            "Δ Elev (°)": delta
        })

    elev_df = pd.DataFrame(elev_rows)

    def highlight_large_delta(row):
        color = 'background-color: #ffcccc' if abs(row["Δ Elev (°)"]) > 5 else ''
        return [color] * len(row)

    st.dataframe(
        elev_df.style.apply(highlight_large_delta, axis=1),
        use_container_width=True, hide_index=True
    )


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

st.markdown("---")
st.markdown("© 2025 Matthias Kronlachner")
