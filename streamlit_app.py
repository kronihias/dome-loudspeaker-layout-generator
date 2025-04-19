import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime

# --- Parameter controls at the top ---
st.title("🎧 Loudspeaker Layout Generator")

col1, col2 = st.columns(2)
with col1:
    N_points = st.number_input("Total Points", min_value=1, max_value=1000, value=23, step=1)
with col2:
    N_rings = st.number_input("Number of Rings", min_value=1, max_value=20, value=4, step=1)
    Voice_of_God = st.checkbox("Include Voice of God", value=True)
    Ring_below_horizon = st.checkbox("Include Ring Below Horizon", value=False)
    
# --- Core logic ---
r = 1  # Radius of the sphere

theta_vals = np.linspace(np.pi/2, 0, N_rings)

if Ring_below_horizon:
    theta_below = np.pi - theta_vals[1]
    theta_vals = np.insert(theta_vals, 0, theta_below)

ring_weights = np.abs(np.sin(theta_vals))
ring_point_counts = np.round(ring_weights / ring_weights.sum() * N_points).astype(int)

if Voice_of_God:
    ring_point_counts[-1] = 1

diff = N_points - ring_point_counts.sum()
ring_point_counts[0] += diff

spherical_coords = []
points = []

for i, (theta, M) in enumerate(zip(theta_vals, ring_point_counts)):
    if M == 0:
        continue

    stagger_offset = 0 if Ring_below_horizon else 1

    if i % 2 == stagger_offset:
        phi_offset = (2 * np.pi) / M / 2
    else:
        phi_offset = 0

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

scale = 1.1
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

# --- Display speaker counts per ring ---
st.subheader("🔊 Speakers per Ring")

ring_info = []
for i, (theta, count) in enumerate(zip(theta_vals, ring_point_counts)):
    elevation_deg = round(90 - np.degrees(theta), 2)
    ring_info.append(f"Ring {i+1} (Elevation: {elevation_deg}°): {count} speakers")

for info in ring_info:
    st.markdown(f"- {info}")


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

# Filename with parameters
filename = f"AllRAD_speakerLayout_N{N_points}_R{N_rings}_VoG{int(Voice_of_God)}_RBH{int(Ring_below_horizon)}.json"

st.download_button(
    label="📥 Download IEM AllRAD Layout JSON",
    data=json_bytes,
    file_name=filename,
    mime="application/json"
)
