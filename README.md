# 🌐 Ambisonic Dome Loudspeaker Layout Generator

A Streamlit app to design and visualise loudspeaker layouts for ambisonic dome systems, with export support for the IEM AllRADecoder plugin.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dome-loudspeaker-layout-generator.streamlit.app/)

## Features

- **Ring-based layout generation** — configure the number of elevation rings and total speaker count; speakers are distributed proportionally by ring area
- **Per-ring settings** — adjust elevation, speaker count, and azimuth offset for each ring individually; actual total speaker count is shown live
- **Voice of God & below-horizon rings** — optional top speaker and sub-horizon ring
- **Dome radius scaling** — set the physical dome radius in metres
- **Listener height** — global reference height used by both the truss and wall planners
- **3D sphere visualisation** — interactive Plotly view with correct ambisonics axis orientation (x = front, y = left, z = up)
- **Mollweide projection** — 2D overview with left/right labelled correctly
- **Loudspeaker coordinates table** — channel, azimuth, elevation, and Cartesian coordinates (x, y, z)
- **IEM AllRADecoder JSON export** — download a layout file ready to import into the [IEM AllRADecoder plugin](https://plugins.iem.at)
- **URL-based config sharing** — encode the full configuration in a shareable URL via base64 query parameter
- **🏗️ Truss Planner** — configure a per-ring rectangular truss (width, depth, height); speakers are projected outward onto the truss while preserving azimuth; view 3D projection and an elevation-change table with heights above floor
- **🏠 Wall Mount Planner** — provide room dimensions (width, length, height) and project speakers onto the nearest wall, ceiling, or floor surface; 3D room visualisation with mount positions and a table of mounting coordinates per channel

## How to run it on your own machine

1. Install the requirements

   ```
   pip install -r requirements.txt
   ```

2. Run the app

   ```
   streamlit run streamlit_app.py
   ```
