"""Convert an IEM AllRADecoder loudspeaker JSON into a share link for the
Dome Loudspeaker Layout Generator.

usage: python iem_json_to_link.py LAYOUT.json [--title T] [--desc D]
                                  [--app URL] [--ring-tol DEG] [--radius M]

Speakers are clustered into rings by elevation. Within a ring the speaker
with the lowest channel number becomes the ring's azimuth-offset anchor and
the others follow in channel order; any deviation from an evenly spaced ring
is stored as per-speaker dAz/dEl offsets, so irregular layouts round-trip
exactly. Channel numbers match the file whenever it numbers the horizon ring
first, then upwards, with an optional below-horizon ring last (the app's own
scheme); otherwise the remapping is printed as a warning.
"""
import argparse
import json
import sys
import urllib.parse

import numpy as np

import share_link

APP_URL = "https://dome-loudspeaker-layout-generator.streamlit.app/"


def wrap180(a):
    """Wrap angle(s) in degrees to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def _cluster_rings(spk, ring_tol):
    spk = sorted(spk, key=lambda s: (float(s["Elevation"]), int(s["Channel"])))
    rings = []
    for s in spk:
        if rings and abs(float(s["Elevation"]) - rings[-1]["el"]) <= ring_tol:
            rings[-1]["spk"].append(s)
            rings[-1]["el"] = float(np.mean([float(x["Elevation"]) for x in rings[-1]["spk"]]))
        else:
            rings.append({"el": float(s["Elevation"]), "spk": [s]})
    return rings


def _by_channel(ring):
    return sorted(ring["spk"], key=lambda x: int(x["Channel"]))


def _anchor_az(ring):
    return 0.0 if ring["el"] >= 90.0 else wrap180(float(_by_channel(ring)[0]["Azimuth"]))


def convert(spk, title="", desc="", ring_tol=3.0, radius=None, listener_height=1.3):
    """Return ``(cfg, warnings)`` where ``cfg`` is the app's share-link dict."""
    spk = [s for s in spk if not s.get("IsImaginary", False)]
    warnings = []

    rings = _cluster_rings(spk, ring_tol)
    above = [r for r in rings if r["el"] >= -1e-6]
    below = [r for r in rings if r["el"] < -1e-6]
    if not above:
        raise SystemExit("no loudspeakers at or above the horizon")
    if len(below) > 1:
        warnings.append(f"{len(below)} below-horizon rings found; the app supports one. "
                        "Merging them into a single ring with per-speaker elevation offsets.")
        merged = {"spk": sum((r["spk"] for r in below), [])}
        merged["el"] = float(np.mean([float(x["Elevation"]) for x in merged["spk"]]))
        below = [merged]

    vog = len(above[-1]["spk"]) == 1 and above[-1]["el"] >= 89.0
    if vog:
        above[-1]["el"] = 90.0
    ordered = below + above                      # widget order (ring index i)

    # Numbering direction: do channel numbers increase with azimuth (CCW)?
    votes = 0
    for r in ordered:
        s = _by_channel(r)
        if len(s) >= 2:
            az = np.array([float(x["Azimuth"]) for x in s])
            votes += int(np.sign(np.median(wrap180(np.diff(az)))))
    clockwise = votes < 0

    # The app numbers the above-horizon rings first (bottom-up), then the
    # below-horizon ring.
    n_above = len(above)
    numbering = (list(range(1, n_above + 1)) + [0]) if below else list(range(n_above))
    base = 1
    spk_offsets, mapping = [], []
    for i in numbering:
        r = ordered[i]
        s = _by_channel(r)
        M = len(s)
        offset = _anchor_az(r)
        if clockwise:
            app_ch = [base] + [base + (M - j) for j in range(1, M)]
        else:
            app_ch = [base + j for j in range(M)]
        for k, x in enumerate(s):                # k-th channel of the ring -> app channel base+k
            j = app_ch.index(base + k)
            gen_az = wrap180(offset + 360.0 * j / M)
            daz = float(x["Azimuth"]) - gen_az   # not wrapped: the app clamps, not wraps
            if abs(daz) > 180:
                daz = wrap180(daz)
                if abs(gen_az + daz) > 180 + 1e-6:
                    warnings.append(f"channel {x['Channel']}: azimuth offset {daz:.1f} deg "
                                    "cannot be represented exactly")
            del_ = float(x["Elevation"]) - r["el"]
            if abs(daz) > 1e-6 or abs(del_) > 1e-6:
                spk_offsets.append({"ch": base + k, "daz": round(daz, 4), "del": round(del_, 4)})
            mapping.append((int(x["Channel"]), base + k))
        base += M

    rings_data = [{"elev": round(r["el"], 4), "count": len(r["spk"]),
                   "az": round(_anchor_az(r), 4), "tw": 0.0, "td": 0.0, "th": 0.0}
                  for r in ordered]

    renumbered = [(a, b) for a, b in mapping if a != b]
    if renumbered:
        warnings.append(f"{len(renumbered)} of {len(spk)} loudspeakers get a different "
                        "channel number in the app (file -> app): " +
                        ", ".join(f"{a}->{b}" for a, b in renumbered[:12]) +
                        (" ..." if len(renumbered) > 12 else ""))

    if radius is None:
        mean_r = float(np.mean([float(s.get("Radius", 1.0)) for s in spk]))
        radius = round(mean_r, 2) if mean_r > 1.05 else 3.0

    cfg = {
        "n": len(spk), "rings": n_above, "vog": int(vog), "rbh": int(bool(below)),
        "r": radius, "lh": listener_height, "title": title, "desc": desc,
        "dir": int(clockwise), "dec_order": 5, "dec_weights": "maxrE",
        "rings_data": rings_data, "spk_offsets": spk_offsets,
        "truss_exp": 0, "wall_exp": 0,
    }
    return cfg, warnings


def make_link(cfg, app=APP_URL):
    return app + "?cfg=" + urllib.parse.quote(share_link.encode_cfg(cfg), safe="")


def load_speakers(path):
    with open(path) as f:
        d = json.load(f)
    if "LoudspeakerLayout" in d:
        d = d["LoudspeakerLayout"]
    return d["Loudspeakers"] if isinstance(d, dict) else d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="IEM AllRADecoder layout .json")
    ap.add_argument("--title", default="", help="layout title shown in the app")
    ap.add_argument("--desc", default="", help="layout description")
    ap.add_argument("--app", default=APP_URL, help="base URL of the app")
    ap.add_argument("--ring-tol", type=float, default=3.0,
                    help="elevation tolerance (deg) for grouping speakers into rings")
    ap.add_argument("--radius", type=float, default=None,
                    help="dome radius in metres (default: from file, or 3.0)")
    a = ap.parse_args(argv)
    cfg, warns = convert(load_speakers(a.path), a.title, a.desc, a.ring_tol, a.radius)
    for w in warns:
        print("WARNING:", w, file=sys.stderr)
    print("rings (elevation, speakers):",
          [(r["elev"], r["count"]) for r in cfg["rings_data"]], file=sys.stderr)
    print(make_link(cfg, a.app))


if __name__ == "__main__":
    main()
