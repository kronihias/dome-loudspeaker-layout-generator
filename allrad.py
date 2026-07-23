"""AllRAD decoder + energy-distribution computation.

Faithful NumPy re-implementation of the decoder calculation performed by the
IEM AllRADecoder plug-in (github.com/IEM-Institute / IEMPluginSuite,
AllRADecoder/Source/PluginProcessor.cpp -> calculateDecoder()).

The result is an All-Round Ambisonic Decoder (AllRAD): a spherical t-design of
5200 virtual sources is decoded with sampling ambisonic decoding (SAD) and each
virtual source is panned to the real loudspeakers with VBAP over the convex hull
of the layout. Imaginary loudspeakers (used to close the hull, e.g. a "voice of
hell" below the horizon) are handled with the same energy-distributing kappa
scheme as the reference implementation and are removed from the final matrix.

Real spherical harmonics use the ACN channel ordering and N3D normalisation and
have been verified bit-for-bit against IEM's SHEval.
"""

import os
import math
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TDESIGN = None

# maxrE per-degree weights, transcribed from IEM resources/Weights.h (Weights::maxRe).
# Row N = decoder order, column l = ambisonic degree.
_MAXRE = np.array([
    [1.0, 0, 0, 0, 0, 0, 0, 0],
    [1.0, 5.7754104119288496e-01, 0, 0, 0, 0, 0, 0],
    [1.0, 7.7520766107019334e-01, 4.0142037667287966e-01, 0, 0, 0, 0, 0],
    [1.0, 8.6155075887658639e-01, 6.1340456518123299e-01, 3.0643144179936538e-01, 0, 0, 0, 0],
    [1.0, 9.0644136637224459e-01, 7.3245392600617265e-01, 5.0224998490808703e-01,
     2.4736484001129033e-01, 0, 0, 0],
    [1.0, 9.3263709143129281e-01, 8.0471791647013236e-01, 6.2909156744472861e-01,
     4.2321128963220900e-01, 2.0719132924646289e-01, 0, 0],
    [1.0, 9.4921830632793713e-01, 8.5152308960211620e-01, 7.1432330396679700e-01,
     5.4794300713180655e-01, 3.6475291657556469e-01, 1.7813609450688817e-01, 0],
    [1.0, 9.6036452263662697e-01, 8.8345002450861454e-01, 7.7381375334313540e-01,
     6.3791321433685355e-01, 4.8368159255186721e-01, 3.2000849790781744e-01,
     1.5616185043093761e-01],
], dtype=np.float64)

# IEM's SHEval scales the raw (orthonormal) real SH by sqrt(4*pi) for "encoding"
# (true N3D, Y_00 = 1). The decoder is normalised against these N3D-encoded
# loudspeaker gains, so the exported matrix — labelled ExpectedInputNormalization
# n3d — is 1/sqrt(4*pi) the size of one built from raw orthonormal SH. Applying
# this factor makes the matrix identical to IEM's (verified against CUBE.json).
_SQRT_4PI = math.sqrt(4.0 * math.pi)


def _inphase_perdegree(N):
    """inPhase (max-directivity-free) per-degree weights, normalised to w_0 = 1.

    w_l = (N+1) * N!^2 / ((N+l+1)! * (N-l)!). Verified against IEM Weights::inPhase.
    """
    fN = math.factorial(N)
    return np.array([
        (N + 1) * fN * fN / (math.factorial(N + l + 1) * math.factorial(N - l))
        for l in range(N + 1)
    ], dtype=np.float64)


MAX_ORDER = 7


def _tdesign():
    global _TDESIGN
    if _TDESIGN is None:
        _TDESIGN = np.load(os.path.join(_HERE, "tdesign5200.npy")).astype(np.float64)
    return _TDESIGN


def _assoc_legendre_no_cs(order, z):
    """Associated Legendre P_l^m(z) for m >= 0 without the Condon-Shortley phase.

    Returns a dict keyed by (l, m) of arrays matching ``z``. Uses the standard
    stable upward recurrences.
    """
    z = np.asarray(z, dtype=np.float64)
    s = np.sqrt(np.clip(1.0 - z * z, 0.0, None))   # sin(theta) = cos(elevation)
    p = {(0, 0): np.ones_like(z)}
    for m in range(1, order + 1):                  # sectoral: P_m^m = (2m-1)!! s^m
        p[(m, m)] = (2 * m - 1) * s * p[(m - 1, m - 1)]
    for m in range(order + 1):
        if m < order:                              # P_{m+1}^m = z (2m+1) P_m^m
            p[(m + 1, m)] = z * (2 * m + 1) * p[(m, m)]
        for l in range(m + 2, order + 1):          # general recurrence in l
            p[(l, m)] = (z * (2 * l - 1) * p[(l - 1, m)]
                         - (l + m - 1) * p[(l - 2, m)]) / (l - m)
    return p


def real_sh(order, x, y, z):
    """Real spherical harmonics (ACN order, N3D normalisation).

    Axis convention matches IEM: x = front, y = left, z = up. Verified
    bit-for-bit against IEM's SHEval. Returns shape (n_points, (order+1)**2).
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    z = np.atleast_1d(np.clip(np.asarray(z, dtype=np.float64), -1.0, 1.0))
    az = np.arctan2(y, x)
    leg = _assoc_legendre_no_cs(order, z)
    sqrt2 = math.sqrt(2.0)
    n = (order + 1) ** 2
    out = np.zeros((x.shape[0], n))
    for l in range(order + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            nlm = math.sqrt((2 * l + 1) / (4 * math.pi)
                            * math.factorial(l - am) / math.factorial(l + am))
            base = nlm * leg[(l, am)]
            if m > 0:
                out[:, l * l + l + m] = sqrt2 * base * np.cos(m * az)
            elif m < 0:
                out[:, l * l + l + m] = sqrt2 * base * np.sin(am * az)
            else:
                out[:, l * l + l + m] = base
    return out


def _convex_hull_triangles(pts, tol=1e-7):
    """Convex-hull triangulation of ``pts`` (n, 3) in pure NumPy.

    Loudspeaker layouts sit on the unit sphere, so all points are in convex
    position. Every supporting plane is found by testing point triples; triples
    on the same plane are merged into one facet and fan-triangulated, so
    coplanar faces (e.g. a rectangular sub-array) yield a clean triangulation
    rather than overlapping triangles. Returns an (T, 3) array of vertex indices.
    """
    n = len(pts)
    facets = {}
    for i in range(n):
        pi = pts - pts[i]
        for j in range(i + 1, n):
            dij = pts[j] - pts[i]
            for k in range(j + 1, n):
                normal = np.cross(dij, pts[k] - pts[i])
                nn = np.linalg.norm(normal)
                if nn < 1e-12:
                    continue                       # collinear
                normal = normal / nn
                d = pi @ normal
                pos = bool((d > tol).any())
                neg = bool((d < -tol).any())
                if pos and neg:
                    continue                       # not a supporting plane
                if pos:                            # orient outward: all d <= 0
                    normal = -normal
                    d = -d
                key = tuple(np.round(np.append(normal, normal @ pts[i]), 5))
                on = np.where(np.abs(d) < tol)[0]
                facets.setdefault(key, set()).update(on.tolist())

    tris = []
    for key, idxs in facets.items():
        idx = sorted(idxs)
        if len(idx) < 3:
            continue
        normal = np.array(key[:3])
        centre = pts[idx].mean(axis=0)
        u = pts[idx[0]] - centre
        if np.linalg.norm(u) < 1e-12:
            u = pts[idx[1]] - centre
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        ang = np.arctan2([(pts[p] - centre) @ v for p in idx],
                         [(pts[p] - centre) @ u for p in idx])
        order = [idx[t] for t in np.argsort(ang)]
        for t in range(1, len(order) - 1):         # fan triangulation
            tris.append((order[0], order[t], order[t + 1]))
    return np.array(tris, dtype=int)


def weight_vector(order, weights):
    """Per-coefficient weighting vector for the given ambisonic order."""
    n = (order + 1) ** 2
    w = np.ones(n)
    if weights == "maxrE":
        per = _MAXRE[order][:order + 1]
    elif weights == "inPhase":
        per = _inphase_perdegree(order)
    else:
        return w
    for l in range(order + 1):
        w[l * l:(l + 1) * (l + 1)] = per[l]
    return w


def _sph_to_cart(az_deg, el_deg, radius=1.0):
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    return np.array([
        radius * np.cos(el) * np.cos(az),
        radius * np.cos(el) * np.sin(az),
        radius * np.sin(el),
    ], dtype=np.float64)


def _get_kappa(g_im, g_re1, g_re2, n_conn):
    p = g_im * (g_re1 + g_re2) / (n_conn * g_im * g_im)
    q = (g_re1 * g_re1 + g_re2 * g_re2 - 1.0) / (n_conn * g_im * g_im)
    return -p + np.sqrt(max(p * p - q, 0.0))


class AllRADResult:
    def __init__(self, order, weights, matrix, routing, real_dirs, real_channels):
        self.order = order
        self.weights = weights          # "maxrE" | "inPhase" | "none"
        self.matrix = matrix            # (n_real, n_coeffs) N3D decoder
        self.routing = routing          # list of 1-based channels, per matrix row
        self.real_dirs = real_dirs      # (n_real, 3) unit vectors of real lsps
        self.real_channels = real_channels  # same as routing, convenience


def calculate_allrad(loudspeakers, order, weights="maxrE"):
    """Compute an AllRAD decoder for a loudspeaker layout.

    ``loudspeakers`` is a list of dicts with keys Azimuth, Elevation, Radius,
    IsImaginary, Channel, Gain (the same structure used for the JSON export).
    Returns an :class:`AllRADResult`.

    Raises ValueError with an IEM-style message if the layout is unsuitable.
    """
    order = int(order)
    if order < 0 or order > MAX_ORDER:
        raise ValueError(f"Order must be between 0 and {MAX_ORDER}.")
    n_coeffs = (order + 1) ** 2

    n_lsps = len(loudspeakers)
    if n_lsps < 4:
        raise ValueError("There are fewer than 4 loudspeakers. Add some more.")

    pts = np.zeros((n_lsps, 3))
    is_imag = np.zeros(n_lsps, dtype=bool)
    gains_lsp = np.zeros(n_lsps)
    real_lsp_num = np.full(n_lsps, -1, dtype=int)
    channels = np.zeros(n_lsps, dtype=int)
    imag_count = 0
    for i, ls in enumerate(loudspeakers):
        rad = float(ls.get("Radius", 1.0)) if ls.get("IsImaginary") else 1.0
        pts[i] = _sph_to_cart(ls["Azimuth"], ls["Elevation"], rad)
        is_imag[i] = bool(ls.get("IsImaginary", False))
        gains_lsp[i] = float(ls.get("Gain", 1.0))
        channels[i] = int(ls.get("Channel", i + 1))
        if is_imag[i]:
            imag_count += 1
            real_lsp_num[i] = -1
        else:
            real_lsp_num[i] = i - imag_count

    n_real = n_lsps - imag_count
    if n_real == 0:
        raise ValueError("There are only imaginary loudspeakers.")

    # Convex hull triangulation of all loudspeakers.
    tris = _convex_hull_triangles(pts)  # (T, 3) point indices
    if len(tris) == 0:
        raise ValueError("Could not build the convex hull. The layout might be "
                         "degenerate — try adding imaginary loudspeakers.")

    # Reject triangles with more than one imaginary loudspeaker (IEM ERROR 5).
    imag_in_tri = is_imag[tris].sum(axis=1)
    if np.any(imag_in_tri > 1):
        raise ValueError("There is a triangle with more than one imaginary "
                         "loudspeaker. Try a different layout.")

    # Per-triangle 3x3 matrix (columns = vertex directions), imaginary vertices
    # normalised to unit length, and its inverse.
    tri_pts = pts[tris]                      # (T, 3, 3): [tri, vertex, xyz]
    norms = np.linalg.norm(tri_pts, axis=2, keepdims=True)
    tri_pts_n = np.where(is_imag[tris][:, :, None], tri_pts / norms, tri_pts)
    L = np.transpose(tri_pts_n, (0, 2, 1))   # columns are vertices
    try:
        inv = np.linalg.inv(L)               # (T, 3, 3)
    except np.linalg.LinAlgError as exc:
        raise ValueError("A hull triangle is degenerate.") from exc

    # Precompute, for every imaginary loudspeaker, the set of real loudspeakers
    # it shares a triangle with (its "connected" speakers).
    connected = {}
    for imag_idx in np.where(is_imag)[0]:
        conn = set()
        for tri in tris:
            if imag_idx in tri:
                for v in tri:
                    if v != imag_idx:
                        conn.add(int(v))
        connected[int(imag_idx)] = sorted(conn)

    # Assign every t-design source to the triangle whose VBAP gains are all >= 0.
    src = _tdesign()                         # (5200, 3)
    eps = 1e-6
    # gains[t, p, :] = inv[t] @ src[p]
    gains_all = np.einsum("tij,pj->tpi", inv, src)   # (T, P, 3)
    valid = np.all(gains_all >= -eps, axis=2)        # (T, P)
    tri_of = np.argmax(valid, axis=0)                # first valid triangle per source
    found = valid[tri_of, np.arange(src.shape[0])]

    sh_src = real_sh(order, src[:, 0], src[:, 1], src[:, 2])  # (P, n_coeffs)

    decoder = np.zeros((n_real, n_coeffs))

    for p in range(src.shape[0]):
        if not found[p]:
            continue  # numerical corner case: source not covered by any triangle
        t = tri_of[p]
        tri = tris[t]
        g = gains_all[t, p].copy()
        g /= np.linalg.norm(g)               # energy normalisation
        sh = sh_src[p]

        tri_imag = is_imag[tri]
        if tri_imag.any():
            im_local = int(np.argmax(tri_imag))       # 0,1,2 -> imaginary vertex
            imag_idx = int(tri[im_local])
            real_local = [k for k in range(3) if k != im_local]
            conn = connected[imag_idx]
            g_im = g[im_local]
            g_re1 = g[real_local[0]]
            g_re2 = g[real_local[1]]
            kappa = _get_kappa(g_im, g_re1, g_re2, len(conn))
            base = g_im * gains_lsp[imag_idx] * kappa
            gain_vec = {c: base for c in conn}
            for k in real_local:
                gain_vec[int(tri[k])] += g[k]
            for c in conn:
                decoder[real_lsp_num[c]] += sh * gain_vec[c]
        else:
            for k in range(3):
                decoder[real_lsp_num[int(tri[k])]] += sh * g[k]

    # Normalise so the loudest loudspeaker gain (evaluated at loudspeaker
    # positions) is unity. IEM measures this gain with N3D-encoded SH
    # (raw * sqrt(4*pi)); using the same factor here fixes the overall matrix
    # level to match IEM exactly.
    real_idx = np.where(~is_imag)[0]
    sh_lsp = real_sh(order, pts[real_idx, 0], pts[real_idx, 1], pts[real_idx, 2]) * _SQRT_4PI
    # decoded gains: (n_real_dirs, n_real_speakers)
    decoded = sh_lsp @ decoder.T
    max_gain = np.sqrt(np.sum(decoded ** 2, axis=1)).max()
    if max_gain > 0:
        decoder = decoder / max_gain

    # Routing: matrix row (realLspNum) -> channel.
    routing = [0] * n_real
    real_dirs = np.zeros((n_real, 3))
    for i in range(n_lsps):
        if not is_imag[i]:
            routing[real_lsp_num[i]] = int(channels[i])
            real_dirs[real_lsp_num[i]] = pts[i] / np.linalg.norm(pts[i])

    return AllRADResult(order, weights, decoder, routing, real_dirs, list(routing))


def energy_distribution(result, n_az=180, n_el=90):
    """Sample the decoded energy over the sphere.

    Returns (az_grid_deg, el_grid_deg, level_db, mean_db) where ``level_db`` has
    shape (n_el, n_az) and holds the decoded energy in dB relative units.
    Mirrors the reference implementation's per-direction energy readout.
    """
    order = result.order
    az = np.linspace(-np.pi, np.pi, n_az)
    el = np.linspace(-np.pi / 2, np.pi / 2, n_el)
    AZ, EL = np.meshgrid(az, el)             # (n_el, n_az)
    x = np.cos(EL) * np.cos(AZ)
    y = np.cos(EL) * np.sin(AZ)
    z = np.sin(EL)
    sh = real_sh(order, x.ravel(), y.ravel(), z.ravel()) * _SQRT_4PI   # N3D encode
    sh = sh * weight_vector(order, result.weights)[None, :]
    decoded = sh @ result.matrix.T           # (n_dir, n_real)
    energy = np.sum(decoded ** 2, axis=1)    # power per direction
    energy = np.maximum(energy, 1e-12)
    level = 10.0 * np.log10(energy)          # 0.5 * 20 log10, energy is power
    level = level.reshape(EL.shape)
    # Reference level: solid-angle-weighted mean (cos(el)) so the poles, which
    # are over-represented on a lon/lat grid, do not bias the baseline.
    weights = np.cos(EL)
    mean = float(np.sum(level * weights) / np.sum(weights))
    return np.degrees(AZ), np.degrees(EL), level, mean


def decoder_to_json(result):
    """Serialise the decoder to IEM's 'Decoder' JSON structure."""
    order = result.order
    order_names = ["zeroth", "first", "second", "third", "fourth",
                   "fifth", "sixth", "seventh"]
    on = order_names[order] if order < len(order_names) else str(order)
    return {
        "Name": "Decoder",
        "Description": f"A {on} order Ambisonics decoder using the AllRAD approach.",
        "ExpectedInputNormalization": "n3d",
        "Weights": result.weights if result.weights in ("maxrE", "inPhase") else "none",
        "WeightsAlreadyApplied": False,
        "Matrix": [[float(v) for v in row] for row in result.matrix],
        "Routing": [int(c) for c in result.routing],
    }
