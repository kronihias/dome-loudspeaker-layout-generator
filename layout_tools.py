"""Layout quality metrics and a ring-elevation optimiser.

Two things:

* ``analyze_decoder`` reports how good a *calculated* AllRAD decoder is —
  energy uniformity, energy-vector rE (localisation), angular error, decoder
  condition number and minimum loudspeaker spacing.

* ``optimize_rings`` searches ring elevations (counts follow from them) to
  minimise a fast surrogate cost — energy fluctuation plus rE loss of a
  max-rE-weighted sampling decoder, evaluated over the loudspeaker-covered
  region. The surrogate needs no convex hull, so thousands of candidate layouts
  can be scored per second; the user then calculates the real AllRAD decoder on
  the winner.
"""

import numpy as np
import allrad

_SQRT_4PI = np.sqrt(4.0 * np.pi)


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def fibonacci_sphere(n):
    """``n`` roughly uniform unit vectors on the sphere (deterministic)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)          # polar angle
    gold = np.pi * (1.0 + 5 ** 0.5)             # golden angle
    theta = gold * i
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def covered_grid(min_el_deg, n=1600):
    """Roughly uniform directions with elevation >= ``min_el_deg`` (degrees)."""
    pts = fibonacci_sphere(n)
    return pts[np.degrees(np.arcsin(np.clip(pts[:, 2], -1, 1))) >= min_el_deg - 1e-9]


def counts_from_elevations(elevs, n_speakers, include_vog):
    """Distribute ``n_speakers`` across rings proportional to ring circumference.

    Mirrors the generator's own heuristic: count ∝ |sin(polar angle)|, the
    highest ring gets a single Voice-of-God speaker when enabled, and any
    rounding remainder goes to the lowest ring.
    """
    elevs = np.asarray(elevs, dtype=float)
    theta = np.radians(90.0 - elevs)
    w = np.abs(np.sin(theta))
    if w.sum() <= 0:
        counts = np.zeros(len(elevs), dtype=int)
    else:
        counts = np.round(w / w.sum() * n_speakers).astype(int)
    if include_vog and len(counts):
        counts[-1] = 1
    counts = np.maximum(counts, 0)
    diff = n_speakers - int(counts.sum())
    if len(counts):
        counts[0] = max(0, counts[0] + diff)
    return counts


def ring_offsets(elevs, counts, ring_below_horizon):
    """Staggered azimuth offsets matching the generator's defaults."""
    stagger = 0 if ring_below_horizon else 1
    offs = np.zeros(len(elevs))
    for i, (el, c) in enumerate(zip(elevs, counts)):
        if el < 90.0 and c > 0 and i % 2 == stagger:
            offs[i] = round(180.0 / c, 4)
    return offs


def speaker_dirs(elevs, counts, offsets):
    """Unit vectors for every real loudspeaker of a ring layout."""
    dirs = []
    for el, c, off in zip(elevs, counts, offsets):
        if c <= 0:
            continue
        th = np.radians(90.0 - el)
        phis = np.linspace(0, 2 * np.pi, int(c), endpoint=False) + np.radians(off)
        for phi in phis:
            dirs.append([np.sin(th) * np.cos(phi),
                         np.sin(th) * np.sin(phi),
                         np.cos(th)])
    return np.array(dirs) if dirs else np.zeros((0, 3))


# --------------------------------------------------------------------------- #
# fast surrogate metrics (sampling decoder, no convex hull)
# --------------------------------------------------------------------------- #
def sampling_metrics(spk_dirs, eval_dirs, order, weights):
    """Energy and rE of a max-rE-weighted sampling ('basic') decoder."""
    w = allrad.weight_vector(order, weights)
    y_spk = allrad.real_sh(order, spk_dirs[:, 0], spk_dirs[:, 1], spk_dirs[:, 2])
    y_ev = allrad.real_sh(order, eval_dirs[:, 0], eval_dirs[:, 1], eval_dirs[:, 2]) * w
    g = y_spk @ y_ev.T                          # (n_spk, n_eval)
    energy = np.maximum((g ** 2).sum(axis=0), 1e-12)
    re_vec = (g ** 2).T @ spk_dirs / energy[:, None]
    return energy, re_vec


def _quality_from(energy, re_vec, eval_dirs):
    level = 10.0 * np.log10(energy)
    re_mag = np.linalg.norm(re_vec, axis=1)
    re_dir = re_vec / np.maximum(re_mag[:, None], 1e-12)
    ang_err = np.degrees(np.arccos(np.clip((re_dir * eval_dirs).sum(axis=1), -1, 1)))
    return {
        "energy_pp_db": float(level.max() - level.min()),
        "energy_std_db": float(level.std()),
        "rE_mean": float(re_mag.mean()),
        "rE_min": float(re_mag.min()),
        "rE_spread_deg": float(np.degrees(2 * np.arccos(np.clip(re_mag, 0, 1))).mean()),
        "err_mean_deg": float(ang_err.mean()),
        "err_max_deg": float(ang_err.max()),
    }


def analyze_layout(spk_dirs, order, weights, min_el, eval_dirs=None, n_eval=1600):
    """Fast surrogate quality of a raw layout (no decoder needed)."""
    if eval_dirs is None:
        eval_dirs = covered_grid(min_el, n_eval)
    energy, re_vec = sampling_metrics(spk_dirs, eval_dirs, order, weights)
    return _quality_from(energy, re_vec, eval_dirs)


# --------------------------------------------------------------------------- #
# real AllRAD decoder metrics
# --------------------------------------------------------------------------- #
def analyze_decoder(result, min_el, n_eval=1600):
    """Quality metrics for a calculated AllRAD decoder (:class:`AllRADResult`)."""
    order = result.order
    eval_dirs = covered_grid(min_el, n_eval)
    w = allrad.weight_vector(order, result.weights)
    sh = allrad.real_sh(order, eval_dirs[:, 0], eval_dirs[:, 1], eval_dirs[:, 2]) * _SQRT_4PI * w
    g = sh @ result.matrix.T                    # (n_dir, n_spk)
    energy = np.maximum((g ** 2).sum(axis=1), 1e-12)
    m = _quality_from(energy, (g ** 2) @ result.real_dirs / energy[:, None], eval_dirs)

    # decoder robustness
    m["condition_number"] = float(np.linalg.cond(result.matrix))

    # closest loudspeaker pair (degrees)
    d = result.real_dirs
    cos = np.clip(d @ d.T, -1, 1)
    np.fill_diagonal(cos, -1.0)
    m["min_spacing_deg"] = float(np.degrees(np.arccos(cos.max()))) if len(d) > 1 else 180.0
    return m


# --------------------------------------------------------------------------- #
# ring-elevation optimiser
# --------------------------------------------------------------------------- #
def _init_elevations(n_rings, include_vog, min_el):
    if include_vog:
        el = np.linspace(min_el, 90.0, n_rings)
    else:
        step = (90.0 - min_el) / n_rings
        el = np.linspace(min_el, 90.0 - step, n_rings)
    return el


def optimize_rings(n_speakers, n_rings, include_vog, min_el, order, weights,
                   ring_below_horizon=False, re_weight=12.0, iterations=60,
                   min_sep=8.0):
    """Search ring elevations that minimise the surrogate cost.

    Returns a dict with the optimised ``elevations`` / ``counts`` / ``offsets``
    and the surrogate metrics ``before`` (generator defaults) and ``after``.
    Counts are derived from the elevations; the Voice-of-God ring (if enabled)
    is pinned to 90 deg. Rings are kept at least ``min_sep`` degrees apart so the
    optimiser cannot collapse two rings onto (almost) the same elevation.
    """
    eval_dirs = covered_grid(min_el, 1600)
    free = list(range(n_rings - 1)) if include_vog else list(range(n_rings))
    lo, hi = float(min_el), 90.0
    # Keep rings distinct; shrink the requested separation if there isn't room.
    if n_rings > 1:
        min_sep = min(min_sep, 0.9 * (hi - lo) / (n_rings - 1))

    def cost(elevs):
        counts = counts_from_elevations(elevs, n_speakers, include_vog)
        offs = ring_offsets(elevs, counts, ring_below_horizon)
        dirs = speaker_dirs(elevs, counts, offs)
        if len(dirs) < 4:
            return 1e9, None
        energy, re_vec = sampling_metrics(dirs, eval_dirs, order, weights)
        q = _quality_from(energy, re_vec, eval_dirs)
        return q["energy_pp_db"] + re_weight * (1.0 - q["rE_mean"]), q

    el = _init_elevations(n_rings, include_vog, min_el)
    init_el = el.copy()
    best_cost, _ = cost(el)

    step = 10.0
    it = 0
    while step > 0.4 and it < iterations:
        improved = False
        for i in free:
            for delta in (step, -step):
                cand = el.copy()
                cand[i] = np.clip(el[i] + delta, lo, hi)
                s = np.sort(cand)               # keep rings ordered and separated
                if np.any(np.diff(s) < min_sep):
                    continue
                cand = s
                if include_vog:
                    cand[-1] = 90.0
                c, _ = cost(cand)
                if c < best_cost - 1e-9:
                    best_cost, el, improved = c, cand, True
            it += 1
        if not improved:
            step *= 0.5

    counts = counts_from_elevations(el, n_speakers, include_vog)
    offs = ring_offsets(el, counts, ring_below_horizon)
    _, after = cost(el)
    _, before = cost(init_el)
    return {
        "elevations": [round(float(e), 2) for e in el],
        "counts": [int(c) for c in counts],
        "offsets": [round(float(o), 4) for o in offs],
        "before": before,
        "after": after,
        "init_elevations": [round(float(e), 2) for e in init_el],
    }
