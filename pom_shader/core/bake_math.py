# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLAG_NO_HIT = 1 << 0
FLAG_MULTI = 1 << 1
FLAG_LATERAL = 1 << 2
FLAG_OUTSIDE = 1 << 3
FLAG_NAMES = {FLAG_NO_HIT: "no hit", FLAG_MULTI: "undercut / multi-valued",
              FLAG_LATERAL: "lateral deviation", FLAG_OUTSIDE: "outside the islands"}


def raster_barycentric(tri_uvs: np.ndarray, width: int, height: int, chunk: int = 200_000):
    tri_id = np.full((height, width), -1, np.int64)
    bary = np.zeros((height, width, 3), np.float64)
    px = tri_uvs.astype(np.float64) * np.array([width, height])
    span = np.maximum(px[:, :, 0].max(1) - px[:, :, 0].min(1), px[:, :, 1].max(1) - px[:, :, 1].min(1))
    size = np.ceil(span).astype(np.int64) + 2
    edges = [0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1 << 30]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = np.nonzero((size > lo) & (size <= hi))[0]
        if sel.size == 0:
            continue
        side = min(hi, int(size[sel].max()))
        step = max(1, chunk * 16 // (side * side))
        for i in range(0, sel.size, step):
            idx = sel[i:i + step]
            _bary_batch(px[idx], idx, side, tri_id, bary)
    return tri_id, bary


def _bary_batch(px, idx, size, tri_id, bary) -> None:
    hh, ww = tri_id.shape
    x0 = np.floor(px[:, :, 0].min(axis=1) - 0.5).astype(np.int64) + 1
    y0 = np.floor(px[:, :, 1].min(axis=1) - 0.5).astype(np.int64) + 1
    gx = x0[:, None, None] + np.arange(size)[None, None, :]
    gy = y0[:, None, None] + np.arange(size)[None, :, None]
    cx, cy = gx + 0.5, gy + 0.5
    a, b, c = px[:, 0], px[:, 1], px[:, 2]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)[:, None, None]
    l1 = ((cx - a[:, 0, None, None]) * (c[:, 1, None, None] - a[:, 1, None, None])
          - (c[:, 0, None, None] - a[:, 0, None, None]) * (cy - a[:, 1, None, None])) * inv
    l2 = ((b[:, 0, None, None] - a[:, 0, None, None]) * (cy - a[:, 1, None, None])
          - (cx - a[:, 0, None, None]) * (b[:, 1, None, None] - a[:, 1, None, None])) * inv
    l0 = 1.0 - l1 - l2
    eps = -1e-9
    inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps) & ok[:, None, None]
    inside &= (gx >= 0) & (gx < ww) & (gy >= 0) & (gy < hh)
    bi, yy, xx = np.nonzero(inside)
    ty, tx = gy[bi, yy, 0], gx[bi, 0, xx]
    tri_id[ty, tx] = idx[bi]
    bary[ty, tx, 0] = l0[bi, yy, xx]
    bary[ty, tx, 1] = l1[bi, yy, xx]
    bary[ty, tx, 2] = l2[bi, yy, xx]


def interpolate(tri_id: np.ndarray, bary: np.ndarray, corner_values: np.ndarray) -> np.ndarray:
    out = np.zeros((*tri_id.shape, corner_values.shape[2]), np.float64)
    m = tri_id >= 0
    v = corner_values[tri_id[m]]
    out[m] = np.einsum("kj,kjc->kc", bary[m], v)
    return out


def signed_distance(p_high: np.ndarray, p_low: np.ndarray, n_low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = p_high - p_low
    t = np.einsum("...c,...c->...", d, n_low)
    lateral = np.linalg.norm(d - t[..., None] * n_low, axis=-1)
    return t, lateral


def flags_from(hit: np.ndarray, inside: np.ndarray, lateral: np.ndarray, lateral_tol: float,
               multi: np.ndarray | None = None) -> np.ndarray:
    f = np.zeros(hit.shape, np.uint8)
    f |= np.where(~inside, FLAG_OUTSIDE, 0).astype(np.uint8)
    f |= np.where(inside & ~hit, FLAG_NO_HIT, 0).astype(np.uint8)
    f |= np.where(inside & hit & (lateral > lateral_tol), FLAG_LATERAL, 0).astype(np.uint8)
    if multi is not None:
        f |= np.where(inside & multi, FLAG_MULTI, 0).astype(np.uint8)
    return f


def flag_percentages(flags: np.ndarray, inside: np.ndarray) -> dict[str, float]:
    n = max(int(inside.sum()), 1)
    return {name: float(((flags & bit) != 0)[inside].sum() / n * 100) for bit, name in FLAG_NAMES.items()
            if bit != FLAG_OUTSIDE}


def slope_undercut_candidates(d: np.ndarray, texel_m: float, valid: np.ndarray, max_deg: float = 85.0) -> np.ndarray:
    gy, gx = np.gradient(np.where(valid, d, np.nan), texel_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return np.nan_to_num(slope, nan=0.0) > max_deg


def push_pull_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = np.where(valid, values, 0.0).astype(np.float64)
    w = valid.astype(np.float64)
    levels = [(v, w)]
    while min(v.shape) > 1:
        hh, ww = v.shape
        ph, pw = hh % 2, ww % 2
        vp = np.pad(v, ((0, ph), (0, pw)))
        wp = np.pad(w, ((0, ph), (0, pw)))
        v = vp.reshape(vp.shape[0] // 2, 2, vp.shape[1] // 2, 2).sum(axis=(1, 3))
        w = wp.reshape(wp.shape[0] // 2, 2, wp.shape[1] // 2, 2).sum(axis=(1, 3))
        levels.append((v, w))
    est = np.where(levels[-1][1] > 0, levels[-1][0] / np.maximum(levels[-1][1], 1e-30), 0.0)
    for v_l, w_l in reversed(levels[:-1]):
        up = np.repeat(np.repeat(est, 2, 0), 2, 1)[: v_l.shape[0], : v_l.shape[1]]
        mean = np.where(w_l > 0, v_l / np.maximum(w_l, 1e-30), up)
        alpha = np.clip(w_l, 0.0, 1.0)
        est = alpha * mean + (1 - alpha) * up
    return np.where(valid, values, est)


def dilate(values: np.ndarray, valid: np.ndarray, iterations: int = 8) -> np.ndarray:
    v = np.where(valid, values, 0.0).astype(np.float64)
    w = valid.astype(np.float64)
    out = v.copy()
    done = valid.copy()
    for _ in range(iterations):
        acc = np.zeros_like(v)
        cnt = np.zeros_like(v)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            acc += np.roll(np.roll(out * done, dy, 0), dx, 1)
            cnt += np.roll(np.roll(done.astype(np.float64), dy, 0), dx, 1)
        grow = (~done) & (cnt > 0)
        out = np.where(grow, acc / np.maximum(cnt, 1), out)
        done = done | grow
        if done.all():
            break
    del w
    return out


@dataclass
class SurfaceSamples:
    points: np.ndarray
    tri: np.ndarray
    bary: np.ndarray


def sample_surface(tri_positions: np.ndarray, count: int, seed: int = 0) -> SurfaceSamples:
    rng = np.random.default_rng(seed)
    a, b, c = tri_positions[:, 0], tri_positions[:, 1], tri_positions[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    p = area / area.sum()
    tri = rng.choice(len(area), size=count, p=p)
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    pts = np.einsum("mj,mjc->mc", bary, tri_positions[tri])
    return SurfaceSamples(pts, tri, bary)


def auto_extrusion(distances: np.ndarray, margin: float) -> float:
    return float(1.2 * np.percentile(np.abs(distances), 99.9) + margin)
