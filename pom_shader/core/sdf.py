# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMPTY = -1


def island_ids(tri_verts: np.ndarray, tri_uvs: np.ndarray) -> np.ndarray:
    t = tri_verts.shape[0]
    q = np.round(tri_uvs.astype(np.float64) * 1e6).astype(np.int64)
    keys = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        va, vb = tri_verts[:, a].astype(np.int64), tri_verts[:, b].astype(np.int64)
        swap = va > vb
        lo = np.where(swap, vb, va)
        hi = np.where(swap, va, vb)
        ua = np.where(swap[:, None], q[:, b], q[:, a])
        ub = np.where(swap[:, None], q[:, a], q[:, b])
        keys.append(np.column_stack([lo, hi, ua, ub, np.arange(t)]))
    k = np.concatenate(keys)
    order = np.lexsort(k[:, :6].T[::-1])
    k = k[order]
    same = np.all(k[1:, :6] == k[:-1, :6], axis=1)
    pa, pb = k[:-1, 6][same], k[1:, 6][same]
    labels = np.arange(t)
    while True:
        lab_min = np.minimum(labels[pa], labels[pb])
        new = labels.copy()
        np.minimum.at(new, pa, lab_min)
        np.minimum.at(new, pb, lab_min)
        new = new[new]
        if np.array_equal(new, labels):
            break
        labels = new
    _, ids = np.unique(labels, return_inverse=True)
    return ids.astype(np.int32)


@dataclass
class Raster:
    labels: np.ndarray
    coverage: np.ndarray
    overlap_texels: int


def _batch(px: np.ndarray, size: int, labels: np.ndarray, cover: np.ndarray, isl: np.ndarray) -> None:
    hh, ww = labels.shape
    x0 = np.floor(px[:, :, 0].min(axis=1) - 0.5).astype(np.int64) + 1
    y0 = np.floor(px[:, :, 1].min(axis=1) - 0.5).astype(np.int64) + 1
    gx = x0[:, None, None] + np.arange(size)[None, None, :]
    gy = y0[:, None, None] + np.arange(size)[None, :, None]
    cx = gx + 0.5
    cy = gy + 0.5
    a, b, c = px[:, 0], px[:, 1], px[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    sgn = np.sign(area)[:, None, None]

    def edge(p, q):
        return ((q[:, 0] - p[:, 0])[:, None, None] * (cy - p[:, 1][:, None, None])
                - (q[:, 1] - p[:, 1])[:, None, None] * (cx - p[:, 0][:, None, None])) * sgn

    inside = (edge(a, b) >= 0) & (edge(b, c) >= 0) & (edge(c, a) >= 0) & (sgn != 0)
    inside &= (gx >= 0) & (gx < ww) & (gy >= 0) & (gy < hh)
    bi, yy, xx = np.nonzero(inside)
    ty, tx = gy[bi, yy, 0], gx[bi, 0, xx]
    np.add.at(cover, (ty, tx), 1)
    labels[ty, tx] = isl[bi]


def rasterize(tri_uvs: np.ndarray, islands: np.ndarray, width: int, height: int,
              chunk: int = 200_000) -> Raster:
    labels = np.full((height, width), EMPTY, np.int32)
    cover = np.zeros((height, width), np.int32)
    px = tri_uvs.astype(np.float64) * np.array([width, height])
    span = np.maximum(px[:, :, 0].max(1) - px[:, :, 0].min(1), px[:, :, 1].max(1) - px[:, :, 1].min(1))
    size = np.ceil(span).astype(np.int64) + 2
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128)):
        sel = np.nonzero((size > lo) & (size <= hi))[0]
        step = max(1, chunk * 16 // (hi * hi))
        for i in range(0, sel.size, step):
            idx = sel[i:i + step]
            _batch(px[idx], hi, labels, cover, islands[idx])
    for i in np.nonzero(size > 128)[0]:
        _large(px[i], labels, cover, int(islands[i]))
    overlap = _overlaps(tri_uvs, islands, labels, cover, width, height)
    return Raster(labels, cover, overlap)


def _large(tri: np.ndarray, labels: np.ndarray, cover: np.ndarray, isl: int) -> None:
    hh, ww = labels.shape
    xmin = max(int(np.floor(tri[:, 0].min())), 0)
    xmax = min(int(np.ceil(tri[:, 0].max())), ww - 1)
    ymin = max(int(np.floor(tri[:, 1].min())), 0)
    ymax = min(int(np.ceil(tri[:, 1].max())), hh - 1)
    if xmin > xmax or ymin > ymax:
        return
    cx = np.arange(xmin, xmax + 1) + 0.5
    a, b, c = tri
    area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    if area == 0:
        return
    s = np.sign(area)
    for y in range(ymin, ymax + 1):
        cy = y + 0.5
        e1 = ((b[0] - a[0]) * (cy - a[1]) - (b[1] - a[1]) * (cx - a[0])) * s
        e2 = ((c[0] - b[0]) * (cy - b[1]) - (c[1] - b[1]) * (cx - b[0])) * s
        e3 = ((a[0] - c[0]) * (cy - c[1]) - (a[1] - c[1]) * (cx - c[0])) * s
        xs = np.nonzero((e1 >= 0) & (e2 >= 0) & (e3 >= 0))[0] + xmin
        cover[y, xs] += 1
        labels[y, xs] = isl


def _overlaps(tri_uvs, islands, labels, cover, width, height) -> int:
    multi = cover > 1
    if not multi.any():
        return 0
    owner = np.full((height, width), EMPTY, np.int64)
    conflict = np.zeros((height, width), bool)
    order = np.argsort(islands, kind="stable")
    sorted_isl = islands[order]
    starts = np.flatnonzero(np.r_[True, sorted_isl[1:] != sorted_isl[:-1]])
    ends = np.r_[starts[1:], sorted_isl.size]
    for s, e in zip(starts, ends):
        sub = rasterize_mask(tri_uvs[order[s:e]], width, height)
        clash = sub & (owner != EMPTY)
        conflict |= clash
        owner[sub] = sorted_isl[s]
    return int((conflict | (cover > 2)).sum())


def rasterize_mask(tri_uvs: np.ndarray, width: int, height: int) -> np.ndarray:
    lab = np.full((height, width), EMPTY, np.int32)
    cov = np.zeros((height, width), np.int32)
    px = tri_uvs.astype(np.float64) * np.array([width, height])
    span = np.maximum(px[:, :, 0].max(1) - px[:, :, 0].min(1), px[:, :, 1].max(1) - px[:, :, 1].min(1))
    size = np.ceil(span).astype(np.int64) + 2
    zeros = np.zeros(tri_uvs.shape[0], np.int32)
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128)):
        sel = np.nonzero((size > lo) & (size <= hi))[0]
        step = max(1, 200_000 * 16 // (hi * hi))
        for i in range(0, sel.size, step):
            _batch(px[sel[i:i + step]], hi, lab, cov, zeros[sel[i:i + step]])
    for i in np.nonzero(size > 128)[0]:
        _large(px[i], lab, cov, 0)
    return cov > 0


_SQRT2 = float(np.sqrt(2.0))


def _boundary_seeds(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hh, ww = labels.shape
    sy = np.full((hh, ww), np.nan)
    sx = np.full((hh, ww), np.nan)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float64)
    dx = labels[:, 1:] != labels[:, :-1]
    for side in (0, 1):
        sl = (slice(None), slice(0, ww - 1)) if side == 0 else (slice(None), slice(1, ww))
        sy[sl] = np.where(dx, yy[:, :-1] + 0.5, sy[sl])
        sx[sl] = np.where(dx, xx[:, :-1] + 1.0, sx[sl])
    dy = labels[1:, :] != labels[:-1, :]
    for side in (0, 1):
        sl = (slice(0, hh - 1), slice(None)) if side == 0 else (slice(1, hh), slice(None))
        sy[sl] = np.where(dy, yy[:-1, :] + 1.0, sy[sl])
        sx[sl] = np.where(dy, xx[:-1, :] + 0.5, sx[sl])
    return sy, sx


def jump_flood(labels: np.ndarray) -> np.ndarray:
    hh, ww = labels.shape
    sy, sx = _boundary_seeds(labels)
    cy, cx = np.mgrid[0:hh, 0:ww].astype(np.float64) + 0.5
    best = np.where(np.isnan(sy), np.inf, np.hypot(sy - cy, sx - cx))
    sy = np.nan_to_num(sy, nan=-1e9)
    sx = np.nan_to_num(sx, nan=-1e9)
    step = 1 << int(np.ceil(np.log2(max(hh, ww)))) >> 1
    steps = []
    while step >= 1:
        steps.append(step)
        step >>= 1
    steps += [1, 1]
    for k in steps:
        for oy in (-k, 0, k):
            for ox in (-k, 0, k):
                if oy == 0 and ox == 0:
                    continue
                ny = np.clip(np.arange(hh) + oy, 0, hh - 1)
                nx = np.clip(np.arange(ww) + ox, 0, ww - 1)
                cand_y = sy[np.ix_(ny, nx)]
                cand_x = sx[np.ix_(ny, nx)]
                d = np.hypot(cand_y - cy, cand_x - cx)
                better = d < best
                best = np.where(better, d, best)
                sy = np.where(better, cand_y, sy)
                sx = np.where(better, cand_x, sx)
    return best


def _pool_labels(labels: np.ndarray, f: int) -> np.ndarray:
    hh, ww = labels.shape
    ph, pw = (-hh) % f, (-ww) % f
    lab = np.pad(labels, ((0, ph), (0, pw)), constant_values=EMPTY)
    blocks = lab.reshape(lab.shape[0] // f, f, lab.shape[1] // f, f)
    first = blocks[:, :1, :, :1]
    uniform = np.all(blocks == first, axis=(1, 3))
    return np.where(uniform, first[:, 0, :, 0], EMPTY)


def border_distance(labels: np.ndarray, max_res: int = 1024) -> np.ndarray:
    hh, ww = labels.shape
    f = 1
    while max(hh, ww) // f > max_res:
        f *= 2
    if f == 1:
        d = jump_flood(labels) - 0.5
    else:
        lo = _pool_labels(labels, f)
        d_lo = jump_flood(lo)
        d_lo = np.where(lo == EMPTY, 0.0, d_lo)
        d = np.repeat(np.repeat(d_lo, f, 0), f, 1)[:hh, :ww] * f - f * (0.5 + _SQRT2 / 2)
    d = d - 1.5 * _SQRT2
    d = np.where(labels == EMPTY, 0.0, np.maximum(d, 0.0))
    return (d / max(hh, ww)).astype(np.float32)


def brute_force_distance(labels: np.ndarray) -> np.ndarray:
    hh, ww = labels.shape
    cy, cx = np.mgrid[0:hh, 0:ww] + 0.5
    out = np.full((hh, ww), np.inf)
    for lab in np.unique(labels):
        inside = labels == lab
        oy, ox = cy[~inside], cx[~inside]
        if oy.size == 0:
            continue
        py, px = cy[inside], cx[inside]
        d = np.min(np.hypot(py[:, None] - oy[None, :], px[:, None] - ox[None, :]), axis=1)
        out[inside] = d
    return out
