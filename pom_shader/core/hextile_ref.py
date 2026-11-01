# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

SQRT3 = float(np.sqrt(3.0))
SKEW = np.array([[1.0, -1.0 / SQRT3], [0.0, 2.0 / SQRT3]])
UNSKEW = np.array([[1.0, 0.5], [0.0, SQRT3 / 2.0]])


@dataclass
class Grid:
    ids: np.ndarray
    bary: np.ndarray
    centers: np.ndarray


def triangle_grid(st: np.ndarray, scale: float = 1.0) -> Grid:
    k = 2.0 * SQRT3 * scale
    sk = (st * k) @ SKEW.T
    base = np.floor(sk)
    f = sk - base
    z = 1.0 - f[:, 0] - f[:, 1]
    s = (z < 0).astype(np.float64)
    pts = np.stack([base + s[:, None], base + [1.0, 0.0], base + [0.0, 1.0]])
    w = np.stack([np.abs(z), f[:, 0] + s * z, f[:, 1] + s * z])
    col = np.mod(base[:, 0] - base[:, 1], 3).astype(int)
    order = (np.arange(3)[:, None] - col[None, :]) % 3
    n = np.arange(len(st))
    ids = pts[order, n]
    centers = (ids @ UNSKEW.T) / k
    return Grid(ids, w[order, n].T, centers)


def rotate(v: np.ndarray, center: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    d = v - center
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1) + center


def taps(st_eval: np.ndarray, grid: Grid, offsets: np.ndarray, angles: np.ndarray) -> np.ndarray:
    return np.stack([rotate(st_eval, grid.centers[i], angles[i]) + offsets[i] for i in range(3)])


def gain3(w: np.ndarray, r: float) -> np.ndarray:
    if abs(r - 0.5) < 1e-9:
        return w
    k = np.log(1 - r) / np.log(0.5)
    s = 2.0 * (w >= 0.5)
    m = 2.0 * (1.0 - s)
    res = 0.5 * s + 0.25 * m * np.power(np.maximum(0.0, s + w * m), k)
    return res / res.sum(axis=1, keepdims=True)


def weights(bary: np.ndarray, drive: np.ndarray | None, contrast: float, exponent: float, r: float) -> np.ndarray:
    dw = np.ones_like(bary) if drive is None else 1.0 + contrast * (drive - 1.0)
    w = dw * np.power(np.maximum(bary, 0.0), exponent)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-20)
    return gain3(w, r)


def blend(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return np.einsum("in,ni->n", values, w)
    return np.einsum("inc,ni->nc", values, w)


def rotate_tangent_normals(n: np.ndarray, angles: np.ndarray) -> np.ndarray:
    out = n.copy()
    for i in range(3):
        c, s = np.cos(-angles[i]), np.sin(-angles[i])
        x, y = n[i, :, 0], n[i, :, 1]
        out[i, :, 0] = c * x - s * y
        out[i, :, 1] = s * x + c * y
    return out


def hex_sample(st: np.ndarray, tex: Callable[[np.ndarray], np.ndarray], offsets_fn, angles_fn, scale=1.0,
               contrast=0.6, exponent=7.0, r=0.5, luminance=True) -> np.ndarray:
    g = triangle_grid(st, scale)
    offs = offsets_fn(g.ids)
    angs = angles_fn(g.ids)
    tp = taps(st, g, offs, angs)
    cols = np.stack([tex(tp[i]) for i in range(3)])
    drive = None
    if luminance:
        lw = np.array([0.299, 0.587, 0.114])
        drive = np.stack([cols[i] @ lw for i in range(3)], 1)
    w = weights(g.bary, drive, contrast, exponent, r)
    return blend(cols, w)
