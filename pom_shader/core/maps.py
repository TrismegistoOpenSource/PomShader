# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.i18n import tr

ALGORITHM_VERSION = 1


@dataclass
class Normalized:
    height: np.ndarray
    depth_m: float | None
    ref_plane: float | None
    warnings: list[str]


def normalize_unit(data: np.ndarray, bit_depth: int | None = None) -> Normalized:
    warnings = []
    lo, hi = float(np.min(data)), float(np.max(data))
    if lo < -1e-4 or hi > 1 + 1e-4:
        warnings.append(tr("values outside 0..1: clipped"))
    if bit_depth == 8:
        warnings.append(tr("8-bit height map: large depths will show steps"))
    return Normalized(np.clip(data, 0.0, 1.0).astype(np.float32), None, None, warnings)


def normalize_meters(data: np.ndarray) -> Normalized:
    finite = np.isfinite(data)
    if not finite.all():
        data = np.where(finite, data, 0.0)
    lo, hi = float(np.min(data)), float(np.max(data))
    span = hi - lo
    if span <= 0:
        return Normalized(np.zeros_like(data, dtype=np.float32), 0.0, 0.0, ["displacement costante"])
    h = ((data - lo) / span).astype(np.float32)
    return Normalized(h, span, float(np.clip(-lo / span, 0.0, 1.0)), [])


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def shift(a: np.ndarray, dy: int, dx: int, wrap: bool, fill: float | None = None) -> np.ndarray:
    if wrap:
        return np.roll(a, shift=(-dy, -dx), axis=(0, 1))
    h, w = a.shape
    if fill is None:
        ys = np.clip(np.arange(h) + dy, 0, h - 1)
        xs = np.clip(np.arange(w) + dx, 0, w - 1)
        return a[np.ix_(ys, xs)]
    out = np.full_like(a, fill)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys0 < ys1 and xs0 < xs1:
        out[ys0:ys1, xs0:xs1] = a[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return out


def gradient(h: np.ndarray, wrap: bool) -> tuple[np.ndarray, np.ndarray]:
    hh, ww = h.shape
    du = (shift(h, 0, 1, wrap) - shift(h, 0, -1, wrap)) * (ww / 2.0)
    dv = (shift(h, 1, 0, wrap) - shift(h, -1, 0, wrap)) * (hh / 2.0)
    if not wrap:
        du[:, 0] *= 2.0
        du[:, -1] *= 2.0
        dv[0, :] *= 2.0
        dv[-1, :] *= 2.0
    return du.astype(np.float32), dv.astype(np.float32)


CONE_MAX = 4.0
_RING = 6


def _cells(d: np.ndarray, wrap: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hh, ww = d.shape
    a = d
    b = shift(d, 0, 1, wrap)
    c = shift(d, 1, 0, wrap)
    e = shift(d, 1, 1, wrap)
    cmin = np.minimum(np.minimum(a, b), np.minimum(c, e))
    cmax = np.maximum(np.maximum(a, b), np.maximum(c, e))
    gx = np.maximum(np.abs(b - a), np.abs(e - c)) * ww
    gy = np.maximum(np.abs(c - a), np.abs(e - b)) * hh
    return cmin, cmax, np.sqrt(gx * gx + gy * gy)


def _max_filter3(a: np.ndarray, wrap: bool) -> np.ndarray:
    out = a.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                np.maximum(out, shift(a, dy, dx, wrap), out=out)
    return out


def _pool(a: np.ndarray, wrap: bool, op) -> tuple[np.ndarray, int, int]:
    hh, ww = a.shape
    py, px = hh % 2, ww % 2
    if py or px:
        a = np.pad(a, ((0, py), (0, px)), mode="wrap" if wrap else "edge")
    a = op(a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2), axis=(1, 3))
    return a, py, px


def _far_level(cmax_l: np.ndarray, cmin_l: np.ndarray, block: int, pad: tuple[int, int],
               full: tuple[int, int], wrap: bool) -> np.ndarray:
    hh, ww = full
    lh, lw = cmin_l.shape
    out = np.full_like(cmax_l, np.inf)
    for oy in range(-_RING, _RING + 1):
        for ox in range(-_RING, _RING + 1):
            if max(abs(oy), abs(ox)) < 2:
                continue
            if wrap and (abs(oy) > lh // 2 + 1 or abs(ox) > lw // 2 + 1):
                continue
            gy = max((abs(oy) - 1) * block - pad[0], 0) / hh
            gx = max((abs(ox) - 1) * block - pad[1], 0) / ww
            gap = np.float32(np.hypot(gx, gy))
            if gap <= 0:
                continue
            drop = cmax_l - shift(cmin_l, oy, ox, wrap, fill=None if wrap else np.inf)
            np.minimum(out, np.where(drop > 0, gap / np.maximum(drop, 1e-9), np.inf), out=out)
    return out


def cone_map(h: np.ndarray, wrap: bool) -> np.ndarray:
    d = (1.0 - np.asarray(h, dtype=np.float32)).astype(np.float32)
    hh, ww = d.shape
    cmin, cmax, lip = _cells(d, wrap)
    w_cell = 1.0 / np.maximum(_max_filter3(lip, wrap), 1e-6)
    np.minimum(w_cell, CONE_MAX, out=w_cell)

    lmin, lmax = cmin, cmax
    block = 1
    pad_y = pad_x = 0
    while True:
        far = _far_level(lmax, lmin, block, (pad_y, pad_x), (hh, ww), wrap)
        if block > 1:
            far = np.repeat(np.repeat(far, block, axis=0), block, axis=1)[:hh, :ww]
        np.minimum(w_cell, far, out=w_cell)
        if block * 4 > max(hh, ww):
            break
        lmin, py, px = _pool(lmin, wrap, np.min)
        lmax, _, _ = _pool(lmax, wrap, np.max)
        pad_y += py * block
        pad_x += px * block
        block *= 2
    covered = (_RING * block) / max(hh, ww)
    drop = cmax - float(cmin.min())
    np.minimum(w_cell, np.where(drop > 0, covered / np.maximum(drop, 1e-9), np.inf), out=w_cell)

    w = w_cell
    for dy, dx in ((-1, -1), (-1, 0), (0, -1)):
        w = np.minimum(w, shift(w_cell, dy, dx, wrap))
    return (np.clip(w, 0.0, CONE_MAX) * (1.0 - 2.0 ** -9)).astype(np.float32)


def resample(a: np.ndarray, height: int, width: int) -> np.ndarray:
    hh, ww = a.shape[:2]
    if (hh, ww) == (height, width):
        return a.astype(np.float32)
    if hh % height == 0 and ww % width == 0 and hh >= height and ww >= width:
        fy, fx = hh // height, ww // width
        return a.reshape(height, fy, width, fx, *a.shape[2:]).mean(axis=(1, 3)).astype(np.float32)
    from .reference import sample_bilinear

    v = (np.arange(height) + 0.5) / height
    u = (np.arange(width) + 0.5) / width
    uu, vv = np.meshgrid(u, v)
    out = sample_bilinear(a, np.stack([uu.ravel(), vv.ravel()], 1), wrap=False)
    return out.reshape(height, width, *a.shape[2:]).astype(np.float32)


def data_size(height: int, width: int, max_res: int) -> tuple[int, int]:
    scale = min(1.0, max_res / max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


@dataclass
class DataMaps:
    march: np.ndarray
    shade: np.ndarray


def build_data_maps(h: np.ndarray, wrap: bool, depth_texels: float, sdf: np.ndarray | None = None,
                    with_cone: bool = False, with_cavity: bool = True, progress=None) -> DataMaps:
    tick = progress or (lambda _f: None)
    hh, ww = h.shape
    march = np.zeros((hh, ww, 4), np.float32)
    shade = np.zeros((hh, ww, 4), np.float32)
    march[..., 0] = h
    tick(0.1)
    du, dv = gradient(h, wrap)
    shade[..., 0] = du
    shade[..., 1] = dv
    tick(0.3)
    shade[..., 2] = cavity(h, wrap, depth_texels) if with_cavity else 1.0
    tick(0.6)
    if with_cone:
        march[..., 1] = cone_map(h, wrap)
    tick(0.9)
    if sdf is not None:
        march[..., 2] = resample(sdf, hh, ww) if sdf.shape != h.shape else sdf
    tick(1.0)
    return DataMaps(march, shade)


_DIRECTIONS = 8


def cavity(h: np.ndarray, wrap: bool, depth_texels: float, radius: int = 16, samples: int = 6) -> np.ndarray:
    hh = np.asarray(h, dtype=np.float32) * np.float32(depth_texels)
    occl = np.zeros_like(hh)
    radii = np.unique(np.round(np.geomspace(1, radius, samples)).astype(int))
    for k in range(_DIRECTIONS):
        ang = 2 * np.pi * k / _DIRECTIONS
        cx, cy = np.cos(ang), np.sin(ang)
        best = np.zeros_like(hh)
        for r in radii:
            dx, dy = int(round(cx * r)), int(round(cy * r))
            dist = float(np.hypot(dx, dy))
            if dist == 0:
                continue
            rise = shift(hh, dy, dx, wrap) - hh
            np.maximum(best, rise / dist, out=best)
        occl += best / np.sqrt(1.0 + best * best)
    return np.clip(1.0 - occl / _DIRECTIONS, 0.0, 1.0).astype(np.float32)
