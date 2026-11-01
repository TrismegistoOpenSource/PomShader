# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np

SCOPE = ("The map measures how faithful the simplified model is to the high-poly model. "
         "It does not measure how accurate the relief is with respect to the real object.")


def stats(values: np.ndarray, thresholds_mm=(0.5, 1.0, 2.0)) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"count": 0}
    a = np.abs(v)
    out = {
        "count": int(v.size),
        "mean": float(v.mean()),
        "mean_abs": float(a.mean()),
        "rms": float(np.sqrt((v * v).mean())),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }
    for t in thresholds_mm:
        out[f"within_{t:g}mm_pct"] = float((a <= t).mean() * 100)
    return out


def histogram(values: np.ndarray, bins: int = 40, limit: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    lim = limit if limit is not None else (float(np.percentile(np.abs(v), 99.5)) if v.size else 1.0) or 1.0
    return np.histogram(np.clip(v, -lim, lim), bins=bins, range=(-lim, lim))


_BLUE = np.array([59, 76, 192]) / 255.0
_WHITE = np.array([242, 242, 242]) / 255.0
_RED = np.array([180, 4, 38]) / 255.0
_D65 = np.array([0.95047, 1.0, 1.08883])


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])


def _to_lab(rgb):
    xyz = _srgb_to_linear(rgb) @ _M.T / _D65
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _from_lab(lab):
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _D65
    return _linear_to_srgb(xyz @ np.linalg.inv(_M).T)


def diverging(values_mm: np.ndarray, limit_mm: float) -> np.ndarray:
    t = np.clip(np.asarray(values_mm, dtype=np.float64) / max(limit_mm, 1e-12), -1.0, 1.0)
    lab_b, lab_w, lab_r = _to_lab(_BLUE), _to_lab(_WHITE), _to_lab(_RED)
    neg = t < 0
    s = np.abs(t)[..., None]
    lab = np.where(neg[..., None], lab_w + (lab_b - lab_w) * s, lab_w + (lab_r - lab_w) * s)
    out = _from_lab(lab)
    return np.where(np.isfinite(values_mm)[..., None], out, 0.5)


def svg_histogram(counts: np.ndarray, edges: np.ndarray, unit: str = "mm", width: int = 560, height: int = 180) -> str:
    peak = max(int(counts.max()), 1)
    n = len(counts)
    bw = (width - 60) / n
    bars = []
    for i, c in enumerate(counts):
        h = (height - 40) * c / peak
        mid = 0.5 * (edges[i] + edges[i + 1])
        color = diverging(np.array([mid]), max(abs(edges[0]), abs(edges[-1])))[0]
        fill = "#%02x%02x%02x" % tuple(int(round(x * 255)) for x in color)
        bars.append(f'<rect x="{50 + i * bw:.1f}" y="{height - 25 - h:.1f}" width="{bw - 1:.1f}" '
                    f'height="{h:.1f}" fill="{fill}" stroke="#888" stroke-width="0.3"/>')
    labels = (f'<text x="50" y="{height - 8}" font-size="11">{edges[0]:.2f} {unit}</text>'
              f'<text x="{width / 2 + 10:.0f}" y="{height - 8}" font-size="11" text-anchor="middle">0</text>'
              f'<text x="{width - 10}" y="{height - 8}" font-size="11" text-anchor="end">{edges[-1]:.2f} {unit}</text>'
              f'<text x="4" y="14" font-size="11">{peak}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">{"".join(bars)}{labels}</svg>')


def island_at(labels: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    uv = np.nan_to_num(np.asarray(uv, dtype=np.float64), nan=-1.0)
    x = np.floor(uv[..., 0] * w).astype(np.int64)
    y = np.floor(uv[..., 1] * h).astype(np.int64)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full(uv.shape[:-1], -1, np.int32)
    out[ok] = labels[y[ok], x[ok]]
    return out


def seam_crossings(labels: np.ndarray, uv_a: np.ndarray, uv_b: np.ndarray) -> np.ndarray:
    return island_at(labels, uv_a) != island_at(labels, uv_b)
