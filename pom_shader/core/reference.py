# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIT_EPS = 1e-6


def sample_bilinear(img: np.ndarray, uv: np.ndarray, wrap: bool) -> np.ndarray:
    h, w = img.shape[:2]
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = x - x0
    fy = y - y0
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    if wrap:
        x0 %= w
        x1 %= w
        y0 %= h
        y1 %= h
    else:
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)
    if img.ndim == 3:
        fx = fx[:, None]
        fy = fy[:, None]
    a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    b = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return a * (1 - fy) + b * fy


@dataclass
class HeightSource:

    height: np.ndarray
    wrap: bool = True
    cone: np.ndarray | None = None
    reads: int = 0

    @property
    def resolution(self) -> int:
        return int(max(self.height.shape[:2]))

    def h(self, uv: np.ndarray) -> np.ndarray:
        self.reads += 1
        return sample_bilinear(self.height, uv, self.wrap)

    def h_and_cone(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.cone is None:
            raise ValueError("cone map missing")
        self.reads += 1
        stacked = np.stack([self.height, self.cone], axis=-1)
        v = sample_bilinear(stacked, uv, self.wrap)
        return v[:, 0], v[:, 1]


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def uv_velocity_exact(tu: np.ndarray, tv: np.ndarray, r: np.ndarray) -> np.ndarray:
    a = _dot(tu, tu)
    b = _dot(tu, tv)
    c = _dot(tv, tv)
    det = a * c - b * b
    pu = _dot(tu, r)
    pv = _dot(tv, r)
    du = (c * pu - b * pv) / det
    dv = (a * pv - b * pu) / det
    return np.stack([du, dv], axis=1)


def uv_velocity_orthogonal(tu: np.ndarray, tv: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.stack([_dot(tu, r) / _dot(tu, tu), _dot(tv, r) / _dot(tv, tv)], axis=1)


@dataclass
class RaySetup:
    uv0: np.ndarray
    uv_top: np.ndarray
    s: np.ndarray
    ref_plane: float


def setup_rays(
    uv0: np.ndarray,
    duv: np.ndarray,
    cos_view: np.ndarray,
    depth: np.ndarray | float,
    ref_plane: float,
    c_min: float = 0.08,
    s_max: float | None = None,
    sdf: np.ndarray | None = None,
) -> RaySetup:
    c = np.maximum(cos_view, c_min)
    depth = np.broadcast_to(np.asarray(depth, dtype=np.float64), c.shape)
    s = limit_offset(duv * (depth / c)[:, None], s_max, sdf)
    uv_top = uv0 - (1.0 - ref_plane) * s
    return RaySetup(uv0=uv0, uv_top=uv_top, s=s, ref_plane=ref_plane)


def limit_offset(s: np.ndarray, s_max: float | None, sdf: np.ndarray | None) -> np.ndarray:
    length = np.linalg.norm(s, axis=1)
    limit = np.full_like(length, np.inf)
    if s_max is not None:
        limit = np.minimum(limit, s_max)
    if sdf is not None:
        limit = np.minimum(limit, np.maximum(sdf, 0.0))
    scale = np.minimum(1.0, limit / np.maximum(length, 1e-12))
    return s * scale[:, None]


def ray_uv(ray: RaySetup, k: np.ndarray) -> np.ndarray:
    return ray.uv_top + k[:, None] * ray.s


def diff(src: HeightSource, ray: RaySetup, k: np.ndarray) -> np.ndarray:
    return (1.0 - k) - src.h(ray_uv(ray, k))


@dataclass
class Bracket:
    k_lo: np.ndarray
    k_hi: np.ndarray
    d_lo: np.ndarray
    d_hi: np.ndarray


def linear_search(src: HeightSource, ray: RaySetup, n: int, xi: np.ndarray | float = 0.0) -> Bracket:
    m = ray.uv0.shape[0]
    xi = np.broadcast_to(np.asarray(xi, dtype=np.float64), (m,))
    delta = 1.0 / (n - xi)
    q = 1.0 + xi * delta
    active = np.ones(m)
    for _ in range(n):
        cand = q - delta * active
        d = cand - src.h(ray.uv_top + (1.0 - cand)[:, None] * ray.s)
        above = (d > HIT_EPS).astype(np.float64)
        active = active * above
        q = q - delta * active
    k_last = 1.0 - q
    k_hi = np.where(active > 0, 1.0, np.minimum(k_last + delta, 1.0))
    k_lo = np.maximum(k_last, 0.0)
    return Bracket(k_lo, k_hi, diff(src, ray, k_lo), diff(src, ray, k_hi))


def _secant(b: Bracket) -> np.ndarray:
    den = b.d_lo - b.d_hi
    t = np.where(den > 1e-12, b.d_lo / np.maximum(den, 1e-12), 0.5)
    return b.k_lo + (b.k_hi - b.k_lo) * t


def refine(src: HeightSource, ray: RaySetup, b: Bracket, m: int, method: str = "illinois") -> np.ndarray:
    k_lo, k_hi, d_lo, d_hi = b.k_lo.copy(), b.k_hi.copy(), b.d_lo.copy(), b.d_hi.copy()
    side = np.zeros_like(k_lo)
    for i in range(m):
        if method == "bisection" or (method == "hybrid" and i > 0):
            k_new = 0.5 * (k_lo + k_hi)
        else:
            k_new = _secant(Bracket(k_lo, k_hi, d_lo, d_hi))
        d_new = diff(src, ray, k_new)
        above = d_new > HIT_EPS
        if method == "illinois":
            d_hi = np.where(above & (side > 0), 0.5 * d_hi, d_hi)
            d_lo = np.where(~above & (side < 0), 0.5 * d_lo, d_lo)
        k_lo = np.where(above, k_new, k_lo)
        d_lo = np.where(above, d_new, d_lo)
        k_hi = np.where(above, k_hi, k_new)
        d_hi = np.where(above, d_hi, d_new)
        side = np.where(above, 1.0, -1.0)
    return _secant(Bracket(k_lo, k_hi, d_lo, d_hi))


def dense_reference(src: HeightSource, ray: RaySetup, n: int = 4096, bisect: int = 30) -> np.ndarray:
    reads = src.reads
    b = linear_search(src, ray, n, 0.0)
    k_lo, k_hi = b.k_lo, b.k_hi
    for _ in range(bisect):
        mid = 0.5 * (k_lo + k_hi)
        above = diff(src, ray, mid) > HIT_EPS
        k_lo = np.where(above, mid, k_lo)
        k_hi = np.where(above, k_hi, mid)
    src.reads = reads
    return 0.5 * (k_lo + k_hi)


def displaced_normal(
    tu: np.ndarray, tv: np.ndarray, n: np.ndarray, depth: np.ndarray | float, dh_du: np.ndarray, dh_dv: np.ndarray
) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float64)
    d = d[:, None] if d.ndim else d
    tu2 = tu + n * d * dh_du[:, None]
    tv2 = tv + n * d * dh_dv[:, None]
    cr = np.cross(tu2, tv2)
    sign = np.sign(_dot(np.cross(tu, tv), n))
    sign = np.where(sign == 0, 1.0, sign)
    return cr / np.linalg.norm(cr, axis=1, keepdims=True) * sign[:, None]


def uv_error_texels(uv_a: np.ndarray, uv_b: np.ndarray, resolution: int, wrap: bool = True) -> np.ndarray:
    d = uv_a - uv_b
    if wrap:
        d = d - np.round(d)
    return np.linalg.norm(d, axis=1) * resolution


def shadow(src: HeightSource, uv_hit: np.ndarray, h_hit: np.ndarray, duv_light: np.ndarray, cos_light: np.ndarray,
           depth: float, steps: int, sharpness: float, strength: float, room: np.ndarray | None = None) -> np.ndarray:
    s_l = duv_light * (depth / np.maximum(cos_light, 0.05))[:, None]
    rise = 1.0 - h_hit
    s_eff = s_l * rise[:, None]
    if room is not None:
        length = np.linalg.norm(s_eff, axis=1)
        s_eff = s_eff * np.minimum(1.0, room / np.maximum(length, 1e-12))[:, None]
    occl = np.zeros(len(h_hit))
    for j in range(steps):
        t = (j + 1) / steps
        h_ray = h_hit + t * rise
        h = src.h(uv_hit + t * s_eff)
        occl = np.maximum(occl, np.clip((h - h_ray) * (1 - t) * sharpness, 0.0, 1.0))
    amount = strength * np.clip(10.0 * cos_light, 0.0, 1.0)
    return 1.0 + amount * ((1.0 - occl) - 1.0)
