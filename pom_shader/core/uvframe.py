# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    dpdu: np.ndarray
    dpdv: np.ndarray
    degenerate: int


def triangle_frames(p: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    d1 = uv[:, 1] - uv[:, 0]
    d2 = uv[:, 2] - uv[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)[:, None]
    tu = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * inv
    tv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * inv
    return tu, tv, ok


def _corner_angles(p: np.ndarray) -> np.ndarray:
    out = np.empty(p.shape[:2])
    for i in range(3):
        a = p[:, (i + 1) % 3] - p[:, i]
        b = p[:, (i + 2) % 3] - p[:, i]
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        cos = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-30)
        out[:, i] = np.arccos(np.clip(cos, -1.0, 1.0))
    return out


def uv_vertex_ids(loop_verts: np.ndarray, loop_uvs: np.ndarray, loop_normals: np.ndarray | None = None) -> np.ndarray:
    q = np.round(loop_uvs.astype(np.float64) * 1e6).astype(np.int64)
    cols = [loop_verts.astype(np.int64), q[:, 0], q[:, 1]]
    if loop_normals is not None:
        qn = np.round(loop_normals.astype(np.float64) * 1e3).astype(np.int64)
        cols += [qn[:, 0], qn[:, 1], qn[:, 2]]
    key = np.stack(cols, axis=1)
    _, ids = np.unique(key, axis=0, return_inverse=True)
    return ids.reshape(-1)


def compute_frame(positions: np.ndarray, tri_loops: np.ndarray, loop_verts: np.ndarray,
                  loop_uvs: np.ndarray, loop_normals: np.ndarray | None = None) -> Frame:
    pos = positions.astype(np.float64)
    uvs = loop_uvs.astype(np.float64)
    p = pos[loop_verts[tri_loops]]
    uv = uvs[tri_loops]
    tu, tv, ok = triangle_frames(p, uv)
    w = _corner_angles(p) * ok[:, None]
    ids = uv_vertex_ids(loop_verts, loop_uvs, loop_normals)
    n_ids = int(ids.max()) + 1 if ids.size else 0
    acc_u = np.zeros((n_ids, 3))
    acc_v = np.zeros((n_ids, 3))
    acc_w = np.zeros(n_ids)
    corner_ids = ids[tri_loops]
    for c in range(3):
        np.add.at(acc_u, corner_ids[:, c], tu * w[:, c:c + 1])
        np.add.at(acc_v, corner_ids[:, c], tv * w[:, c:c + 1])
        np.add.at(acc_w, corner_ids[:, c], w[:, c])
    acc_w = np.maximum(acc_w, 1e-30)[:, None]
    dpdu = (acc_u / acc_w)[ids]
    dpdv = (acc_v / acc_w)[ids]
    return Frame(dpdu.astype(np.float32), dpdv.astype(np.float32), int((~ok).sum()))


def compact_scale(frame: Frame, loop_normals: np.ndarray) -> np.ndarray:
    su = np.linalg.norm(frame.dpdu, axis=1)
    sv = np.linalg.norm(frame.dpdv, axis=1)
    side = np.einsum("ij,ij->i", np.cross(frame.dpdu, frame.dpdv), loop_normals)
    return np.stack([su, np.where(side < 0, -sv, sv)], axis=1).astype(np.float32)


def uniform_scale(positions: np.ndarray, tri_loops: np.ndarray, loop_verts: np.ndarray,
                  loop_uvs: np.ndarray) -> float:
    p = positions.astype(np.float64)[loop_verts[tri_loops]]
    uv = loop_uvs.astype(np.float64)[tri_loops]
    a3 = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    a2 = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1])
    ok = a2 > 1e-14
    if not ok.any():
        return 1.0
    return float(np.sqrt(np.median(a3[ok] / a2[ok])))


def topology_hash(positions: np.ndarray, loop_verts: np.ndarray, loop_uvs: np.ndarray) -> str:
    h = hashlib.sha1()
    for arr in (positions.astype(np.float32), loop_verts.astype(np.int32), loop_uvs.astype(np.float32)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def quick_signature(positions: np.ndarray, loop_uvs: np.ndarray, samples: int = 997) -> str:
    h = hashlib.sha1()
    h.update(np.array([len(positions), len(loop_uvs)], np.int64).tobytes())
    for arr in (positions, loop_uvs):
        step = max(1, len(arr) // samples)
        h.update(np.ascontiguousarray(arr[::step], dtype=np.float32).tobytes())
    return h.hexdigest()[:16]
