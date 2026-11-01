# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json

import bpy
import numpy as np

from .. import preferences
from ..core import uvframe
from ..graph.signature import ATTR_DPDU, ATTR_DPDV, ATTR_UVSCALE
from ..utils.i18n import tr
from . import common

FRAME_KEY = "rpom_frame"


def _set_attr(mesh: bpy.types.Mesh, name: str, kind: str, data: np.ndarray) -> None:
    at = mesh.attributes.get(name)
    if at is not None and (at.domain != "CORNER" or at.data_type != kind):
        mesh.attributes.remove(at)
        at = None
    if at is None:
        at = mesh.attributes.new(name, kind, "CORNER")
    field = "vector"
    at.data.foreach_set(field, np.ascontiguousarray(data, dtype=np.float32).ravel())


def _drop(mesh: bpy.types.Mesh, names) -> None:
    for name in names:
        at = mesh.attributes.get(name)
        if at is not None:
            mesh.attributes.remove(at)


def choose_mode(requested: str, border_mode: str, arr: common.MeshArrays, frame: uvframe.Frame) -> str:
    if requested != "AUTO":
        return requested
    prefs = preferences.get()
    threshold = (prefs.compact_threshold if prefs else 2) * 1_000_000
    if len(arr.loop_verts) > threshold:
        return "COMPACT"
    if border_mode == "ATLAS":
        return "EXACT"
    su = np.linalg.norm(frame.dpdu, axis=1)
    sv = np.linalg.norm(frame.dpdv, axis=1)
    ok = (su > 0) & (sv > 0)
    if not ok.any():
        return "UNIFORM"
    ratio = su[ok] / sv[ok]
    density = np.sqrt(su[ok] * sv[ok])
    cos = np.abs(np.einsum("ij,ij->i", frame.dpdu[ok], frame.dpdv[ok])) / (su[ok] * sv[ok])
    uniform = (np.std(density) / max(np.mean(density), 1e-12) < 0.02 and np.abs(ratio - 1).max() < 0.02
               and cos.max() < 0.02)
    return "UNIFORM" if uniform else "EXACT"


def write_frame(mesh: bpy.types.Mesh, uv_map: str, requested: str, border_mode: str) -> tuple[str, float]:
    uv_name = common.resolve_uv_name(mesh, uv_map)
    if uv_name is None:
        raise ValueError(tr("{}: no UV map '{}'", mesh.name, uv_map) if uv_map else tr("{}: no active UV map", mesh.name))
    arr = common.mesh_arrays(mesh, uv_name)
    frame = uvframe.compute_frame(arr.positions, arr.tri_loops, arr.loop_verts, arr.loop_uvs, arr.loop_normals)
    scale = uvframe.uniform_scale(arr.positions, arr.tri_loops, arr.loop_verts, arr.loop_uvs)
    mode = choose_mode(requested, border_mode, arr, frame)
    if mode == "EXACT":
        _set_attr(mesh, ATTR_DPDU, "FLOAT_VECTOR", frame.dpdu)
        _set_attr(mesh, ATTR_DPDV, "FLOAT_VECTOR", frame.dpdv)
        _drop(mesh, (ATTR_UVSCALE,))
    elif mode == "COMPACT":
        _set_attr(mesh, ATTR_UVSCALE, "FLOAT2", uvframe.compact_scale(frame, arr.loop_normals))
        _drop(mesh, (ATTR_DPDU, ATTR_DPDV))
    else:
        _drop(mesh, (ATTR_DPDU, ATTR_DPDV, ATTR_UVSCALE))
    mesh[FRAME_KEY] = json.dumps({
        "mode": mode, "uv": uv_name, "scale": scale, "degenerate": frame.degenerate,
        "hash": uvframe.topology_hash(arr.positions, arr.loop_verts, arr.loop_uvs),
        "quick": uvframe.quick_signature(arr.positions, arr.loop_uvs),
    })
    mesh.update()
    return mode, scale


def frame_info(mesh: bpy.types.Mesh) -> dict | None:
    raw = mesh.get(FRAME_KEY)
    return json.loads(raw) if raw else None


def is_stale(mesh: bpy.types.Mesh, quick: bool = True) -> bool:
    info = frame_info(mesh)
    if info is None or info.get("uv") not in mesh.uv_layers:
        return info is not None
    if quick:
        pos = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", pos)
        uvs = np.empty(len(mesh.loops) * 2, np.float32)
        mesh.uv_layers[info["uv"]].data.foreach_get("uv", uvs)
        return uvframe.quick_signature(pos.reshape(-1, 3), uvs.reshape(-1, 2)) != info.get("quick")
    arr = common.mesh_arrays(mesh, info["uv"])
    return uvframe.topology_hash(arr.positions, arr.loop_verts, arr.loop_uvs) != info["hash"]


class RPOM_OT_compute_frame(bpy.types.Operator):
    """Compute the tangent frame attributes on the meshes using the material"""

    bl_idname = "rpom.compute_frame"
    bl_label = "Compute UV Frame"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        mat = getattr(context, "material", None) or _active_material(context)
        return mat is not None and mat.library is None

    def execute(self, context):
        mat = getattr(context, "material", None) or _active_material(context)
        st = mat.rpom
        modes = []
        for me in common.meshes_with(mat):
            try:
                modes.append(write_frame(me, st.uv_map, st.frame_mode, st.border_mode)[0])
            except ValueError as e:
                self.report({"WARNING"}, str(e))
        self.report({"INFO"}, tr("frame: {}", ", ".join(sorted(set(modes))) or tr("no mesh")))
        return {"FINISHED"}


def _active_material(context):
    ob = context.active_object
    return ob.active_material if ob is not None else None


CLASSES = (RPOM_OT_compute_frame,)
