# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import os
from dataclasses import dataclass, field

import bpy
import numpy as np

from .. import preferences
from ..core import maps, sdf
from ..graph import material_patch as mp
from ..storage import cache, image_io
from ..utils import compat, log
from ..utils.i18n import tr

LOG = log.get("ops")


class NeedsChoice(Exception):

    def __init__(self, options: list[str]) -> None:
        super().__init__("ambiguous height source")
        self.options = options


def objects_with(mat: bpy.types.Material) -> list[bpy.types.Object]:
    out = []
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        if any(slot.material == mat for slot in ob.material_slots):
            out.append(ob)
    return out


def meshes_with(mat: bpy.types.Material) -> list[bpy.types.Mesh]:
    seen, out = set(), []
    for ob in objects_with(mat):
        if ob.data.name not in seen:
            seen.add(ob.data.name)
            out.append(ob.data)
    return out


def default_uv(mat: bpy.types.Material) -> str:
    meshes = meshes_with(mat)
    return compat.active_render_uv(meshes[0]) if meshes else ""


def resolve_uv_name(mesh: bpy.types.Mesh, uv_map: str) -> str | None:
    name = uv_map or compat.active_render_uv(mesh)
    return name if name in mesh.uv_layers else None


@dataclass
class MeshArrays:
    positions: np.ndarray
    loop_verts: np.ndarray
    loop_uvs: np.ndarray
    loop_normals: np.ndarray
    tri_loops: np.ndarray
    tri_material: np.ndarray


def mesh_arrays(mesh: bpy.types.Mesh, uv_name: str) -> MeshArrays:
    n_v, n_l = len(mesh.vertices), len(mesh.loops)
    pos = np.empty(n_v * 3, np.float32)
    mesh.vertices.foreach_get("co", pos)
    lv = np.empty(n_l, np.int32)
    mesh.loops.foreach_get("vertex_index", lv)
    uvs = np.empty(n_l * 2, np.float32)
    mesh.uv_layers[uv_name].data.foreach_get("uv", uvs)
    nrm = np.empty(n_l * 3, np.float32)
    mesh.corner_normals.foreach_get("vector", nrm)
    mesh.calc_loop_triangles()
    n_t = len(mesh.loop_triangles)
    tris = np.empty(n_t * 3, np.int32)
    mesh.loop_triangles.foreach_get("loops", tris)
    mats = np.empty(n_t, np.int32)
    mesh.loop_triangles.foreach_get("material_index", mats)
    return MeshArrays(pos.reshape(-1, 3), lv, uvs.reshape(-1, 2), nrm.reshape(-1, 3), tris.reshape(-1, 3), mats)


def material_slots(mesh: bpy.types.Mesh, mat: bpy.types.Material) -> list[int]:
    return [i for i, m in enumerate(mesh.materials) if m == mat]


def resolve_height(mat: bpy.types.Material, analysis: mp.Analysis) -> tuple[bpy.types.Image, str]:
    st = mat.rpom
    node = mat.node_tree.nodes.get(st.height_node) if st.height_node else None
    node_img = getattr(node, "image", None) if node is not None else None
    if st.height_image is not None and (node_img is None or st.height_image.name != st.height_used):
        img = st.height_image
        if node is None:
            node = next((t.node for t in analysis.all_textures if t.node.image == img), None)
        elif node.image != img:
            node.image = img
    elif node_img is not None:
        img = node_img
    else:
        cands = analysis.height_candidates
        if not cands:
            raise ValueError(tr("no height found: choose it in the panel (name with height/disp, or linked to "
                                "Displacement/Bump)"))
        if len(cands) > 1 and _score_tie(cands, analysis):
            raise NeedsChoice([n.name for n in cands])
        img, node = cands[0].image, cands[0]
    st.height_image = img
    st.height_used = img.name
    return img, node.name if node is not None else st.height_node


def _score_tie(cands: list[bpy.types.Node], analysis: mp.Analysis) -> bool:
    linked = [n for n in cands if mp._feeds(n, "ShaderNodeDisplacement", "Height") or mp._feeds(n, "ShaderNodeBump", "Height")]
    return len(linked) != 1 and len({n.image.name for n in cands}) > 1


@dataclass
class MapsJob:
    mat_name: str
    image_name: str
    pixels: np.ndarray
    bit_depth: int
    channel: str
    source_kind: str
    wrap: bool
    data_res: int
    depth_m: float
    meters_per_uv: float
    labels_tris: tuple[np.ndarray, np.ndarray] | None
    entry: cache.CacheEntry
    cache_warning: str | None
    result: maps.DataMaps | None = None
    depth_out: float = 0.0
    ref_out: float | None = None
    warnings: list[str] = field(default_factory=list)
    cancelled: bool = False


def collect(mat: bpy.types.Material, image: bpy.types.Image, uv_scale: float) -> MapsJob:
    st = mat.rpom
    prefs = preferences.get()
    max_res = st.data_resolution or (prefs.data_resolution if prefs else 4096)
    px = image_io.read_pixels(image)
    depth_bits = image_io.bit_depth(image)
    wrap = st.border_mode == "TILING"
    tris = None
    if st.border_mode == "ATLAS":
        tris = atlas_triangles(mat, st.uv_map)
    directory, warning = cache.cache_dir(prefs.cache_dir if prefs else "")
    ident = cache.image_identity(image, px)
    params = {"res": max_res, "wrap": wrap, "mode": st.border_mode, "channel": st.channel, "kind": st.source_kind,
              "depth": round(st.depth, 6), "mpu": round(uv_scale, 6),
              "atlas": cache.pixel_checksum(tris[0]) if tris is not None else ""}
    stem = os.path.splitext(os.path.basename(image.filepath or image.name))[0]
    entry = cache.entry_for(directory, stem, ident, params)
    return MapsJob(mat.name, image.name, px, depth_bits, st.channel, st.source_kind, wrap, max_res, st.depth,
                   uv_scale, tris, entry, warning)


def atlas_triangles(mat: bpy.types.Material, uv_map: str) -> tuple[np.ndarray, np.ndarray]:
    all_uv, all_isl, offset = [], [], 0
    for me in meshes_with(mat):
        uv_name = resolve_uv_name(me, uv_map)
        if uv_name is None:
            continue
        arr = mesh_arrays(me, uv_name)
        sel = np.isin(arr.tri_material, material_slots(me, mat))
        tri = arr.tri_loops[sel]
        tri_uv = arr.loop_uvs[tri]
        isl = sdf.island_ids(arr.loop_verts[tri], tri_uv) + offset
        offset = int(isl.max()) + 1 if isl.size else offset
        all_uv.append(tri_uv)
        all_isl.append(isl)
    if not all_uv:
        raise ValueError(tr("no mesh with UVs uses this material"))
    return np.concatenate(all_uv), np.concatenate(all_isl)


def compute(job: MapsJob, progress=None) -> None:
    tick = progress or (lambda _f: None)
    h_raw = image_io.height_from_pixels(job.pixels, job.channel)
    kind = job.source_kind
    if kind == "AUTO":
        is_meters = job.bit_depth == 32 and (float(h_raw.min()) < -1e-6 or float(h_raw.max()) > 1 + 1e-6)
        kind = "METERS" if is_meters else "UNIT"
    if kind == "METERS":
        norm = maps.normalize_meters(h_raw)
        job.depth_out, job.ref_out = norm.depth_m, norm.ref_plane
    else:
        norm = maps.normalize_unit(h_raw, job.bit_depth)
        job.depth_out = job.depth_m
    job.warnings += norm.warnings
    tick(0.1)
    hh, ww = maps.data_size(*norm.height.shape, job.data_res)
    h = maps.resample(norm.height, hh, ww)
    tick(0.2)
    border = None
    if job.labels_tris is not None:
        tri_uv, isl = job.labels_tris
        r = sdf.rasterize(tri_uv, isl, ww, hh)
        if r.overlap_texels:
            job.warnings.append(tr("atlas with {} overlapping texels: the island border is unreliable there", r.overlap_texels))
        border = sdf.border_distance(r.labels)
    tick(0.4)
    texel_m = job.meters_per_uv / max(hh, ww)
    depth_texels = job.depth_out / texel_m if texel_m > 0 else 1.0
    job.result = maps.build_data_maps(h, job.wrap, depth_texels, sdf=border,
                                      progress=lambda f: tick(0.4 + 0.6 * f))


def finish(job: MapsJob, half: bool = True) -> tuple[bpy.types.Image, bpy.types.Image]:
    image_io.write_exr(job.entry.march_path, job.result.march, half)
    image_io.write_exr(job.entry.shade_path, job.result.shade, half)
    job.entry.write_meta({"depth": job.depth_out, "ref_plane": job.ref_out, "source": job.image_name,
                          "warnings": job.warnings, "shape": list(job.result.march.shape[:2])})
    return load_cached(job.entry, job.image_name)


def load_cached(entry: cache.CacheEntry, image_name: str) -> tuple[bpy.types.Image, bpy.types.Image]:
    base = os.path.splitext(image_name)[0]
    march = image_io.load_data_image(entry.march_path, f"{base}_rpom_march")
    shade = image_io.load_data_image(entry.shade_path, f"{base}_rpom_shade")
    return march, shade
