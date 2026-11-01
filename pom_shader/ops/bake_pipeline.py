# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from ..core import bake_math as bm
from ..core import sdf
from ..storage import image_io
from ..utils import log
from ..utils.i18n import tr
from . import common

LOG = log.get("bake")
FAR_FROM_ORIGIN = 10_000.0
VERSION = "1.0.0"


@dataclass
class BakeSettings:
    resolution: int = 2048
    extrusion: float = 0.0
    margin: float = 0.002
    uv_map: str = ""
    method: str = "CYCLES"
    validate_fraction: float = 0.01
    undercut_mode: str = "AUTO"
    lateral_tolerance: float = 0.0
    bake_albedo: bool = True
    float32_maps: bool = False
    apply_pom: bool = True


@dataclass
class BakeResult:
    directory: str
    displacement_path: str
    flags_path: str
    meta: dict
    warnings: list[str] = field(default_factory=list)


def _bbox_world(ob: bpy.types.Object) -> tuple[Vector, Vector]:
    pts = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def checks(high: bpy.types.Object, low: bpy.types.Object, s: BakeSettings) -> tuple[list[str], list[str]]:
    blocking, warnings = [], []
    for ob in (high, low):
        sc = ob.matrix_world.to_scale()
        if max(sc) - min(sc) > 1e-4 * max(abs(max(sc)), 1e-9):
            blocking.append(tr("{}: non-uniform scale not applied ({:.4g}, {:.4g}, {:.4g})", ob.name, sc.x, sc.y, sc.z))
        if ob.matrix_world.translation.length > FAR_FROM_ORIGIN or _bbox_world(ob)[1].length > FAR_FROM_ORIGIN:
            warnings.append(tr("{}: more than 10 km from the origin, working in a local frame", ob.name))
    if bpy.context.scene.unit_settings.system != "METRIC" or abs(bpy.context.scene.unit_settings.scale_length - 1) > 1e-9:
        warnings.append(tr("scene units are not meters: depths will be in Blender units"))
    if common.resolve_uv_name(low.data, s.uv_map) is None:
        blocking.append(tr("{}: no UV map", low.name))
    if any(not p.use_smooth for p in low.data.polygons):
        warnings.append(tr("{}: flat shading; the bake uses the same normal as the shader, smooth is better", low.name))
    return blocking, warnings


def uv_report(low: bpy.types.Object, s: BakeSettings) -> tuple[dict, np.ndarray, np.ndarray]:
    uv_name = common.resolve_uv_name(low.data, s.uv_map)
    arr = common.mesh_arrays(low.data, uv_name)
    tri_uv = arr.loop_uvs[arr.tri_loops].astype(np.float64)
    isl = sdf.island_ids(arr.loop_verts[arr.tri_loops], tri_uv)
    r = sdf.rasterize(tri_uv, isl, s.resolution, s.resolution)
    scale = low.matrix_world.to_scale().x
    p = arr.positions[arr.loop_verts[arr.tri_loops]].astype(np.float64) * scale
    a3 = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    d1, d2 = tri_uv[:, 1] - tri_uv[:, 0], tri_uv[:, 2] - tri_uv[:, 0]
    a2 = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1])
    ok = a2 > 1e-14
    mm_per_texel = np.sqrt(a3[ok] / a2[ok]) / s.resolution * 1000 if ok.any() else np.array([0.0])
    info = {"islands": int(isl.max() + 1 if isl.size else 0), "overlap_texels": r.overlap_texels,
            "coverage": float((r.labels != sdf.EMPTY).mean()),
            "mm_per_texel_p50": float(np.median(mm_per_texel)), "mm_per_texel_p95": float(np.percentile(mm_per_texel, 95))}
    return info, r.labels, isl


def _depsgraph():
    return bpy.context.evaluated_depsgraph_get()


def alignment(high: bpy.types.Object, low: bpy.types.Object, count: int = 10_000) -> tuple[float, np.ndarray]:
    dg = _depsgraph()
    bvh = BVHTree.FromObject(high, dg)
    to_high = high.matrix_world.inverted() @ low.matrix_world
    arr = common.mesh_arrays(low.data, low.data.uv_layers[0].name if low.data.uv_layers else None) \
        if low.data.uv_layers else None
    me = low.data
    pos = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", pos)
    me.calc_loop_triangles()
    tris = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tris)
    tp = pos.reshape(-1, 3)[tris.reshape(-1, 3)].astype(np.float64)
    smp = bm.sample_surface(tp, count)
    nrm = np.cross(tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    m3 = np.array(to_high.to_3x3())
    dist, signed = [], []
    for p, t in zip(smp.points, smp.tri):
        q = to_high @ Vector(p)
        hit = bvh.find_nearest(q)
        if hit[0] is None:
            continue
        dvec = (np.array(hit[0]) - np.array(q)) @ np.linalg.inv(m3).T
        dist.append(np.linalg.norm(dvec))
        signed.append(float(np.dot(dvec, nrm[t])))
    dist = np.array(dist) * low.matrix_world.to_scale().x
    return float(np.median(dist)) if dist.size else float("inf"), np.array(signed) * low.matrix_world.to_scale().x


class _LocalFrame:

    def __init__(self, high, low):
        self.objs = (high, low)
        lo, hi = _bbox_world(low)
        self.center = (lo + hi) / 2
        self.saved = {}

    def __enter__(self):
        vl = bpy.context.view_layer
        self.selection = [o for o in bpy.context.scene.objects if o.select_get()]
        self.active = vl.objects.active
        shift = Matrix.Translation(-self.center)
        for ob in self.objs:
            self.saved[ob.name] = (ob.matrix_basis.copy(), [s.material for s in ob.material_slots])
            ob.matrix_world = shift @ ob.matrix_world
        vl.update()
        return self

    def __exit__(self, *exc):
        for ob in self.objs:
            basis, mats = self.saved[ob.name]
            ob.matrix_basis = basis
            for slot, m in zip(ob.material_slots, mats):
                slot.material = m
        for o in bpy.context.scene.objects:
            o.select_set(o in self.selection)
        bpy.context.view_layer.objects.active = self.active
        bpy.context.view_layer.update()


def _emit_material(name: str, socket: str, image: bpy.types.Image | None) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = mat.node_tree
    nt.nodes.clear()
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    em = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(geo.outputs[socket], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    if image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        nt.nodes.active = tex
    return mat


def _assign(ob: bpy.types.Object, mat: bpy.types.Material) -> None:
    if not ob.material_slots:
        ob.data.materials.append(mat)
    for slot in ob.material_slots:
        slot.material = mat


def _bake_image(name: str, res: int) -> bpy.types.Image:
    img = bpy.data.images.get(name)
    if img is not None:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, res, res, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    img.alpha_mode = "CHANNEL_PACKED"
    img.pixels.foreach_set(np.full(res * res * 4, -1.0, np.float32))
    return img


def _bake(high, low, image, socket, selected: bool, e: float, uv_name: str) -> np.ndarray:
    target = high if selected else low
    _assign(target, _emit_material(f"_rpom_bake_{socket}", socket, None))
    _assign(low, _emit_material(f"_rpom_bake_low_{socket}", socket, image))
    for o in bpy.context.scene.objects:
        o.select_set(False)
    low.select_set(True)
    if selected:
        high.select_set(True)
    bpy.context.view_layer.objects.active = low
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 0
    bpy.ops.object.bake(type="EMIT", use_selected_to_active=selected, cage_extrusion=e if selected else 0.0,
                        max_ray_distance=2 * e if selected else 0.0, margin=0, use_clear=False, uv_layer=uv_name,
                        target="IMAGE_TEXTURES")
    return image_io.read_pixels(image).astype(np.float64)


def bake_cycles(high, low, s: BakeSettings, e: float, uv_name: str) -> dict:
    res = s.resolution
    img = _bake_image("_rpom_bake", res)
    try:
        ph = _bake(high, low, img, "Position", True, e, uv_name)
        hit = ph[..., 3] > 0
        img.pixels.foreach_set(np.full(res * res * 4, -1.0, np.float32))
        pl = _bake(high, low, img, "Position", False, e, uv_name)
        img.pixels.foreach_set(np.full(res * res * 4, -1.0, np.float32))
        nl = _bake(high, low, img, "Normal", False, e, uv_name)
    finally:
        bpy.data.images.remove(img)
        for name in ("_rpom_bake_Position", "_rpom_bake_low_Position", "_rpom_bake_Normal", "_rpom_bake_low_Normal"):
            m = bpy.data.materials.get(name)
            if m is not None:
                bpy.data.materials.remove(m)
    inside = pl[..., 3] > 0
    n = nl[..., :3]
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-30)
    return {"p_high": ph[..., :3], "hit": hit & inside, "p_low": pl[..., :3], "n_low": n, "inside": inside}


def low_texels(low: bpy.types.Object, res: int, uv_name: str) -> dict:
    me = low.data
    arr = common.mesh_arrays(me, uv_name)
    tri_uv = arr.loop_uvs[arr.tri_loops].astype(np.float64)
    tri_id, bary = bm.raster_barycentric(tri_uv, res, res)
    mw = np.array(low.matrix_world)
    pos = arr.positions.astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
    nrm = arr.loop_normals.astype(np.float64) @ np.linalg.inv(mw[:3, :3])
    p_corner = pos[arr.loop_verts[arr.tri_loops]]
    n_corner = nrm[arr.tri_loops]
    p_l = bm.interpolate(tri_id, bary, p_corner)
    n_l = bm.interpolate(tri_id, bary, n_corner)
    n_l /= np.maximum(np.linalg.norm(n_l, axis=-1, keepdims=True), 1e-30)
    return {"p_low": p_l, "n_low": n_l, "inside": tri_id >= 0}


def bvh_cast(high: bpy.types.Object, p_low: np.ndarray, n_low: np.ndarray, e: float,
             max_hits: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bvh = BVHTree.FromObject(high, _depsgraph())
    inv = high.matrix_world.inverted()
    inv3 = inv.to_3x3()
    fwd = high.matrix_world
    n = len(p_low)
    first = np.full((n, 3), np.nan)
    count = np.zeros(n, np.int32)
    for i in range(n):
        o = inv @ Vector(p_low[i] + n_low[i] * e)
        d = (inv3 @ Vector(-n_low[i])).normalized()
        dist_left = 2 * e * (inv3 @ Vector(n_low[i])).length
        eps = max(1e-6, 8.0 * float(np.spacing(np.float32(max(abs(o.x), abs(o.y), abs(o.z))))))
        hits = 0
        while dist_left > 0 and hits < max_hits:
            loc, _nor, _idx, dist = bvh.ray_cast(o, d, dist_left)
            if loc is None:
                break
            if hits == 0:
                first[i] = np.array(fwd @ loc)
            hits += 1
            step = dist + eps
            o = o + d * step
            dist_left -= step
        count[i] = hits
    return first, count, ~np.isnan(first[:, 0])


def output_dir(low: bpy.types.Object) -> str:
    base = bpy.path.abspath("//rpom_bake") if bpy.data.filepath else os.path.join(
        bpy.app.tempdir or ".", "rpom_bake")
    safe = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in low.name)
    path = os.path.join(base, safe)
    os.makedirs(path, exist_ok=True)
    return path


def run(high: bpy.types.Object, low: bpy.types.Object, s: BakeSettings, progress=None) -> BakeResult:
    tick = progress or (lambda _f, _msg="": None)
    t0 = time.perf_counter()
    blocking, warnings = checks(high, low, s)
    if blocking:
        raise ValueError("; ".join(blocking))
    uv_name = common.resolve_uv_name(low.data, s.uv_map)
    with _LocalFrame(high, low) as frame:
        tick(0.05, tr("UV check"))
        uvinfo, labels, _ = uv_report(low, s)
        if uvinfo["overlap_texels"]:
            warnings.append(tr("low UVs with {} overlapping texels", uvinfo['overlap_texels']))
        tick(0.1, tr("alignment"))
        median, signed = alignment(high, low)
        diag = (_bbox_world(low)[1] - _bbox_world(low)[0]).length
        if median > 0.02 * diag:
            warnings.append(tr("high and low may not be aligned: median distance {:.4g} m", median))
        e = s.extrusion or bm.auto_extrusion(signed, s.margin)
        tick(0.2, tr("bake"))
        timings = {}
        t = time.perf_counter()
        if s.method == "CYCLES":
            data = bake_cycles(high, low, s, e, uv_name)
            p_high, hit = data["p_high"], data["hit"]
        else:
            data = low_texels(low, s.resolution, uv_name)
            m = data["inside"]
            first, _count, ok = bvh_cast(high, data["p_low"][m], data["n_low"][m], e)
            p_high = np.zeros_like(data["p_low"])
            p_high[m] = np.nan_to_num(first)
            hit = np.zeros(m.shape, bool)
            hit[m] = ok
        timings["bake_s"] = time.perf_counter() - t
        inside = data["inside"]
        p_low, n_low = data["p_low"], data["n_low"]
        d, lateral = bm.signed_distance(p_high, p_low, n_low)
        texel_m = uvinfo["mm_per_texel_p50"] / 1000 or 1e-3
        tol = s.lateral_tolerance or texel_m
        tick(0.6, tr("undercuts"))
        t = time.perf_counter()
        multi, validation = _undercuts_and_validation(high, s, e, inside, hit, p_low, n_low, p_high, texel_m)
        timings["undercut_s"] = time.perf_counter() - t
        flags = bm.flags_from(hit, inside, lateral, tol, multi)
        good = inside & hit & ((flags & (bm.FLAG_LATERAL | bm.FLAG_MULTI)) == 0)
        tick(0.8, tr("filling"))
        d_filled = bm.push_pull_fill(np.where(good, d, 0.0), good)
        d_filled = np.where(inside, d_filled, 0.0)
        d_final = bm.dilate(d_filled, inside, iterations=16)
        albedo = None
        if s.bake_albedo:
            tick(0.9, tr("albedo"))
            albedo = _bake_albedo(high, low, frame, s, e, uv_name)
    out = output_dir(low)
    disp_path = os.path.join(out, "rpom_disp.exr")
    flags_path = os.path.join(out, "rpom_bakeflags.exr")
    image_io.write_exr(disp_path, np.repeat(d_final[..., None], 4, 2).astype(np.float32), half=False)
    image_io.write_exr(flags_path, np.repeat(flags[..., None].astype(np.float32), 4, 2), half=True)
    lo, hi = float(d_final[inside].min(initial=0.0)), float(d_final[inside].max(initial=0.0))
    meta = {
        "high": {"name": high.name, "triangles": _tri_count(high)},
        "low": {"name": low.name, "triangles": _tri_count(low)},
        "extrusion_m": e, "resolution": s.resolution, "depth_m": hi - lo,
        "ref_plane": (-lo / (hi - lo)) if hi > lo else 0.0, "method": s.method,
        "flags_pct": bm.flag_percentages(flags, inside), "uv": uvinfo, "alignment_median_m": median,
        "validation": validation, "timings_s": {k: round(v, 3) for k, v in timings.items()},
        "half_step_mm": (hi - lo) * 2.0 ** -11 * 1000, "addon_version": VERSION,
        "date": time.strftime("%Y-%m-%d %H:%M"), "settings": asdict(s), "local_center": list(frame_center(high, low)),
        "total_s": round(time.perf_counter() - t0, 3),
    }
    with open(os.path.join(out, "bake.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    if albedo is not None:
        path = os.path.join(out, "rpom_albedo.png")
        albedo.filepath_raw = path
        albedo.file_format = "PNG"
        albedo.save()
        meta["albedo"] = path
    tick(1.0, tr("done"))
    return BakeResult(out, disp_path, flags_path, meta, warnings)


def frame_center(high, low) -> Vector:
    lo, hi = _bbox_world(low)
    return (lo + hi) / 2


def _tri_count(ob: bpy.types.Object) -> int:
    ob.data.calc_loop_triangles()
    return len(ob.data.loop_triangles)


def _block_samples(idx: np.ndarray, shape: tuple[int, int], size: int):
    grid = (-(-shape[0] // size), -(-shape[1] // size))
    y, x = np.divmod(idx, shape[1])
    block = (y // size) * grid[1] + x // size
    c = (size - 1) / 2
    dist = (y % size - c) ** 2 + (x % size - c) ** 2
    order = np.lexsort((dist, block))
    blocks, first = np.unique(block[order], return_index=True)
    return idx[order[first]], blocks, block, grid


def _spread_to_neighbours(flag: np.ndarray) -> np.ndarray:
    p = np.pad(flag, 1)
    out = np.zeros_like(flag)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + flag.shape[0], dx:dx + flag.shape[1]]
    return out


def _undercuts_and_validation(high, s, e, inside, hit, p_low, n_low, p_high, texel_m):
    idx = np.flatnonzero(inside)
    multi = np.zeros(inside.shape, bool)
    full = s.undercut_mode == "FULL" or (s.undercut_mode == "AUTO" and idx.size <= 300_000)
    if full:
        _first, count, _ok = bvh_cast(high, p_low.reshape(-1, 3)[idx], n_low.reshape(-1, 3)[idx], e)
        multi.reshape(-1)[idx] = count > 1
    else:
        sample, blocks, block_of, grid = _block_samples(idx, inside.shape, 4)
        _first, count, _ok = bvh_cast(high, p_low.reshape(-1, 3)[sample], n_low.reshape(-1, 3)[sample], e)
        flag = np.zeros(grid[0] * grid[1], bool)
        flag[blocks] = count > 1
        multi.reshape(-1)[idx] = _spread_to_neighbours(flag.reshape(grid)).reshape(-1)[block_of]
    if not full:
        d, _ = bm.signed_distance(p_high, p_low, n_low)
        multi |= bm.slope_undercut_candidates(d, texel_m, inside & hit)
    validation = {"undercut_mode": "completo" if full else "campione a blocchi 4×4 + pendenza"}
    if s.method == "CYCLES" and idx.size:
        rng = np.random.default_rng(0)
        pick = rng.choice(idx, size=max(1, int(idx.size * s.validate_fraction)), replace=False)
        first, _c, ok = bvh_cast(high, p_low.reshape(-1, 3)[pick], n_low.reshape(-1, 3)[pick], e)
        both = ok & hit.reshape(-1)[pick]
        diff = np.linalg.norm(first[both] - p_high.reshape(-1, 3)[pick][both], axis=1)
        validation.update({"texels": int(pick.size), "agree_pct": float(both.mean() * 100) if pick.size else 0.0,
                           "diff_p99_mm": float(np.percentile(diff, 99) * 1000) if diff.size else 0.0})
    return multi, validation


def apply_to_low(low: bpy.types.Object, result: BakeResult, albedo: bpy.types.Image | None = None) -> bpy.types.Material:
    from . import apply as apply_op

    mat = bpy.data.materials.new(f"RPOM_{low.name}")
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    disp = bpy.data.images.load(result.displacement_path, check_existing=True)
    disp.colorspace_settings.name = "Non-Color"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = disp
    tex.name = "Height"
    tex.label = "Displacement (m) from bake"
    tex.location = (-700, -300)
    if albedo is not None:
        col = nt.nodes.new("ShaderNodeTexImage")
        col.image = albedo
        col.location = (-700, 200)
        nt.links.new(col.outputs[0], bsdf.inputs["Base Color"])
    if low.material_slots:
        low.material_slots[0].material = mat
    else:
        low.data.materials.append(mat)
    st = mat.rpom
    st.height_image = disp
    st.source_kind = "METERS"
    st.border_mode = "ATLAS"
    st.frame_mode = "EXACT"
    st.uv_map = ""
    apply_op.apply_sync(mat)
    return mat


class RPOM_OT_bake_apply(bpy.types.Operator):
    """Bake the height in meters from the high (selected) to the low (active) and apply the POM"""

    bl_idname = "rpom.bake_apply"
    bl_label = "Bake and Apply"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        sel = [o for o in context.selected_objects if o.type == "MESH"]
        return len(sel) == 2 and context.active_object in sel

    def execute(self, context):
        low = context.active_object
        high = next(o for o in context.selected_objects if o != low and o.type == "MESH")
        sc = context.scene.rpom
        s = BakeSettings(resolution=sc.bake_resolution, extrusion=sc.bake_extrusion, method=sc.bake_method,
                         uv_map=sc.bake_uv_map, apply_pom=sc.bake_apply_pom)
        try:
            result = run(high, low, s)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        for w in result.warnings:
            self.report({"WARNING"}, w)
        if s.apply_pom:
            apply_to_low(low, result)
        self.report({"INFO"}, tr("bake in {:.1f} s: {}", result.meta["total_s"], result.directory))
        return {"FINISHED"}


CLASSES = (RPOM_OT_bake_apply,)


def _bake_albedo(high, low, frame: _LocalFrame, s: BakeSettings, e: float, uv_name: str) -> bpy.types.Image | None:
    _basis, mats = frame.saved[high.name]
    if not any(mats):
        return None
    for slot, m in zip(high.material_slots, mats):
        slot.material = m
    img = bpy.data.images.new(f"{low.name}_rpom_albedo", s.resolution, s.resolution, alpha=True)
    _assign(low, _emit_material("_rpom_bake_low_albedo", "Position", img))
    for o in bpy.context.scene.objects:
        o.select_set(False)
    high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    try:
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, use_selected_to_active=True,
                            cage_extrusion=e, max_ray_distance=2 * e, margin=16, uv_layer=uv_name,
                            target="IMAGE_TEXTURES")
    finally:
        m = bpy.data.materials.get("_rpom_bake_low_albedo")
        if m is not None:
            bpy.data.materials.remove(m)
    return img
