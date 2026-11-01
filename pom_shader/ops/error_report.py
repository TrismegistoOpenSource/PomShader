# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import base64
import csv
import json
import os
import string
import time
from dataclasses import dataclass

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..core import bake_math as bm
from ..core import error_metrics as em
from ..core import reference as ref
from ..core import sdf
from ..graph import material_patch as mp
from ..storage import image_io
from ..utils import i18n, log
from ..utils.i18n import tr
from . import common
from .bake_pipeline import VERSION, _bbox_world, _LocalFrame, output_dir

LOG = log.get("error")
TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "report", "template.html")


@dataclass
class ReportSettings:
    samples: int = 200_000
    reverse_max: int = 250_000
    thresholds_mm: tuple = (0.5, 1.0, 2.0)
    heat_limit_mm: float = 0.0
    screen_views: tuple = (0.0, 45.0, 70.0)
    screen_resolution: int = 192
    with_screen: bool = True


def _world_tris(ob: bpy.types.Object, evaluated: bool) -> np.ndarray:
    if evaluated:
        dg = bpy.context.evaluated_depsgraph_get()
        eob = ob.evaluated_get(dg)
        me = eob.to_mesh()
    else:
        eob, me = ob, ob.data
    me.calc_loop_triangles()
    pos = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", pos)
    tris = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tris)
    mw = np.array(ob.matrix_world)
    p = pos.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
    out = p[tris.reshape(-1, 3)]
    if evaluated:
        eob.to_mesh_clear()
    return out


@dataclass
class LowData:
    tri_p: np.ndarray
    tri_n: np.ndarray
    tri_uv: np.ndarray
    bvh: BVHTree


def low_data(low: bpy.types.Object, uv_map: str) -> LowData:
    uv_name = common.resolve_uv_name(low.data, uv_map)
    arr = common.mesh_arrays(low.data, uv_name)
    mw = np.array(low.matrix_world)
    pos = arr.positions.astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
    nrm = arr.loop_normals.astype(np.float64) @ np.linalg.inv(mw[:3, :3])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    tri_p = pos[arr.loop_verts[arr.tri_loops]]
    bvh = BVHTree.FromPolygons(tri_p.reshape(-1, 3).tolist(), np.arange(len(tri_p) * 3).reshape(-1, 3).tolist(),
                               all_triangles=True)
    return LowData(tri_p, nrm[arr.tri_loops], arr.loop_uvs[arr.tri_loops].astype(np.float64), bvh)


def _barycentric(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - v - w, v, w], 1)


@dataclass
class ShaderMap:

    h: np.ndarray
    depth: float
    ref_plane: float
    wrap: bool

    def d(self, uv: np.ndarray) -> np.ndarray:
        return (ref.sample_bilinear(self.h, uv, self.wrap) - self.ref_plane) * self.depth


def shader_map(mat: bpy.types.Material) -> ShaderMap:
    patch = mp.read_patch(mat)
    if patch is None:
        raise ValueError(tr("{}: POM not applied", mat.name))
    node = mat.node_tree.nodes["rpom_main"]
    img = bpy.data.images[patch.settings["march_image"]]
    h = image_io.read_pixels(img)[..., 0].astype(np.float64)
    return ShaderMap(h, float(node.inputs["Depth"].default_value), float(node.inputs["Reference Plane"].default_value),
                     patch.settings["border_mode"] == "TILING")


def _adaptive_modifier(ob: bpy.types.Object, pixel_size: float) -> bpy.types.Modifier:
    mod = ob.modifiers.get("RPOM_E2_adaptive") or ob.modifiers.new("RPOM_E2_adaptive", "SUBSURF")
    mod.subdivision_type = "SIMPLE"
    mod.levels = 0
    mod.render_levels = 1
    mod.use_adaptive_subdivision = True
    mod.adaptive_space = "PIXEL"
    mod.adaptive_pixel_size = pixel_size
    mod.show_viewport = False
    mod.show_render = True
    return mod


def e1_forward(high: bpy.types.Object, ld: LowData, sm: ShaderMap, count: int) -> dict:
    tri_h = _world_tris(high, evaluated=True)
    smp = bm.sample_surface(tri_h, count)
    loc = np.zeros((count, 3))
    idx = np.zeros(count, np.int64)
    ok = np.zeros(count, bool)
    for i, p in enumerate(smp.points):
        hit = ld.bvh.find_nearest(Vector(p))
        if hit[0] is not None:
            loc[i] = hit[0]
            idx[i] = hit[2]
            ok[i] = True
    bary = _barycentric(loc[ok], ld.tri_p[idx[ok]])
    uv = np.einsum("nj,njc->nc", bary, ld.tri_uv[idx[ok]])
    n = np.einsum("nj,njc->nc", bary, ld.tri_n[idx[ok]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    p_rec = loc[ok] + n * sm.d(uv)[:, None]
    err = smp.points[ok] - p_rec
    return {"uv": uv, "signed": np.einsum("nc,nc->n", err, n), "total": np.linalg.norm(err, axis=1)}


def e1_reverse(high: bpy.types.Object, low: bpy.types.Object, sm: ShaderMap, uv_map: str, res: int,
               max_texels: int) -> dict:
    from .bake_pipeline import low_texels

    tex = low_texels(low, res, common.resolve_uv_name(low.data, uv_map))
    inside = tex["inside"]
    c = (np.arange(res) + 0.5) / res
    uu, vv = np.meshgrid(c, c)
    uv = np.stack([uu, vv], -1)[inside]
    idx = np.flatnonzero(inside)
    step = max(1, idx.size // max_texels)
    sel = np.arange(0, idx.size, step)
    p_l = tex["p_low"][inside][sel]
    n_l = tex["n_low"][inside][sel]
    p_rec = p_l + n_l * sm.d(uv[sel])[:, None]
    bvh = BVHTree.FromObject(high, bpy.context.evaluated_depsgraph_get())
    inv = high.matrix_world.inverted()
    fwd = high.matrix_world
    signed = np.full(len(sel), np.nan)
    total = np.full(len(sel), np.nan)
    for i, p in enumerate(p_rec):
        hit = bvh.find_nearest(inv @ Vector(p))
        if hit[0] is not None:
            q = np.array(fwd @ hit[0])
            total[i] = np.linalg.norm(q - p)
            signed[i] = np.dot(q - p, n_l[i])
    heat = np.full((res, res), np.nan)
    heat.reshape(-1)[idx[sel]] = signed
    return {"signed": signed, "total": total, "heat": heat, "uv": uv[sel]}


def _uv_emission_material(name: str, uv_socket_fn):
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    nt.nodes.clear()
    em_node = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em_node.outputs[0], out.inputs["Surface"])
    uv_socket_fn(mat, nt, em_node.inputs["Color"], out)
    return mat


def e2_screen(low: bpy.types.Object, mat: bpy.types.Material, sm: ShaderMap, rs: ReportSettings,
              mm_per_texel: float, out_dir: str) -> dict:
    from ..graph.builder import Builder
    from ..graph.source import SourceSpec, build_source

    patch = mp.read_patch(mat)
    g_src = mat.node_tree.nodes["rpom_main"]
    march = bpy.data.images[patch.settings["march_image"]]
    shade = bpy.data.images.get(patch.settings["shade_image"])
    ext = "REPEAT" if sm.wrap else "EXTEND"
    res_map = max(march.size)
    uv_name = common.resolve_uv_name(low.data, patch.settings["uv_map"]) or ""
    labels = None
    if not sm.wrap and uv_name:
        arr = common.mesh_arrays(low.data, uv_name)
        tri_uv = arr.loop_uvs[arr.tri_loops].astype(np.float64)
        isl = sdf.island_ids(arr.loop_verts[arr.tri_loops], tri_uv)
        labels = sdf.rasterize(tri_uv, isl, march.size[0], march.size[1]).labels

    def pom_fn(m, nt, color_in, _out):
        b = Builder(nt, "e2_")
        uvn = b.node("ShaderNodeUVMap", uv_map=uv_name)
        tan = b.node("ShaderNodeTangent", direction_type="UV_MAP", uv_map=uv_name)
        g = b.node("ShaderNodeGroup")
        g.node_tree = mp.pom_node(mat).node_tree
        for s in g_src.inputs:
            if hasattr(s, "default_value") and not s.is_linked and s.name in g.inputs:
                g.inputs[s.name].default_value = s.default_value
        g.inputs["Dither"].default_value = 0.0
        nt.links.new(build_source(b, SourceSpec(march, shade, ext)), g.inputs["Source"])
        nt.links.new(uvn.outputs["UV"], g.inputs["UV"])
        nt.links.new(tan.outputs["Tangent"], g.inputs["Tangent"])
        nt.links.new(g.outputs["Parallax UV"], color_in)

    def disp_fn(m, nt, color_in, out):
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = uv_name
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image, tex.extension, tex.interpolation = march, ext, "Linear"
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        disp = nt.nodes.new("ShaderNodeDisplacement")
        disp.space = "WORLD"
        disp.inputs["Midlevel"].default_value = sm.ref_plane
        disp.inputs["Scale"].default_value = sm.depth
        nt.links.new(uvn.outputs[0], tex.inputs[0])
        nt.links.new(tex.outputs[0], sep.inputs[0])
        nt.links.new(sep.outputs[0], disp.inputs["Height"])
        nt.links.new(disp.outputs[0], out.inputs["Displacement"])
        nt.links.new(uvn.outputs[0], color_in)
        m.displacement_method = "DISPLACEMENT"

    m_pom = _uv_emission_material("_rpom_e2_pom", pom_fn)
    m_disp = _uv_emission_material("_rpom_e2_disp", disp_fn)
    saved = [s.material for s in low.material_slots]
    results = []
    sc = bpy.context.scene
    cam_saved = sc.camera
    lo, hi = _bbox_world(low)
    center = (lo + hi) / 2
    diag = (hi - lo).length
    normal = _mean_normal(low)
    side = normal.orthogonal().normalized()
    cam_data = bpy.data.cameras.new("_rpom_e2_cam")
    cam = bpy.data.objects.new("_rpom_e2_cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    mod = None
    hidden = {ob.name: ob.hide_render for ob in sc.objects if ob not in (low, cam)}
    for name in hidden:
        sc.objects[name].hide_render = True
    try:
        for ang in rs.screen_views:
            a = np.radians(ang)
            direction = normal * float(np.cos(a)) + side * float(np.sin(a))
            cam.location = center + direction * diag * 1.2
            cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
            cam_data.lens = 50
            imgs = {}
            for key, m in (("pom", m_pom), ("disp", m_disp)):
                for slot in low.material_slots:
                    slot.material = m
                if key == "disp":
                    mod = _adaptive_modifier(low, 0.5)
                imgs[key] = _render_uv(rs.screen_resolution)
                if mod is not None:
                    low.modifiers.remove(mod)
                    mod = None
            pom, disp = imgs["pom"], imgs["disp"]
            both = (pom[..., 3] > 0.999) & (disp[..., 3] > 0.999)
            sil = (pom[..., 3] > 0.5) != (disp[..., 3] > 0.5)
            seam = both & em.seam_crossings(labels, pom[..., :2], disp[..., :2]) if labels is not None \
                else np.zeros_like(both)
            same = both & ~seam
            texels = np.linalg.norm(pom[..., :2] - disp[..., :2], axis=-1) * res_map
            err_mm = np.where(same, texels * mm_per_texel, np.nan)
            path = os.path.join(out_dir, f"rpom_e2_{int(ang)}.png")
            img = em.diverging(err_mm, max(np.nanpercentile(err_mm, 99) if same.any() else 1, 1e-6))
            img = np.where(seam[..., None], np.array([0.45, 0.45, 0.45]), img)
            _save_png(path, np.where(sil[..., None], np.array([0.1, 0.1, 0.1]), img))
            results.append({"angle": ang, "texel": em.stats(texels[same], (0.5, 1.0, 2.0)),
                            "mm": em.stats(err_mm[same], rs.thresholds_mm),
                            "seam_pct": float(seam.sum() / max(both.sum(), 1) * 100),
                            "texel_seam": em.stats(texels[seam], (0.5, 1.0, 2.0)),
                            "silhouette_pct": float(sil.mean() * 100), "image": path})
    finally:
        for name, flag in hidden.items():
            sc.objects[name].hide_render = flag
        for slot, m in zip(low.material_slots, saved):
            slot.material = m
        if mod is not None:
            low.modifiers.remove(mod)
        sc.camera = cam_saved
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(cam_data)
        for m in (m_pom, m_disp):
            bpy.data.materials.remove(m)
    return {"views": results}


def _mean_normal(ob: bpy.types.Object) -> Vector:
    n = Vector((0.0, 0.0, 0.0))
    m3 = ob.matrix_world.to_3x3()
    for p in ob.data.polygons:
        n += (m3 @ p.normal) * p.area
    return n.normalized() if n.length > 0 else Vector((0, 0, 1))


def _render_uv(res: int) -> np.ndarray:
    import tempfile

    import OpenImageIO as oiio

    from ..utils import compat

    sc = bpy.context.scene
    old = (sc.render.engine, sc.render.resolution_x, sc.render.resolution_y, sc.render.film_transparent,
           sc.render.filepath, sc.view_settings.view_transform)
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    sc.cycles.filter_width = 0.01
    sc.cycles.max_bounces = 0
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "Standard"
    compat.set_multilayer_exr(sc.render.image_settings)
    fd, path = tempfile.mkstemp(suffix=".exr")
    os.close(fd)
    sc.render.filepath = path
    try:
        bpy.ops.render.render(write_still=True)
        inp = oiio.ImageInput.open(path)
        spec = inp.spec()
        data = np.asarray(inp.read_image(oiio.FLOAT)).reshape(spec.height, spec.width, -1)[..., :4]
        inp.close()
    finally:
        os.remove(path)
        (sc.render.engine, sc.render.resolution_x, sc.render.resolution_y, sc.render.film_transparent,
         sc.render.filepath, sc.view_settings.view_transform) = old
    return data


def _save_png(path: str, rgb: np.ndarray) -> None:
    import OpenImageIO as oiio

    data = np.ascontiguousarray(np.clip(rgb, 0, 1).astype(np.float32))
    h, w = data.shape[:2]
    out = oiio.ImageOutput.create(path)
    out.open(path, oiio.ImageSpec(w, h, 3, oiio.UINT8))
    out.write_image(data)
    out.close()


def _png_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _stats_table(title: str, st: dict) -> str:
    if not st.get("count"):
        return f"<h3>{title}</h3><p>{tr('no data')}</p>"
    rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" if isinstance(v, float) else
                   f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in st.items())
    return f"<h3>{title}</h3><table>{rows}</table>"


def write_report(out_dir: str, data: dict, heat_png: str) -> str:
    with open(TEMPLATE, encoding="utf-8") as f:
        tpl = string.Template(f.read())
    e1 = data["e1"]
    hist_f = em.svg_histogram(*em.histogram(np.array(e1["forward_signed_mm"]), 40))
    hist_r = em.svg_histogram(*em.histogram(np.array(e1["reverse_signed_mm"]), 40))
    sections = [
        _stats_table(tr("E1 forward, normal component (mm)"), e1["forward_signed"]),
        _stats_table(tr("E1 forward, total distance (mm)"), e1["forward_total"]),
        _stats_table(tr("E1 reverse, total distance (mm)"), e1["reverse_total"]),
        f"<p>{tr('Symmetric distance (Hausdorff-like): <b>{:.4f} mm</b>', e1['hausdorff_mm'])}</p>",
        f"<h3>{tr('E1 forward histogram')}</h3>{hist_f}<h3>{tr('E1 reverse histogram')}</h3>{hist_r}",
    ]
    if e1.get("unrepresentable"):
        sections.append(_stats_table(tr("E1 in non-representable zones (undercuts, no hit)"), e1["unrepresentable"]))
    e2 = data.get("e2", {}).get("views", [])
    e2_html = "".join(
        f"<div class='view'><h3>{tr('View at {:.0f}°', v['angle'])}</h3><img src='data:image/png;base64,{_png_b64(v['image'])}'/>"
        f"{_stats_table(tr('error (texels), same island'), v['texel'])}{_stats_table(tr('error (mm), same island'), v['mm'])}"
        f"<p>{tr('Pixels across a UV seam, counted separately: {:.2f}%', v.get('seam_pct', 0.0))}"
        + (tr(" (apparent distance p99 {:.1f} texels: points close on the model, far apart on the atlas)",
              v["texel_seam"]["p99"]) if v.get("texel_seam", {}).get("count") else "")
        + f"</p><p>{tr('Pixels with a mismatched silhouette: {:.2f}%', v['silhouette_pct'])}</p></div>" for v in e2)
    e3 = "".join(f"<tr><td>{tr(k)}</td><td>{v:.3f}%</td></tr>" for k, v in data.get("e3", {}).items())
    params = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data["params"].items())
    html = tpl.substitute(
        lang="en",
        title=tr("POM Shader: metric error of {}", data["params"]["low"]),
        intro=tr("POM Shader report generated on {}. All measures are in millimeters unless stated otherwise.",
                 data["params"]["date"]),
        scope_label=tr("Scope."), scope=tr(em.SCOPE),
        h_e1=tr("E1: geometric representation error"),
        p_e1=tr("Low + height field against the high-poly, with the map and the precision the shader sees. Forward: "
                "points sampled on the high, nearest point on the low. Reverse: reconstructed texels of the low, "
                "nearest point on the high. Zones the bake marked as not representable (undercuts, no hit) are "
                "reported separately."),
        h_heat=tr("Heatmap on the low (reverse, normal component)"),
        h_e2=tr("E2: on-screen POM error"),
        p_e2=tr("Parallax UVs against true displacement (adaptive subdivision, 0.5 px dicing), converted to mm with "
                "the texel density, on the pixels where both points fall in the same UV island. Mid gray: pixels "
                "across a seam (counted separately); dark gray: pixels with a mismatched silhouette."),
        h_e3=tr("E3: bake coverage"), h_params=tr("Parameters"),
        e1="".join(sections), e2=e2_html or f"<p>{tr('not computed')}</p>",
        e3=f"<table>{e3}</table>" if e3 else f"<p>{tr('no bake.json')}</p>", params=f"<table>{params}</table>",
        heat=_png_b64(heat_png), heat_limit=f"{data['heat_limit_mm']:.3f}",
    )
    path = os.path.join(out_dir, "rpom_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def run(high: bpy.types.Object, low: bpy.types.Object, rs: ReportSettings | None = None) -> dict:
    rs = rs or ReportSettings()
    t0 = time.perf_counter()
    mat = next((s.material for s in low.material_slots if s.material and mp.is_patched(s.material)), None)
    if mat is None:
        raise ValueError(tr("{}: no material with the POM applied", low.name))
    sm = shader_map(mat)
    patch = mp.read_patch(mat)
    out_dir = output_dir(low)
    bake_meta = {}
    meta_path = os.path.join(out_dir, "bake.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            bake_meta = json.load(f)
    flags = None
    flags_path = os.path.join(out_dir, "rpom_bakeflags.exr")
    if os.path.exists(flags_path):
        import OpenImageIO as oiio

        inp = oiio.ImageInput.open(flags_path)
        spec = inp.spec()
        flags = np.asarray(inp.read_image(oiio.FLOAT)).reshape(spec.height, spec.width, -1)[::-1, :, 0].astype(np.uint8)
        inp.close()
    with _LocalFrame(high, low):
        ld = low_data(low, patch.settings["uv_map"])
        fwd = e1_forward(high, ld, sm, rs.samples)
        res = sm.h.shape[0]
        rev = e1_reverse(high, low, sm, patch.settings["uv_map"], res, rs.reverse_max)
        texel_mm = bake_meta.get("uv", {}).get("mm_per_texel_p50") or (patch.settings["uniform_scale"] / res * 1000)
        e2 = e2_screen(low, mat, sm, rs, texel_mm, out_dir) if rs.with_screen else {}
    bad = np.zeros(len(fwd["uv"]), bool)
    if flags is not None:
        fy = np.clip((fwd["uv"][:, 1] * flags.shape[0]).astype(int), 0, flags.shape[0] - 1)
        fx = np.clip((fwd["uv"][:, 0] * flags.shape[1]).astype(int), 0, flags.shape[1] - 1)
        bad = (flags[fy, fx] & (bm.FLAG_MULTI | bm.FLAG_NO_HIT | bm.FLAG_LATERAL)) != 0
    f_signed = fwd["signed"] * 1000
    f_total = fwd["total"] * 1000
    r_signed = rev["signed"] * 1000
    r_total = rev["total"] * 1000
    heat_mm = rev["heat"] * 1000
    limit = rs.heat_limit_mm or float(np.nanpercentile(np.abs(heat_mm), 99) if np.isfinite(heat_mm).any() else 1.0)
    heat_png = os.path.join(out_dir, "rpom_e1_heatmap.png")
    _save_png(heat_png, em.diverging(heat_mm, limit)[::-1])
    data = {
        "e1": {
            "forward_signed": em.stats(f_signed[~bad], rs.thresholds_mm),
            "forward_total": em.stats(f_total[~bad], rs.thresholds_mm),
            "reverse_total": em.stats(r_total, rs.thresholds_mm),
            "reverse_signed": em.stats(r_signed, rs.thresholds_mm),
            "unrepresentable": em.stats(f_total[bad], rs.thresholds_mm) if bad.any() else {},
            "hausdorff_mm": float(max(np.nanmax(f_total[~bad]) if (~bad).any() else 0, np.nanmax(r_total))),
            "forward_signed_mm": f_signed[~bad][:20000].tolist(),
            "reverse_signed_mm": r_signed[np.isfinite(r_signed)][:20000].tolist(),
        },
        "e2": e2,
        "e3": bake_meta.get("flags_pct", {}),
        "heat_limit_mm": limit,
        "params": {
            "high": high.name, "low": low.name, "samples": rs.samples, "map_resolution": res,
            "depth_m": sm.depth, "ref_plane": sm.ref_plane, "steps": patch.settings["steps"],
            "refine": patch.settings["refine"], "addon_version": VERSION, "blender": bpy.app.version_string,
            "date": time.strftime("%Y-%m-%d %H:%M"), "half_step_mm": sm.depth * 2.0 ** -11 * 1000,
            "duration_s": round(time.perf_counter() - t0, 2),
        },
    }
    html = write_report(out_dir, data, heat_png)
    slim = dict(data)
    slim["e1"] = {k: v for k, v in data["e1"].items() if not k.endswith("_mm") or k == "hausdorff_mm"}
    with open(os.path.join(out_dir, "rpom_report.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "rpom_report.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["misura", "statistica", "valore"])
        for key in ("forward_signed", "forward_total", "reverse_total"):
            for k, v in data["e1"][key].items():
                wr.writerow([f"E1 {key}", k, v])
        for v in data["e2"].get("views", []) if data["e2"] else []:
            for k, val in v["mm"].items():
                wr.writerow([f"E2 {v['angle']:.0f}°", k, val])
            wr.writerow([f"E2 {v['angle']:.0f}°", "seam_pct", v.get("seam_pct", 0.0)])
        for k, v in data["e3"].items():
            wr.writerow(["E3", k, v])
    data["html"] = html
    return data


class RPOM_OT_error_report(bpy.types.Operator):
    """Metric error map between the high (selected) and the low with POM (active): HTML, JSON and CSV report"""

    bl_idname = "rpom.error_report"
    bl_label = "Error Report"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        sel = [o for o in context.selected_objects if o.type == "MESH"]
        return len(sel) == 2 and context.active_object in sel

    def execute(self, context):
        low = context.active_object
        high = next(o for o in context.selected_objects if o != low and o.type == "MESH")
        sc = context.scene.rpom
        try:
            data = run(high, low, ReportSettings(samples=sc.report_samples, with_screen=sc.report_screen))
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, tr("report: {}", data["html"]))
        return {"FINISHED"}


CLASSES = (RPOM_OT_error_report,)
