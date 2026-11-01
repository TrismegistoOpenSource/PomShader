# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import threading

import bpy
from bpy.props import EnumProperty, StringProperty

from ..core import presets
from ..graph import material_patch as mp
from ..utils.i18n import tr
from . import common, light_link, uv_frame

LOG = common.LOG


def _material(context) -> bpy.types.Material | None:
    mat = getattr(context, "material", None)
    if mat is None and context.active_object is not None:
        mat = context.active_object.active_material
    return mat


def _poll(context) -> bool:
    mat = _material(context)
    return mat is not None and mat.library is None and mat.node_tree is not None


def _analysis_for(mat: bpy.types.Material) -> mp.Analysis:
    a = mp.analyze(mat)
    patch = mp.read_patch(mat)
    if patch is not None and patch.settings.get("height_node") and not mat.rpom.height_node:
        mat.rpom.height_node = patch.settings["height_node"]
    return a


def prepare_frames(mat: bpy.types.Material) -> tuple[list[str], float]:
    st = mat.rpom
    modes, scales = [], []
    meshes = common.meshes_with(mat)
    if not meshes:
        raise ValueError(tr("no mesh uses this material"))
    for me in meshes:
        mode, scale = uv_frame.write_frame(me, st.uv_map, st.frame_mode, st.border_mode)
        modes.append(mode)
        scales.append(scale)
    scales.sort()
    return modes, scales[len(scales) // 2]


def settings_for(mat: bpy.types.Material, march: bpy.types.Image, shade: bpy.types.Image, height_node: str,
                 uniform_scale: float, depth: float, ref: float | None, resolution: int) -> mp.Settings:
    st = mat.rpom
    pr = presets.PRESETS[st.preset]
    helper = ""
    if st.shadow and st.light is None:
        st.light = light_link.pick_light(common.objects_with(mat))
    if st.shadow and st.light is not None:
        helper = light_link.ensure_helper(st.light).name
    return mp.Settings(
        march_image=march.name, shade_image=shade.name, uv_map=st.uv_map, border_mode=st.border_mode,
        depth=depth, ref_plane=st.ref_plane if ref is None else ref,
        steps=st.steps_override or presets.steps_for(st.preset, st.target),
        refine=st.refine_override or pr.refine, dither=1.0 if st.dither else 0.0, max_offset=st.max_offset,
        edge_fade=st.edge_fade, resolution=float(resolution), fade_start=st.fade_start, fade_end=st.fade_end,
        c_min=st.grazing_min, uniform_scale=uniform_scale, normal_policy=st.normal_policy, shadow=st.shadow,
        shadow_strength=st.shadow_strength, shadow_steps=pr.shadow_steps or 8,
        shadow_sharpness=st.shadow_sharpness, light_helper=helper, cavity=st.cavity,
        cavity_strength=st.cavity_strength, height_node=height_node,
    )


class _Run:

    def __init__(self, mat: bpy.types.Material) -> None:
        self.mat = mat
        mp.sync_from_node(mat)
        a = _analysis_for(mat)
        self.image, self.height_node = common.resolve_height(mat, a)
        self.modes, self.scale = prepare_frames(mat)
        self.job = common.collect(mat, self.image, self.scale)
        self.cached = self.job.entry.exists()

    def compute(self, progress=None) -> None:
        if not self.cached:
            common.compute(self.job, progress)

    def finish(self) -> list[str]:
        mat, st = self.mat, self.mat.rpom
        if self.cached:
            march, shade = common.load_cached(self.job.entry, self.job.image_name)
            meta = self.job.entry.read_meta()
            depth_data, ref_data, warnings = meta.get("depth", st.depth), meta.get("ref_plane"), meta.get("warnings", [])
            shape = meta.get("shape", [march.size[1], march.size[0]])
        else:
            march, shade = common.finish(self.job)
            depth_data, ref_data, warnings = self.job.depth_out, self.job.ref_out, self.job.warnings
            shape = list(self.job.result.march.shape[:2])
        if ref_data is not None:
            st.depth, st.ref_plane = depth_data, ref_data
        st.maps_depth = st.depth
        st.cache_key = self.job.entry.key
        st.height_node = self.height_node
        s = settings_for(mat, march, shade, self.height_node, self.scale, st.depth, ref_data, max(shape))
        s.frame_mode = self.modes[0] if len(set(self.modes)) == 1 else "AUTO"
        patch = mp.build(mat, s, _hex_settings(mat), common.default_uv(mat))
        all_warnings = list(dict.fromkeys(warnings + patch.warnings + ([self.job.cache_warning] if self.job.cache_warning else [])))
        st.last_warnings = json.dumps(all_warnings, ensure_ascii=False)
        return all_warnings


def _hex_settings(mat: bpy.types.Material) -> dict | None:
    from dataclasses import asdict

    from ..graph import hextile_patch as hp

    st = mat.rpom
    if not st.hex_enabled:
        return None
    return asdict(hp.HexSettings(uv_map=st.uv_map, scale=st.hex_scale, rotation=st.hex_rotation,
                                 seed=float(st.hex_seed), contrast=st.hex_contrast, exponent=st.hex_exponent,
                                 gain=st.hex_gain, weights=st.hex_weights, fast=st.hex_quality == "FAST"))


def _hex_only(mat: bpy.types.Material) -> list[str] | None:
    if not mat.rpom.hex_enabled:
        return None
    patch = mp.build(mat, None, _hex_settings(mat), common.default_uv(mat))
    mat.rpom.last_warnings = json.dumps(patch.warnings, ensure_ascii=False)
    return patch.warnings


def apply_sync(mat: bpy.types.Material) -> list[str]:
    run = _Run(mat)
    run.compute()
    return run.finish()


class RPOM_OT_apply(bpy.types.Operator):
    """Apply (or update) the POM relief on the material"""

    bl_idname = "rpom.apply"
    bl_label = "Apply"
    bl_options = {"REGISTER", "UNDO"}

    choice: StringProperty(name="Height", description="Image Texture holding the height (when ambiguous)")

    @classmethod
    def poll(cls, context):
        return _poll(context)

    def draw(self, context):
        mat = _material(context)
        self.layout.prop_search(self, "choice", mat.node_tree, "nodes", text="Height")

    def _report_done(self, warnings: list[str]) -> None:
        for w in warnings:
            self.report({"WARNING"}, w)
        self.report({"INFO"}, tr("POM Shader applied"))

    def execute(self, context):
        mat = _material(context)
        if self.choice and mat.node_tree.nodes.get(self.choice) is not None:
            mat.rpom.height_node = self.choice
        try:
            self._report_done(apply_sync(mat))
        except common.NeedsChoice as e:
            self.report({"ERROR"}, tr("ambiguous height: choose it in the panel ({})", ", ".join(e.options)))
            return {"CANCELLED"}
        except ValueError as e:
            warnings = _hex_only(mat)
            if warnings is None:
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            self._report_done(warnings)
        return {"FINISHED"}

    def invoke(self, context, event):
        mat = _material(context)
        try:
            self._run = _Run(mat)
        except common.NeedsChoice as e:
            self.choice = e.options[0]
            return context.window_manager.invoke_props_dialog(self, title="Which image is the height?")
        except ValueError as e:
            warnings = _hex_only(mat)
            if warnings is None:
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            self._report_done(warnings)
            return {"FINISHED"}
        if self._run.cached:
            self._report_done(self._run.finish())
            return {"FINISHED"}
        self._progress = 0.0
        self._error = None
        self._thread = threading.Thread(target=self._work, daemon=True)
        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        self._thread.start()
        return {"RUNNING_MODAL"}

    def _work(self) -> None:
        def progress(f):
            if self._run.job.cancelled:
                raise InterruptedError
            self._progress = f

        try:
            self._run.compute(progress)
        except InterruptedError:
            pass
        except Exception as e:
            self._error = e

    def modal(self, context, event):
        wm = context.window_manager
        if event.type == "ESC":
            self._run.job.cancelled = True
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        wm.progress_update(int(self._progress * 100))
        if context.area:
            context.area.header_text_set(tr("POM Shader: maps {:.0f}% (Esc to cancel)", self._progress * 100))
        if self._thread.is_alive():
            return {"RUNNING_MODAL"}
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        if context.area:
            context.area.header_text_set(None)
        if self._run.job.cancelled:
            self.report({"WARNING"}, tr("cancelled"))
            return {"CANCELLED"}
        if self._error is not None:
            self.report({"ERROR"}, tr("map preparation failed: {}", self._error))
            return {"CANCELLED"}
        self._report_done(self._run.finish())
        return {"FINISHED"}


class RPOM_OT_apply_batch(bpy.types.Operator):
    """Apply the POM to every material using the same height, or to the selected objects' materials"""

    bl_idname = "rpom.apply_batch"
    bl_label = "Batch Apply"
    bl_options = {"REGISTER", "UNDO"}

    scope: EnumProperty(name="Scope", items=[
        ("SAME_HEIGHT", "Same Height", "Materials using the same height image"),
        ("SELECTED", "Selected Objects", "All materials of the selected objects"),
    ])

    @classmethod
    def poll(cls, context):
        return _poll(context)

    def execute(self, context):
        src = _material(context)
        targets = []
        if self.scope == "SELECTED":
            for ob in context.selected_objects:
                targets += [s.material for s in ob.material_slots if s.material is not None]
        else:
            img = src.rpom.height_image
            for m in bpy.data.materials:
                if m.node_tree is None or m.library is not None:
                    continue
                if any(getattr(n, "image", None) == img for n in m.node_tree.nodes if img is not None):
                    targets.append(m)
        done = 0
        for m in dict.fromkeys(targets):
            if m != src:
                _copy_settings(src.rpom, m.rpom)
            try:
                apply_sync(m)
                done += 1
            except (ValueError, common.NeedsChoice) as e:
                self.report({"WARNING"}, f"{m.name}: {e}")
        self.report({"INFO"}, tr("applied to {} materials", done))
        return {"FINISHED"}


def _copy_settings(a, b) -> None:
    for prop in a.bl_rna.properties:
        if prop.identifier in {"rna_type", "height_node", "cache_key", "last_warnings", "maps_depth"} or prop.is_readonly:
            continue
        setattr(b, prop.identifier, getattr(a, prop.identifier))


CLASSES = (RPOM_OT_apply, RPOM_OT_apply_batch)
