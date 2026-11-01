# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json

import bpy

from ..graph import material_patch as mp
from ..graph import registry

CATEGORY = "POM Shader"


def _material(context):
    sp = context.space_data
    if sp is not None and getattr(sp, "type", "") == "NODE_EDITOR" and getattr(sp, "tree_type", "") == "ShaderNodeTree":
        ob = context.active_object
        return ob.active_material if ob is not None else None
    return getattr(context, "material", None)


def _warnings(mat) -> list[str]:
    try:
        return json.loads(mat.rpom.last_warnings or "[]")
    except ValueError:
        return []


def _editable(context) -> bool:
    mat = _material(context)
    return mat is not None and mat.library is None


def _status(layout, mat) -> None:
    from ..ops import uv_frame
    from ..ops.common import meshes_with

    box = layout.column()
    if registry.outdated():
        row = box.row()
        row.alert = True
        row.label(text="Node groups from an older version", icon="ERROR")
        row.operator("rpom.upgrade", text="Upgrade Groups")
    if mp.is_patched(mat):
        st = mat.rpom
        if st.cavity and st.maps_depth > 0 and abs(st.depth - st.maps_depth) > 0.2 * st.maps_depth:
            box.label(text="Cavity map outdated: press Update", icon="INFO")
    stale = [me.name for me in meshes_with(mat) if uv_frame.frame_info(me) and uv_frame.is_stale(me)]
    if stale:
        row = box.row()
        row.alert = True
        row.label(text="UV frame out of date: " + ", ".join(stale[:3]), icon="ERROR")
        row.operator("rpom.compute_frame", text="", icon="FILE_REFRESH")


class _Base:
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = CATEGORY

    @classmethod
    def poll(cls, context):
        sp = context.space_data
        return sp is not None and getattr(sp, "tree_type", "") == "ShaderNodeTree" and _editable(context)


class RPOM_PT_main(_Base, bpy.types.Panel):
    bl_label = "POM Shader"
    bl_idname = "RPOM_PT_main"

    @classmethod
    def poll(cls, context):
        sp = context.space_data
        return sp is not None and getattr(sp, "tree_type", "") == "ShaderNodeTree" and _material(context) is not None

    def draw(self, context):
        mat = _material(context)
        layout = self.layout
        if mat.library is not None:
            layout.label(text="Material linked from a library: read only", icon="LIBRARY_DATA_DIRECT")
            return
        _status(layout, mat)
        applied = mp.read_patch(mat) is not None
        row = layout.row(align=True)
        row.scale_y = 1.6
        row.operator("rpom.apply", text="Update" if applied else "Apply", icon="FILE_REFRESH" if applied else "MOD_DISPLACE")
        sub = row.row(align=True)
        sub.scale_x = 0.8
        sub.operator("rpom.remove", text="Remove", icon="LOOP_BACK")


class _Sub(_Base):
    bl_parent_id = "RPOM_PT_main"


class RPOM_PT_texture(_Sub, bpy.types.Panel):
    bl_label = "Texture"

    def draw(self, context):
        st = _material(context).rpom
        layout = self.layout
        layout.label(text="Height Map")
        layout.template_ID(st, "height_image", open="image.open")
        col = layout.column(align=True)
        col.prop(st, "tiling_scale")
        col.prop(st, "tiling_offset")
        col.prop(st, "tiling_rotation")
        ob = context.active_object
        if ob is not None and ob.type == "MESH":
            layout.prop_search(st, "uv_map", ob.data, "uv_layers", text="UV Map")


class RPOM_PT_displacement(_Sub, bpy.types.Panel):
    bl_label = "Displacement"

    def draw(self, context):
        st = _material(context).rpom
        col = self.layout.column(align=True)
        col.prop(st, "depth")
        col.prop(st, "ref_plane")


class RPOM_PT_shadows(_Sub, bpy.types.Panel):
    bl_label = "Shadows"

    def draw(self, context):
        st = _material(context).rpom
        layout = self.layout
        layout.prop(st, "shadow")
        col = layout.column(align=True)
        col.active = st.shadow
        col.prop(st, "light")
        col.prop(st, "shadow_strength")
        col.prop(st, "shadow_sharpness")
        layout.prop(st, "cavity")
        col = layout.column(align=True)
        col.active = st.cavity
        col.prop(st, "cavity_strength", text="Crevice Strength")


class RPOM_PT_hextile(_Sub, bpy.types.Panel):
    bl_label = "Anti-Tiling"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.prop(_material(context).rpom, "hex_enabled", text="")

    def draw(self, context):
        st = _material(context).rpom
        layout = self.layout
        layout.active = st.hex_enabled
        col = layout.column(align=True)
        col.prop(st, "hex_scale")
        col.prop(st, "hex_rotation")
        col.prop(st, "hex_seed")
        col = layout.column(align=True)
        col.label(text="Blend")
        col.prop(st, "hex_weights")
        col.prop(st, "hex_contrast")
        col.prop(st, "hex_exponent")
        col.prop(st, "hex_gain")
        col.prop(st, "hex_quality")


class RPOM_PT_quality(_Sub, bpy.types.Panel):
    bl_label = "Quality"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = _material(context).rpom
        layout = self.layout
        col = layout.column(align=True)
        col.prop(st, "preset")
        col.prop(st, "target")
        col = layout.column(align=True)
        col.prop(st, "max_offset")
        col.prop(st, "grazing_min")
        col = layout.column(align=True)
        col.prop(st, "steps_override")
        col.prop(st, "refine_override")
        col.prop(st, "dither")
        col = layout.column(align=True)
        col.prop(st, "fade_start")
        col.prop(st, "fade_end")


class RPOM_PT_advanced(_Sub, bpy.types.Panel):
    bl_label = "Advanced"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = _material(context).rpom
        layout = self.layout
        col = layout.column(align=True)
        col.prop(st, "border_mode")
        col.prop(st, "normal_policy")
        col.prop(st, "frame_mode")
        col = layout.column(align=True)
        col.prop(st, "channel")
        col.prop(st, "source_kind")
        col.prop(st, "data_resolution")
        col.prop(st, "edge_fade")
        sc = context.scene.rpom
        box = layout.box()
        box.label(text="Depth from Unreal Engine")
        row = box.row(align=True)
        row.prop(sc, "ue_height_ratio", text="Height Ratio")
        row.prop(sc, "ue_meters_per_uv", text="m per UV")
        row.operator("rpom.from_ue", text="", icon="IMPORT")
        layout.operator_menu_enum("rpom.apply_batch", "scope", text="Apply to Other Materials")


class RPOM_PT_photogrammetry(_Sub, bpy.types.Panel):
    bl_label = "From Photogrammetry"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        sc = context.scene.rpom
        layout = self.layout
        sel = [o for o in context.selected_objects if o.type == "MESH"]
        low = context.active_object
        high = next((o for o in sel if o != low), None)
        box = layout.box()
        for label, ob in (("High (selected)", high), ("Low (active)", low)):
            row = box.row()
            row.label(text=label)
            row.label(text=ob.name if ob is not None else "—", translate=False)
        col = layout.column(align=True)
        col.prop(sc, "bake_resolution")
        col.prop(sc, "bake_extrusion")
        col.prop(sc, "bake_method")
        if low is not None and low.type == "MESH":
            col.prop_search(sc, "bake_uv_map", low.data, "uv_layers", text="UV Map")
        layout.prop(sc, "bake_apply_pom")
        row = layout.row()
        row.scale_y = 1.3
        row.operator("rpom.bake_apply", icon="RENDER_STILL")


class RPOM_PT_report(_Sub, bpy.types.Panel):
    bl_label = "Error Report"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        sc = context.scene.rpom
        layout = self.layout
        layout.label(text="Select high, then low (active)")
        layout.prop(sc, "report_samples")
        layout.prop(sc, "report_screen")
        layout.operator("rpom.error_report", icon="TEXT")


class RPOM_PT_diagnostics(_Sub, bpy.types.Panel):
    bl_label = "Diagnostics"

    @classmethod
    def poll(cls, context):
        return _Base.poll.__func__(cls, context) and bool(_warnings(_material(context)))

    def draw(self, context):
        mat = _material(context)
        nodes = mat.node_tree.nodes if mat.node_tree else {}
        for w in _warnings(mat):
            row = self.layout.row()
            name = w.split(":", 1)[0]
            row.label(text=w, icon="ERROR")
            if ":" in w and name in nodes:
                row.operator("rpom.select_node", text="", icon="RESTRICT_SELECT_OFF").node = name


CLASSES = [RPOM_PT_main, RPOM_PT_texture, RPOM_PT_displacement, RPOM_PT_shadows, RPOM_PT_hextile, RPOM_PT_quality,
           RPOM_PT_advanced, RPOM_PT_photogrammetry, RPOM_PT_report, RPOM_PT_diagnostics]


def register() -> None:
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
