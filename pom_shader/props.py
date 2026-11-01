# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty,
                       StringProperty)

from .core.presets import DEFAULT_PRESET, PRESETS

BORDER_ITEMS = [
    ("TILING", "Repeating", "The texture repeats: no borders (floors, walls)"),
    ("SINGLE", "One Island", "One UV island in 0..1: the relief stops at its border"),
    ("CLIP", "Cut at UV Edge", "Outside 0..1 the surface becomes transparent"),
    ("ATLAS", "Unwrapped Atlas", "Many unique UV islands (scans, bakes): the relief never leaves its island"),
]
FRAME_ITEMS = [
    ("AUTO", "Automatic", "Picks the right one for the mesh"),
    ("EXACT", "Exact", "Best for distorted UVs and atlases (more memory)"),
    ("COMPACT", "Compact", "For very large meshes (less memory)"),
    ("UNIFORM", "Uniform", "Same UV size everywhere (simple tiling)"),
]
SOURCE_ITEMS = [
    ("AUTO", "Automatic", "Meters if the image holds values outside 0..1, otherwise 0..1"),
    ("UNIT", "Gray 0..1", "Usual height map: you set the depth"),
    ("METERS", "Meters", "Signed EXR in meters (scans): depth comes from the data"),
]
CHANNEL_ITEMS = [(c, c, "") for c in ("AUTO", "R", "G", "B", "A")] + [("LUMINANCE", "Luminance", "")]
NORMAL_ITEMS = [
    ("AUTO", "Automatic", "Your normal map if you have one, otherwise from the height"),
    ("HEIGHT", "From Height", "Light follows the displacement"),
    ("NORMAL_MAP", "Normal Map", "Light follows your normal map only"),
    ("NONE", "Flat", "Light ignores the relief (for checks)"),
]
TARGET_ITEMS = [
    ("EEVEE", "EEVEE", "Tuned for EEVEE"),
    ("CYCLES", "Cycles", "Tuned for Cycles"),
    ("BOTH", "Both", "Works in both"),
]
PRESET_ITEMS = [(k, p.label, "") for k, p in PRESETS.items()]

LIVE_INPUTS = {
    "depth": "Depth", "ref_plane": "Reference Plane", "max_offset": "Max Offset", "edge_fade": "Edge Fade",
    "fade_start": "Fade Start", "fade_end": "Fade End", "grazing_min": "Grazing Min",
    "shadow_strength": "Shadow Strength", "shadow_sharpness": "Shadow Sharpness",
    "cavity_strength": "Cavity Strength",
}


def _group_node(mat: bpy.types.Material):
    if mat is None or mat.node_tree is None:
        return None
    return mat.node_tree.nodes.get("rpom_main")


def _live(prop: str):
    def update(self, context):
        mat = self.id_data
        node = _group_node(mat)
        if node is None:
            return
        value = getattr(self, prop)
        if prop == "shadow_strength" and not self.shadow:
            value = 0.0
        if prop == "cavity_strength" and not self.cavity:
            value = 0.0
        node.inputs[LIVE_INPUTS[prop]].default_value = value
    return update


def _live_dither(self, context):
    node = _group_node(self.id_data)
    if node is not None:
        node.inputs["Dither"].default_value = 1.0 if self.dither else 0.0


def _live_tiling(self, context):
    node = _group_node(self.id_data)
    if node is None or "Tiling Scale" not in node.inputs:
        return
    sx, sy = self.tiling_scale
    ox, oy = self.tiling_offset
    node.inputs["Tiling Scale"].default_value = (sx, sy, 1.0)
    node.inputs["Tiling Offset"].default_value = (ox, oy, 0.0)
    node.inputs["Tiling Rotation"].default_value = self.tiling_rotation


def _poll_light(self, obj):
    return obj.type == "LIGHT"


class RPOM_MaterialSettings(bpy.types.PropertyGroup):
    height_image: PointerProperty(name="Height Map", type=bpy.types.Image,
                                  description="Gray image of the relief (displacement). Empty = found by itself")
    height_node: StringProperty(name="Height Node", description="Image Texture node holding the height")
    height_used: StringProperty(options={"HIDDEN"})
    channel: EnumProperty(name="Height Channel", items=CHANNEL_ITEMS, default="AUTO",
                          description="Which color channel of the height map holds the height")
    source_kind: EnumProperty(name="Height Type", items=SOURCE_ITEMS, default="AUTO")
    uv_map: StringProperty(name="UV Map", description="UV map of the textures. Empty = the active one")
    border_mode: EnumProperty(name="UV Layout", items=BORDER_ITEMS, default="TILING")
    frame_mode: EnumProperty(name="Surface Frame", items=FRAME_ITEMS, default="AUTO",
                             description="How the add-on measures the UVs on the mesh")
    preset: EnumProperty(name="Quality", items=PRESET_ITEMS, default=DEFAULT_PRESET,
                         description="Higher = cleaner relief, slower")
    target: EnumProperty(name="Render Engine", items=TARGET_ITEMS, default="BOTH")
    steps_override: IntProperty(name="Steps", min=0, max=256, default=0,
                                description="Samples along the view ray. 0 = from Quality")
    refine_override: IntProperty(name="Refinement", min=0, max=16, default=0,
                                 description="Extra precision passes. 0 = from Quality")

    tiling_scale: FloatVectorProperty(name="Repeat", size=2, default=(1.0, 1.0), min=1e-4, soft_max=100.0,
                                      update=_live_tiling, description="How many times the textures repeat")
    tiling_offset: FloatVectorProperty(name="Shift", size=2, default=(0.0, 0.0), update=_live_tiling,
                                       description="Moves the textures on the surface")
    tiling_rotation: FloatProperty(name="Rotate", subtype="ANGLE", default=0.0, update=_live_tiling,
                                   description="Turns the textures on the surface")

    depth: FloatProperty(name="Depth", subtype="DISTANCE", unit="LENGTH", min=0.0, soft_max=0.5,
                         default=0.02, precision=4, update=_live("depth"),
                         description="How deep the displacement goes, from black to white of the height map")
    ref_plane: FloatProperty(name="Surface Level", min=0.0, max=1.0, default=1.0, subtype="FACTOR",
                             update=_live("ref_plane"),
                             description="Which gray sits on the real surface: 1 = white (relief goes in), 0 = black")
    max_offset: FloatProperty(name="Max Shift at Side Views (px)", min=0.0, soft_max=1024.0, default=256.0,
                              update=_live("max_offset"),
                              description="Limit to how far the texture moves when seen from the side. "
                                          "Higher = deeper look at low angles")
    edge_fade: FloatProperty(name="Island Edge Fade (px)", min=0.0, soft_max=64.0, default=4.0,
                             update=_live("edge_fade"), description="Atlas only: fades the relief near UV seams")
    fade_start: FloatProperty(name="Fade From", subtype="DISTANCE", unit="LENGTH", min=0.0, default=20.0,
                              update=_live("fade_start"), description="Distance where the relief starts to fade")
    fade_end: FloatProperty(name="Gone At", subtype="DISTANCE", unit="LENGTH", min=0.0, default=60.0,
                            update=_live("fade_end"), description="Distance where the relief is gone (saves time)")
    grazing_min: FloatProperty(name="Side-View Limit", min=0.001, max=1.0, default=0.04,
                               update=_live("grazing_min"),
                               description="Below this viewing angle the effect stops growing. Lower = deeper at "
                                           "low angles")
    dither: BoolProperty(name="Smooth Steps", default=True, update=_live_dither,
                         description="Adds fine noise that hides the steps of the relief")
    normal_policy: EnumProperty(name="Lighting", items=NORMAL_ITEMS, default="AUTO",
                                description="Which shape the light follows")

    shadow: BoolProperty(name="Cast Shadows", default=True,
                         description="The displacement casts shadows from one light (picked by itself at Apply)")
    light: PointerProperty(name="Light", type=bpy.types.Object, poll=_poll_light,
                           description="The light that makes the shadows. Empty = the strongest one")
    shadow_strength: FloatProperty(name="Strength", min=0.0, max=1.0, default=0.8, subtype="FACTOR",
                                   update=_live("shadow_strength"), description="How dark the shadows are")
    shadow_sharpness: FloatProperty(name="Sharpness", min=0.0, soft_max=64.0, default=8.0,
                                    update=_live("shadow_sharpness"), description="Higher = harder shadow edges")
    cavity: BoolProperty(name="Darken Crevices", default=False, description="Darkens the deep parts")
    cavity_strength: FloatProperty(name="Strength", min=0.0, max=1.0, default=0.5, subtype="FACTOR",
                                   update=_live("cavity_strength"), description="How dark the crevices are")

    data_resolution: IntProperty(name="Map Resolution", min=0, max=16384, default=0,
                                 description="Size of the data computed from the height map. 0 = from the preferences")

    hex_enabled: BoolProperty(name="Anti-Tiling", default=False,
                              description="Hides the repetition of the textures (applied by Update)")
    hex_scale: FloatProperty(name="Patch Size", min=0.01, soft_max=8.0, default=1.0,
                             description="Higher = smaller patches")
    hex_rotation: FloatProperty(name="Random Rotation", min=0.0, max=1.0, default=1.0, subtype="FACTOR",
                                description="0 for bricks and planks, 1 for soil, gravel, plaster")
    hex_seed: IntProperty(name="Random Seed", default=0, description="Change it for a different pattern")
    hex_contrast: FloatProperty(name="Contrast", min=0.0, max=1.0, default=0.6, subtype="FACTOR",
                                description="How much brighter parts win in the blend")
    hex_exponent: FloatProperty(name="Sharpness", min=0.1, max=32.0, default=7.0,
                                description="Higher = harder patch edges")
    hex_gain: FloatProperty(name="Balance", min=0.05, max=0.95, default=0.5, subtype="FACTOR")
    hex_weights: EnumProperty(name="Blend By", items=[
        ("LUMINANCE", "Brightness", "Brighter parts win"),
        ("BARY", "Position", "Only the position in the patch"),
    ], default="LUMINANCE")
    hex_quality: EnumProperty(name="With Displacement", items=[
        ("FULL", "Full", "Best look (3 reads per step)"),
        ("FAST", "Fast", "Faster, a bit less precise"),
    ], default="FULL")

    maps_depth: FloatProperty(name="Computed Depth", default=0.0, options={"HIDDEN"})
    cache_key: StringProperty(options={"HIDDEN"})
    last_warnings: StringProperty(options={"HIDDEN"})


class RPOM_SceneSettings(bpy.types.PropertyGroup):
    bake_resolution: IntProperty(name="Bake Resolution", min=64, max=16384, default=2048)
    bake_extrusion: FloatProperty(name="Extrusion", subtype="DISTANCE", unit="LENGTH", min=0.0, default=0.0,
                                  description="0 = automatic (1.2 × p99.9 of the distance + margin)")
    bake_method: EnumProperty(name="Method", items=[
        ("CYCLES", "Cycles (GPU)", "Selected to Active bake on 32-bit float images"),
        ("BVH", "BVH (CPU)", "Rays cast with mathutils: slow, for small maps or without a GPU"),
    ], default="CYCLES")
    bake_uv_map: StringProperty(name="Low UV Map", description="Empty = active render UV map")
    bake_apply_pom: BoolProperty(name="Apply POM", default=True)
    report_samples: IntProperty(name="Samples on High", min=1000, max=20_000_000, default=200_000)
    report_screen: BoolProperty(name="Screen Error (E2)", default=True,
                                description="Cycles renders of the POM against real displacement, 3 views")
    ue_height_ratio: FloatProperty(name="Height Ratio (UE)", min=0.0, default=0.05, precision=4)
    ue_meters_per_uv: FloatProperty(name="Meters per UV Unit", min=0.0, default=1.0, subtype="DISTANCE",
                                    unit="LENGTH")


CLASSES = (RPOM_MaterialSettings, RPOM_SceneSettings)


def register() -> None:
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Material.rpom = PointerProperty(type=RPOM_MaterialSettings)
    bpy.types.Scene.rpom = PointerProperty(type=RPOM_SceneSettings)


def unregister() -> None:
    del bpy.types.Scene.rpom
    del bpy.types.Material.rpom
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
