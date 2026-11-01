# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass

import bpy

from .builder import Builder
from .signature import SOURCE_INPUTS, SOURCE_OUTPUTS


@dataclass
class SourceSpec:
    march: bpy.types.Image
    shade: bpy.types.Image | None
    extension: str = "REPEAT"
    interpolation: str = "Linear"
    mapping: bpy.types.Node | None = None
    uniform_scale: object = 1.0
    pre: object = None
    origin: str = ""


def _scale(value):
    return value if isinstance(value, bpy.types.NodeSocket) else (float(value) or 1.0)


def _texture(b: Builder, image: bpy.types.Image, spec: SourceSpec, vector, label: str):
    n = b.node("ShaderNodeTexImage", label, image=image, interpolation=spec.interpolation,
               extension=spec.extension, projection="FLAT")
    b.link(vector, n.inputs["Vector"])
    return b.separate_color(n.outputs["Color"])


def _copy_mapping(b: Builder, src: bpy.types.Node, vector):
    n = b.node("ShaderNodeMapping", "Height Mapping", vector_type=src.vector_type)
    for name in ("Location", "Rotation", "Scale"):
        if name in src.inputs:
            s_in = src.inputs[name]
            if s_in.is_linked:
                b.link(s_in.links[0].from_socket, n.inputs[name])
            else:
                n.inputs[name].default_value = s_in.default_value
    b.link(vector, n.inputs["Vector"])
    return n.outputs[0]


def _zone(b: Builder, label: str):
    zin = b.node("NodeClosureInput", label)
    zout = b.node("NodeClosureOutput", label)
    zin.pair_with_output(zout)
    zout.input_items.clear()
    zout.output_items.clear()
    for name, kind in SOURCE_INPUTS:
        zout.input_items.new(kind, name)
    for name, kind in SOURCE_OUTPUTS:
        zout.output_items.new(kind, name)
    return zin, zout


def _hex_weights0(b: Builder, bary, exponent):
    comps = [b.math("POWER", b.math("MAXIMUM", c, 0.0), exponent) for c in b.separate(bary)]
    vec = b.combine(*comps)
    return b.scale(vec, b.div(1.0, b.math("MAXIMUM", b.dot(vec, (1.0, 1.0, 1.0)), 1e-20)))


def build_hex_source(b: Builder, spec: SourceSpec, hexs: dict, base_uv: bpy.types.NodeSocket,
                     hex_groups: dict) -> bpy.types.NodeSocket:
    from .groups import hextile

    plain = build_source(b, SourceSpec(spec.march, spec.shade, spec.extension, spec.interpolation, None, 1.0))
    st0 = spec.pre(base_uv) if spec.pre is not None else base_uv
    if spec.mapping is not None:
        st0 = _copy_mapping(b, spec.mapping, st0)
    tile_in = {"Scale": hexs["scale"], "Rotation": hexs["rotation"], "Seed": hexs["seed"]}
    grid0 = b.group(hex_groups[hextile.TILE], {"Grid UV": st0, "Eval UV": st0, **tile_in}, label="Hex: grid at uv0")
    w0 = _hex_weights0(b, grid0.outputs["Bary"], hexs["exponent"])
    zin, zout = _zone(b, "POM Shader: hex source")
    st = zin.outputs["uv"]
    if spec.pre is not None:
        st = spec.pre(st)
    if spec.mapping is not None:
        st = _copy_mapping(b, spec.mapping, st)
    tile = b.group(hex_groups[hextile.TILE], {"Grid UV": st0, "Eval UV": st, **tile_in}, label="Hex: taps along the ray")
    hh = b.group(hex_groups[hextile.HEIGHT], {
        "Source": plain, **{f"Tap {i}": tile.outputs[f"Tap {i}"] for i in (1, 2, 3)},
        "Angles": tile.outputs["Angles"], "Weights": w0, "Fast": 1.0 if hexs.get("fast") else 0.0,
    }, label="Hex: blended height")
    s = _scale(spec.uniform_scale)
    for name, _ in SOURCE_OUTPUTS:
        val = hh.outputs[name]
        if name == "grad" and s != 1.0:
            val = b.scale(val, s)
        elif name in {"cone", "sdf"} and s != 1.0:
            val = b.div(val, s)
        b.link(val, zout.inputs[name])
    return zout.outputs["Closure"]


def build_source(b: Builder, spec: SourceSpec) -> bpy.types.NodeSocket:
    zin = b.node("NodeClosureInput", "POM Shader: source")
    zout = b.node("NodeClosureOutput", "POM Shader: source")
    zin.pair_with_output(zout)
    zout.input_items.clear()
    zout.output_items.clear()
    for name, kind in SOURCE_INPUTS:
        zout.input_items.new(kind, name)
    for name, kind in SOURCE_OUTPUTS:
        zout.output_items.new(kind, name)
    uv = zin.outputs["uv"]
    if spec.pre is not None:
        uv = spec.pre(uv)
    if spec.mapping is not None:
        uv = _copy_mapping(b, spec.mapping, uv)
    s = _scale(spec.uniform_scale)
    src = f"from {spec.origin}: change it outside, then Update" if spec.origin else "POM Shader"
    h, cone, sdf = _texture(b, spec.march, spec, uv, f"Height data {src}")
    b.link(h, zout.inputs["height"])
    b.link(cone if s == 1.0 else b.div(cone, s), zout.inputs["cone"])
    b.link(sdf if s == 1.0 else b.div(sdf, s), zout.inputs["sdf"])
    if spec.shade is not None:
        gu, gv, cav = _texture(b, spec.shade, spec, uv, f"Slope data {src}")
        grad = b.combine(gu, gv, 0.0)
        b.link(grad if s == 1.0 else b.scale(grad, s), zout.inputs["grad"])
        b.link(cav, zout.inputs["cavity"])
    else:
        zout.inputs["cavity"].default_value = 1.0
    zout.inputs["has_maps"].default_value = 1.0
    return zout.outputs["Closure"]
