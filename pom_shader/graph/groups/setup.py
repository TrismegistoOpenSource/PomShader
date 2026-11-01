# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy

from ..builder import Builder
from ..signature import BIG_UV, BORDER_ATLAS

NAME = "RPOM_Setup"
DEPENDS: tuple[str, ...] = ()


def _uv_velocity(b: Builder, tu, tv, r):
    a = b.dot(tu, tu)
    g = b.dot(tu, tv)
    c = b.dot(tv, tv)
    pu = b.dot(tu, r)
    pv = b.dot(tv, r)
    det = b.math("MAXIMUM", b.sub(b.mul(a, c), b.mul(g, g)), 1e-20)
    du = b.div(b.sub(b.mul(c, pu), b.mul(g, pv)), det)
    dv = b.div(b.sub(b.mul(a, pv), b.mul(g, pu)), det)
    return b.combine(du, dv, 0.0)


def build(tree: bpy.types.NodeTree) -> None:
    b = Builder(tree, "su")
    uv0 = b.inp("UV", "VECTOR")
    tu = b.inp("Tu", "VECTOR")
    tv = b.inp("Tv", "VECTOR")
    normal = b.inp("Normal", "VECTOR")
    depth = b.inp("Depth", "FLOAT", 0.02, 0.0, 1e6)
    ref_plane = b.inp("Reference Plane", "FLOAT", 1.0, 0.0, 1.0)
    c_min = b.inp("Grazing Min", "FLOAT", 0.08, 0.001, 1.0)
    max_off = b.inp("Max Offset", "FLOAT", 64.0, 0.0, 1e6)
    res = b.inp("Resolution", "FLOAT", 1024.0, 1.0, 1e6)
    fade_start = b.inp("Fade Start", "FLOAT", 20.0, 0.0, 1e9)
    fade_end = b.inp("Fade End", "FLOAT", 60.0, 0.0, 1e9)
    mode = b.inp("Border Mode", "INT", 0, 0, 3)
    sdf0 = b.inp("SDF", "FLOAT", BIG_UV)
    edge_fade = b.inp("Edge Fade", "FLOAT", 4.0, 0.0, 1e6)
    dither = b.inp("Dither", "FLOAT", 1.0, 0.0, 1.0)

    geo = b.node("ShaderNodeNewGeometry")
    with b.frame("UV velocity of the ray"):
        incoming = geo.outputs["Incoming"]
        r = b.scale(incoming, -1.0)
        duv = _uv_velocity(b, tu, tv, r)
        c = b.math("MAXIMUM", b.dot(incoming, normal), c_min, label="c = max(V·N, c_min)")

    with b.frame("Effective depth"):
        cam = b.node("ShaderNodeCameraData")
        dist_fade = b.sub(1.0, b.smoothstep(cam.outputs["View Distance"], fade_start, fade_end))
        front = b.sub(1.0, geo.outputs["Backfacing"])
        is_atlas = b.math("COMPARE", mode, float(BORDER_ATLAS), 0.1, label="atlas")
        edge = b.smoothstep(sdf0, 0.0, b.div(edge_fade, res))
        atlas_fade = b.mix(is_atlas, 1.0, edge)
        d_eff = b.mul(b.mul(depth, dist_fade), b.mul(front, atlas_fade), label="D_eff")

    with b.frame("Shift S and limit"):
        s_raw = b.scale(duv, b.div(d_eff, c))
        length = b.vmath("LENGTH", s_raw)
        limit = b.math("MINIMUM", b.div(max_off, res), b.mix(is_atlas, BIG_UV, sdf0))
        fac = b.math("MINIMUM", 1.0, b.div(limit, b.math("MAXIMUM", length, 1e-12)))
        s = b.scale(s_raw, fac)
        uv_top = b.vmadd(s, b.sub(ref_plane, 1.0), uv0, label="uv_top = uv0 - (1-r)·S")
        uv_bot = b.vmath("ADD", uv_top, s, label="uv_bot")

    with b.frame("Dither"):
        noise = b.node("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
        b.link(b.scale(geo.outputs["Position"], 1000.0), noise.inputs["Vector"])
        xi = b.mul(noise.outputs["Value"], dither, clamp=True)
        xi = b.math("MINIMUM", xi, 0.999)

    b.out("UV Top", "VECTOR", uv_top)
    b.out("UV Bottom", "VECTOR", uv_bot)
    b.out("S", "VECTOR", s)
    b.out("Depth", "FLOAT", d_eff)
    b.out("Xi", "FLOAT", xi)
    b.layout()
