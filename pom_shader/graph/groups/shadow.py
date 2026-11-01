# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy

from ..builder import Builder
from ..signature import BIG_UV, BORDER_ATLAS, SOURCE_OUTPUTS
from .setup import _uv_velocity

NAME = "RPOM_Shadow"
DEPENDS: tuple[str, ...] = ()


def build(tree: bpy.types.NodeTree) -> None:
    b = Builder(tree, "sh")
    source = b.inp("Source", "CLOSURE")
    uv0 = b.inp("UV", "VECTOR")
    uv_hit = b.inp("UV Hit", "VECTOR")
    h_hit = b.inp("Height Hit", "FLOAT", 1.0)
    tu = b.inp("Tu", "VECTOR")
    tv = b.inp("Tv", "VECTOR")
    normal = b.inp("Normal", "VECTOR")
    depth = b.inp("Depth", "FLOAT", 0.0)
    light = b.inp("Light Vector", "VECTOR", description="Toward the light, not normalized")
    steps = b.inp("Steps", "INT", 8, 1, 128)
    sharp = b.inp("Sharpness", "FLOAT", 8.0, 0.0, 1e4)
    strength = b.inp("Strength", "FLOAT", 0.0, 0.0, 1.0)
    mode = b.inp("Border Mode", "INT", 0, 0, 3)
    sdf0 = b.inp("SDF", "FLOAT", BIG_UV)

    with b.frame("Ray toward the light"):
        lv = b.vmath("NORMALIZE", light)
        ln = b.dot(lv, normal)
        s_l = b.scale(_uv_velocity(b, tu, tv, lv), b.div(depth, b.math("MAXIMUM", ln, 0.05)))
        rise = b.sub(1.0, h_hit)
        s_eff = b.scale(s_l, rise)
        is_atlas = b.math("COMPARE", mode, float(BORDER_ATLAS), 0.1)
        room = b.math("MAXIMUM", b.sub(sdf0, b.vmath("DISTANCE", uv_hit, uv0)), 0.0)
        room = b.mix(is_atlas, BIG_UV, room)
        fac = b.math("MINIMUM", 1.0, b.div(room, b.math("MAXIMUM", b.vmath("LENGTH", s_eff), 1e-12)))
        s_eff = b.scale(s_eff, fac)
        inv_n = b.div(1.0, steps)

    with b.frame("Shadow march"):
        zone = b.repeat(steps, [("occl", "FLOAT", 0.0)], label="Shadow")
        t = b.madd(zone.iteration, inv_n, inv_n, label="t = (j+1)/Ns")
        h_ray = b.madd(t, rise, h_hit)
        h = b.evaluate(source, [("uv", "VECTOR", b.vmadd(t, s_eff, uv_hit))], SOURCE_OUTPUTS)["height"]
        weight = b.mul(b.sub(1.0, t), sharp)
        occ = b.mul(b.sub(h, h_ray), weight, clamp=True)
        b.close(zone, {"occl": b.math("MAXIMUM", zone.state["occl"], occ)})

    with b.frame("Apply"):
        shade = b.sub(1.0, zone.result["occl"])
        amount = b.mul(strength, b.clamp(b.mul(ln, 10.0)))
        b.out("Shadow", "FLOAT", b.mix(amount, 1.0, shade))
    b.layout()
