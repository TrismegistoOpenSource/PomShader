# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import math

import bpy

from ..builder import Builder
from ..signature import BIG_UV, COLOR_OUTPUTS, SOURCE_OUTPUTS

TILE = "RPOM_HexTile"
SAMPLE = "RPOM_HexSample"
NORMAL = "RPOM_HexNormal"
HEIGHT = "RPOM_HexHeight"
NAMES = (TILE, SAMPLE, NORMAL, HEIGHT)
SQRT3 = math.sqrt(3.0)
LUMA = (0.299, 0.587, 0.114)


def _vec_rotate(b: Builder, v, center, angle):
    n = b.node("ShaderNodeVectorRotate", rotation_type="Z_AXIS")
    b.link(v, n.inputs["Vector"])
    b.link(center, n.inputs["Center"])
    b.link(angle, n.inputs["Angle"])
    return n.outputs[0]


def build_tile(tree: bpy.types.NodeTree) -> None:
    b = Builder(tree, "hx")
    grid_uv = b.inp("Grid UV", "VECTOR")
    eval_uv = b.inp("Eval UV", "VECTOR")
    scale = b.inp("Scale", "FLOAT", 1.0, 1e-4, 1e4, description="Hexagon density (1 = one hexagon ~ one tile)")
    rotation = b.inp("Rotation", "FLOAT", 1.0, 0.0, 1.0, subtype="Factor",
                     description="0 for directional textures, 1 for organic textures")
    seed = b.inp("Seed", "FLOAT", 0.0)

    with b.frame("Triangle grid (skewed coordinates)"):
        k = b.mul(scale, 2.0 * SQRT3)
        x, y, _ = b.separate(b.scale(grid_uv, k))
        sk = b.combine(b.madd(y, -1.0 / SQRT3, x), b.mul(y, 2.0 / SQRT3))
        base = b.vmath("FLOOR", sk)
        fx, fy, _ = b.separate(b.vmath("FRACTION", sk))
        z = b.sub(b.sub(1.0, fx), fy)
        s = b.math("LESS_THAN", z, 0.0)
        corners = ((b.vmath("ADD", base, b.combine(s, s)), b.math("ABSOLUTE", z)),
                   (b.vmath("ADD", base, (1.0, 0.0, 0.0)), b.madd(s, z, fx)),
                   (b.vmath("ADD", base, (0.0, 1.0, 0.0)), b.madd(s, z, fy)))

    with b.frame("Tap by vertex color: (i − j) mod 3"):
        bx, by, _ = b.separate(base)
        t = b.sub(bx, by)
        col = b.madd(b.math("FLOOR", b.div(b.add(t, 0.5), 3.0)), -3.0, t)
        m0 = b.math("LESS_THAN", col, 0.5)
        m2 = b.math("GREATER_THAN", col, 1.5)
        m1 = b.sub(b.sub(1.0, m0), m2)
        verts, bary = [], []
        for i in range(3):
            (v0, w0), (v1, w1), (v2, w2) = corners[i], corners[(i + 2) % 3], corners[(i + 1) % 3]
            verts.append(b.vmadd(v0, m0, b.vmadd(v1, m1, b.scale(v2, m2))))
            bary.append(b.madd(w0, m0, b.madd(w1, m1, b.mul(w2, m2))))

    taps, angles = [], []
    with b.frame("Tap: R·(st − c) + c + o"):
        inv_k = b.div(1.0, k)
        for v in verts:
            vx, vy, _ = b.separate(v)
            center = b.scale(b.combine(b.madd(vy, 0.5, vx), b.mul(vy, SQRT3 / 2.0)), inv_k)
            noise = b.node("ShaderNodeTexWhiteNoise", noise_dimensions="4D")
            b.link(v, noise.inputs["Vector"])
            b.link(seed, noise.inputs["W"])
            r, g, bl = b.separate_color(noise.outputs["Color"])
            ang = b.mul(b.madd(bl, 2.0, -1.0), b.mul(rotation, math.pi))
            taps.append(b.vmath("ADD", _vec_rotate(b, eval_uv, center, ang), b.combine(r, g)))
            angles.append(ang)
    for i, t in enumerate(taps):
        b.out(f"Tap {i + 1}", "VECTOR", t)
    b.out("Bary", "VECTOR", b.combine(*bary))
    b.out("Angles", "VECTOR", b.combine(*angles))
    b.layout()


def _weights(b: Builder, bary, drive, contrast, exponent, gain):
    bx, by, bz = b.separate(bary)
    dx, dy, dz = b.separate(drive)
    raw = []
    for w, d in ((bx, dx), (by, dy), (bz, dz)):
        dw = b.madd(contrast, b.sub(d, 1.0), 1.0)
        raw.append(b.mul(dw, b.math("POWER", b.math("MAXIMUM", w, 0.0), exponent)))
    vec = b.combine(*raw)
    vec = b.scale(vec, b.div(1.0, b.math("MAXIMUM", b.dot(vec, (1.0, 1.0, 1.0)), 1e-20)))
    kk = b.div(b.math("LOGARITHM", b.math("MAXIMUM", b.sub(1.0, gain), 1e-6), math.e), math.log(0.5))
    comps = []
    for c in b.separate(vec):
        lo = b.mul(0.5, b.math("POWER", b.mul(c, 2.0), kk))
        hi = b.sub(1.0, b.mul(0.5, b.math("POWER", b.math("MAXIMUM", b.madd(c, -2.0, 2.0), 0.0), kk)))
        comps.append(b.mix(b.math("GREATER_THAN", c, 0.5 - 1e-7), lo, hi))
    gained = b.combine(*comps)
    gained = b.scale(gained, b.div(1.0, b.math("MAXIMUM", b.dot(gained, (1.0, 1.0, 1.0)), 1e-20)))
    plain = b.math("LESS_THAN", b.math("ABSOLUTE", b.sub(gain, 0.5)), 1e-6)
    return b.mix(plain, gained, vec, "VECTOR")


def _blend3(b: Builder, w, a, c, d, kind: str = "VECTOR"):
    wx, wy, wz = b.separate(w)
    if kind == "FLOAT":
        return b.madd(d, wz, b.madd(c, wy, b.mul(a, wx)))
    return b.vmadd(d, wz, b.vmadd(c, wy, b.scale(a, wx)))


def build_sample(tree: bpy.types.NodeTree) -> None:
    b = Builder(tree, "hs")
    source = b.inp("Source", "CLOSURE", description="Color closure: uv → color, alpha")
    taps = [b.inp(f"Tap {i}", "VECTOR") for i in (1, 2, 3)]
    bary = b.inp("Bary", "VECTOR")
    w_in = b.inp("Weights In", "VECTOR")
    use_in = b.inp("Use Weights In", "FLOAT", 0.0, 0.0, 1.0)
    contrast = b.inp("Contrast", "FLOAT", 0.6, 0.0, 1.0, subtype="Factor")
    exponent = b.inp("Exponent", "FLOAT", 7.0, 0.0, 64.0)
    gain = b.inp("Gain", "FLOAT", 0.5, 0.01, 0.99, subtype="Factor")
    ev = [b.evaluate(source, [("uv", "VECTOR", t)], COLOR_OUTPUTS) for t in taps]
    lum = b.combine(*(b.dot(e["color"], LUMA) for e in ev))
    w = b.mix(use_in, _weights(b, bary, lum, contrast, exponent, gain), w_in, "VECTOR")
    b.out("Color", "COLOR", _blend3(b, w, *(e["color"] for e in ev)))
    b.out("Alpha", "FLOAT", _blend3(b, w, *(e["alpha"] for e in ev), kind="FLOAT"))
    b.out("Weights", "VECTOR", w)
    b.layout()


def build_normal(tree: bpy.types.NodeTree) -> None:
    b = Builder(tree, "hn")
    source = b.inp("Source", "CLOSURE", description="Normal map closure: uv → color, alpha")
    taps = [b.inp(f"Tap {i}", "VECTOR") for i in (1, 2, 3)]
    angles = b.inp("Angles", "VECTOR")
    w = b.inp("Weights", "VECTOR")
    ang = b.separate(angles)
    slopes = []
    for t, a in zip(taps, ang):
        c = b.evaluate(source, [("uv", "VECTOR", t)], COLOR_OUTPUTS)["color"]
        n = b.vmadd(c, 2.0, -1.0)
        n = _vec_rotate(b, n, (0.0, 0.0, 0.0), b.mul(a, -1.0))
        nz = b.math("MAXIMUM", b.separate(n)[2], 1e-4)
        slopes.append(b.scale(n, b.div(1.0, nz)))
    blended = _blend3(b, w, *slopes)
    sx, sy, _ = b.separate(blended)
    n = b.vmath("NORMALIZE", b.combine(sx, sy, 1.0))
    b.out("Color", "COLOR", b.vmadd(n, 0.5, 0.5))
    b.layout()


def build_height(tree: bpy.types.NodeTree) -> None:
    b = Builder(tree, "hh")
    source = b.inp("Source", "CLOSURE", description="Plain height source, read in the taps")
    taps = [b.inp(f"Tap {i}", "VECTOR") for i in (1, 2, 3)]
    angles = b.inp("Angles", "VECTOR")
    w = b.inp("Weights", "VECTOR", description="Fixed weights, computed at uv0")
    fast = b.inp("Fast", "FLOAT", 0.0, 0.0, 1.0, description="1 = height from the dominant tap only")
    ev = [b.evaluate(source, [("uv", "VECTOR", t)], SOURCE_OUTPUTS) for t in taps]
    wx, wy, wz = b.separate(w)
    top_x = b.mul(b.math("GREATER_THAN", wx, wy), b.math("GREATER_THAN", wx, wz))
    top_y = b.mul(b.sub(1.0, top_x), b.math("GREATER_THAN", wy, wz))
    dominant = b.combine(top_x, top_y, b.sub(b.sub(1.0, top_x), top_y))
    w_h = b.mix(fast, w, dominant, "VECTOR")
    grads = [_vec_rotate(b, e["grad"], (0.0, 0.0, 0.0), b.mul(a, -1.0)) for e, a in zip(ev, b.separate(angles))]
    b.out("height", "FLOAT", _blend3(b, w_h, *(e["height"] for e in ev), kind="FLOAT"))
    b.out("cone", "FLOAT", b.math("MINIMUM", b.math("MINIMUM", ev[0]["cone"], ev[1]["cone"]), ev[2]["cone"]))
    b.out("sdf", "FLOAT", BIG_UV)
    b.out("grad", "VECTOR", _blend3(b, w, *grads))
    b.out("cavity", "FLOAT", _blend3(b, w, *(e["cavity"] for e in ev), kind="FLOAT"))
    b.out("has_maps", "FLOAT", 1.0)
    b.layout()


BUILDERS = {TILE: build_tile, SAMPLE: build_sample, NORMAL: build_normal, HEIGHT: build_height}
