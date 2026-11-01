# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy

from ..builder import Builder
from ..signature import BORDER_CLIP, BORDER_SINGLE, SOURCE_OUTPUTS
from . import frame, march_linear, normal, refine, setup, shadow

NAME = "RPOM"
DEPENDS = (frame.NAME, setup.NAME, march_linear.NAME, refine.NAME, normal.NAME, shadow.NAME)


def _interface(b: Builder) -> dict:
    s = {
        "source": b.inp("Source", "CLOSURE", description="Height source closure of the material"),
        "uv": b.inp("UV", "VECTOR", description="Base UV (UV Map node)"),
        "tangent": b.inp("Tangent", "VECTOR", description="Tangent node of the same UV map"),
    }
    p = b.panel("Depth")
    s["depth"] = b.inp("Depth", "FLOAT", 0.02, 0.0, 1e6, p, "Relief thickness between h=0 and h=1, in meters",
                       subtype="Distance")
    s["ref"] = b.inp("Reference Plane", "FLOAT", 1.0, 0.0, 1.0, p, "Normalized height of the actual polygon",
                     subtype="Factor")
    p = b.panel("Quality")
    s["steps"] = b.inp("Steps", "INT", 16, 1, 256, p, "Linear steps (fixed per material)")
    s["refine"] = b.inp("Refine", "INT", 2, 0, 16, p, "Bisections after the linear search")
    s["dither"] = b.inp("Dither", "FLOAT", 1.0, 0.0, 1.0, p, "Random offset of the steps", subtype="Factor")
    p = b.panel("Borders")
    s["mode"] = b.inp("Border Mode", "INT", 0, 0, 3, p, "0 tiling, 1 single island, 2 clip, 3 atlas")
    s["max_off"] = b.inp("Max Offset", "FLOAT", 64.0, 0.0, 1e6, p, "Maximum shift in texels")
    s["edge"] = b.inp("Edge Fade", "FLOAT", 4.0, 0.0, 1e6, p, "Fade near the borders, in texels")
    s["res"] = b.inp("Resolution", "FLOAT", 1024.0, 1.0, 1e6, p, "Data map resolution (px)")
    p = b.panel("Distance", closed=True)
    s["fade_start"] = b.inp("Fade Start", "FLOAT", 20.0, 0.0, 1e9, p, subtype="Distance")
    s["fade_end"] = b.inp("Fade End", "FLOAT", 60.0, 0.0, 1e9, p, subtype="Distance")
    p = b.panel("Shadow", closed=True)
    s["sh_strength"] = b.inp("Shadow Strength", "FLOAT", 0.0, 0.0, 1.0, p, subtype="Factor")
    s["sh_steps"] = b.inp("Shadow Steps", "INT", 8, 1, 128, p)
    s["sh_sharp"] = b.inp("Shadow Sharpness", "FLOAT", 8.0, 0.0, 1e4, p)
    s["light"] = b.inp("Light Vector", "VECTOR", None, None, None, p, "Texture Coordinate (Object) of the helper, negated")
    s["cav_strength"] = b.inp("Cavity Strength", "FLOAT", 0.0, 0.0, 1.0, p, subtype="Factor")
    p = b.panel("Advanced", closed=True)
    s["c_min"] = b.inp("Grazing Min", "FLOAT", 0.08, 0.001, 1.0, p, "Threshold for grazing incidence")
    s["uniform"] = b.inp("Uniform Scale", "FLOAT", 1.0, 1e-9, 1e9, p, "Meters per UV unit when the attributes are missing")
    s["frame_mode"] = b.inp("Frame Mode", "INT", 0, 0, 3, p, "0 automatic, 1 exact, 2 compact, 3 uniform")
    return s


def _border(b: Builder, s: dict, uv_hit):
    is_single = b.math("COMPARE", s["mode"], float(BORDER_SINGLE), 0.1)
    is_clip = b.math("COMPARE", s["mode"], float(BORDER_CLIP), 0.1)
    clamped = b.vmath("MINIMUM", b.vmath("MAXIMUM", uv_hit, 0.0), 1.0)
    uv = b.mix(is_single, uv_hit, clamped, "VECTOR")
    u, v, _ = b.separate(uv_hit)
    inside = b.math("MINIMUM", b.math("MINIMUM", u, b.sub(1.0, u)), b.math("MINIMUM", v, b.sub(1.0, v)))
    width = b.div(b.math("MAXIMUM", s["edge"], 0.5), s["res"])
    soft = b.clamp(b.madd(inside, b.div(1.0, width), 0.5))
    return uv, b.mix(is_clip, 1.0, soft)


def build(tree: bpy.types.NodeTree, groups: dict[str, bpy.types.NodeTree]) -> None:
    b = Builder(tree, "rp")
    s = _interface(b)
    geo = b.node("ShaderNodeNewGeometry")
    n = geo.outputs["Normal"]

    with b.frame("Tangent frame"):
        fr = b.group(groups[frame.NAME], {"Tangent": s["tangent"], "Normal": n, "Uniform Scale": s["uniform"],
                                         "Mode": s["frame_mode"]})
        tu, tv = fr.outputs["Tu"], fr.outputs["Tv"]

    with b.frame("Ray"):
        at_uv0 = b.evaluate(s["source"], [("uv", "VECTOR", s["uv"])], SOURCE_OUTPUTS, label="in uv0")
        st = b.group(groups[setup.NAME], {
            "UV": s["uv"], "Tu": tu, "Tv": tv, "Normal": n, "Depth": s["depth"], "Reference Plane": s["ref"],
            "Grazing Min": s["c_min"], "Max Offset": s["max_off"], "Resolution": s["res"],
            "Fade Start": s["fade_start"], "Fade End": s["fade_end"], "Border Mode": s["mode"],
            "SDF": at_uv0["sdf"], "Edge Fade": s["edge"], "Dither": s["dither"],
        })

    with b.frame("Search and refinement"):
        ml = b.group(groups[march_linear.NAME], {
            "Source": s["source"], "UV Bottom": st.outputs["UV Bottom"], "S": st.outputs["S"],
            "Steps": s["steps"], "Xi": st.outputs["Xi"],
        })
        rf = b.group(groups[refine.NAME], {
            "Source": s["source"], "UV Bottom": st.outputs["UV Bottom"], "S": st.outputs["S"],
            "Q Lo": ml.outputs["Q Lo"], "Q Hi": ml.outputs["Q Hi"], "Iterations": s["refine"],
        })
        uv_hit, clip = _border(b, s, rf.outputs["UV"])
        q_hit = rf.outputs["Q"]

    with b.frame("Hit point"):
        at_hit = b.evaluate(s["source"], [("uv", "VECTOR", uv_hit)], SOURCE_OUTPUTS, label="in uv_hit")
        d_eff = st.outputs["Depth"]
        nm = b.group(groups[normal.NAME], {"Tu": tu, "Tv": tv, "Normal": n, "Depth": d_eff,
                                           "Gradient": at_hit["grad"]})
        sh = b.group(groups[shadow.NAME], {
            "Source": s["source"], "UV": s["uv"], "UV Hit": uv_hit, "Height Hit": q_hit, "Tu": tu, "Tv": tv,
            "Normal": n, "Depth": d_eff, "Light Vector": s["light"], "Steps": s["sh_steps"],
            "Sharpness": s["sh_sharp"], "Strength": s["sh_strength"], "Border Mode": s["mode"],
            "SDF": at_uv0["sdf"],
        })
        cavity = b.mix(s["cav_strength"], 1.0, at_hit["cavity"])
        depth_m = b.mul(b.sub(s["ref"], q_hit), d_eff, label="depth below the polygon")

    b.out("Parallax UV", "VECTOR", uv_hit)
    b.out("Delta UV", "VECTOR", b.vmath("SUBTRACT", uv_hit, s["uv"]))
    b.out("Height", "FLOAT", q_hit)
    b.out("Normal", "VECTOR", nm.outputs["Normal"])
    b.out("Shadow", "FLOAT", sh.outputs["Shadow"])
    b.out("Cavity", "FLOAT", cavity)
    b.out("Clip Mask", "FLOAT", clip)
    b.out("Depth", "FLOAT", depth_m)
    b.layout()
