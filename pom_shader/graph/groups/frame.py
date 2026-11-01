# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy

from ..builder import Builder
from ..signature import ATTR_DPDU, ATTR_DPDV, ATTR_UVSCALE

NAME = "RPOM_Frame"
DEPENDS: tuple[str, ...] = ()


def _attribute(b: Builder, name: str) -> bpy.types.NodeSocket:
    n = b.node("ShaderNodeAttribute", name, attribute_type="GEOMETRY", attribute_name=name)
    return n.outputs["Vector"]


def _to_world(b: Builder, v, inverse: bool = False) -> bpy.types.NodeSocket:
    n = b.node("ShaderNodeVectorTransform", "", vector_type="VECTOR",
               convert_from="WORLD" if inverse else "OBJECT", convert_to="OBJECT" if inverse else "WORLD")
    b.link(v, n.inputs[0])
    return n.outputs[0]


def _world_scale(b: Builder, axis) -> bpy.types.NodeSocket:
    return b.div(1.0, b.math("MAXIMUM", b.vmath("LENGTH", _to_world(b, axis, inverse=True)), 1e-12))


def build(tree: bpy.types.NodeTree) -> None:
    b = Builder(tree, "fr")
    tangent = b.inp("Tangent", "VECTOR", description="Tangent node (UV Map) of the material")
    normal = b.inp("Normal", "VECTOR")
    uniform = b.inp("Uniform Scale", "FLOAT", 1.0, 1e-9, 1e9,
                    description="Meters per UV unit in object space, when the attributes are missing")
    mode = b.inp("Mode", "INT", 0, 0, 3, description="0 automatic, 1 exact, 2 compact, 3 uniform")

    with b.frame("Exact: dP/du, dP/dv attributes"):
        tu_exact = _to_world(b, _attribute(b, ATTR_DPDU))
        tv_exact = _to_world(b, _attribute(b, ATTR_DPDV))
        exact_ok = b.math("GREATER_THAN", b.vmath("LENGTH", tu_exact), 1e-12, label="exact present")

    with b.frame("Compact / Uniform: Tangent × scale"):
        t = b.vmath("NORMALIZE", tangent)
        bt = b.vmath("NORMALIZE", b.vmath("CROSS_PRODUCT", normal, t))
        k_t = _world_scale(b, t)
        k_b = _world_scale(b, bt)
        su, sv, _ = b.separate(_attribute(b, ATTR_UVSCALE))
        compact_ok = b.math("GREATER_THAN", b.math("ABSOLUTE", su), 1e-12, label="compact present")
        scale_u = b.mix(compact_ok, uniform, su)
        scale_v = b.mix(compact_ok, uniform, sv)
        tu_c = b.scale(t, b.mul(scale_u, k_t))
        tv_c = b.scale(bt, b.mul(scale_v, k_b))

    with b.frame("Automatic choice"):
        tu_auto = b.mix(exact_ok, tu_c, tu_exact, "VECTOR")
        tv_auto = b.mix(exact_ok, tv_c, tv_exact, "VECTOR")

    with b.frame("Mode fixed by the operator (unused branches are pruned)"):
        tu_cmp = b.scale(t, b.mul(su, k_t))
        tv_cmp = b.scale(bt, b.mul(sv, k_b))
        tu_uni = b.scale(t, b.mul(uniform, k_t))
        tv_uni = b.scale(bt, b.mul(uniform, k_b))
        m1 = b.math("COMPARE", mode, 1.0, 0.1)
        m2 = b.math("COMPARE", mode, 2.0, 0.1)
        m3 = b.math("COMPARE", mode, 3.0, 0.1)
        tu = b.mix(m1, b.mix(m2, b.mix(m3, tu_auto, tu_uni, "VECTOR"), tu_cmp, "VECTOR"), tu_exact, "VECTOR")
        tv = b.mix(m1, b.mix(m2, b.mix(m3, tv_auto, tv_uni, "VECTOR"), tv_cmp, "VECTOR"), tv_exact, "VECTOR")

    with b.frame("Tangent plane of the shading normal"):
        tu = b.vmath("SUBTRACT", tu, b.scale(normal, b.dot(normal, tu)))
        tv = b.vmath("SUBTRACT", tv, b.scale(normal, b.dot(normal, tv)))
    b.out("Tu", "VECTOR", tu)
    b.out("Tv", "VECTOR", tv)
    b.layout()
