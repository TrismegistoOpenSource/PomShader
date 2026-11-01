# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy

from ..builder import Builder

NAME = "RPOM_Normal"
DEPENDS: tuple[str, ...] = ()


def build(tree: bpy.types.NodeTree) -> None:
    b = Builder(tree, "nm")
    tu = b.inp("Tu", "VECTOR")
    tv = b.inp("Tv", "VECTOR")
    normal = b.inp("Normal", "VECTOR")
    depth = b.inp("Depth", "FLOAT", 0.0)
    grad = b.inp("Gradient", "VECTOR")

    gu, gv, _ = b.separate(grad)
    tu2 = b.vmadd(normal, b.mul(depth, gu), tu, label="Tu'")
    tv2 = b.vmadd(normal, b.mul(depth, gv), tv, label="Tv'")
    n2 = b.vmath("NORMALIZE", b.vmath("CROSS_PRODUCT", tu2, tv2))
    side = b.math("SIGN", b.dot(b.vmath("CROSS_PRODUCT", tu, tv), normal))
    side = b.mix(b.math("COMPARE", side, 0.0, 0.5), side, 1.0)
    b.out("Normal", "VECTOR", b.scale(n2, side))
    b.layout()
