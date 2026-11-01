# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy

from ..builder import Builder
from ..signature import HIT_EPS, SOURCE_OUTPUTS

NAME = "RPOM_MarchLinear"
DEPENDS: tuple[str, ...] = ()


def build(tree: bpy.types.NodeTree) -> None:
    b = Builder(tree, "ml")
    source = b.inp("Source", "CLOSURE")
    uv_bot = b.inp("UV Bottom", "VECTOR")
    s = b.inp("S", "VECTOR")
    steps = b.inp("Steps", "INT", 16, 1, 256)
    xi = b.inp("Xi", "FLOAT", 0.0, 0.0, 1.0)

    with b.frame("Grid k_j = (j - ξ)·Δ"):
        delta = b.div(1.0, b.sub(steps, xi), label="Δ = 1/(N - ξ)")
        neg_delta = b.mul(delta, -1.0)
        q0 = b.madd(xi, delta, 1.0, label="q0 = 1 + ξΔ")
        neg_s = b.scale(s, -1.0)

    with b.frame("Linear search"):
        zone = b.repeat(steps, [("q", "FLOAT", q0), ("active", "FLOAT", 1.0)], label="March")
        q, active = zone.state["q"], zone.state["active"]
        cand = b.madd(active, neg_delta, q)
        uv = b.vmadd(cand, neg_s, uv_bot)
        h = b.evaluate(source, [("uv", "VECTOR", uv)], SOURCE_OUTPUTS)["height"]
        above = b.math("GREATER_THAN", b.sub(cand, h), HIT_EPS)
        active2 = b.mul(active, above)
        b.close(zone, {"q": b.madd(active2, neg_delta, q), "active": active2})

    with b.frame("Interval"):
        q_end = zone.result["q"]
        q_lo = b.math("MINIMUM", q_end, 1.0, label="q_lo (k_lo ≥ 0)")
        q_hi = b.mix(zone.result["active"], b.math("MAXIMUM", b.sub(q_end, delta), 0.0), 0.0, label="q_hi")
    b.out("Q Lo", "FLOAT", q_lo)
    b.out("Q Hi", "FLOAT", q_hi)
    b.layout()
