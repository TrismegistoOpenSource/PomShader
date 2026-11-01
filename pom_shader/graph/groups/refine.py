# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy

from ..builder import Builder
from ..signature import HIT_EPS, SOURCE_OUTPUTS

NAME = "RPOM_Refine"
DEPENDS: tuple[str, ...] = ()


def _height(b: Builder, source, uv):
    return b.evaluate(source, [("uv", "VECTOR", uv)], SOURCE_OUTPUTS)["height"]


def build(tree: bpy.types.NodeTree) -> None:
    b = Builder(tree, "rf")
    source = b.inp("Source", "CLOSURE")
    uv_bot = b.inp("UV Bottom", "VECTOR")
    s = b.inp("S", "VECTOR")
    q_lo = b.inp("Q Lo", "FLOAT", 1.0)
    q_hi = b.inp("Q Hi", "FLOAT", 0.0)
    iters = b.inp("Iterations", "INT", 2, 0, 16)

    with b.frame("Bounds"):
        neg_s = b.scale(s, -1.0)
        d_lo = b.sub(q_lo, _height(b, source, b.vmadd(q_lo, neg_s, uv_bot)))
        d_hi = b.sub(q_hi, _height(b, source, b.vmadd(q_hi, neg_s, uv_bot)))
        lo0 = b.combine(q_lo, d_lo)
        hi0 = b.combine(q_hi, d_hi)

    with b.frame("Bisection"):
        zone = b.repeat(iters, [("lo", "VECTOR", lo0), ("hi", "VECTOR", hi0)], label="Refine")
        lo, hi = zone.state["lo"], zone.state["hi"]
        qa = b.separate(lo)[0]
        qb = b.separate(hi)[0]
        mid = b.mix(0.5, qa, qb)
        d_mid = b.sub(mid, _height(b, source, b.vmadd(mid, neg_s, uv_bot)))
        above = b.math("GREATER_THAN", d_mid, HIT_EPS)
        new = b.combine(mid, d_mid)
        b.close(zone, {"lo": b.mix(above, lo, new, "VECTOR"), "hi": b.mix(above, new, hi, "VECTOR")})

    with b.frame("Final interpolation"):
        qa, da, _ = b.separate(zone.result["lo"])
        qb, db, _ = b.separate(zone.result["hi"])
        den = b.sub(da, db)
        t = b.div(da, b.math("MAXIMUM", den, 1e-12))
        t = b.mix(b.math("GREATER_THAN", den, 1e-12), 0.5, t)
        q_hit = b.madd(b.sub(qb, qa), t, qa, label="q*")
        uv_hit = b.vmadd(q_hit, neg_s, uv_bot, label="uv_hit")
    b.out("Q", "FLOAT", q_hit)
    b.out("UV", "VECTOR", uv_hit)
    b.layout()
