# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

SOURCE_INPUTS = [("uv", "VECTOR")]
SOURCE_OUTPUTS = [
    ("height", "FLOAT"),
    ("cone", "FLOAT"),
    ("sdf", "FLOAT"),
    ("grad", "VECTOR"),
    ("cavity", "FLOAT"),
    ("has_maps", "FLOAT"),
]

COLOR_INPUTS = [("uv", "VECTOR")]
COLOR_OUTPUTS = [("color", "RGBA"), ("alpha", "FLOAT")]

ATTR_DPDU = "rpom_dpdu"
ATTR_DPDV = "rpom_dpdv"
ATTR_UVSCALE = "rpom_uvscale"

BORDER_TILING = 0
BORDER_SINGLE = 1
BORDER_CLIP = 2
BORDER_ATLAS = 3
BORDER_MODES = {
    "TILING": BORDER_TILING,
    "SINGLE": BORDER_SINGLE,
    "CLIP": BORDER_CLIP,
    "ATLAS": BORDER_ATLAS,
}

HIT_EPS = 1e-6
BIG_UV = 1.0e6
