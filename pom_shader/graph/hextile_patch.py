# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import asdict, dataclass

import bpy

from . import material_patch as mp


@dataclass
class HexSettings:
    uv_map: str = ""
    scale: float = 1.0
    rotation: float = 1.0
    seed: float = 0.0
    contrast: float = 0.6
    exponent: float = 7.0
    gain: float = 0.5
    weights: str = "LUMINANCE"
    fast: bool = False


def is_applied(mat: bpy.types.Material) -> bool:
    p = mp.read_patch(mat) if mat.node_tree else None
    return p is not None and p.hex is not None


def read(mat: bpy.types.Material) -> HexSettings | None:
    p = mp.read_patch(mat)
    return HexSettings(**p.hex) if p is not None and p.hex is not None else None


def _pom(patch: mp.Patch | None, mat: bpy.types.Material) -> mp.Settings | None:
    return mp.refresh(mp.Settings(**patch.settings), mat) if patch is not None and patch.settings else None


def apply(mat: bpy.types.Material, s: HexSettings, default_uv: str = "") -> mp.Patch:
    mp.sync_from_node(mat)
    patch = mp.read_patch(mat)
    return mp.build(mat, _pom(patch, mat), asdict(s), default_uv or (patch.default_uv if patch else ""))


def remove(mat: bpy.types.Material) -> bool:
    patch = mp.read_patch(mat)
    if patch is None or patch.hex is None:
        return False
    mp.sync_from_node(mat)
    pom = _pom(patch, mat)
    if pom is not None:
        mp.build(mat, pom, None, patch.default_uv)
    else:
        mp.teardown(mat)
    return True
