# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    steps_eevee: int
    steps_cycles: int
    refine: int
    shadow_steps: int


PRESETS: dict[str, Preset] = {
    p.key: p
    for p in (
        Preset("DRAFT", "Draft", 8, 8, 2, 0),
        Preset("MEDIUM", "Medium", 16, 12, 2, 8),
        Preset("HIGH", "High", 32, 20, 3, 16),
        Preset("ULTRA", "Ultra", 64, 32, 4, 24),
    )
}

DEFAULT_PRESET = "MEDIUM"

TARGETS = ("EEVEE", "CYCLES", "BOTH")


def steps_for(preset_key: str, target: str) -> int:
    p = PRESETS[preset_key]
    return p.steps_eevee if target == "EEVEE" else p.steps_cycles
