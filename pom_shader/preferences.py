# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy
from bpy.props import EnumProperty, IntProperty, StringProperty

from .core.presets import DEFAULT_PRESET, PRESETS

PACKAGE = __package__


class RPOM_Preferences(bpy.types.AddonPreferences):
    bl_idname = PACKAGE

    cache_dir: StringProperty(name="Cache Folder", subtype="DIR_PATH", default="",
                              description="Empty = //rpom_cache next to the .blend")
    data_resolution: IntProperty(name="Max Data Resolution", min=64, max=16384, default=4096)
    default_preset: EnumProperty(name="Default Preset", items=[(k, p.label, "") for k, p in PRESETS.items()],
                                 default=DEFAULT_PRESET)
    compact_threshold: IntProperty(name="Compact Frame Threshold (million corners)", min=1, max=1000, default=2)

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "cache_dir")
        col.prop(self, "data_resolution")
        col.prop(self, "default_preset")
        col.prop(self, "compact_threshold")


def get(context=None) -> RPOM_Preferences | None:
    ctx = context or bpy.context
    addon = ctx.preferences.addons.get(PACKAGE)
    return addon.preferences if addon else None


def register() -> None:
    bpy.utils.register_class(RPOM_Preferences)


def unregister() -> None:
    bpy.utils.unregister_class(RPOM_Preferences)
