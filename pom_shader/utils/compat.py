# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy

VERSION = bpy.app.version
MIN_VERSION = (5, 1, 0)


def set_multilayer_exr(settings: bpy.types.ImageFormatSettings, depth: str = "32") -> None:
    if hasattr(settings, "media_type"):
        settings.media_type = "MULTI_LAYER_IMAGE"
        settings.file_format = "OPEN_EXR_MULTILAYER"
    else:
        settings.file_format = "OPEN_EXR_MULTILAYER"
    settings.color_depth = depth


def active_render_uv(mesh: bpy.types.Mesh) -> str:
    for uv in mesh.uv_layers:
        if uv.active_render:
            return uv.name
    return mesh.uv_layers[0].name if len(mesh.uv_layers) else ""


