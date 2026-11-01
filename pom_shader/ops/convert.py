# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy

from ..utils.i18n import tr
from .apply import _material


class RPOM_OT_from_ue(bpy.types.Operator):
    """Set the depth from an Unreal Engine Height Ratio"""

    bl_idname = "rpom.from_ue"
    bl_label = "From UE Height Ratio"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _material(context) is not None

    def execute(self, context):
        sc = context.scene.rpom
        mat = _material(context)
        mat.rpom.depth = sc.ue_height_ratio * sc.ue_meters_per_uv
        self.report({"INFO"}, tr("depth {:.4f} m", mat.rpom.depth))
        return {"FINISHED"}


CLASSES = (RPOM_OT_from_ue,)
