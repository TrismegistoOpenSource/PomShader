# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json

import bpy

from ..graph import material_patch as mp
from ..utils.i18n import tr
from .apply import _material


class RPOM_OT_remove(bpy.types.Operator):
    """Put the material back exactly as it was"""

    bl_idname = "rpom.remove"
    bl_label = "Remove"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        mat = _material(context)
        return mat is not None and mat.library is None and mp.read_patch(mat) is not None

    def execute(self, context):
        mat = _material(context)
        mp.remove(mat)
        mat.rpom.last_warnings = json.dumps([])
        self.report({"INFO"}, tr("material restored"))
        return {"FINISHED"}


class RPOM_OT_select_node(bpy.types.Operator):
    """Select the node the warning refers to"""

    bl_idname = "rpom.select_node"
    bl_label = "Select Node"
    bl_options = {"REGISTER", "UNDO"}

    node: bpy.props.StringProperty()

    def execute(self, context):
        mat = _material(context)
        if mat is None or mat.node_tree is None or self.node not in mat.node_tree.nodes:
            return {"CANCELLED"}
        nt = mat.node_tree
        for n in nt.nodes:
            n.select = n.name == self.node
        nt.nodes.active = nt.nodes[self.node]
        return {"FINISHED"}


CLASSES = (RPOM_OT_remove, RPOM_OT_select_node)
