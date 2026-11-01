# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

from ..graph import registry
from ..utils import log
from ..utils.i18n import tr

LOG = log.get("upgrade")


def _group_nodes(tree_names: set[str]):
    for mat in bpy.data.materials:
        if mat.node_tree is None or mat.library is not None:
            continue
        for n in mat.node_tree.nodes:
            if n.bl_idname == "ShaderNodeGroup" and n.node_tree is not None and n.node_tree.name in tree_names:
                yield mat, n


def _save(node: bpy.types.Node) -> dict:
    ins, outs = {}, {}
    for s in node.inputs:
        if s.is_linked:
            ins[s.name] = ("link", s.links[0].from_node.name, s.links[0].from_socket.identifier)
        elif hasattr(s, "default_value"):
            v = s.default_value
            ins[s.name] = ("value", tuple(v) if hasattr(v, "__len__") else v)
    for s in node.outputs:
        outs[s.name] = [(lk.to_node.name, lk.to_socket.identifier) for lk in s.links]
    return {"in": ins, "out": outs}


def _restore(mat: bpy.types.Material, node: bpy.types.Node, saved: dict) -> list[str]:
    nt = mat.node_tree
    lost = []
    for name, val in saved["in"].items():
        s = node.inputs.get(name)
        if s is None:
            lost.append(tr("{}: input '{}' no longer exists", mat.name, name))
            continue
        if val[0] == "link":
            src = nt.nodes.get(val[1])
            out = next((o for o in src.outputs if o.identifier == val[2]), None) if src else None
            if out is not None:
                nt.links.new(out, s)
        elif hasattr(s, "default_value"):
            try:
                s.default_value = val[1]
            except (TypeError, ValueError):
                lost.append(tr("{}: value of '{}' cannot be restored", mat.name, name))
    for name, targets in saved["out"].items():
        s = node.outputs.get(name)
        for to_node, to_sock in targets:
            dst = nt.nodes.get(to_node)
            inp = next((i for i in dst.inputs if i.identifier == to_sock), None) if dst else None
            if s is not None and inp is not None:
                nt.links.new(s, inp)
            else:
                lost.append(tr("{}: output '{}' no longer exists", mat.name, name))
    return lost


def upgrade_all() -> list[str]:
    names = {t.name for t in bpy.data.node_groups if t.get("rpom_group")}
    saved = [(mat, n.name, _save(n)) for mat, n in _group_nodes(names)]
    registry.ensure_groups(force=True)
    if any(t.get("rpom_group", "").startswith("RPOM_Hex") for t in bpy.data.node_groups):
        registry.ensure_hex_groups(force=True)
    lost = []
    for mat, node_name, data in saved:
        node = mat.node_tree.nodes.get(node_name)
        if node is not None:
            lost += _restore(mat, node, data)
    return lost


@persistent
def on_load_post(*_args) -> None:
    old = registry.outdated()
    if old:
        LOG.warning("POM Shader: %d groups from an older version; use 'Upgrade Groups' in the panel",
                    len(old))


class RPOM_OT_upgrade(bpy.types.Operator):
    """Rebuild the POM Shader node groups at the add-on version, keeping the links"""

    bl_idname = "rpom.upgrade"
    bl_label = "Upgrade Groups"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        for msg in upgrade_all():
            self.report({"WARNING"}, msg)
        self.report({"INFO"}, tr("groups at version {}", registry.GROUP_VERSION))
        return {"FINISHED"}


CLASSES = (RPOM_OT_upgrade,)


def register_handlers() -> None:
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)


def unregister_handlers() -> None:
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
