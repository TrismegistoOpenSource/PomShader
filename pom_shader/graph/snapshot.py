# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
from typing import Any

import bpy

_NODE_PROPS = (
    "operation", "data_type", "blend_type", "interpolation", "interpolation_type", "projection", "extension",
    "use_clamp", "clamp", "clamp_factor", "attribute_type", "attribute_name", "vector_type", "convert_from",
    "convert_to", "direction_type", "uv_map", "target", "aov_name", "noise_dimensions", "distribution",
    "subsurface_method", "vector_type", "space", "is_active_output", "label", "mute",
)
_SKIP_SOCKET_TYPES = {"CUSTOM", "SHADER", "CLOSURE", "BUNDLE", "GEOMETRY"}


def _value(v: Any) -> Any:
    if hasattr(v, "__len__") and not isinstance(v, str):
        return [round(float(x), 6) for x in v]
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, bpy.types.ID):
        return f"ID:{v.name}"
    return v


def _sockets(sockets) -> dict:
    out = {}
    for s in sockets:
        if s.type in _SKIP_SOCKET_TYPES or s.is_linked or not hasattr(s, "default_value"):
            continue
        out[socket_key(s)] = _value(s.default_value)
    return out


def socket_key(socket: bpy.types.NodeSocket) -> str:
    coll = socket.node.outputs if socket.is_output else socket.node.inputs
    same = [x for x in coll if x.name == socket.name]
    return f"{socket.name}#{same.index(socket)}" if len(same) > 1 else socket.name


def _items(node: bpy.types.Node) -> dict:
    out = {}
    for attr in ("repeat_items", "input_items", "output_items"):
        coll = getattr(node, attr, None)
        if coll is not None:
            out[attr] = [[it.name, it.socket_type] for it in coll]
    if getattr(node, "paired_output", None) is not None:
        out["paired_output"] = node.paired_output.name
    return out


def node_dict(node: bpy.types.Node) -> dict:
    d: dict[str, Any] = {"type": node.bl_idname}
    for p in _NODE_PROPS:
        if hasattr(node, p):
            d[p] = _value(getattr(node, p))
    if node.parent is not None:
        d["parent"] = node.parent.name
    if getattr(node, "node_tree", None) is not None:
        d["node_tree"] = node.node_tree.name
    if getattr(node, "image", None) is not None:
        d["image"] = node.image.name
    if getattr(node, "object", None) is not None:
        d["object"] = node.object.name
    d["inputs"] = _sockets(node.inputs)
    d.update(_items(node))
    return d


def _interface(tree: bpy.types.NodeTree) -> list:
    out = []
    for item in tree.interface.items_tree:
        if item.item_type == "PANEL":
            out.append(["PANEL", item.name])
        else:
            entry = ["SOCKET", item.in_out, item.name, item.socket_type]
            if hasattr(item, "default_value"):
                entry.append(_value(item.default_value))
            out.append(entry)
    return out


def snapshot(tree: bpy.types.NodeTree, with_interface: bool = True) -> dict:
    nodes = {n.name: node_dict(n) for n in tree.nodes}
    links = sorted(
        [lk.from_node.name, socket_key(lk.from_socket), lk.to_node.name, socket_key(lk.to_socket)]
        for lk in tree.links
    )
    snap = {"nodes": dict(sorted(nodes.items())), "links": links}
    if with_interface:
        snap["interface"] = _interface(tree)
    return snap


def dumps(tree: bpy.types.NodeTree) -> str:
    return json.dumps(snapshot(tree), indent=1, sort_keys=True, ensure_ascii=False)


