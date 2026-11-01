# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import bpy

Value = Any

_SOCKET_TYPES = {
    "FLOAT": "NodeSocketFloat",
    "INT": "NodeSocketInt",
    "VECTOR": "NodeSocketVector",
    "COLOR": "NodeSocketColor",
    "SHADER": "NodeSocketShader",
    "CLOSURE": "NodeSocketClosure",
    "BOOLEAN": "NodeSocketBool",
    "OBJECT": "NodeSocketObject",
}

COLUMN_WIDTH = 220.0
ROW_HEIGHT = 190.0


@dataclass
class RepeatZone:

    input_node: bpy.types.Node
    output_node: bpy.types.Node
    state: dict[str, bpy.types.NodeSocket] = field(default_factory=dict)
    result: dict[str, bpy.types.NodeSocket] = field(default_factory=dict)
    iteration: bpy.types.NodeSocket | None = None


def sock(collection, key: str) -> bpy.types.NodeSocket:
    for s in collection:
        if s.identifier == key:
            return s
    return collection[key]


def set_subtype(item, subtype: str) -> None:
    try:
        item.subtype = subtype.upper()
    except (TypeError, AttributeError):
        pass


class Builder:
    def __init__(self, tree: bpy.types.NodeTree, prefix: str = "") -> None:
        self.tree = tree
        self.prefix = prefix
        self._count = 0
        self._frame: bpy.types.Node | None = None
        self._frames: dict[str, bpy.types.Node] = {}
        self._group_in: bpy.types.Node | None = None
        self._group_out: bpy.types.Node | None = None
        self.external = None

    def node(self, bl_idname: str, label: str = "", **props: Any) -> bpy.types.Node:
        n = self.tree.nodes.new(bl_idname)
        self._count += 1
        short = bl_idname.replace("ShaderNode", "").replace("GeometryNode", "").replace("Node", "")
        n.name = f"{self.prefix}{self._count:03d}_{short}"
        if label:
            n.label = label
        for key, val in props.items():
            setattr(n, key, val)
        if self._frame is not None:
            n.parent = self._frame
        return n

    def link(self, value: Value, socket: bpy.types.NodeSocket) -> None:
        if value is None:
            return
        if isinstance(value, bpy.types.NodeSocket):
            if value.id_data != self.tree and self.external is not None:
                value = self.external(value)
            self.tree.links.new(value, socket)
            return
        if isinstance(value, (int, float)) and socket.type in {"VECTOR", "RGBA"}:
            n = len(socket.default_value)
            value = (float(value),) * n
            if socket.type == "RGBA":
                value = value[:3] + (1.0,)
        socket.default_value = value

    def _set_inputs(self, n: bpy.types.Node, values: dict[str, Value]) -> None:
        for key, val in values.items():
            self.link(val, n.inputs[key])

    def math(self, op: str, a: Value = 0.0, b: Value = 0.0, c: Value = 0.0, clamp: bool = False,
             label: str = "") -> bpy.types.NodeSocket:
        n = self.node("ShaderNodeMath", label, operation=op, use_clamp=clamp)
        self.link(a, n.inputs[0])
        self.link(b, n.inputs[1])
        self.link(c, n.inputs[2])
        return n.outputs[0]

    def vmath(self, op: str, a: Value = 0.0, b: Value = 0.0, c: Value = 0.0, scale: Value = 1.0,
              label: str = "") -> bpy.types.NodeSocket:
        n = self.node("ShaderNodeVectorMath", label, operation=op)
        self.link(a, n.inputs[0])
        self.link(b, n.inputs[1])
        self.link(c, n.inputs[2])
        self.link(scale, n.inputs[3])
        if op in {"DOT_PRODUCT", "LENGTH", "DISTANCE"}:
            return n.outputs["Value"]
        return n.outputs["Vector"]

    def add(self, a: Value, b: Value, **kw) -> bpy.types.NodeSocket:
        return self.math("ADD", a, b, **kw)

    def sub(self, a: Value, b: Value, **kw) -> bpy.types.NodeSocket:
        return self.math("SUBTRACT", a, b, **kw)

    def mul(self, a: Value, b: Value, **kw) -> bpy.types.NodeSocket:
        return self.math("MULTIPLY", a, b, **kw)

    def div(self, a: Value, b: Value, **kw) -> bpy.types.NodeSocket:
        return self.math("DIVIDE", a, b, **kw)

    def madd(self, a: Value, b: Value, c: Value, **kw) -> bpy.types.NodeSocket:
        return self.math("MULTIPLY_ADD", a, b, c, **kw)

    def vmadd(self, a: Value, b: Value, c: Value, **kw) -> bpy.types.NodeSocket:
        return self.vmath("MULTIPLY_ADD", a, b, c, **kw)

    def dot(self, a: Value, b: Value, **kw) -> bpy.types.NodeSocket:
        return self.vmath("DOT_PRODUCT", a, b, **kw)

    def scale(self, v: Value, s: Value, **kw) -> bpy.types.NodeSocket:
        return self.vmath("SCALE", v, scale=s, **kw)

    def clamp(self, x: Value, lo: Value = 0.0, hi: Value = 1.0, label: str = "") -> bpy.types.NodeSocket:
        n = self.node("ShaderNodeClamp", label)
        self._set_inputs(n, {"Value": x, "Min": lo, "Max": hi})
        return n.outputs[0]

    def mix(self, fac: Value, a: Value, b: Value, data_type: str = "FLOAT", label: str = "") -> bpy.types.NodeSocket:
        n = self.node("ShaderNodeMix", label, data_type=data_type, clamp_factor=False)
        suffix = {"FLOAT": "Float", "VECTOR": "Vector", "RGBA": "Color"}[data_type]
        self.link(fac, sock(n.inputs, "Factor_Float"))
        self.link(a, sock(n.inputs, f"A_{suffix}"))
        self.link(b, sock(n.inputs, f"B_{suffix}"))
        return sock(n.outputs, f"Result_{suffix}")

    def smoothstep(self, x: Value, lo: Value, hi: Value, label: str = "") -> bpy.types.NodeSocket:
        n = self.node("ShaderNodeMapRange", label, interpolation_type="SMOOTHSTEP", clamp=True)
        self._set_inputs(n, {"Value": x, "From Min": lo, "From Max": hi, "To Min": 0.0, "To Max": 1.0})
        return n.outputs["Result"]

    def combine(self, x: Value = 0.0, y: Value = 0.0, z: Value = 0.0, label: str = "") -> bpy.types.NodeSocket:
        n = self.node("ShaderNodeCombineXYZ", label)
        self._set_inputs(n, {"X": x, "Y": y, "Z": z})
        return n.outputs[0]

    def separate(self, v: Value, label: str = "") -> tuple[bpy.types.NodeSocket, ...]:
        n = self.node("ShaderNodeSeparateXYZ", label)
        self.link(v, n.inputs[0])
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def separate_color(self, c: Value, label: str = "") -> tuple[bpy.types.NodeSocket, ...]:
        n = self.node("ShaderNodeSeparateColor", label)
        self.link(c, n.inputs[0])
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def group_input(self) -> bpy.types.Node:
        if self._group_in is None:
            self._group_in = self.node("NodeGroupInput")
        return self._group_in

    def group_output(self) -> bpy.types.Node:
        if self._group_out is None:
            self._group_out = self.node("NodeGroupOutput")
            self._group_out.is_active_output = True
        return self._group_out

    def panel(self, name: str, closed: bool = False):
        p = self.tree.interface.new_panel(name, default_closed=closed)
        return p

    def inp(self, name: str, kind: str, default: Any = None, lo: Any = None, hi: Any = None,
            panel: Any = None, description: str = "", subtype: str = "") -> bpy.types.NodeSocket:
        item = self.tree.interface.new_socket(name, in_out="INPUT", socket_type=_SOCKET_TYPES[kind], parent=panel,
                                              description=description)
        if subtype:
            set_subtype(item, subtype)
        if default is not None:
            item.default_value = default
        if lo is not None:
            item.min_value = lo
        if hi is not None:
            item.max_value = hi
        if kind in {"VECTOR"} and default is None:
            item.hide_value = True
        return self.group_input().outputs[item.identifier]

    def out(self, name: str, kind: str, value: Value, panel: Any = None, description: str = "") -> None:
        item = self.tree.interface.new_socket(name, in_out="OUTPUT", socket_type=_SOCKET_TYPES[kind],
                                              parent=panel, description=description)
        self.link(value, self.group_output().inputs[item.identifier])

    def repeat(self, iterations: Value, items: list[tuple[str, str, Value]], label: str = "") -> RepeatZone:
        zin = self.node("GeometryNodeRepeatInput", label)
        zout = self.node("GeometryNodeRepeatOutput", label)
        zin.pair_with_output(zout)
        zout.repeat_items.clear()
        for name, kind, _ in items:
            zout.repeat_items.new("FLOAT" if kind == "FLOAT" else kind, name)
        self.link(iterations, zin.inputs["Iterations"])
        zone = RepeatZone(zin, zout, iteration=zin.outputs["Iteration"])
        for name, _, init in items:
            self.link(init, zin.inputs[name])
            zone.state[name] = zin.outputs[name]
            zone.result[name] = zout.outputs[name]
        return zone

    def close(self, zone: RepeatZone, values: dict[str, Value]) -> None:
        for name, val in values.items():
            self.link(val, zone.output_node.inputs[name])

    def evaluate(self, closure: Value, inputs: list[tuple[str, str, Value]], outputs: list[tuple[str, str]],
                 label: str = "") -> dict[str, bpy.types.NodeSocket]:
        n = self.node("NodeEvaluateClosure", label)
        n.input_items.clear()
        n.output_items.clear()
        for name, kind, _ in inputs:
            n.input_items.new(kind, name)
        for name, kind in outputs:
            n.output_items.new(kind, name)
        self.link(closure, n.inputs["Closure"])
        for name, _, val in inputs:
            self.link(val, n.inputs[name])
        return {name: n.outputs[name] for name, _ in outputs}

    def group(self, tree: bpy.types.NodeTree, inputs: dict[str, Value], label: str = "") -> bpy.types.Node:
        n = self.node("ShaderNodeGroup", label)
        n.node_tree = tree
        for key, val in inputs.items():
            self.link(val, n.inputs[key])
        return n

    @contextmanager
    def frame(self, label: str) -> Iterator[bpy.types.Node]:
        prev = self._frame
        fr = self._frames.get(label)
        if fr is None:
            fr = self.tree.nodes.new("NodeFrame")
            fr.name = f"{self.prefix}frame_{len(self._frames) + 1:02d}"
            fr.label = label
            fr.label_size = 16
            self._frames[label] = fr
        self._frame = fr
        try:
            yield fr
        finally:
            self._frame = prev

    def layout(self, only: set[str] | None = None, origin: tuple[float, float] = (0.0, 0.0)) -> None:
        nodes = [n for n in self.tree.nodes if n.bl_idname != "NodeFrame" and (only is None or n.name in only)]
        depth: dict[str, int] = {}
        incoming: dict[str, list[str]] = {n.name: [] for n in nodes}
        for lk in self.tree.links:
            if lk.to_node.name in incoming and lk.from_node.name in incoming:
                incoming[lk.to_node.name].append(lk.from_node.name)

        def d(name: str, guard: int = 0) -> int:
            if name in depth:
                return depth[name]
            if guard > 400:
                return 0
            depth[name] = 0
            val = 1 + max((d(src, guard + 1) for src in incoming[name]), default=-1)
            depth[name] = val
            return val

        rows: dict[int, int] = {}
        for n in nodes:
            col = d(n.name)
            row = rows.get(col, 0)
            rows[col] = row + 1
            n.location = (origin[0] + col * COLUMN_WIDTH, origin[1] - row * ROW_HEIGHT)
