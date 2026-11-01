# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import bpy

from ..utils.i18n import tr
from . import registry
from .builder import _SOCKET_TYPES, Builder, set_subtype, sock
from .signature import BORDER_MODES, COLOR_INPUTS, COLOR_OUTPUTS
from .source import SourceSpec, build_hex_source, build_source

PATCH_KEY = "rpom_patch"
FRAME_MODES = {"AUTO": 0, "EXACT": 1, "COMPACT": 2, "UNIFORM": 3}
PATCH_VERSION = 2
PREFIX = "rpom_"
HEIGHT_HINTS = ("height", "disp", "displacement", "_h.", "_h_", "-h.", "bump", "depth")


@dataclass
class TextureUse:
    node: bpy.types.Node
    uv_map: str
    source: bpy.types.NodeSocket | None
    mapping: bpy.types.Node | None
    other_coords: bool = False
    parallax: bool = False


@dataclass
class Analysis:
    output: bpy.types.Node | None
    upstream: set[str] = field(default_factory=set)
    textures: list[TextureUse] = field(default_factory=list)
    all_textures: list[TextureUse] = field(default_factory=list)
    height_candidates: list[bpy.types.Node] = field(default_factory=list)
    shaders: list[bpy.types.Node] = field(default_factory=list)
    displacement_link: bpy.types.NodeLink | None = None
    has_normal_map: bool = False
    warnings: list[str] = field(default_factory=list)


def active_output(nt: bpy.types.NodeTree) -> bpy.types.Node | None:
    outs = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and not n.name.startswith(PREFIX)]
    for n in outs:
        if n.is_active_output:
            return n
    return outs[0] if outs else None


def _upstream(start: bpy.types.Node) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        n = stack.pop()
        if n.name in seen:
            continue
        seen.add(n.name)
        for s in n.inputs:
            for lk in s.links:
                if lk.is_valid and not lk.is_muted:
                    stack.append(lk.from_node)
    return seen


def _trace_uv(tex: bpy.types.Node) -> TextureUse:
    s_in = tex.inputs.get("Vector")
    if s_in is None or not s_in.is_linked:
        return TextureUse(tex, "", None, None)
    mapping = None
    lk = s_in.links[0]
    for _ in range(64):
        n = lk.from_node
        if n.name == PREFIX + "main":
            return TextureUse(tex, "", lk.from_socket, mapping, parallax=True)
        if n.bl_idname == "NodeReroute":
            nxt = n.inputs[0]
        elif n.bl_idname == "ShaderNodeMapping":
            mapping = mapping or n
            nxt = n.inputs["Vector"]
        elif n.bl_idname == "ShaderNodeTexCoord":
            if lk.from_socket.identifier == "UV":
                return TextureUse(tex, "", lk.from_socket, mapping)
            return TextureUse(tex, "", lk.from_socket, mapping, other_coords=True)
        elif n.bl_idname == "ShaderNodeUVMap":
            return TextureUse(tex, n.uv_map, lk.from_socket, mapping)
        else:
            return TextureUse(tex, "", None, mapping, other_coords=True)
        if not nxt.is_linked:
            return TextureUse(tex, "", None, mapping)
        lk = nxt.links[0]
    return TextureUse(tex, "", None, mapping, other_coords=True)


def _feeds(node: bpy.types.Node, target_bl_idname: str, socket_name: str) -> bool:
    for out in node.outputs:
        for lk in out.links:
            if lk.to_node.bl_idname == target_bl_idname and lk.to_socket.name == socket_name:
                return True
            if lk.to_node.bl_idname == "NodeReroute" and _feeds(lk.to_node, target_bl_idname, socket_name):
                return True
    return False


def analyze(mat: bpy.types.Material) -> Analysis:
    nt = mat.node_tree
    out = active_output(nt)
    a = Analysis(out)
    if out is None:
        a.warnings.append(tr("no Material Output"))
        return a
    surf = out.inputs["Surface"]
    if surf.is_linked:
        a.upstream = _upstream(surf.links[0].from_node)
    disp = out.inputs["Displacement"]
    if disp.is_linked:
        a.displacement_link = disp.links[0]
        a.upstream |= _upstream(disp.links[0].from_node)
    for n in sorted(nt.nodes, key=lambda x: x.name):
        if n.bl_idname == "ShaderNodeTexImage" and n.image is not None and not n.name.startswith(PREFIX):
            use = _trace_uv(n)
            a.all_textures.append(use)
            if n.name in a.upstream:
                a.textures.append(use)
    for name in sorted(a.upstream):
        n = nt.nodes[name]
        if n.name.startswith(PREFIX):
            continue
        if any(o.type == "SHADER" for o in n.outputs) and n.inputs.get("Normal") is not None:
            a.shaders.append(n)
        if n.bl_idname == "ShaderNodeNormalMap" and any(lk.to_node.name in a.upstream for o in n.outputs for lk in o.links):
            a.has_normal_map = True
    a.height_candidates = _height_candidates(a)
    return a


def _height_candidates(a: Analysis) -> list[bpy.types.Node]:
    scored = []
    for t in a.all_textures:
        n = t.node
        text = f"{n.name} {n.label} {n.image.name} {n.image.filepath}".lower()
        score = 0
        if _feeds(n, "ShaderNodeDisplacement", "Height"):
            score += 4
        if _feeds(n, "ShaderNodeBump", "Height"):
            score += 3
        if any(h in text for h in HEIGHT_HINTS):
            score += 2
        if score:
            scored.append((score, n.name, n))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, _, n in scored]


@dataclass
class Settings:
    march_image: str
    shade_image: str
    uv_map: str = ""
    border_mode: str = "TILING"
    depth: float = 0.02
    ref_plane: float = 1.0
    steps: int = 16
    refine: int = 2
    dither: float = 1.0
    max_offset: float = 256.0
    edge_fade: float = 4.0
    resolution: float = 1024.0
    fade_start: float = 20.0
    fade_end: float = 60.0
    c_min: float = 0.04
    uniform_scale: float = 1.0
    frame_mode: str = "AUTO"
    normal_policy: str = "AUTO"
    shadow: bool = False
    shadow_strength: float = 0.8
    shadow_steps: int = 8
    shadow_sharpness: float = 8.0
    light_helper: str = ""
    cavity: bool = False
    cavity_strength: float = 0.5
    height_node: str = ""
    disconnect_displacement: bool = True
    hex: dict | None = None

    def group_inputs(self) -> dict:
        return {
            "Depth": self.depth, "Reference Plane": self.ref_plane, "Steps": self.steps, "Refine": self.refine,
            "Dither": self.dither, "Border Mode": BORDER_MODES[self.border_mode], "Max Offset": self.max_offset,
            "Edge Fade": self.edge_fade, "Resolution": self.resolution, "Fade Start": self.fade_start,
            "Fade End": self.fade_end,
            "Shadow Strength": self.shadow_strength if self.shadow and self.light_helper else 0.0,
            "Shadow Steps": self.shadow_steps, "Shadow Sharpness": self.shadow_sharpness,
            "Cavity Strength": self.cavity_strength if self.cavity else 0.0, "Grazing Min": self.c_min,
            "Uniform Scale": self.uniform_scale, "Frame Mode": FRAME_MODES.get(self.frame_mode, 0),
        }


def _link_record(lk: bpy.types.NodeLink) -> list[str]:
    return [lk.from_node.name, lk.from_socket.identifier, lk.to_node.name, lk.to_socket.identifier]


@dataclass
class Patch:
    nodes: list[str] = field(default_factory=list)
    links_removed: list[list[str]] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    hex: dict | None = None
    container: str = ""
    default_uv: str = ""
    warnings: list[str] = field(default_factory=list)
    props: list[list] = field(default_factory=list)
    version: int = PATCH_VERSION
    group_version: int = registry.GROUP_VERSION


def read_patch(mat: bpy.types.Material) -> Patch | None:
    raw = mat.get(PATCH_KEY)
    if not raw:
        return None
    data = json.loads(raw)
    if data.get("version") != PATCH_VERSION:
        return None
    return Patch(**data)


def is_patched(mat: bpy.types.Material) -> bool:
    p = read_patch(mat) if mat.node_tree else None
    return p is not None and bool(p.settings)


def container(mat: bpy.types.Material) -> bpy.types.NodeTree | None:
    p = read_patch(mat)
    return bpy.data.node_groups.get(p.container) if p is not None else None


def pom_node(mat: bpy.types.Material) -> bpy.types.Node | None:
    tree = container(mat)
    return tree.nodes.get(PREFIX + "pom") if tree is not None else None


def _write_patch(mat: bpy.types.Material, patch: Patch) -> None:
    mat[PATCH_KEY] = json.dumps(asdict(patch), ensure_ascii=False)


LEGACY_KEYS = ("rpom_hex_patch", PATCH_KEY)


def _legacy_teardown(mat: bpy.types.Material) -> bool:
    found = False
    nt = mat.node_tree
    for key in LEGACY_KEYS:
        raw = mat.get(key)
        if not raw:
            continue
        data = json.loads(raw)
        if key == PATCH_KEY and data.get("version") == PATCH_VERSION:
            continue
        found = True
        for name in data.get("nodes", []):
            n = nt.nodes.get(name)
            if n is not None:
                nt.nodes.remove(n)
        for fn, fs, tn, ts in data.get("links_removed", []):
            a, b = nt.nodes.get(fn), nt.nodes.get(tn)
            if a is not None and b is not None:
                nt.links.new(sock(a.outputs, fs), sock(b.inputs, ts))
        for kind, id_name, node_name, prop, value in reversed(data.get("props", [])):
            owner = _prop_owner(mat, kind, id_name, node_name)
            if owner is not None:
                setattr(owner, prop, value)
        for ob_name, mod_name in data.get("modifiers", []):
            ob = bpy.data.objects.get(ob_name)
            mod = ob.modifiers.get(mod_name) if ob is not None else None
            if mod is not None:
                ob.modifiers.remove(mod)
        del mat[key]
    return found


def teardown(mat: bpy.types.Material) -> Patch | None:
    _legacy_teardown(mat)
    patch = read_patch(mat)
    if patch is None:
        return None
    nt = mat.node_tree
    for name in patch.nodes:
        n = nt.nodes.get(name)
        if n is not None:
            nt.nodes.remove(n)
    tree = bpy.data.node_groups.get(patch.container)
    if tree is not None and tree.get("rpom_container") is not None:
        bpy.data.node_groups.remove(tree)
    for fn, fs, tn, ts in patch.links_removed:
        a, b = nt.nodes.get(fn), nt.nodes.get(tn)
        if a is None or b is None:
            continue
        nt.links.new(sock(a.outputs, fs), sock(b.inputs, ts))
    for kind, id_name, node_name, prop, value in reversed(patch.props):
        owner = _prop_owner(mat, kind, id_name, node_name)
        if owner is not None:
            setattr(owner, prop, value)
    del mat[PATCH_KEY]
    return patch


def remove(mat: bpy.types.Material) -> bool:
    return teardown(mat) is not None


def _prop_owner(mat, kind: str, id_name: str, node_name: str):
    if kind == "MATERIAL":
        return mat
    if kind == "NODE":
        return mat.node_tree.nodes.get(node_name)
    return None


def tiling_values(mat: bpy.types.Material) -> tuple[tuple, tuple, float]:
    rp = getattr(mat, "rpom", None)
    if rp is None:
        return (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.0
    sx, sy = rp.tiling_scale
    ox, oy = rp.tiling_offset
    return (sx, sy, 1.0), (ox, oy, 0.0), rp.tiling_rotation


def sync_from_node(mat: bpy.types.Material) -> None:
    from ..props import LIVE_INPUTS

    rp = getattr(mat, "rpom", None)
    node = mat.node_tree.nodes.get(PREFIX + "main") if mat.node_tree else None
    if rp is None or node is None or node.node_tree is None or node.node_tree.get("rpom_container") is None:
        return
    for prop, name in LIVE_INPUTS.items():
        s_in = node.inputs.get(name)
        if s_in is None or (prop == "shadow_strength" and not rp.shadow) or (prop == "cavity_strength" and not rp.cavity):
            continue
        if abs(s_in.default_value - getattr(rp, prop)) > 1e-7:
            setattr(rp, prop, s_in.default_value)
    if "Tiling Scale" in node.inputs:
        sx, sy, _ = node.inputs["Tiling Scale"].default_value
        ox, oy, _ = node.inputs["Tiling Offset"].default_value
        rot = node.inputs["Tiling Rotation"].default_value
        rp.tiling_scale = (sx, sy)
        rp.tiling_offset = (ox, oy)
        rp.tiling_rotation = rot


def refresh(s: Settings, mat: bpy.types.Material) -> Settings:
    from ..props import LIVE_INPUTS

    rp = getattr(mat, "rpom", None)
    if rp is not None:
        names = {"ref_plane": "ref_plane", "grazing_min": "c_min"}
        for prop in LIVE_INPUTS:
            setattr(s, names.get(prop, prop), getattr(rp, prop))
    return s


_KINDS = {"VALUE": "FLOAT", "INT": "INT", "BOOLEAN": "BOOLEAN", "VECTOR": "VECTOR", "RGBA": "COLOR",
          "SHADER": "SHADER", "CLOSURE": "CLOSURE"}


class _Patcher:

    def __init__(self, mat: bpy.types.Material, s: Settings | None, hexs: dict | None, default_uv: str) -> None:
        self.mat = mat
        self.nt = mat.node_tree
        self.s = s
        self.hexs = hexs
        self.patch = Patch(settings=asdict(s) if s is not None else {}, hex=hexs, default_uv=default_uv)
        self.outer = Builder(self.nt, PREFIX)
        self.tree = bpy.data.node_groups.new(f"POM Shader {mat.name}", "ShaderNodeTree")
        self.tree["rpom_container"] = mat.name
        self.tree.description = "POM Shader: generated by the add-on (use Remove in the panel to restore the material)"
        self.patch.container = self.tree.name
        self.b = Builder(self.tree, PREFIX)
        self.b.external = self.feed
        self._before = set(self.nt.nodes.keys())
        self._feeds: dict[int, bpy.types.NodeSocket] = {}
        self._pending_in: list[tuple[bpy.types.NodeSocket, str]] = []
        self._pending_out: list[tuple[str, list[bpy.types.NodeSocket]]] = []
        self._params: list[tuple[str, object]] = []
        self._names: set[str] = set()
        self._settings_panel = None
        self._tiling = self._tiling_inputs()

    def _unique(self, name: str) -> str:
        base, i = name, 2
        while name in self._names:
            name = f"{base} {i}"
            i += 1
        self._names.add(name)
        return name

    def feed(self, outer: bpy.types.NodeSocket, name: str = "") -> bpy.types.NodeSocket:
        key = outer.as_pointer()
        if key in self._feeds:
            return self._feeds[key]
        kind = _KINDS.get(outer.type, "FLOAT")
        label = name or f"{outer.node.label or outer.node.name}: {outer.name}"
        item = self.tree.interface.new_socket(self._unique(label), in_out="INPUT", socket_type=_SOCKET_TYPES[kind])
        if kind == "VECTOR":
            item.hide_value = True
        self._pending_in.append((outer, item.identifier))
        inner = self.b.group_input().outputs[item.identifier]
        self._feeds[key] = inner
        return inner

    def give(self, value, name: str, kind: str, targets: list[bpy.types.NodeSocket]) -> None:
        item = self.tree.interface.new_socket(self._unique(name), in_out="OUTPUT", socket_type=_SOCKET_TYPES[kind])
        self.b.link(value, self.b.group_output().inputs[item.identifier])
        self._pending_out.append((item.identifier, list(targets)))

    def param(self, template, value) -> bpy.types.NodeSocket:
        if self._settings_panel is None:
            self._settings_panel = self.tree.interface.new_panel("Settings", default_closed=True)
        item = self.tree.interface.new_socket(template.name, in_out="INPUT", socket_type=template.socket_type,
                                              parent=self._settings_panel, description=template.description)
        for attr in ("min_value", "max_value", "subtype", "default_value"):
            if hasattr(template, attr):
                try:
                    setattr(item, attr, getattr(template, attr))
                except (AttributeError, TypeError, ValueError):
                    pass
        self._names.add(template.name)
        self._params.append((item.identifier, value))
        return self.b.group_input().outputs[item.identifier]

    def _tiling_inputs(self) -> dict[str, bpy.types.NodeSocket]:
        panel = self.tree.interface.new_panel("Tiling")
        scale, offset, rotation = tiling_values(self.mat)
        live = {}
        for name, kind, subtype, value in (("Tiling Scale", "NodeSocketVector", "XYZ", scale),
                                           ("Tiling Offset", "NodeSocketVector", "TRANSLATION", offset),
                                           ("Tiling Rotation", "NodeSocketFloat", "ANGLE", rotation)):
            item = self.tree.interface.new_socket(name, in_out="INPUT", socket_type=kind, parent=panel)
            set_subtype(item, subtype)
            item.default_value = value
            self._names.add(name)
            self._params.append((item.identifier, value))
            live[name] = self.b.group_input().outputs[item.identifier]
        return live

    def tile(self, uv) -> bpy.types.NodeSocket:
        b = self.b
        m = b.node("ShaderNodeMapping", "Tiling", vector_type="POINT")
        b.link(uv, m.inputs["Vector"])
        b.link(self._tiling["Tiling Scale"], m.inputs["Scale"])
        b.link(self._tiling["Tiling Offset"], m.inputs["Location"])
        b.link(b.combine(0.0, 0.0, self._tiling["Tiling Rotation"]), m.inputs["Rotation"])
        return m.outputs[0]

    def tiling_scale(self) -> bpy.types.NodeSocket:
        return _live_scale(self.b, self._tiling["Tiling Scale"], "POINT")

    def _place(self, node: bpy.types.Node, anchor: bpy.types.Node | None) -> None:
        def pos(n):
            return getattr(n, "location_absolute", n.location)

        user = [n for n in self.nt.nodes if n.name in self._before and n.bl_idname != "NodeFrame"]
        extra = [n for n in self.nt.nodes if n.name not in self._before and n != node
                 and n.bl_idname not in {"NodeClosureInput", "NodeClosureOutput"}]
        x0 = pos(anchor).x - 300.0 if anchor is not None else 0.0
        y0 = min((pos(n).y - (n.dimensions.y or 280.0) for n in user), default=0.0) - 120.0
        frame = self.nt.nodes.new("NodeFrame")
        frame.name = PREFIX + "frame"
        frame.label = "POM Shader"
        frame.label_size = 16
        for i, n in enumerate([node] + extra):
            n.parent = frame
            n.location_absolute = (x0 + i * 260.0, y0)

    def unlink(self, lk: bpy.types.NodeLink) -> None:
        self.patch.links_removed.append(_link_record(lk))
        self.nt.links.remove(lk)

    def finish(self, anchor: bpy.types.Node | None) -> Patch:
        node = self.nt.nodes.new("ShaderNodeGroup")
        node.name = PREFIX + "main"
        node.label = "POM Shader"
        node.node_tree = self.tree
        node.width = 220
        for outer, ident in self._pending_in:
            self.nt.links.new(outer, sock(node.inputs, ident))
        for ident, targets in self._pending_out:
            for target in targets:
                self.nt.links.new(sock(node.outputs, ident), target)
        for ident, value in self._params:
            sock(node.inputs, ident).default_value = value
        self._place(node, anchor)
        self.b.layout()
        self.patch.nodes = sorted(set(self.nt.nodes.keys()) - self._before)
        _write_patch(self.mat, self.patch)
        return self.patch


def _same_map(a: str, b: str, default: str) -> bool:
    return (a or default) == (b or default)


def _mapping_scale(mapping: bpy.types.Node | None, warnings: list[str]) -> float:
    if mapping is None:
        return 1.0
    sc = mapping.inputs.get("Scale")
    rot = mapping.inputs.get("Rotation")
    if sc is None or sc.is_linked or (rot is not None and rot.is_linked):
        warnings.append(tr("height Mapping with linked inputs: scale taken as 1, approximate normal"))
        return 1.0
    sx, sy, _ = sc.default_value
    rx, ry, rz = rot.default_value if rot is not None else (0, 0, 0)
    if abs(sx - sy) > 1e-6 * max(abs(sx), 1) or abs(rx) > 1e-6 or abs(ry) > 1e-6 or abs(rz) > 1e-6:
        warnings.append(tr("height Mapping rotated or non-uniformly scaled: approximate normal"))
    s = (abs(sx) + abs(sy)) / 2 or 1.0
    return 1.0 / s if mapping.vector_type == "TEXTURE" else s


def _live_scale(b: Builder, scale, vector_type: str) -> bpy.types.NodeSocket:
    sx, sy, _ = b.separate(scale)
    avg = b.math("MAXIMUM", b.mul(b.add(b.math("ABSOLUTE", sx), b.math("ABSOLUTE", sy)), 0.5), 1e-6)
    return b.div(1.0, avg) if vector_type == "TEXTURE" else avg


def _from_height(n: bpy.types.Node, height_use: TextureUse | None) -> bool:
    if height_use is None:
        return False
    hin = n.inputs.get("Height")
    return bool(hin and hin.is_linked and height_use.node.name in _upstream(hin.links[0].from_node))


def _vector_chain(tex: bpy.types.Node) -> list[bpy.types.Node]:
    chain = []
    s_in = tex.inputs.get("Vector")
    while s_in is not None and s_in.is_linked:
        n = s_in.links[0].from_node
        if n.bl_idname == "NodeReroute":
            s_in = n.inputs[0]
        elif n.bl_idname == "ShaderNodeMapping":
            chain.append(n)
            s_in = n.inputs["Vector"]
        else:
            break
    return chain[::-1]


def _copy_chain(p: _Patcher, chain: list[bpy.types.Node], uv) -> bpy.types.NodeSocket:
    b = p.b
    v = p.tile(uv)
    for src in chain:
        m = b.node("ShaderNodeMapping", f"Mapping ({src.label or src.name})", vector_type=src.vector_type)
        for key in ("Location", "Rotation", "Scale"):
            if key in src.inputs:
                s_in = src.inputs[key]
                if s_in.is_linked:
                    b.link(s_in.links[0].from_socket, m.inputs[key])
                else:
                    m.inputs[key].default_value = s_in.default_value
        b.link(v, m.inputs["Vector"])
        v = m.outputs[0]
    return v


def _feeds_only(node: bpy.types.Node, kinds: set[str]) -> bool:
    links = [lk for o in node.outputs for lk in o.links]
    return bool(links) and all(lk.to_node.bl_idname in kinds for lk in links)


def _feeds_normal_map(node: bpy.types.Node) -> bool:
    return any(lk.to_node.bl_idname == "ShaderNodeNormalMap" for o in node.outputs for lk in o.links)


def _feeds_base_color(node: bpy.types.Node) -> bool:
    return any(lk.to_socket.name in {"Base Color", "Color"} for o in node.outputs for lk in o.links)


def _wrap(p: _Patcher, tex: bpy.types.Node):
    zin = p.outer.node("NodeClosureInput", f"POM Shader: {tex.label or tex.name}")
    zout = p.outer.node("NodeClosureOutput", f"POM Shader: {tex.label or tex.name}")
    zin.pair_with_output(zout)
    zout.input_items.clear()
    zout.output_items.clear()
    for name, kind in COLOR_INPUTS:
        zout.input_items.new(kind, name)
    for name, kind in COLOR_OUTPUTS:
        zout.output_items.new(kind, name)
    consumers = {"Color": [], "Alpha": []}
    for out_name in ("Color", "Alpha"):
        for lk in list(tex.outputs[out_name].links):
            consumers[out_name].append(lk.to_socket)
            p.unlink(lk)
    if tex.inputs["Vector"].is_linked:
        p.unlink(tex.inputs["Vector"].links[0])
    p.nt.links.new(zin.outputs["uv"], tex.inputs["Vector"])
    p.nt.links.new(tex.outputs["Color"], zout.inputs["color"])
    p.nt.links.new(tex.outputs["Alpha"], zout.inputs["alpha"])
    at = getattr(tex, "location_absolute", tex.location)
    zin.location = (at.x - 160.0, at.y)
    zout.location = (at.x + tex.width + 40.0, at.y)
    return zout.outputs["Closure"], consumers


def _shadow_targets(a: Analysis) -> set[int]:
    return {(n.inputs.get("Base Color") or n.inputs.get("Color")).as_pointer() for n in a.shaders
            if (n.inputs.get("Base Color") or n.inputs.get("Color")) is not None}


def _emit(p: _Patcher, value, name: str, kind: str, targets: list, darken, shadow_ptrs: set[int]) -> None:
    plain = [t for t in targets if darken is None or t.as_pointer() not in shadow_ptrs]
    dark = [t for t in targets if darken is not None and t.as_pointer() in shadow_ptrs]
    if plain or not dark:
        p.give(value, name, kind, plain)
    if dark:
        mixed = p.b.mix(1.0, value, None, data_type="RGBA", label="POM Shader: shadow")
        mix_node = mixed.node
        mix_node.blend_type = "MULTIPLY"
        p.b.link(darken, sock(mix_node.inputs, "B_Color"))
        p.give(mixed, f"{name} (shadow)", kind, dark)


def _textures(p: _Patcher, a: Analysis, base_uv, eval_uv, uv_name: str, darken) -> set[int]:
    s, hexs = p.s, p.hexs
    default_uv = p.patch.default_uv
    shadow_ptrs = _shadow_targets(a) if darken is not None else set()
    served: set[int] = set()
    by_chain: dict[tuple[str, ...], list[TextureUse]] = {}
    for t in a.textures:
        if t.node.name.startswith(PREFIX):
            continue
        if t.other_coords:
            p.patch.warnings.append(tr("{}: non-UV coordinates, left without parallax", t.node.name))
            continue
        if not _same_map(t.uv_map, uv_name, default_uv):
            p.patch.warnings.append(tr("{}: uses UV map '{}', left without parallax", t.node.name, t.uv_map))
            continue
        if s is not None and t.node.name == s.height_node and _feeds_only(
                t.node, {"ShaderNodeDisplacement", "ShaderNodeBump"}):
            continue
        key = tuple(n.name for n in _vector_chain(t.node))
        by_chain.setdefault(key, []).append(t)
    if not by_chain and s is None:
        raise ValueError(tr("no Image Texture with UVs to process"))
    b = p.b
    for key, uses in sorted(by_chain.items()):
        chain = _vector_chain(uses[0].node)
        st = _copy_chain(p, chain, eval_uv)
        tile = None
        if hexs is not None:
            from .groups import hextile

            groups = registry.ensure_hex_groups()
            grid = _copy_chain(p, chain, base_uv) if eval_uv is not base_uv else st
            tile = b.group(groups[hextile.TILE], {"Grid UV": grid, "Eval UV": st, "Scale": hexs["scale"],
                                                  "Rotation": hexs["rotation"], "Seed": hexs["seed"]},
                           label="Hex tiling")
        ordered = sorted(uses, key=lambda t: (not _feeds_base_color(t.node), _feeds_normal_map(t.node), t.node.name))
        ref_weights = None
        for t in ordered:
            is_normal = _feeds_normal_map(t.node)
            name = t.node.label or t.node.name
            closure, consumers = _wrap(p, t.node)
            clos = p.feed(closure, name)
            if tile is None:
                ev = b.evaluate(clos, [("uv", "VECTOR", st)], COLOR_OUTPUTS, label=f"Read: {name}")
                outs = {"Color": ev["color"], "Alpha": ev["alpha"]}
            elif is_normal:
                g = b.group(groups[hextile.NORMAL], {
                    "Source": clos, **{f"Tap {i}": tile.outputs[f"Tap {i}"] for i in (1, 2, 3)},
                    "Angles": tile.outputs["Angles"],
                    "Weights": ref_weights if ref_weights is not None else (1 / 3,) * 3}, label=f"Hex normal: {name}")
                outs = {"Color": g.outputs["Color"], "Alpha": None}
            else:
                ins = {"Source": clos, **{f"Tap {i}": tile.outputs[f"Tap {i}"] for i in (1, 2, 3)},
                       "Bary": tile.outputs["Bary"],
                       "Contrast": hexs["contrast"] if hexs["weights"] == "LUMINANCE" else 0.0,
                       "Exponent": hexs["exponent"], "Gain": hexs["gain"]}
                if ref_weights is not None:
                    ins.update({"Weights In": ref_weights, "Use Weights In": 1.0})
                g = b.group(groups[hextile.SAMPLE], ins, label=f"Hex: {name}")
                outs = {"Color": g.outputs["Color"], "Alpha": g.outputs["Alpha"]}
                if ref_weights is None:
                    ref_weights = g.outputs["Weights"]
            for out_name, targets in consumers.items():
                if targets and outs[out_name] is not None:
                    kind = "COLOR" if out_name == "Color" else "FLOAT"
                    _emit(p, outs[out_name], f"{name} {out_name}", kind, targets, darken, shadow_ptrs)
                    served |= {x.as_pointer() for x in targets}
    return served


def _pom(p: _Patcher, a: Analysis):
    s, b = p.s, p.b
    groups = registry.ensure_groups()
    height_use = next((t for t in a.all_textures if t.node.name == s.height_node), None)
    uvn = b.node("ShaderNodeUVMap", "Base UV", uv_map=s.uv_map)
    tan = b.node("ShaderNodeTangent", "Tangent", direction_type="UV_MAP", uv_map=s.uv_map)
    mapping = height_use.mapping if height_use else None
    spec = SourceSpec(
        march=bpy.data.images[s.march_image],
        shade=bpy.data.images.get(s.shade_image) if s.shade_image else None,
        extension="REPEAT" if s.border_mode == "TILING" else "EXTEND",
        mapping=mapping,
        uniform_scale=_mapping_scale(mapping, p.patch.warnings),
        origin=height_use.node.image.name if height_use else "",
    )
    spec.pre = p.tile
    spec.uniform_scale = b.mul(p.tiling_scale(), spec.uniform_scale)
    if s.hex:
        closure = build_hex_source(b, spec, s.hex, uvn.outputs["UV"], registry.ensure_hex_groups())
    else:
        closure = build_source(b, spec)
    values = s.group_inputs()
    main = groups[registry.MAIN]
    params = {item.name: p.param(item, values[item.name]) for item in main.interface.items_tree
              if item.item_type == "SOCKET" and item.in_out == "INPUT" and item.name in values}
    g = b.group(main, {"Source": closure, "UV": uvn.outputs["UV"], "Tangent": tan.outputs["Tangent"], **params},
                label="POM Shader")
    g.name = PREFIX + "pom"
    g.width = 240
    if s.shadow and s.light_helper and s.light_helper in bpy.data.objects:
        tc = b.node("ShaderNodeTexCoord", "Light helper", object=bpy.data.objects[s.light_helper])
        b.link(b.scale(tc.outputs["Object"], -1.0), g.inputs["Light Vector"])
    return g, uvn.outputs["UV"], height_use


def _normals(p: _Patcher, a: Analysis, height_use: TextureUse | None) -> list:
    pol = p.s.normal_policy
    targets = []
    if pol == "NONE":
        return targets
    for n in a.shaders:
        s_in = n.inputs["Normal"]
        if not s_in.is_linked:
            if pol in {"AUTO", "HEIGHT"}:
                targets.append(s_in)
            continue
        src = s_in.links[0].from_node
        if src.bl_idname == "ShaderNodeBump" and _from_height(src, height_use):
            p.unlink(s_in.links[0])
            targets.append(s_in)
        elif src.bl_idname == "ShaderNodeNormalMap" and pol == "HEIGHT":
            p.unlink(s_in.links[0])
            targets.append(s_in)
    return targets


def _clip_group() -> bpy.types.NodeTree:
    tree = next((t for t in bpy.data.node_groups if t.get("rpom_clip")), None)
    if tree is not None:
        return tree
    tree = bpy.data.node_groups.new("POM Shader Clip", "ShaderNodeTree")
    tree["rpom_clip"] = 1
    tree.description = "POM Shader: generated by the add-on"
    b = Builder(tree, PREFIX)
    shader = b.inp("Shader", "SHADER")
    mask = b.inp("Mask", "FLOAT", 1.0, 0.0, 1.0)
    transp = b.node("ShaderNodeBsdfTransparent")
    mix = b.node("ShaderNodeMixShader")
    b.link(mask, mix.inputs["Fac"])
    b.link(transp.outputs[0], mix.inputs[1])
    b.link(shader, mix.inputs[2])
    b.out("Shader", "SHADER", mix.outputs[0])
    b.layout()
    return tree


def _darken_rest(p: _Patcher, a: Analysis, darken, served: set[int]) -> None:
    for n in a.shaders:
        c_in = n.inputs.get("Base Color") or n.inputs.get("Color")
        if c_in is None or c_in.type != "RGBA" or c_in.as_pointer() in served:
            continue
        if not c_in.is_linked:
            val = p.b.mix(1.0, tuple(c_in.default_value), darken, data_type="RGBA", label="POM Shader: shadow")
            val.node.blend_type = "MULTIPLY"
            p.give(val, f"{n.label or n.name} Color (shadow)", "COLOR", [c_in])
            continue
        lk = c_in.links[0]
        src = lk.from_socket
        p.unlink(lk)
        mix = p.outer.node("ShaderNodeMix", "POM Shader: shadow", data_type="RGBA", blend_type="MULTIPLY")
        sock(mix.inputs, "Factor_Float").default_value = 1.0
        p.nt.links.new(src, sock(mix.inputs, "A_Color"))
        p.nt.links.new(sock(mix.outputs, "Result_Color"), c_in)
        p.give(darken, "Shadow", "FLOAT", [sock(mix.inputs, "B_Color")])
        mix.location = (n.location.x - 200.0, n.location.y + 120.0)


def build(mat: bpy.types.Material, s: Settings | None, hexs: dict | None, default_uv: str = "") -> Patch | None:
    if mat.library is not None:
        raise ValueError(tr("material linked from a library: read only"))
    teardown(mat)
    if s is None and hexs is None:
        return None
    a = analyze(mat)
    if a.output is None:
        raise ValueError(tr("the material has no Material Output"))
    if s is not None:
        s.hex = hexs
    p = _Patcher(mat, s, hexs, default_uv)
    try:
        p.patch.warnings += a.warnings
        anchor = a.shaders[0] if a.shaders else a.output
        darken = None
        if s is not None:
            g, base_uv, height_use = _pom(p, a)
            eval_uv = g.outputs["Parallax UV"]
            normal_targets = _normals(p, a, height_use)
            if s.shadow or s.cavity:
                darken = p.b.mul(g.outputs["Shadow"], g.outputs["Cavity"], label="shadow × cavity")
            uv_name = s.uv_map
        else:
            base_uv = p.b.node("ShaderNodeUVMap", "Base UV", uv_map=hexs.get("uv_map", "")).outputs["UV"]
            eval_uv = base_uv
            uv_name = hexs.get("uv_map", "")
        served = _textures(p, a, base_uv, eval_uv, uv_name, darken)
        if s is not None:
            p.give(g.outputs["Normal"], "Normal", "VECTOR", normal_targets)
            if darken is not None:
                _darken_rest(p, a, darken, served)
            p.give(eval_uv, "Parallax UV", "VECTOR", [])
            p.give(g.outputs["Height"], "Height", "FLOAT", [])
            if s.border_mode == "CLIP" and a.output.inputs["Surface"].is_linked:
                surf = a.output.inputs["Surface"]
                lk = surf.links[0]
                src = lk.from_socket
                p.unlink(lk)
                clip = p.outer.node("ShaderNodeGroup", "POM Shader: clip")
                clip.node_tree = _clip_group()
                p.nt.links.new(src, clip.inputs["Shader"])
                p.nt.links.new(clip.outputs["Shader"], surf)
                p.give(g.outputs["Clip Mask"], "Clip Mask", "FLOAT", [clip.inputs["Mask"]])
                clip.location = (a.output.location.x - 200.0, a.output.location.y)
            if a.displacement_link is not None and s.disconnect_displacement:
                p.unlink(a.displacement_link)
                p.patch.warnings.append(tr("Displacement output disconnected: it would add to the POM"))
        return p.finish(anchor)
    except Exception:
        p.patch.nodes = sorted(set(mat.node_tree.nodes.keys()) - p._before)
        _write_patch(mat, p.patch)
        teardown(mat)
        raise


def apply(mat: bpy.types.Material, s: Settings, default_uv: str = "") -> Patch:
    patch = read_patch(mat)
    hexs = patch.hex if patch is not None else None
    return build(mat, s, hexs, default_uv)
