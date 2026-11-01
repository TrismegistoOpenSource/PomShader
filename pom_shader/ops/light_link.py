# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy


HELPER_KEY = "rpom_light_helper"
SUN_DISTANCE = 10_000.0


def helper_for(light: bpy.types.Object) -> bpy.types.Object | None:
    for ob in bpy.data.objects:
        if ob.get(HELPER_KEY) == light.name and ob.type == "EMPTY":
            return ob
    return None


def _constraints(helper: bpy.types.Object, light: bpy.types.Object) -> None:
    helper.constraints.clear()
    helper.rotation_euler = (0.0, 0.0, 0.0)
    helper.scale = (1.0, 1.0, 1.0)
    if light.data.type == "SUN":
        helper.location = (0.0, 0.0, SUN_DISTANCE)
        c = helper.constraints.new("CHILD_OF")
        c.target = light
        c.set_inverse_pending = False
        c.inverse_matrix.identity()
        lr = helper.constraints.new("LIMIT_ROTATION")
        lr.use_limit_x = lr.use_limit_y = lr.use_limit_z = True
        lr.min_x = lr.max_x = lr.min_y = lr.max_y = lr.min_z = lr.max_z = 0.0
        lr.owner_space = "WORLD"
        ls = helper.constraints.new("LIMIT_SCALE")
        ls.use_min_x = ls.use_min_y = ls.use_min_z = True
        ls.use_max_x = ls.use_max_y = ls.use_max_z = True
        ls.min_x = ls.max_x = ls.min_y = ls.max_y = ls.min_z = ls.max_z = 1.0
        ls.owner_space = "WORLD"
    else:
        helper.location = (0.0, 0.0, 0.0)
        c = helper.constraints.new("COPY_LOCATION")
        c.target = light


def pick_light(objects: list[bpy.types.Object]) -> bpy.types.Object | None:
    import math

    from mathutils import Vector

    if not objects:
        return None
    center = sum((ob.matrix_world @ (sum((Vector(c) for c in ob.bound_box), Vector()) / 8) for ob in objects),
                 Vector()) / len(objects)
    best, best_score = None, 0.0
    for ob in bpy.context.scene.objects:
        if ob.type != "LIGHT" or ob.hide_render:
            continue
        d = ob.data
        if d.type == "SUN":
            score = d.energy
        else:
            to_center = center - ob.matrix_world.translation
            score = d.energy / (4.0 * math.pi * max(to_center.length_squared, 1e-4))
            if d.type == "SPOT":
                axis = ob.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
                if axis.angle(to_center, 0.0) > d.spot_size / 2:
                    score *= 0.05
        if score > best_score:
            best, best_score = ob, score
    return best


def ensure_helper(light: bpy.types.Object) -> bpy.types.Object:
    helper = helper_for(light)
    if helper is None:
        helper = bpy.data.objects.new(f"RPOM_helper_{light.name}", None)
        helper.empty_display_type = "SPHERE"
        helper.empty_display_size = 0.1
        helper.hide_render = True
        helper[HELPER_KEY] = light.name
        cols = light.users_collection or (bpy.context.scene.collection,)
        cols[0].objects.link(helper)
    _constraints(helper, light)
    return helper


CLASSES = ()
