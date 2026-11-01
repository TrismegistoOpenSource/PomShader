# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bpy

from . import apply, bake_pipeline, convert, error_report, light_link, remove, upgrade, uv_frame

_MODULES = [apply, remove, uv_frame, light_link, upgrade, convert, bake_pipeline, error_report]


def _classes():
    for mod in _MODULES:
        yield from getattr(mod, "CLASSES", ())


def register() -> None:
    for c in _classes():
        bpy.utils.register_class(c)
    upgrade.register_handlers()


def unregister() -> None:
    upgrade.unregister_handlers()
    for c in reversed(list(_classes())):
        bpy.utils.unregister_class(c)
