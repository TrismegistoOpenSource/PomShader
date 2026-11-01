# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from . import panels


def register() -> None:
    panels.register()


def unregister() -> None:
    panels.unregister()
