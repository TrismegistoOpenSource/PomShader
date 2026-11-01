# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations


def tr(msg: str, *args) -> str:
    return msg.format(*args) if args else msg
