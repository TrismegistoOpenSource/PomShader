# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging

LOGGER_NAME = "pom_shader"


def get(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
