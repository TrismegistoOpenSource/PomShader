# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass

import bpy
import numpy as np

from ..core.maps import ALGORITHM_VERSION
from ..utils.i18n import tr

CACHE_DIRNAME = "rpom_cache"


@dataclass
class CacheEntry:
    key: str
    march_path: str
    shade_path: str
    meta_path: str

    def exists(self) -> bool:
        return all(os.path.exists(p) for p in (self.march_path, self.shade_path, self.meta_path))

    def read_meta(self) -> dict:
        with open(self.meta_path, encoding="utf-8") as f:
            return json.load(f)

    def write_meta(self, meta: dict) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)


def cache_dir(preferred: str = "") -> tuple[str, str | None]:
    if preferred:
        path = bpy.path.abspath(preferred)
        os.makedirs(path, exist_ok=True)
        return path, None
    if bpy.data.filepath:
        path = bpy.path.abspath("//" + CACHE_DIRNAME)
        os.makedirs(path, exist_ok=True)
        return path, None
    try:
        addon = __package__.rpartition(".")[0] if __package__ else "pom_shader"
        path = bpy.utils.extension_path_user(addon, path="cache", create=True)
    except (ValueError, AttributeError, TypeError):
        path = os.path.join(tempfile.gettempdir(), "pom_shader_cache")
        os.makedirs(path, exist_ok=True)
    return path, tr(".blend file not saved: maps go to the user folder, save the file to keep them with the project")


def pixel_checksum(pixels: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(pixels, dtype=np.float32).tobytes()).hexdigest()


def image_identity(image: bpy.types.Image, pixels: np.ndarray) -> dict:
    path = bpy.path.abspath(image.filepath) if image.filepath else ""
    stat = os.stat(path) if path and os.path.exists(path) else None
    return {
        "path": os.path.normcase(os.path.abspath(path)) if path else f"<{image.name}>",
        "mtime": int(stat.st_mtime) if stat else 0,
        "size": int(stat.st_size) if stat else 0,
        "pixels": pixel_checksum(pixels),
    }


def entry_for(directory: str, stem: str, identity: dict, params: dict) -> CacheEntry:
    payload = json.dumps({"id": identity, "params": params, "alg": ALGORITHM_VERSION}, sort_keys=True)
    key = hashlib.sha1(payload.encode()).hexdigest()[:12]
    safe = "".join(ch if ch.isalnum() or ch in "-" else "-" for ch in stem)[:48] or "height"
    base = os.path.join(directory, f"{safe}-{key}")
    return CacheEntry(key, base + "_rpom_march.exr", base + "_rpom_shade.exr", base + "_rpom.json")
