# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import os

import bpy
import numpy as np

from ..utils.i18n import tr

log = logging.getLogger(__name__)


def read_pixels(image: bpy.types.Image) -> np.ndarray:
    w, h = image.size
    if w == 0 or h == 0:
        raise ValueError(tr("empty or not loaded image: {}", image.name))
    c = image.channels
    buf = np.empty(w * h * c, np.float32)
    image.pixels.foreach_get(buf)
    return buf.reshape(h, w, c)


def bit_depth(image: bpy.types.Image) -> int:
    if image.is_float:
        return 32
    per_pixel = image.depth or 32
    return max(8, per_pixel // max(image.channels, 1))


def height_from_pixels(px: np.ndarray, channel: str = "AUTO") -> np.ndarray:
    if px.shape[2] == 1 or channel == "R":
        return px[..., 0]
    if channel in "GBA" and len(channel) == 1:
        return px[..., "RGBA".index(channel)]
    if channel == "LUMINANCE":
        return px[..., 0] * 0.2126 + px[..., 1] * 0.7152 + px[..., 2] * 0.0722
    rgb = px[..., :3]
    if np.allclose(rgb[..., 0], rgb[..., 1], atol=1e-4) and np.allclose(rgb[..., 0], rgb[..., 2], atol=1e-4):
        return px[..., 0]
    return px[..., 0] * 0.2126 + px[..., 1] * 0.7152 + px[..., 2] * 0.0722


def write_exr(path: str, rgba: np.ndarray, half: bool = True) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = np.ascontiguousarray(rgba[::-1], dtype=np.float32)
    try:
        import OpenImageIO as oiio
    except ImportError:
        _write_exr_blender(path, rgba, half)
        return
    h, w, c = data.shape
    spec = oiio.ImageSpec(w, h, c, oiio.HALF if half else oiio.FLOAT)
    spec.attribute("compression", "zip")
    out = oiio.ImageOutput.create(path)
    if out is None or not out.open(path, spec):
        raise OSError(tr("cannot write {}: {}", path, oiio.geterror()))
    ok = out.write_image(data)
    out.close()
    if not ok:
        raise OSError(tr("write failed {}: {}", path, oiio.geterror()))


def _write_exr_blender(path: str, rgba: np.ndarray, half: bool) -> None:
    h, w, _ = rgba.shape
    img = bpy.data.images.new("_rpom_tmp_write", w, h, alpha=True, float_buffer=True)
    try:
        img.colorspace_settings.name = "Non-Color"
        img.alpha_mode = "CHANNEL_PACKED"
        img.pixels.foreach_set(np.ascontiguousarray(rgba, dtype=np.float32).ravel())
        sc = bpy.context.scene
        s = sc.render.image_settings
        old = (s.media_type, s.file_format, s.color_depth, s.color_mode, s.exr_codec)
        s.media_type, s.file_format = "IMAGE", "OPEN_EXR"
        s.color_depth, s.color_mode, s.exr_codec = ("16" if half else "32"), "RGBA", "ZIP"
        try:
            img.save_render(path, scene=sc)
        finally:
            s.media_type, s.file_format, s.color_depth, s.color_mode, s.exr_codec = old
    finally:
        bpy.data.images.remove(img)


def load_data_image(path: str, name: str) -> bpy.types.Image:
    img = bpy.data.images.get(name)
    rel = bpy.path.relpath(path) if bpy.data.filepath else path
    if img is not None and bpy.path.abspath(img.filepath) == os.path.abspath(path):
        img.reload()
    else:
        if img is not None:
            img.name = name + ".old"
        img = bpy.data.images.load(path, check_existing=False)
        img.name = name
        img.filepath = rel
    img.colorspace_settings.name = "Non-Color"
    img.alpha_mode = "CHANNEL_PACKED"
    img["rpom_data_map"] = True
    return img


def check_data_image(img: bpy.types.Image) -> list[str]:
    issues = []
    if img.colorspace_settings.name != "Non-Color":
        issues.append(tr("{}: color space {}, Non-Color needed", img.name, img.colorspace_settings.name))
    if img.alpha_mode != "CHANNEL_PACKED":
        issues.append(tr("{}: alpha {}, Channel Packed needed", img.name, img.alpha_mode))
    if img.packed_file is not None:
        issues.append(tr("{}: packed (the Cycles 5.2 texture cache does not support it)", img.name))
    return issues
