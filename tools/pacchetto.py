# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "pom_shader"
DATA = (1980, 1, 1, 0, 0, 0)


def manifest_value(key: str) -> str:
    text = (SRC / "blender_manifest.toml").read_text(encoding="utf-8")
    m = re.search(rf'^{key} = "([^"]+)"', text, re.M)
    if not m:
        raise SystemExit(f"blender_manifest.toml: missing {key}")
    return m.group(1)


def files() -> list[str]:
    out = []
    for p in SRC.rglob("*"):
        rel = p.relative_to(SRC)
        if not p.is_file() or p.suffix == ".zip" or any(part == "__pycache__" or part.startswith(".") for part in rel.parts):
            continue
        out.append(rel.as_posix())
    return sorted(out)


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest_value('id')}-{manifest_value('version')}.zip"
    tmp = path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for name in files():
            info = zipfile.ZipInfo(name, DATA)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            zf.writestr(info, (SRC / name).read_bytes())
    tmp.replace(path)
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "build"
    print(build(out))
