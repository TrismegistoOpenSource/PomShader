# POM Shader

Extension for **Blender 5.1 or later** that adds height-map relief to a material with **parallax occlusion
mapping**, in EEVEE and in Cycles, using **native nodes only**: the saved file opens and renders the same where
the extension is not installed.

- Live relief (depth in meters, reference plane), linear search plus bisection, directional self-shadow and
  cavity.
- Borders: tiling, single island, clip and atlas (the ray never leaves its UV island).
- A clean shader: everything the add-on makes stays in one node group per material; your textures stay outside
  and editable.
- Non-destructive Apply, Update and Remove: Remove puts the material back exactly as it was.
- Hex-tiling anti-repetition of the textures, with or without the POM.
- From photogrammetry: bake of the height in meters from a high-poly to the low-poly model and a metric error
  report (HTML, JSON, CSV).
- Interface in English.

## Install

Download `pom_shader-<version>.zip` from the latest release and drag it into the Blender window, or Preferences
→ Extensions → top-right menu → **Install from Disk…**. One zip for macOS, Windows and Linux.

Each release publishes `SHA256SUMS.txt` with the SHA-256 of the zip and its signature `SHA256SUMS.txt.sig`.

How to use it: [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## License

GPL-3.0-or-later (`LICENSE`).
