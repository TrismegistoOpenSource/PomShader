# POM Shader: user guide

POM Shader adds height-map relief with parallax occlusion mapping to a material, in EEVEE and in Cycles. It
creates only native nodes: the saved file opens and renders the same where the add-on is not installed.

Requires Blender 5.1 or later (tested on 5.1 and 5.2 LTS, on macOS, Windows and Linux).

## Install

Download `pom_shader-<version>.zip` from the releases page and drag it into the Blender window. Or: Edit →
Preferences → Extensions → top-right menu → **Install from Disk…** → pick the zip. The extension turns on by
itself ("POM Shader").

## What the add-on puts in your material

Everything the add-on creates lives in **one node group per material**, named `POM Shader <material>`. In your
shader you only see:

- the **POM Shader** group node, with one output for each texture output it serves (Color, Alpha), plus
  Normal, Parallax UV and Height. Its **Tiling** panel (Scale, Offset, Rotation) sets how often the textures
  repeat, live (the same values as **Texture** → Repeat, Shift, Rotate in the add-on panel); the other settings
  sit in the closed **Settings** panel;
- a **closure zone** around each of your Image Textures. Your textures stay where they are and you can still
  change them: the group reads them where it needs (once with the POM, three times with anti-tiling);
- in **Clip** mode, one small node that makes the surface transparent outside the UV range.

The add-on's own nodes sit in a **POM Shader** frame below your nodes, so they do not mix with yours.

Inside the group, the two images "Height data" and "Slope data" are **computed** from your height. To change the
height, change the image in your height node outside the group, or in **Source** → *Height* (the two always
show the same image; the last one you change wins), then press **Update**: the data is computed again.

**Repeat**, **Shift** and **Rotate** (panel **Texture**) are the same values as the node's Tiling. They apply to every texture the group reads and to the height, on top of your own Mapping node if you have
one; Update and anti-tiling keep it. Press **Tab** on the group node to look inside. **Remove** deletes the group and the zones and puts the material
back exactly as it was.

## Basic use

1. In the **Shader Editor** press `N` and open the **POM Shader** tab on the right.
2. If the material already has the height (an Image Texture with "height", "disp" or "bump" in its name, or
   linked to Displacement or Bump), nothing else is needed. Otherwise pick it in **Texture** → *Height Map*.
   Set the height texture to **Non-Color**.
3. Press **Apply**. The first time the add-on prepares its maps (progress bar, Esc to cancel) and saves them in
   `rpom_cache` next to the `.blend` file: save the file first, so the maps travel with the project.
4. Adjust **Displacement** → *Depth* (in meters) and *Surface Level*: they change at once, without applying again.
5. **Remove** puts the material back exactly as it was.

There is **one Update button** for everything: after you change an option in **Quality**, **Advanced**,
**Shadows** (the light) or **Anti-Tiling**, press **Update**. The sliders (repeat, depth, shadow strength…) change
live.

If the height is linked to the material's Displacement output, Apply disconnects that link (a warning says so):
real displacement and POM would add up.

### Which UV layout (Advanced)

| Case | Mode |
|---|---|
| Repeating texture (floors, walls) | Repeating |
| One UV island in 0..1 | One Island (or Cut at UV Edge for transparent borders) |
| Photogrammetry or baked model, unique UV islands | Unwrapped Atlas |

### Depth from Unreal

**Advanced** → *Depth from Unreal Engine*: enter the Height Ratio and the meters per UV unit, then press the
button on the right to set the depth.

### Displacement in meters

A signed EXR (for example exported by photogrammetry software) is recognized by itself: depth and reference
plane come from the data and the relief is at true scale, as long as the object scale is applied.

## Shadow and cavity

The relief casts shadows from **one light**, and they move with it. *Cast Shadows* is on by default: at Apply
the add-on picks the light that lights the object most and links it with a small helper Empty (a shader cannot
read the scene's lights by itself). **Shadows** shows which light it is; pick another one, then **Update**.
*Darken Crevices* darkens the recessed parts. Both are approximations: they darken the color,
they do not block light.

## Anti-tiling

Tick **Anti-Tiling** and press **Update**: it breaks up the repetition of the textures with hex-tiling (three reads per pixel,
hexagons with random offset and rotation). It works on any material with repeating textures, with or without
the POM. With the POM on, it also acts on the relief, in step with the color.

- *Random Rotation*: 0 for bricks and planks, 1 for soil, gravel, plaster.
- *Patch Size*: higher = smaller patches.
- Normal maps are rotated and blended correctly. If the normal map does not come from a tiling height map,
  rotating it can show a slope: then use the normal from height.

## From photogrammetry

1. Select the **high-poly**, then Ctrl+click the **low-poly** (it must be the active object, with UV islands
   that do not overlap).
2. **POM Shader** tab → **From Photogrammetry**: resolution, extrusion (0 = automatic), method (Cycles is the
   fastest).
3. **Bake and Apply** bakes the height in meters from high to low, marks the texels that cannot be represented
   (undercuts, no hit) and applies the POM to the low with a new material `RPOM_<low name>`.

The files go to `rpom_bake/<object>/` next to the `.blend`: `rpom_disp.exr` (meters), `rpom_bakeflags.exr`,
`bake.json` with all the parameters.

Georeferenced models (millions of meters from the origin): the bake works in a local frame and puts the objects
back where they were.

## Error report

With the high selected and the low (with the POM) active, **Error Report** writes in `rpom_bake/<object>/` a
standalone `rpom_report.html`, plus JSON and CSV.

- **E1**: how well low + height field reproduce the high, in mm, both ways, with a blue-white-red heat map on the
  low.
- **E2**: screen error of the POM against real displacement, for three views. Pixels across a UV seam (close on
  the model, far apart on the atlas) are counted apart and shown in mid gray: they are not shape errors.
- **E3**: share of texels marked by the bake.

The map measures how faithful the light model is to the high-poly model. It does not measure how accurate the
relief is against the real object.

## Limits

- The POM does not move geometry: silhouettes, intersections and cast shadows stay those of the polygon.
- Correct on flat or moderately curved surfaces.
- In atlases the relief fades near the seams.
- Panoramic cameras are not supported.
- In EEVEE the step count is fixed per material.
- In EEVEE, far from the camera, the relief is a little softer than in Cycles: the GPU reads the height at a
  reduced resolution (mip). The difference stays below a pixel and reduces flicker.
- With "Both" the material uses the Cycles step count.
- Seen from the side the effect grows up to **Quality** → *Max Shift at Side Views* (256 px by default) and
  *Side-View Limit* (0.04): raise the first, lower the second for a deeper look at low angles.
