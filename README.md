# ccf-glass-render

![support](https://img.shields.io/badge/support-unsupported-red)

Glass renders of the Allen Mouse Common Coordinate Framework with single-neuron
reconstructions inside.

![sagittal view](docs/example.png)

The brain is shaded as a transparent dielectric — Fresnel mixing of a reflected
studio environment against a refracted backdrop, with chromatic dispersion,
far-wall contours and procedural surface irregularity. Neurons are drawn as z-buffered 3D tubes composited over it.

## Install

```bash
uv pip install git+https://github.com/peter-grotz/ccf-glass-render.git
```

## Use

Each camera is rendered once
and cached; figures then composite onto it in seconds.

```bash
ccf-render brain --views sagittal,iso --resolution 10

ccf-render cells \
  --asset s3://aind-open-data/exaSPIM_685221_2024-04-12_11-46-38_reconstructions_2026-08-28_23-04-54 \
  --views sagittal,iso --sample 8 --label LC --out figures/
```

`--asset` accepts an S3 reconstruction asset or a local directory and repeats,
so several subjects can share a figure. Each run writes PNG and SVG per view,
plus `provenance.json` recording the input assets, package version and profile
digest.

| Flag | Effect |
|---|---|
| `--compartment axon\|dendrite\|soma` | Restrict to one compartment; dendrite carries the soma |
| `--cell ID` | Named cells, repeatable |
| `--sample N` | Seeded random subset |
| `--resolution` | Pixel size in µm, a multiple of 10 |
| `--colors NAME` | Per-cell colour scheme (below) |
| `--profile FILE` | TOML overriding any shader, palette or geometry constant |

## Colour schemes

`--colors` sets how cells are coloured. Every scheme is chosen to read against
the white glass; pale hues disappear on it.

| Scheme | |
|---|---|
| `vivid` | high-chroma hues around the circle (default) |
| `allen` | Allen Institute brand primaries and accents |
| `okabe-ito` | colour-vision-deficiency safe |
| `tab10`, `set1`, `dark2` | matplotlib default and two ColorBrewer qualitative sets |
| `dark`, `husl`, `viridis`, `turbo` | generated for any cell count |

Fixed schemes cap a figure at their length and raise rather than reusing a hue
on two cells; the generated schemes size themselves to the selection. A profile
may instead give `palette` as explicit hex values, which overrides the scheme.

## Resolution

`--resolution 10` is the native template sampling; 20 µm is ~8× cheaper and
adequate while composing a figure. A 10 µm build rotates 1.2 × 10⁹ voxels. The
occupancy and its rotation (4.8 GB each) are memory-mapped rather than held
resident, which keeps the resample at minutes instead of hours on a 19 GB
machine; it needs ~10 GB of scratch disk, released on completion.

## Reconstruction formats

Both exaSPIM SWC conventions are read, distinguished by inspection rather than
assumption:

- **split** — one file per compartment (`<cell>-axon-<INITIALS>.swc`), type
  column uniformly 0, compartment carried by the filename.
- **combined** — one file per cell with standard SWC type codes.

Reading the type column without this check fails silently: every node in a
split file is "undefined", and a compartment filter returns nothing.

## Atlas

`average_template_10` is fetched from the Allen informatics archive on first
use, cached under `~/.cache/ccf-glass-render` (`CCF_CACHE` overrides), and
shape-checked, since a different atlas or axis order would otherwise yield a
silently misoriented render. Coordinates are CCF microns on array axes
(anterior–posterior, inferior–superior, left–right); midline is z = 5700 µm.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
ruff check . && pytest
```

MIT. Not an officially supported Allen Institute product.
