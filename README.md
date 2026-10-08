# ccf-glass-render

![support](https://img.shields.io/badge/support-unsupported-red)

Volumetric glass renders of the Allen Mouse Common Coordinate Framework, with
single-neuron reconstructions inside.

![sagittal and iso views](docs/example.png)

The brain is shaded as glass — Fresnel mixing of a reflected studio environment
against a refracted backdrop, with chromatic dispersion, far-wall contours and
procedural surface irregularity — rather than as a lit solid. Neurons are drawn
as z-buffered 3D tubes composited over it.

## Install

```bash
uv pip install git+https://github.com/peter-grotz/ccf-glass-render.git
```

## Use

Rendering is two stages, because the glass brain does not depend on which cells
are drawn. Build each camera once, then composite cells onto it in seconds.

```bash
# once per view: ~13 min and ~10 GB of scratch disk at 10 um
ccf-render brain --views sagittal,iso --resolution 10

# per figure: seconds
ccf-render cells \
  --asset s3://aind-open-data/exaSPIM_685221_2024-04-12_11-46-38_reconstructions_2026-08-28_23-04-54 \
  --views sagittal,iso --sample 8 --label LC --out figures/
```

`--asset` takes an S3 reconstruction asset or a local directory, and repeats, so
several subjects can go into one figure. Each run writes PNG + SVG per view and
a `provenance.json` naming the input assets, the package version and the profile
digest — the three things needed to regenerate the figure exactly.

### Options worth knowing

| Flag | Effect |
|---|---|
| `--compartment axon\|dendrite\|soma` | Render one compartment. Dendrite selections carry the soma. |
| `--cell N004-685221 --cell …` | Restrict to named cells. |
| `--sample 8` | Seeded random subset. |
| `--resolution 20` | Coarser and ~8× faster to build; good for iterating. |
| `--profile profiles/vivid.toml` | Swap the whole look. |

## Resolution

`--resolution` is the template downsample, so 10 is native. A 10 µm build
rotates a 1.2-billion-voxel volume; it is memory-mapped rather than held in RAM,
because holding it resident drives a 19 GB machine into swap and the rotation
goes from minutes to hours. Expect ~10 GB of scratch disk, freed on completion.
20 µm needs well under a gigabyte and is the right choice while composing a
figure.

## Reconstruction formats

Two SWC conventions appear in exaSPIM assets and both are read:

- **split** — one file per compartment (`<cell>-axon-<INITIALS>.swc`), with the
  type column uniformly 0; the compartment comes from the filename.
- **combined** — one file per cell with real SWC type codes.

Reading the type column without sniffing the convention is a silent failure: on
a split file every node is "undefined" and a compartment filter returns nothing.

## Atlas

The 10 µm `average_template` is downloaded from the Allen informatics archive on
first use, cached under `~/.cache/ccf-glass-render` (override with `CCF_CACHE`),
and shape-checked — a different atlas or axis order would otherwise produce a
silently misoriented render. Coordinates are CCF microns; array axes are
(anterior–posterior, inferior–superior, left–right), midline at z = 5700 µm.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
ruff check . && pytest
```

MIT licensed. Not an officially supported Allen Institute product.
