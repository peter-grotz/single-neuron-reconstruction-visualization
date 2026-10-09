# Contributing

```bash
uv venv && uv pip install -e ".[dev]"
uv run ruff check . && uv run pytest
```

Style follows the AIND software practices: 100-character lines, ruff, numpydoc
docstrings, type hints on parameters and returns, functions under 100 lines.

## Verifying a change to the shader

The renderer has no reference images in the repo — they would be large and would
churn. Instead, verify against a view you have already built:

1. Build or keep a cached view (`ccf-render brain --views sagittal --resolution 20`).
2. Render a fixed cell set before and after the change, with the same profile.
3. Diff the PNGs. A refactor that is meant to preserve output should come back
   under ~0.001% of pixels differing by more than 0.02, which is antialiasing.

That is how this package was validated against the scripts it replaced.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`,
`docs:`, `refactor:`, `perf:`, `test:`, `build:`, `ci:`.

## Code Ocean: bump the pin when the library changes

`environment/Dockerfile` installs the renderer from a pinned commit, so a push
updates `code/` but leaves the installed library behind. A run then fails with
an `ImportError` for whatever the new app code uses.

Before pushing to the capsule, set `RENDERER_REF` to the commit being pushed:

```bash
sed -i '' "s/ARG RENDERER_REF=.*/ARG RENDERER_REF=$(git rev-parse HEAD)/" environment/Dockerfile
```

The pin is deliberate: a released capsule should name exactly the code it ran.
