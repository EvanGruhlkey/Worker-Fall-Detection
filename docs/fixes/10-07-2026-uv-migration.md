# 10-07-2026 - Migrate dependency management to uv

## Problem

Installing the project's dependencies with `pip install -r requirements.txt`
failed inconsistently depending on which Python version a teammate had
installed. The failures always traced back to PyTorch (pulled in by
`ultralytics`), which would either refuse to install or install a build that
crashed at import time.

## Root cause

The development machine is an Intel (x86_64) Mac. PyTorch stopped publishing
macOS x86_64 wheels after **torch 2.2.2** (and **torchvision 0.17.2**). That
release:

- supports Python 3.8 through 3.12 only, so Python 3.13 and 3.14 have no
  installable torch build on Intel Macs;
- was compiled against NumPy 1.x, so it fails at runtime when NumPy 2 is
  installed alongside it.

`requirements.txt` had no Python version pin and no lockfile, so every machine
resolved its own set of versions and only some of them happened to work.

## Fix

Replaced `requirements.txt` with a uv project.

- **`pyproject.toml`**
  - `requires-python = ">=3.12,<3.13"` pins the project to the newest Python
    that PyTorch supports on Intel Macs.
  - Dependencies: `ultralytics`, `opencv-python`, `lap>=0.5.12`, plus
    `numpy<2` restricted by environment marker to `darwin` / `x86_64`.
  - `[tool.uv] package = false` because the project is a set of scripts, not
    an installable package.
  - `[tool.uv] required-environments` lists the Intel Mac platform so the
    lockfile keeps torch 2.2.2 / torchvision 0.17.2 for that platform while
    Apple Silicon, Linux, and Windows resolve to the current torch release
    from the same lockfile.
- **`.python-version`** set to `3.12` so uv picks (or downloads) the right
  interpreter automatically.
- **`uv.lock`** generated and committed so every clone installs identical
  versions.
- **`README.md`** setup and run instructions updated to `uv sync` and
  `uv run`.
- **`.gitignore`** now ignores `__pycache__/`.
- `requirements.txt` removed.

## Resulting versions on Intel macOS

| Package      | Version   |
|--------------|-----------|
| Python       | 3.12.11   |
| torch        | 2.2.2     |
| torchvision  | 0.17.2    |
| numpy        | 1.26.4    |
| opencv-python| 4.11.0.86 |
| ultralytics  | 8.4.174   |
| lap          | 0.5.13    |

Other platforms resolve to torch 2.14.1 / torchvision 0.29.1 from the same
lockfile.

## Verification

- `uv sync` resolved 63 packages with no errors.
- All dependencies import under Python 3.12.11, and tensor-to-NumPy
  conversion works (the failure mode with NumPy 2).
- `models/yolo11n-pose.pt` loads and `process_frame` runs a tracking pass on
  a test frame.
- `uv run src/yolo_pose.py --help` runs.

## How to use going forward

```bash
uv sync                                                  # after cloning or after a pull changes uv.lock
uv run src/yolo_pose.py --model models/yolo11n-pose.pt   # run scripts without activating a venv
uv add <package>                                         # add a dependency; commit pyproject.toml and uv.lock
```

Do not raise the Python ceiling above 3.12 or drop the `numpy<2` marker while
anyone on the team is on an Intel Mac.
