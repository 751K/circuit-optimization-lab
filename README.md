# Circuit Optimization

[![CI](https://github.com/751K/circuit-optimization-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/751K/circuit-optimization-lab/actions/workflows/ci.yml)
[![Docs](https://github.com/751K/circuit-optimization-lab/actions/workflows/docs.yml/badge.svg)](https://751k.github.io/circuit-optimization-lab/)
[![Version](https://img.shields.io/badge/version-v1.1.0-blue)](CHANGELOG.md)

[English](docs/README.md) | [中文说明](docs/README_zh.md)

A local, Cadence-calibrated framework for analog circuit simulation and ML-driven
design optimization. The open core needs no commercial simulator; optional licensed
foundry models such as TSMC28HPC+ remain local and are never committed.

一个本地、以 Cadence 校准的模拟电路仿真与机器学习驱动设计优化框架。开放核心无需商业仿真器；
TSMC28HPC+ 等可选 foundry 模型保留在本机，不进入 Git。

Full documentation lives in [`docs/README.md`](docs/README.md) (中文见
[`docs/README_zh.md`](docs/README_zh.md)).

```bash
pip install -e .            # solver + CLI
pip install -e ".[demo]"    # + Flask AFE Tuner web app
circuit-opt run examples/periodic_rc.json --analysis ac
```

## Releasing / 发版

1. Keep the release version synchronized in `pyproject.toml`, `frontend/package.json`,
   `frontend/package-lock.json`, `frontend/src-tauri/Cargo.toml`, and
   `frontend/src-tauri/tauri.conf.json`. The installed Python package reads its
   runtime version from the metadata generated from `pyproject.toml`.
2. Move the new items in `CHANGELOG.md` from `[Unreleased]` into a new `[X.Y.Z] - <date>` section.
3. `git commit` the two files, then `git tag vX.Y.Z` and `git push --tags`.
4. The `Release` workflow builds the sdist + wheel and attaches them to a GitHub Release automatically.

The project follows semantic versioning. The public API is the `circuitopt/__init__` export surface,
the circuit JSON format, and CLI flags. PyPI publishing remains deferred until the project is
registered and trusted publishing is configured; tagged releases currently publish artifacts to
GitHub Releases only.

Licensed under the [MIT License](LICENSE).
