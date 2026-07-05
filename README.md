# Circuit Optimization

[![CI](https://github.com/751K/circuit-optimization-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/751K/circuit-optimization-lab/actions/workflows/ci.yml)
[![Docs](https://github.com/751K/circuit-optimization-lab/actions/workflows/docs.yml/badge.svg)](https://751k.github.io/circuit-optimization-lab/)
[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](CHANGELOG.md)

[English](docs/README.md) | [中文说明](docs/README_zh.md)

A local, license-free, Cadence-calibrated framework for analog circuit simulation and
ML-driven design optimization — the DC/AC/Noise/PSS analysis loop plus a dataset
generator and ML surrogate optimizer, all on the local machine with no commercial
simulator in the loop.

一个本地、无需商业许可、以 Cadence 校准的模拟电路仿真与机器学习驱动设计优化框架 —
在本地完成 DC/AC/Noise/PSS 分析回路，并附带数据集生成器与 ML 代理优化器，全程无需商业仿真器。

Full documentation lives in [`docs/README.md`](docs/README.md) (中文见
[`docs/README_zh.md`](docs/README_zh.md)).

```bash
pip install -e .            # solver + CLI
pip install -e ".[demo]"    # + Flask AFE Tuner web app
circuit-opt run examples/periodic_rc.json --analysis ac
```

## Releasing / 发版

1. Bump `version` in `pyproject.toml` (it is the single source; `core.__version__` reads it back).
2. Move the new items in `CHANGELOG.md` from `[Unreleased]` into a new `[X.Y.Z] - <date>` section.
3. `git commit` the two files, then `git tag vX.Y.Z` and `git push --tags`.
4. The `Release` workflow builds the sdist + wheel and attaches them to a GitHub Release automatically.

Semantic versioning: in the 0.x phase, `minor` = new capability and `patch` = fix. The public API
is the `core/__init__` export surface + the circuit JSON format + CLI flags. (PyPI publishing is
deferred until the generic top-level `core` package is renamed — see `pyproject.toml`.)

Licensed under the [MIT License](LICENSE).
