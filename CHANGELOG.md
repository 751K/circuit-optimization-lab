# Changelog

All notable changes to this project are documented here.
本项目的所有重要变更都记录在此。

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循
[语义化版本](https://semver.org/lang/zh-CN/)。

在 0.x 阶段，公共 API 定义为 `circuitopt/__init__` 的导出面 + circuit JSON 格式 + CLI flags；
minor 版本引入新能力，patch 版本为修复。See the "Releasing / 发版" section of the root
`README.md` for the release convention.

## [Unreleased]

### Changed
- **破坏性 / BREAKING** — 顶层导入包由泛化的 `core` 更名为 `circuitopt`
  （`import circuitopt`、`python -m circuitopt …`、`python -m circuitopt.calibration` 等）。
  PyPI 分发名（`circuit-optimization`）与 `circuit-opt` 命令行入口不变。此前因包名
  过于通用而推迟的 PyPI 发布随之解锁。
  Top-level import package renamed from the generic `core` to `circuitopt`; the PyPI
  distribution name and the `circuit-opt` console script are unchanged. This unblocks
  public PyPI publishing.

### Added
- **本地服务层 / Local service layer** — `circuitopt/service/` 下的 FastAPI 本地
  HTTP 层，架在既有求解器栈之上，是桌面 GUI 与 MCP server 的共用底座；`serve`
  可选依赖组（`pip install -e ".[serve]"`）与 `circuit-opt serve` / `python -m
  circuitopt.service` 两个等价入口。同步端点 `health` / `capabilities` /
  `validate` / `solve`，与后台任务端点 `jobs/explore` / `jobs/mc`（提交/轮询/
  WebSocket 进度流/取消），均为薄适配，与 CLI 共用同一批入口函数
  （`run_analysis_suite`、`explore_from_dict`、`mismatch_mc_from_dict`），
  语义不会漂移。新增 25 项测试（`tests/test_service.py`、
  `tests/test_service_jobs.py`）。完整参考见
  [Service API](docs/service_api.md)。
  A local FastAPI HTTP layer over the existing solver stack
  (`circuitopt/service/`) — the shared base for a future desktop GUI and MCP
  server; a new `serve` extra (`pip install -e ".[serve]"`) and two equivalent
  entry points (`circuit-opt serve` / `python -m circuitopt.service`).
  Synchronous endpoints (`health` / `capabilities` / `validate` / `solve`) plus
  background-job endpoints (`jobs/explore` / `jobs/mc` — submit/poll/WebSocket
  progress stream/cancel) are thin adapters sharing the same entry points as
  the CLI (`run_analysis_suite`, `explore_from_dict`, `mismatch_mc_from_dict`),
  so semantics can't drift. 25 new tests
  (`tests/test_service.py`, `tests/test_service_jobs.py`). Full reference in
  [Service API](docs/service_api.md).

## [0.1.0] - 2026-07-05

初始公开版本 / Initial public release.

### Added
- **三工艺器件栈 / Three-process device stack** — 通过统一的 `TransistorModel` 接口驱动
  AT4000TG PMOS-OTFT（标定锚点）、SKY130（130 nm，OpenVAF 编译的 BSIM4 经 OSDI 宿主）、
  FreePDK45（45 nm，ngspice-C 作为精确器件求值器）；OTFT 及全部分析无需任何外部工具链。
- **全分析栈 / Full analysis stack** — DC / AC / Noise / Transient 以及 chopper 放大器的
  PSS / PAC / PNoise 周期分析，对标 Cadence Spectre RF；热点路径以 Numba JIT 内核加速。
- **Cadence 校准与 byte-gate / Cadence calibration & byte-gate** — 求解器栈对 Spectre 24.1
  标定（AT4000TG AFE 上增益/带宽/IRN 通常 <1%），`core.calibration --all` 作为可复现的
  漂移硬门禁。
- **数据集 → 代理 → 优化的 ML 闭环 / Dataset → surrogate → optimize ML loop** — 带 provenance
  的带标签数据集生成器（`core/dataset.py`）、GBT/torch 代理（`core/surrogate.py`、
  `surrogate_torch.py`）、以及"代理筛选 + 求解器校验"的优化器（`core/optimize.py`），
  以量级更高的吞吐筛选候选，最终由校准求解器为设计决策把关。
- **统一 API / Unified API** — `CircuitBinding` 把拓扑、尺寸、偏置、器件模型绑定成一次求解器
  调用；JSON 电路格式让新电路无需改动求解器源码，`models` 字段可将具体器件绑定到非默认 PDK。
- **CLI / 命令行** — `circuit-opt` 入口脚本与 `python -m core` 子命令
  （run / explore / corners / mc / chopper / dataset），可被 LLM/agent 从本地 shell 驱动整个设计闭环。
- **工艺角与失配 / Corners & mismatch** — 工艺角扫描、逐器件失配 Monte Carlo、latch 筛查。
- **测试 / Tests** — 359 项测试（含 Cadence 回归与 byte-gate 复现），CI 三作业
  （lint / test 矩阵 / byte-gate）。

[Unreleased]: https://github.com/751K/circuit-optimization-lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/751K/circuit-optimization-lab/releases/tag/v0.1.0
