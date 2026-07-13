# Changelog

All notable changes to this project are documented here.
本项目的所有重要变更都记录在此。

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循
[语义化版本](https://semver.org/lang/zh-CN/)。

公共 API 定义为 `circuitopt/__init__` 的导出面、circuit JSON 格式与 CLI flags。
版本遵循 Semantic Versioning；发布约定见根目录 `README.md` 的 "Releasing / 发版" 一节。

## [Unreleased]

## [1.1.0] - 2026-07-13

### Added

#### Process Integration / 工艺接入
- **TSMC28HPC+ local adapter / 本地工艺适配** — Added the generic
  `NgspiceProcessAdapter` boundary and registered `tsmc28hpcp.nmos` /
  `tsmc28hpcp.pmos` for the licensed 0.9 V `nch_mac` / `pch_mac` core wrappers.
  Supports tt/ss/ff/sf/fs corners, temperature, native `NF`, hierarchical `.op`,
  cached DC/AC/noise characterization, full-deck `.tran`/ `.ac`/ `.noise`, and
  per-instance `_delvto`. The licensed model payload stays local and Git-ignored.
- **Full-circuit ngspice oracles and PVT / 全电路 oracle 与 PVT** — Added shared
  `ac_ngspice`, `noise_ngspice`, `op_ngspice`, and `loop_gain_ngspice` paths,
  plus FreePDK45 mixed `sf`/`fs` corners and strict corner validation. AC, noise,
  operating-region checks, transient, temperature, corner, and supply now use one
  common deck renderer.

#### ADC Design Cases / ADC 设计案例
- **14-bit pipeline-ADC MDAC OTA / 14-bit 流水线 ADC MDAC OTA** — Added a
  fully-differential two-stage FreePDK45 OTA and six generated testbenches for
  residue settling, open-loop AC, differential/CMFB loop gain, and noise. All 11
  mini-PVT points pass; see `docs/mdac_ota_derivation.md` and
  `tests/test_mdac_ota.py`.
- **6-bit differential SAR ADC / 6-bit 差分 SAR ADC** — Added a common-mode-switching
  CDAC with a clocked StrongARM comparator and backward-compatible `adc.clock`
  strobes. All 64 code centers pass at nom/ss/ff; measured SNDR 36.9 dB, ENOB 5.84
  bit, and SFDR 44.1 dB. See `docs/freepdk45_sar_design.md`.

#### ADC Tooling / ADC 工具链
- **SAR plotting and CLI / SAR 出图与 CLI** — Added transfer/DNL/INL, spectrum,
  conversion-timeline, and mismatch-MC plots; `circuit-opt adc` gains `--plot`
  and `--mc` modes.
- **SAR mismatch Monte Carlo / SAR 失配蒙特卡洛** — Added Pelgrom-scaled transistor
  Vth mismatch, CDAC capacitor mismatch, yield summaries, and the optional
  `adc.mismatch` JSON block.
- **SAR design-space exploration / SAR 设计空间探索** — Added capacitor and MOS
  geometry variables, static/dynamic ADC objectives, Pareto/feasibility output,
  CSV/JSONL export, and `circuit-opt adc --explore`.

#### Documentation / 文档
- Added English and Chinese TSMC28HPC+ setup guides, portable model-entry rules,
  JSON binding reference, architecture notes, ngspice-oracle coverage, verification
  matrix, and explicit foundry license/NDA boundaries.

### Changed
- **ngspice transient options / 瞬态容差直通** — `transient_ngspice` and the
  renderer accept `extra_options` (for example tighter `reltol`/`vntol`/`abstol`)
  while preserving byte-identical default decks.
- **Parallel SAR conversions / SAR 转换并行化** — Added deterministic, ordered
  `workers` support to SAR sweeps, signal runs, mismatch MC, and exploration.
  Per-bit decisions remain serial; independent conversions run concurrently.

### Fixed
- **Circuit JSON schema completion / JSON schema 补全** — Added the previously
  supported `vcvs`, `cccs`, and `ccvs` blocks plus `adc.clock` and
  `adc.mismatch`, preventing valid circuits from being rejected by schema checks.

## [1.0.5] - 2026-07-13

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
- **桌面 / 浏览器电路编辑器 / Desktop & browser circuit editor** — `frontend/` 下的
  React + React Flow 画布编辑器，配 Tauri 桌面壳。在画布上绘制电路、对本地 service
  实时 `validate`/`solve`、运行分析并查看 Bode / noise / transient 图；电路 JSON 是
  唯一真源，编辑器对其无损往返（round-trip）。离线可用，仅 `validate`/`solve` 需要后端。
  A React + React Flow canvas editor (`frontend/`) with a Tauri desktop shell: draw a
  circuit, live-`validate`/`solve` against the local service, run analyses and view
  Bode / noise / transient plots. The circuit JSON is the single source of truth and
  round-trips losslessly; the editor is usable offline, only `validate`/`solve` need
  the backend.
- **晶体管级 ADC / SAR 工作流 / Transistor-level ADC & SAR workflow** — `circuitopt/adc.py`
  与 `circuitopt/sar.py`：由全电荷 transient 仿真驱动的闭环 SAR 转换，输出静态（INL/DNL）
  与动态（SNDR/ENOB）性能指标。新增 `circuit-opt adc` 子命令、电路 JSON 的 `adc` 配置块
  （含 schema）、`examples/freepdk45_sar3.json` 示例，以及 `tests/test_adc.py` /
  `tests/test_sar.py`。
  `circuitopt/adc.py` + `circuitopt/sar.py`: closed-loop SAR conversion driven by
  full-charge transient simulations, yielding static (INL/DNL) and dynamic (SNDR/ENOB)
  metrics. Adds a `circuit-opt adc` subcommand, an `adc` circuit-JSON block (with
  schema), the `examples/freepdk45_sar3.json` example, and new tests.
- **FreePDK45 全电路 ngspice 瞬态后端 / FreePDK45 full-circuit ngspice transient backend**
  — `circuitopt/ngspice_transient.py` 将完整 `Topology` 渲染为原始 model card 的 `.tran`
  网表，以 ngspice 作为 FreePDK45 大信号 oracle（快速 `ngspice_device` 适配器只存 DC/小信号
  栅格，不含四端 BSIM4 电荷态），结果映射回 circuitopt 标准 transient 结果结构。
  `circuitopt/ngspice_transient.py` renders the full `Topology` to a `.tran` netlist
  with the original model cards, keeping ngspice as the FreePDK45 large-signal oracle,
  and maps the waveforms back to circuitopt's standard transient result shape.
- **`plot` 子命令 / `plot` subcommand** — `circuit-opt plot` 将 transient 波形与 AC/PAC
  Bode 图渲染为 PNG。 Renders transient waveforms and AC/PAC Bode plots to PNG.
- **SLiCAP 符号分析技能 / SLiCAP symbolic-analysis skill** — 用于从 SPICE 类网表推导传递函数、
  极零点与设计方程的符号分析工具链（`.agents/skills/slicap/`、`tools/slicap/`）。

### Changed
- **破坏性 / BREAKING** — 顶层导入包由泛化的 `core` 更名为 `circuitopt`
  （`import circuitopt`、`python -m circuitopt …`、`python -m circuitopt.calibration` 等）。
  PyPI 分发名（`circuit-optimization`）与 `circuit-opt` 命令行入口不变。此前因包名
  过于通用而推迟的 PyPI 发布随之解锁。
  Top-level import package renamed from the generic `core` to `circuitopt`; the PyPI
  distribution name and the `circuit-opt` console script are unchanged. This unblocks
  public PyPI publishing.
- **工具链可移植性 / Toolchain portability** — 新增 `circuitopt/toolchain.py`，从显式环境变量
  （`NGSPICE_BIN` / `PDK_ROOT`）、当前/项目 virtualenv、再到 `PATH` 依次解析可选的 ngspice
  二进制与 PDK 安装，移除硬编码本地路径，使 OSDI/ngspice 相关分析在他人机器上可开箱运行。
  A new `circuitopt/toolchain.py` resolves optional ngspice binaries and PDK installs
  from explicit env vars, the active/project virtualenv, then `PATH` — removing
  hardcoded local paths so OSDI/ngspice-backed analyses run portably.
- **测试增长 / Test growth** — 测试数由 0.1.0 的 359 增至约 400（新增 service、ADC/SAR、
  FreePDK45 transient、toolchain 等覆盖）。
  Test count grew from 359 (0.1.0) to ~400, covering the service layer, ADC/SAR,
  FreePDK45 transient, and toolchain resolution.

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

[Unreleased]: https://github.com/751K/circuit-optimization-lab/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.0.5...v1.1.0
[1.0.5]: https://github.com/751K/circuit-optimization-lab/compare/v0.1.0...v1.0.5
[0.1.0]: https://github.com/751K/circuit-optimization-lab/releases/tag/v0.1.0
