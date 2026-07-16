# Changelog / 更新日志

All notable changes to this project are documented in this file.

本文件记录项目的所有重要变更。

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

本文档格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

The public API includes the exports from `circuitopt/__init__.py`, the circuit
JSON format, and CLI flags. See [Development](docs/development.md) for the
release checklist.

公共 API 包括 `circuitopt/__init__.py` 的导出接口、电路 JSON 格式和命令行参数。
发布检查流程见[开发指南](docs/development.md)。

## [Unreleased] / 未发布

## [1.2.0] - 2026-07-17

### Added / 新增

- **Native TSMC28 BSIM4 simulation / 原生 TSMC28 BSIM4 仿真**

  **English:** Added an internal HSPICE frontend that resolves `.lib` and
  `.include` closures, parameter expressions, foundry MOS macros, and model
  bins. A bundled Berkeley BSIM4.5 backend now evaluates four-terminal
  currents, charges, conductance, capacitance, and correlated noise for the
  default `tsmc28hpcp.nmos` and `tsmc28hpcp.pmos` models without launching
  ngspice. The native library is compiled and cached on first use; macOS and
  Linux require a C99 compiler selected through `BSIM4_CC`, `CC`, or `PATH`.

  **中文：** 新增内部 HSPICE 前端，可解析 `.lib`、`.include` 的递归依赖、参数
  表达式、代工厂 MOS 宏模型和模型分档。默认的 `tsmc28hpcp.nmos` 与
  `tsmc28hpcp.pmos` 现由内置 Berkeley BSIM4.5 后端计算四端电流、电荷、电导、
  电容和相关噪声，不再需要启动 ngspice。原生库会在首次使用时编译并缓存；
  macOS 和 Linux 需要可通过 `BSIM4_CC`、`CC` 或 `PATH` 找到的 C99 编译器。

- **TSMC28 5T OTA cross-check / TSMC28 五管 OTA 交叉验证**

  **English:** Added `examples/tsmc28hpcp_5t_ota.json`,
  `experiments/tsmc28_5t_ota_compare.py`, and regression tests that compare
  device `Id/gm/gds`, differential AC response, integrated output noise from
  1 kHz to 10 GHz, and a 2 mV differential-step transient against the explicit
  ngspice oracle.

  **中文：** 新增 `examples/tsmc28hpcp_5t_ota.json`、
  `experiments/tsmc28_5t_ota_compare.py` 和对应回归测试，以显式 ngspice
  oracle 为基准，对比器件 `Id/gm/gds`、差分 AC 响应、1 kHz 至 10 GHz
  积分输出噪声，以及 2 mV 差分阶跃瞬态。

- **TSMC28HPC+ pipeline-MDAC OTA / TSMC28HPC+ 流水线 MDAC OTA**

  **English:** Added a fully transistorized, fully differential OTA powered
  from one 20 uA reference current, together with generated open-loop,
  differential-loop, two-CMFB-loop, closed-loop noise, five-level residue, and
  split-CDAC `0111 -> 1000` testbenches. A resumable 45-point foundry-model PVT
  campaign driver and an ADC-to-OTA design record are included. A complete
  45-point result set is not versioned or claimed for this release.

  **中文：** 新增仅由一个 20 uA 参考电流供电的全晶体管、全差分 OTA，以及自动
  生成的开环、差模环路、双 CMFB 环路、闭环噪声、五级 residue 和分裂 CDAC
  `0111 -> 1000` 测试台。同时提供可断点续跑的 45 点代工厂模型 PVT 驱动器和
  ADC 到 OTA 的设计记录。本次发布尚未纳入或宣称完整的 45 点结果集。

- **Parallel device multiplicity / 并联器件倍乘**

  **English:** Circuit JSON device objects now accept SPICE-style `M >= 1`
  parallel-instance multiplicity independently of `NF`. The loader stores the
  value in `Topology.device_mult`, and supported native and full-circuit
  ngspice paths preserve it during device construction or as rendered `m=`
  parameters.

  **中文：** 电路 JSON 的器件对象现支持独立于 `NF` 的 SPICE 风格 `M >= 1`
  并联实例倍乘。加载器将其保存到 `Topology.device_mult`；受支持的原生路径和
  全电路 ngspice 路径会在器件构造时或渲染为 `m=` 参数时保留该值。

- **Third-party licensing index / 第三方许可证索引**

  **English:** Added a prominent bilingual third-party notice covering the
  vendored UC Berkeley BSIM4.5.0 equations, ngspice compatibility sources,
  CircuitOpt adapter modifications, and the boundary for licensed foundry
  models. Linked the notice from the repository README, documentation site,
  and package metadata, and included it in source and wheel distributions.

  **中文：** 新增醒目的双语第三方软件声明，集中说明仓库内 UC Berkeley
  BSIM4.5.0 方程、ngspice 兼容源码、CircuitOpt 适配修改，以及受许可代工厂
  模型的边界。该声明已从仓库 README、文档站和包元数据建立入口，并随源码包和
  wheel 分发。

### Changed / 变更

- **Documentation reorganization / 文档重组**

  **English:** Reorganized `docs/` into maintained paths for getting started,
  PDK integration, architecture, design records, and developer handoff.
  Simplified the root README, replaced oversized overview and CLI pages with
  navigable references, removed the completed native-BSIM implementation plan
  and stale roadmap, and clarified the actual coverage of partial MDAC PVT
  campaigns.

  **中文：** 重组 `docs/`，建立面向快速入门、PDK 接入、系统架构、设计记录和
  开发交接的维护路径。精简根目录 README，将过大的概览和 CLI 页面改为可导航
  的参考文档，删除已完成的原生 BSIM 实施计划和过时路线图，并明确标注部分
  MDAC PVT campaign 的实际覆盖范围。

- **TSMC28 MDAC C1 sizing update / TSMC28 MDAC C1 尺寸更新**

  **English:** Regenerated all TSMC28 MDAC testbenches with the iteration-C1
  device and compensation values, including parallel `M9/M10`, a shorter
  second-stage channel length, and an updated nulling resistor. Structural
  tests now verify the multiplicity mechanism instead of freezing one
  optimization iteration's transistor widths.

  **中文：** 使用 C1 迭代的器件和补偿参数重新生成全部 TSMC28 MDAC 测试台，
  包括并联的 `M9/M10`、更短的第二级沟道长度和更新后的调零电阻。结构测试现
  验证倍乘机制，不再固化某一次优化迭代的晶体管宽度。

- **TSMC28 default backend / TSMC28 默认后端**

  **English:** `tsmc28hpcp.*` now selects the native BSIM4 implementation.
  The subprocess-backed implementation remains available as
  `tsmc28hpcp_ngspice.*` for independent oracle comparisons.

  **中文：** `tsmc28hpcp.*` 现默认选择原生 BSIM4 实现。基于子进程的原实现仍
  以 `tsmc28hpcp_ngspice.*` 保留，用于独立 oracle 对比。

- **Full-terminal periodic analysis / 完整端口周期分析**

  **English:** PSS and PAC now use native four-terminal conductance and charge
  linearization. PNoise folds the full Hermitian terminal-noise covariance,
  preserves cross-terminal correlation, and extracts the foundry model's
  flicker-noise exponent instead of assuming exact `1/f` behavior.

  **中文：** PSS 和 PAC 现使用原生四端电导与电荷线性化。PNoise 会折叠完整的
  Hermitian 端口噪声协方差，保留跨端口相关性，并从代工厂模型中提取闪烁噪声
  指数，不再假设严格的 `1/f` 特性。

- **Chained same-process ngspice analyses / 同进程串联 ngspice 分析**

  **English:** Added same-topology analysis chaining to amortize foundry-macro
  parsing. `loop_gain_tian_ngspice` combines voltage- and current-injection
  sweeps, `transient_ngspice_chain` runs input-only variants after one parse,
  and the PVT campaign combines open-loop `.ac` with power and saturation
  `.op`. A measured TSMC28 MDAC PVT point drops from 15 to 7 ngspice processes
  and from 28.8 to 13.3 minutes, with bit-identical chained results.
  `CIRCUITOPT_NGSPICE_CHAIN=0` restores the previous behavior.

  **中文：** 新增同拓扑分析串联机制，以分摊代工厂宏模型的解析开销。
  `loop_gain_tian_ngspice` 合并电压和电流注入扫描，
  `transient_ngspice_chain` 在一次解析后运行仅输入不同的多个变体，PVT campaign
  则合并开环 `.ac` 与功耗、饱和区检查所需的 `.op`。实测单个 TSMC28 MDAC
  PVT 点从 15 个 ngspice 进程降至 7 个，耗时从 28.8 分钟降至 13.3 分钟，
  串联结果与逐进程路径逐位一致。设置 `CIRCUITOPT_NGSPICE_CHAIN=0` 可恢复原行为。

- **Transient operating-point vectors / 瞬态工作点向量**

  **English:** `transient_ngspice` can optionally return per-device `vds`,
  `vgs`, `vdsat`, `id`, `gm`, and `gds` waveforms and final values. Saturation
  can therefore be checked at the actual end of a charge-transfer transient
  instead of through a replacement DC solve.

  **中文：** `transient_ngspice` 现可选返回每个器件的 `vds`、`vgs`、`vdsat`、
  `id`、`gm` 和 `gds` 波形及终值，因此可在电荷转移瞬态的真实结束时刻检查
  饱和状态，而不必使用替代性的 DC 求解。

- **Frontend chart dependency / 前端图表依赖**

  **English:** Upgraded Apache ECharts to 6.1.0. The production dependency
  audit now reports zero known vulnerabilities.

  **中文：** 将 Apache ECharts 升级到 6.1.0，前端生产依赖审计现无已知漏洞。

### Fixed / 修复

- **Frontend result module tracking / 前端结果模块跟踪**

  **English:** Scoped the generated-result ignore rule to the repository root
  so it no longer hides `frontend/src/results/`. Restored the result panel's
  AC/PAC, noise, transient/PSS plotting, scalar metrics, JSON tree, and JSON
  download module, with transform regression tests. Partial resumable MDAC PVT
  CSV files now skip the 45-point sign-off gate until the campaign is complete.

  **中文：** 将生成结果目录的忽略规则限定在仓库根目录，避免继续误伤
  `frontend/src/results/`。恢复结果面板的 AC/PAC、噪声、transient/PSS 曲线、
  标量指标、JSON 树和 JSON 下载模块，并补充转换逻辑回归测试。可断点续跑的
  MDAC PVT CSV 在未满 45 点时会跳过签核门禁，完成后才执行完整断言。

## [1.1.0] - 2026-07-13

### Added / 新增

- **TSMC28HPC+ local adapter / TSMC28HPC+ 本地适配器**

  **English:** Added the generic `NgspiceProcessAdapter` boundary and
  registered `tsmc28hpcp.nmos` and `tsmc28hpcp.pmos` for licensed 0.9 V
  `nch_mac` and `pch_mac` core wrappers. The adapter supports TT, SS, FF, SF,
  and FS corners, temperature, native `NF`, hierarchical `.op`, cached
  DC/AC/noise characterization, full-deck transient/AC/noise analyses, and
  per-instance `_delvto`. Licensed model payloads remain local and Git-ignored.

  **中文：** 新增通用 `NgspiceProcessAdapter` 边界，并为受许可的 0.9 V
  `nch_mac` 和 `pch_mac` 核心封装注册 `tsmc28hpcp.nmos` 与
  `tsmc28hpcp.pmos`。适配器支持 TT、SS、FF、SF、FS 工艺角、温度、原生
  `NF`、层级 `.op`、缓存的 DC/AC/噪声表征、完整网表瞬态/AC/噪声分析，以及
  逐实例 `_delvto`。受许可模型文件仅保留在本地并由 Git 忽略。

- **Full-circuit ngspice oracles and PVT / 全电路 ngspice oracle 与 PVT**

  **English:** Added shared `ac_ngspice`, `noise_ngspice`, `op_ngspice`, and
  `loop_gain_ngspice` paths, together with FreePDK45 mixed SF/FS corners and
  strict corner validation. AC, noise, operating-region checks, transient,
  temperature, process corner, and supply now share one deck renderer.

  **中文：** 新增共享的 `ac_ngspice`、`noise_ngspice`、`op_ngspice` 和
  `loop_gain_ngspice` 路径，同时补充 FreePDK45 混合 SF/FS 工艺角与严格的
  corner 校验。AC、噪声、工作区检查、瞬态、温度、工艺角和电源扫描现共用同一
  网表渲染器。

- **14-bit pipeline-ADC MDAC OTA / 14 位流水线 ADC MDAC OTA**

  **English:** Added a fully differential two-stage FreePDK45 OTA and six
  generated testbenches for residue settling, open-loop AC,
  differential/CMFB loop gain, and noise. All 11 mini-PVT points pass the
  recorded checks.

  **中文：** 新增全差分两级 FreePDK45 OTA，以及六个用于 residue 建立、开环
  AC、差模/CMFB 环路增益和噪声的自动生成测试台。记录中的 11 个 mini-PVT
  点均通过检查。

- **6-bit differential SAR ADC / 6 位差分 SAR ADC**

  **English:** Added a common-mode-switching CDAC with a clocked StrongARM
  comparator and backward-compatible `adc.clock` strobes. All 64 code centers
  pass at nominal, SS, and FF corners; the recorded result is 36.9 dB SNDR,
  5.84-bit ENOB, and 44.1 dB SFDR.

  **中文：** 新增采用共模切换 CDAC 和时钟控制 StrongARM 比较器的 6 位差分
  SAR ADC，并提供向后兼容的 `adc.clock` 选通信号。64 个码中心在 nominal、
  SS 和 FF 工艺角均通过；记录结果为 36.9 dB SNDR、5.84 bit ENOB 和
  44.1 dB SFDR。

- **SAR plotting and CLI / SAR 绘图与命令行**

  **English:** Added transfer, DNL/INL, spectrum, conversion-timeline, and
  mismatch Monte Carlo plots. `circuit-opt adc` gained `--plot` and `--mc`
  modes.

  **中文：** 新增传输曲线、DNL/INL、频谱、转换时间线和失配蒙特卡洛图；
  `circuit-opt adc` 新增 `--plot` 与 `--mc` 模式。

- **SAR mismatch Monte Carlo / SAR 失配蒙特卡洛**

  **English:** Added Pelgrom-scaled transistor threshold mismatch, CDAC
  capacitor mismatch, yield summaries, and the optional `adc.mismatch` JSON
  block.

  **中文：** 新增按 Pelgrom 模型缩放的晶体管阈值失配、CDAC 电容失配、良率
  汇总，以及可选的 `adc.mismatch` JSON 配置块。

- **SAR design-space exploration / SAR 设计空间探索**

  **English:** Added capacitor and MOS geometry variables, static and dynamic
  ADC objectives, Pareto and feasibility output, CSV/JSONL export, and
  `circuit-opt adc --explore`.

  **中文：** 新增电容与 MOS 几何尺寸变量、ADC 静态和动态目标、Pareto 与可行性
  输出、CSV/JSONL 导出，以及 `circuit-opt adc --explore`。

- **TSMC28 integration documentation / TSMC28 接入文档**

  **English:** Added English and Chinese setup guides, portable model-entry
  rules, JSON binding references, architecture notes, ngspice-oracle coverage,
  a verification matrix, and explicit foundry license and NDA boundaries.

  **中文：** 新增中英文安装指南、可迁移模型入口规范、JSON 绑定参考、架构说明、
  ngspice oracle 覆盖范围、验证矩阵，以及明确的代工厂许可与 NDA 边界。

### Changed / 变更

- **ngspice transient options / ngspice 瞬态选项**

  **English:** `transient_ngspice` and the renderer now accept
  `extra_options`, such as tighter `reltol`, `vntol`, and `abstol`, while
  preserving byte-identical default decks.

  **中文：** `transient_ngspice` 和网表渲染器现支持 `extra_options`，例如更严格的
  `reltol`、`vntol` 和 `abstol`，同时保持默认网表逐字节不变。

- **Parallel SAR conversions / SAR 转换并行化**

  **English:** Added deterministic and ordered `workers` support to SAR
  sweeps, signal runs, mismatch Monte Carlo, and exploration. Per-bit
  decisions remain serial, while independent conversions run concurrently.

  **中文：** 为 SAR 扫描、信号仿真、失配蒙特卡洛和设计空间探索新增确定性且有序
  的 `workers` 支持。单次转换内的逐位判决仍保持串行，彼此独立的转换可并发运行。

### Fixed / 修复

- **Circuit JSON schema completion / 电路 JSON schema 补全**

  **English:** Added the already-supported `vcvs`, `cccs`, and `ccvs` blocks,
  together with `adc.clock` and `adc.mismatch`, preventing valid circuits from
  being rejected during schema validation.

  **中文：** 在 schema 中补充已受支持的 `vcvs`、`cccs`、`ccvs` 配置块，以及
  `adc.clock` 和 `adc.mismatch`，避免合法电路在 schema 校验阶段被拒绝。

## [1.0.5] - 2026-07-13

### Added / 新增

- **Local service layer / 本地服务层**

  **English:** Added a FastAPI HTTP layer in `circuitopt/service/` over the
  existing solver stack, serving as the shared backend for the desktop GUI and
  MCP server. The optional `serve` dependency group and the equivalent
  `circuit-opt serve` and `python -m circuitopt.service` entry points expose
  synchronous health, capability, validation, and solve endpoints, plus
  background exploration and mismatch jobs with polling, WebSocket progress,
  and cancellation.

  **中文：** 在既有求解器栈之上新增位于 `circuitopt/service/` 的 FastAPI HTTP
  服务层，作为桌面 GUI 和 MCP server 的共享后端。可选的 `serve` 依赖组，以及
  等价的 `circuit-opt serve` 和 `python -m circuitopt.service` 入口，提供同步的
  健康检查、能力查询、校验和求解接口，并支持带轮询、WebSocket 进度和取消功能
  的后台探索与失配任务。

- **Desktop and browser circuit editor / 桌面与浏览器电路编辑器**

  **English:** Added a React and React Flow canvas editor in `frontend/` with
  a Tauri desktop shell. It draws circuits, validates and solves them through
  the local service, runs analyses, and displays Bode, noise, and transient
  plots. Circuit JSON remains the losslessly round-tripped source of truth.

  **中文：** 在 `frontend/` 新增基于 React 和 React Flow 的画布编辑器，并提供
  Tauri 桌面壳。编辑器可绘制电路，通过本地服务完成校验和求解，运行分析并显示
  Bode、噪声和瞬态图。电路 JSON 仍是可无损往返的唯一数据源。

- **Transistor-level ADC and SAR workflow / 晶体管级 ADC 与 SAR 工作流**

  **English:** Added closed-loop SAR conversion driven by full-charge
  transient simulation in `circuitopt/adc.py` and `circuitopt/sar.py`, with
  static DNL/INL and dynamic SNDR/ENOB metrics. Added the `circuit-opt adc`
  command, the circuit JSON `adc` block, schema support, and a FreePDK45 SAR
  example.

  **中文：** 在 `circuitopt/adc.py` 和 `circuitopt/sar.py` 中新增由完整电荷瞬态
  仿真驱动的闭环 SAR 转换，可输出 DNL/INL 静态指标和 SNDR/ENOB 动态指标。
  同时新增 `circuit-opt adc` 命令、电路 JSON 的 `adc` 配置块、schema 支持和
  FreePDK45 SAR 示例。

- **FreePDK45 full-circuit ngspice transient backend / FreePDK45 全电路 ngspice 瞬态后端**

  **English:** Added `circuitopt/ngspice_transient.py` to render a complete
  `Topology` as a `.tran` deck using the original model cards, execute
  ngspice as the FreePDK45 large-signal oracle, and map waveforms back into
  circuitopt's standard transient result structure.

  **中文：** 新增 `circuitopt/ngspice_transient.py`，将完整 `Topology` 使用原始
  model card 渲染为 `.tran` 网表，以 ngspice 作为 FreePDK45 大信号 oracle，
  并将波形映射回 circuitopt 标准瞬态结果结构。

- **Plot command / 绘图命令**

  **English:** Added `circuit-opt plot` for rendering transient waveforms and
  AC/PAC Bode plots as PNG files.

  **中文：** 新增 `circuit-opt plot`，可将瞬态波形和 AC/PAC Bode 图渲染为
  PNG 文件。

- **SLiCAP symbolic-analysis skill / SLiCAP 符号分析技能**

  **English:** Added a symbolic-analysis workflow for deriving transfer
  functions, poles, zeros, and design equations from SPICE-like netlists.

  **中文：** 新增从 SPICE 类网表推导传递函数、极点、零点和设计方程的符号分析
  工作流。

### Changed / 变更

- **Breaking package rename / 破坏性包名变更**

  **English:** Renamed the top-level import package from the generic `core` to
  `circuitopt`. The PyPI distribution name `circuit-optimization` and the
  `circuit-opt` console command remain unchanged.

  **中文：** 顶层导入包由通用名称 `core` 更名为 `circuitopt`。PyPI 分发名
  `circuit-optimization` 和 `circuit-opt` 命令行入口保持不变。

- **Toolchain portability / 工具链可迁移性**

  **English:** Added `circuitopt/toolchain.py` to resolve optional ngspice
  binaries and PDK installations from explicit environment variables, the
  active or project virtual environment, and then `PATH`, removing hard-coded
  local paths.

  **中文：** 新增 `circuitopt/toolchain.py`，依次从显式环境变量、当前或项目虚拟
  环境以及 `PATH` 解析可选的 ngspice 二进制和 PDK 安装位置，移除硬编码本地路径。

- **Test growth / 测试增长**

  **English:** Expanded the suite from 359 tests in v0.1.0 to about 400,
  covering the service layer, ADC/SAR workflows, FreePDK45 transient
  simulation, and toolchain resolution.

  **中文：** 测试数量从 v0.1.0 的 359 项增长至约 400 项，新增服务层、ADC/SAR
  工作流、FreePDK45 瞬态仿真和工具链解析等覆盖。

## [0.1.0] - 2026-07-05

Initial public release.

初始公开版本。

### Added / 新增

- **Three-process device stack / 三工艺器件栈**

  **English:** Added a unified `TransistorModel` interface for the AT4000TG
  PMOS OTFT calibration anchor, SKY130 BSIM4 through an OpenVAF-compiled OSDI
  host, and FreePDK45 using ngspice-C as an accurate device evaluator. OTFT
  simulation and the general analysis stack require no external toolchain.

  **中文：** 新增统一的 `TransistorModel` 接口，支持作为标定锚点的 AT4000TG
  PMOS OTFT、通过 OpenVAF 编译 OSDI 宿主运行的 SKY130 BSIM4，以及使用
  ngspice-C 进行精确器件求值的 FreePDK45。OTFT 仿真和通用分析栈无需外部
  工具链。

- **Full analysis stack / 完整分析栈**

  **English:** Added DC, AC, noise, and transient analyses, together with PSS,
  PAC, and PNoise periodic analyses for chopper amplifiers. Performance-critical
  paths use Numba JIT kernels.

  **中文：** 新增 DC、AC、噪声和瞬态分析，以及面向斩波放大器的 PSS、PAC 和
  PNoise 周期分析。性能关键路径使用 Numba JIT 内核加速。

- **Cadence calibration and byte gate / Cadence 标定与字节门禁**

  **English:** Calibrated the solver stack against Spectre 24.1, with typical
  gain, bandwidth, and input-referred-noise error below 1% on the AT4000TG AFE.
  The then-current `core.calibration --all` command provided a reproducible
  drift gate.

  **中文：** 使用 Spectre 24.1 标定求解器栈；在 AT4000TG AFE 上，增益、带宽和
  输入等效噪声的典型误差低于 1%。当时的 `core.calibration --all` 命令提供可复现
  的漂移硬门禁。

- **Dataset-to-optimization ML loop / 数据集到优化的机器学习闭环**

  **English:** Added provenance-aware labeled dataset generation, GBT and
  PyTorch surrogates, and a surrogate-screened, solver-verified optimizer for
  high-throughput candidate selection.

  **中文：** 新增带 provenance 的标注数据集生成、GBT 与 PyTorch 代理模型，以及
  先由代理筛选、再由求解器校验的优化器，用于高吞吐候选设计筛选。

- **Unified circuit API / 统一电路 API**

  **English:** Added `CircuitBinding` to bind topology, sizing, bias, and
  device models into one solver call. The JSON circuit format allows new
  circuits and per-device PDK bindings without changing solver source code.

  **中文：** 新增 `CircuitBinding`，将拓扑、尺寸、偏置和器件模型绑定为一次求解器
  调用。电路 JSON 格式允许在不修改求解器源码的情况下添加新电路，并为具体器件
  绑定非默认 PDK。

- **Command-line interface / 命令行接口**

  **English:** Added the `circuit-opt` entry point and the then-current
  `python -m core` commands for run, exploration, corners, mismatch Monte
  Carlo, chopper analysis, and dataset generation.

  **中文：** 新增 `circuit-opt` 入口，以及当时用于运行、探索、工艺角、失配
  蒙特卡洛、斩波分析和数据集生成的 `python -m core` 命令。

- **Process corners and mismatch / 工艺角与失配**

  **English:** Added process-corner sweeps, per-device mismatch Monte Carlo,
  and latch screening.

  **中文：** 新增工艺角扫描、逐器件失配蒙特卡洛和 latch 筛查。

- **Tests and CI / 测试与持续集成**

  **English:** Added 359 tests, including Cadence regressions and byte-gate
  reproduction, plus lint, test-matrix, and byte-gate CI jobs.

  **中文：** 新增 359 项测试，包括 Cadence 回归和字节门禁复现，并建立 lint、
  测试矩阵和字节门禁三类 CI 作业。

[Unreleased]: https://github.com/751K/circuit-optimization-lab/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.0.5...v1.1.0
[1.0.5]: https://github.com/751K/circuit-optimization-lab/compare/v0.1.0...v1.0.5
[0.1.0]: https://github.com/751K/circuit-optimization-lab/releases/tag/v0.1.0
