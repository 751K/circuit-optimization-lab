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

## [2.1.5] - 2026-07-24

### Added / 新增

- **PVT corner grid: full silicon corner sets + temperature & supply-scale axes on `corner_table` / PVT 网格：硅工艺全 corner 集 + `corner_table` 温度与电压轴**

  **English:** `corners.corner_table` grows from a process-only sweep into a PVT
  grid. (1) The per-family default silicon corner set (`silicon_corner_names`) is
  now the full process set — freepdk45 `nom/ss/ff/sf/fs` and tsmc28
  `tt/ss/ff/sf/fs` (five each; the freepdk45 `sf/fs` cross corners reuse the `ss/ff`
  per-polarity dirs, tsmc28's added `ff/sf/fs` are core-`.l` sections), while sky130
  stays `tt/ss` (the bundled-card data boundary). A geometry that selects **zero
  bins** in some corner (tsmc28 `ff/sf/fs`, or an out-of-grid width) is recorded as
  `None` and counted (`corners.corner_zero_bin_skip`) instead of raising the PDK bin
  error and sinking the whole sweep — both arms reject it identically. (2) A new
  `temps=` argument adds a **temperature axis** in °C; (3) a new `vdd_scale=`
  argument adds a **supply-scale axis** that multiplies the whole bias dict
  uniformly (the established `scale = vdd/VDD; bias = {k: v*scale}` convention). Both
  are **silicon-only** (an OTFT/default-PDK binding rejects them) and reuse frozen
  primitives with **no Rust change**: temperature rides the silicon-device
  `temperature` ctor kwarg (Kelvin) onto `device_kwargs`, so both the compiled
  campaign (`CompiledPdk::numeric_card` card selection + `co_bsim4::create`) and the
  scalar reference see it. Each `(temp, vdd)` slice is one compiled campaign (the R9
  dataset-layering precedent), the grid parallelises across its independent slices,
  and every point inherits the 0-bin skip + non-convergence rollback. The result
  nests under each corner in axis order `[temp_c, vdd_scale]` (`{corner: metrics}`
  when neither is given). The `corners` CLI gains `--temps` / `--vdd-scale`
  (comma-separated); **with neither flag the printed table and `-o` CSV are
  byte-for-byte unchanged**, and with them the print groups by slice and the CSV adds
  `temp_c`/`vdd_scale` columns for the active axes. `mismatch_mc` keeps its frozen
  behaviour — the T/V *grid* is deferred (its per-slice nominal-seed recomputation is
  a distinct change surface), though a single (temp, vdd) point is already reachable
  by composing a temperature-baked binding + scaled bias. Default `corner_table` and
  `mismatch_mc` are byte-for-byte identical to 2.1.0. Parity (compiled campaign vs
  the frozen scalar reference, per corner × the temperature/supply axes, incl. the
  tsmc28 ff/sf/fs bins and the −40/+125 °C extremes): worst rel ~2e-15 freepdk45,
  ~1.4e-16 sky130, ≤~1.5e-9 tsmc28 (the cold Newton-vs-fsolve DC-root floor) — no
  divergence, no rollback; byte-identical across workers {1, 2, 8}; golden corpus
  reproduces bit-exactly (no re-freeze). 45-point grid (5 corner × 3 temp × 3 vdd)
  speedup, `workers=1` → `8` (median of 3): freepdk45 5T OTA **3.5×**, tsmc28 5T OTA
  **2.5×** (per-candidate macro-expansion-bound); 0-bin skips 0/45 for these OTA
  geometries.

  **中文：** `corners.corner_table` 由纯工艺 corner 扫描扩为 PVT 网格。①各族默认硅
  corner 集（`silicon_corner_names`）改为完整工艺集——freepdk45 `nom/ss/ff/sf/fs`、
  tsmc28 `tt/ss/ff/sf/fs`（各 5 个；freepdk45 `sf/fs` 交叉 corner 复用 `ss/ff` 逐极性
  目录，tsmc28 新增 `ff/sf/fs` 为核心 `.l` section），sky130 维持 `tt/ss`（捆绑卡数据
  边界）。某几何在某 corner **无 bin**（tsmc28 `ff/sf/fs` 或超网格宽度）时标 `None` 并
  计数（`corners.corner_zero_bin_skip`），而非抛 PDK bin 错拖垮整表——两臂拒绝方式一致。
  ②新增 `temps=` **温度轴**（°C）；③新增 `vdd_scale=` **电压轴**，对整个 bias 字典统一
  缩放（沿用 `scale = vdd/VDD; bias = {k: v*scale}` 既有约定）。二者均**限硅族**
  （OTFT/默认 PDK binding 报错），复用冻结原语且**不改 Rust**：温度经硅器件
  `temperature` 构造参数（开尔文）落到 `device_kwargs`，编译 campaign
  （`CompiledPdk::numeric_card` 选卡 + `co_bsim4::create`）与标量参考同见。每个
  `(temp, vdd)` 切片为一个编译 campaign（R9 数据集分层先例），网格沿独立切片并行，每点
  继承 0-bin 跳过 + 不收敛回退。结果按轴序 `[temp_c, vdd_scale]` 在各 corner 下嵌套
  （均不传时为 `{corner: metrics}`）。`corners` CLI 新增 `--temps` / `--vdd-scale`
  （逗号分隔）；**不传时打印表与 `-o` CSV 逐字节不变**，传入时按切片分组打印、CSV 为
  启用的轴增列 `temp_c`/`vdd_scale`。`mismatch_mc` 保持冻结行为——其 T/V *网格*留待
  下期（逐切片名义种子重算是独立改动面），但单个 (temp, vdd) 点已可经"温度 binding +
  缩放 bias"组合达成。默认 `corner_table` 与 `mismatch_mc` 对 2.1.0 逐字节一致。Parity
  （编译 campaign 对冻结标量参考，逐 corner × 温度/电压轴，含 tsmc28 ff/sf/fs bin 与
  −40/+125 °C 极点）：最差相对 ~2e-15 freepdk45、~1.4e-16 sky130、≤~1.5e-9 tsmc28
  （冷牛顿 vs fsolve 的 DC 根下限）——无分叉、无回退；workers {1, 2, 8} 逐字节一致；
  golden 语料逐位复现（无重冻）。45 点网格（5 corner × 3 温 × 3 压）加速，`workers=1`
  → `8`（3 次中位）：freepdk45 5T OTA **3.5×**、tsmc28 5T OTA **2.5×**（逐候选受宏
  展开支配）；上述 OTA 几何 0-bin 跳过 0/45。

## [2.1.0] - 2026-07-24

### Changed / 性能

- **Silicon corners / mismatch-MC / dataset route through the compiled campaign / 硅工艺 corners / 失配 MC / 数据集接入编译 campaign**

  **English:** The silicon (BSIM4) paths of `corners.corner_table`,
  `corners.mismatch_mc`, and the `dataset` size-grid builder now evaluate their
  candidate matrix through the compiled campaign (`circuitopt._rust_campaign`) —
  one Rayon pool, per-candidate corner, `workers` scaled, and **no per-candidate
  Python callback** — instead of a per-candidate Python solve. The frozen scalar
  path (`ac_solve` / `noise_analysis` under the same binding; `delvto` mismatch via
  the device `delvto` knob; `explore._supply_power_uW` / `_area` post-batch
  reductions) is the reference the campaign is validated bit-for-bit against and the
  per-corner / per-layer fallback. **AFE / mixed circuits are untouched and stay on
  the scalar path**: a cold campaign cannot reproduce the multistable OTFT basin and
  would under-report the latch rate, so only the monostable, cold-DC-consistent
  silicon families route (guard tests pin this). No result key, CLI flag, or JSON
  contract changes; the CLI `corners`/`mc` and the service MC job auto-benefit for
  silicon. `corner_table`/`mismatch_mc` gain a `binding=` argument; the silicon
  campaign result additionally exposes `gain_dB` (DC gain) and per-device `ich`
  (channel current) — both already computed in the pipeline, surfaced for the
  dataset `power_uW`/`gain_dB` labels. Parity: campaign vs the frozen scalar path is
  bit-for-bit on freepdk45/sky130 and ≤1e-9 relative on tsmc28 (the cold
  Newton-vs-fsolve DC-root floor, far inside the 1e-3 calibration tolerance);
  byte-identical across workers {1, 2, 8}; golden corpus reproduces bit-exactly (no
  re-freeze). Measured speedup vs the scalar `workers=1` baseline (median of 3):
  mismatch-MC N=200 freepdk45 **26.8×** at 8 workers (5.4× at 1), tsmc28 **4.8×**
  (macro-expansion-bound per candidate); dataset build (freepdk45, 120 rows)
  **5.0×**; `corner_table` **2.1×** (its parallelism is capped at the corner count).

  **中文：** `corners.corner_table`、`corners.mismatch_mc` 与 `dataset` 尺寸网格构建
  的**硅工艺（BSIM4）**路径现将候选矩阵交由编译 campaign（`circuitopt._rust_campaign`）
  求值——单 Rayon 池、逐候选 corner、`workers` 可扩、**无逐候选 Python 回调**——取代
  原先的逐候选 Python 求解。冻结标量路径（同 binding 下的 `ac_solve` / `noise_analysis`；
  `delvto` 失配走器件 `delvto` 端；`explore._supply_power_uW` / `_area` 批后归约）作为
  campaign 逐位对照的参考及逐 corner / 逐层回退。**AFE / 混合电路一字不动，保留标量
  路径**：冷 campaign 复现不了多稳 OTFT 盆地、会把 latch_rate 低报，故仅单稳、冷 DC
  一致的硅族接入（守卫测试钉死）。结果键、CLI 参数、JSON 契约均不变；CLI `corners`/`mc`
  与 service MC 作业对硅自动受益。`corner_table`/`mismatch_mc` 新增 `binding=` 参数；
  硅 campaign 结果另导出 `gain_dB`（直流增益）与逐器件 `ich`（沟道电流）——二者本已在
  流水线中算出，为数据集 `power_uW`/`gain_dB` 标签疏通。Parity：campaign 对冻结标量
  在 freepdk45/sky130 逐位、tsmc28 ≤1e-9 相对（冷牛顿 vs fsolve 的 DC 根下限，远在 1e-3
  校准容差内）；workers {1, 2, 8} 逐字节一致；golden 语料逐位复现（无重冻）。相对标量
  `workers=1` 基线实测加速（3 次中位）：失配 MC N=200 freepdk45 8 workers **26.8×**
  （1 worker 5.4×），tsmc28 **4.8×**（逐候选受宏展开支配）；数据集构建（freepdk45,
  120 行）**5.0×**；`corner_table` **2.1×**（并行度上限为 corner 数）。

- **BSIM4 DC Newton skips capacitance extraction (D6 acLoad-skip) / BSIM4 DC 牛顿迭代跳过电容抽取（D6 acLoad-skip）**

  **English:** The BSIM4 DC operating-point Newton (`bsim_transient::solve_dc`)
  consumes only the terminal currents and conductance, but every device eval had
  been running the full host.c tail — a `MODEINITSMSIG` reload plus `acLoad` and a
  complex Schur reduction — to extract the small-signal capacitance nobody reads
  until the final operating-point eval. That tail is now split out
  (`co_bsim4::eval_vp_dc`, gated by the new `Evaluator::evaluate_dc`) and skipped
  during DC iterations; the one-shot small-signal eval, the transient
  (`solve_fixed_grid`), and every scalar/reference entry point still run the full
  eval. Currents and conductance (and their conservation snap) are bit-for-bit
  identical — the frozen engine-parity golden corpus reproduces bit-exactly, no
  re-freeze — because `acLoad` writes only the per-call-cleared matrix and the DC
  load's charge terms stamp `0` into the Jacobian. Measured on the pure DC solve
  (median of 3, N=4000): freepdk45 5T OTA **1.88×** (178.1 → 94.5 µs/solve),
  tsmc28 5T OTA **1.60×** (137.7 → 86.1 µs/solve). Set the escape hatch
  `CIRCUIT_BSIM4_FULL_EVAL=1` to force the full extraction on the DC path (exact
  rollback; also verified bit-exact).

  **中文：** BSIM4 直流工作点牛顿迭代（`bsim_transient::solve_dc`）只消费端口电流与
  电导，但此前每次器件求值都白跑 host.c 的完整尾段——一次 `MODEINITSMSIG` 重载加
  `acLoad` 与复数 Schur 消元——去抽取直到最终工作点求值才会被读取的小信号电容。现在
  该尾段被拆出（`co_bsim4::eval_vp_dc`，经新增的 `Evaluator::evaluate_dc` 分派）并在
  DC 迭代期跳过；工作点处的单次小信号求值、瞬态（`solve_fixed_grid`）以及所有标量/
  参考入口仍走完整求值。电流与电导（含守恒修正）逐位不变——冻结的 engine-parity
  golden 语料逐位复现，无需重冻——因为 `acLoad` 只写每次调用前已清零的矩阵，且 DC
  载入的电荷项在雅可比里恒印 `0`。纯 DC 求解实测（3 次中位，N=4000）：freepdk45 5T
  OTA **1.88×**（178.1 → 94.5 µs/次），tsmc28 5T OTA **1.60×**（137.7 → 86.1
  µs/次）。设 `CIRCUIT_BSIM4_FULL_EVAL=1` 可强制 DC 路径走完整抽取（精确回退，亦已
  验证逐位一致）。

## [2.0.2] - 2026-07-23

### Added / 新增

- **Windows CI and wheels (first Windows build) / Windows CI 与 wheel（首个 Windows 版本）**

  **English:** The `test` matrix (`ci.yml`) and the `build-wheels` matrix
  (`release.yml`) gained a `windows-latest` leg, so `circuitopt-core` now targets
  Windows (`win_amd64`, MSVC ABI, one abi3-py310 wheel) alongside Linux and macOS.
  `rust/crates/co-bsim4/build.rs` is now toolchain-conditional: the clang/gcc
  invocation (`-std=c99 -Wno-error=implicit-function-declaration`) is preserved
  byte-for-byte for macOS/Linux, while MSVC uses its permissive default C mode and
  an out-of-vendor config shim (`co-bsim4/msvc_shim/ngspice/config.h`, prepended to
  the include path only for the `msvc` target env) so the vendored ngspice headers
  resolve without POSIX-only headers. The vendored Berkeley BSIM4.5 C is unchanged
  and macOS/Linux builds are bit-for-bit identical (golden corpus reproduces
  bit-exactly). A repo-wide `.gitattributes` (`* text=auto eol=lf`) keeps the
  byte-exact goldens and POSIX scripts LF on the Windows runner. The Windows legs were promoted to required gates after the first green
  Windows runner run confirmed the MSVC build.

  **中文：** `ci.yml` 的 `test` 矩阵与 `release.yml` 的 `build-wheels` 矩阵新增
  `windows-latest` 腿，`circuitopt-core` 现在除 Linux/macOS 外也覆盖 Windows
  （`win_amd64`，MSVC ABI，单个 abi3-py310 wheel）。`rust/crates/co-bsim4/build.rs`
  改为按工具链条件化：macOS/Linux 的 clang/gcc 编译参数
  （`-std=c99 -Wno-error=implicit-function-declaration`）逐字保持不变，MSVC 则使用其
  宽松的默认 C 模式，并通过一个位于 vendor 之外的 config 垫片
  （`co-bsim4/msvc_shim/ngspice/config.h`，仅在 `msvc` 目标下前置到 include 路径），
  让随附的 ngspice 头文件在不引入 POSIX-only 头的情况下解析。随附的 Berkeley BSIM4.5 C
  一字未改，macOS/Linux 构建逐位一致（golden 语料逐位复现）。仓库级 `.gitattributes`
  （`* text=auto eol=lf`）保证 byte-exact golden 与 POSIX 脚本在 Windows runner 上仍为
  LF。Windows 腿已在真实 runner 首次全绿后转为必过门。

## [2.0.1] - 2026-07-22

### Fixed / 修复

- **`run --workers` removed (ineffective since introduction) / 移除 `run --workers`（自引入起无效）**

  **English:** The `run` subcommand listed a `--workers` flag in `--help`, but
  `_cmd_run` never read it and `run_analysis_suite` has no `workers` parameter —
  `run` executes a single analysis suite for one circuit at one corner, with no
  independent corner/sample sweep to parallelize. The flag was copy-pasted into
  `_add_run_parser` alongside the R5-A `corners` parser fix (v2.0.0, commit
  `9e333e6`) and had zero effect from the moment it was introduced; its help
  text ("Parallel corner workers") was even the verbatim corner-specific
  wording. Removed it. Parallel batch execution stays on the subcommands that
  actually implement it: `corners`, `mc`, `dataset`, and `adc`.

  **中文：** `run` 子命令曾在 `--help` 中列出 `--workers`，但 `_cmd_run` 从不
  读取它，`run_analysis_suite` 也没有 `workers` 参数——`run` 只对单个电路、单个
  工艺角执行一次分析套件，没有可并行的独立 corner/样本扫描。该 flag 是在 R5-A
  修 `corners` parser 时（v2.0.0，提交 `9e333e6`）连带粘贴进 `_add_run_parser`
  的，自引入起零效果；其 help 文案（"Parallel corner workers"）甚至就是 corner
  专用措辞的原文照搬。现予移除。并行批处理仍保留在真正实现它的子命令上：
  `corners`、`mc`、`dataset`、`adc`。

- **`tools/profile_hotspots.py` repaired (stale library calls) / 修复 `tools/profile_hotspots.py`（过期库调用）**

  **English:** The profiling script crashed at its explore step (it passed
  `ExploreConfig` a dict of variables where the library has always taken a
  `list[Variable]`, plus operator-form constraints and list-form objectives
  the library never supported) and, once past that, at its corners/MC step
  (positional `freqs` landing on `nf`; a nonexistent `corner=` kwarg). Both
  call sites were updated to the canonical API; the library was untouched.
  The script now runs end-to-end on the compiled core.

  **中文：** 性能剖析脚本在 explore 步崩溃（给 `ExploreConfig` 传了 dict 形式
  的 variables，而库自始至终只接受 `list[Variable]`，另有库从不支持的运算符
  形式约束与 list 形式目标），越过后又在 corners/MC 步崩溃（位置实参 `freqs`
  落到 `nf` 上、不存在的 `corner=` 关键字）。两处调用点均改为正统 API，库
  零改动；脚本现可在编译核上端到端运行。

## [2.0.0] - 2026-07-22

### Changed (breaking) / 破坏性变更

- **The compute core is now Rust / 计算核心整体切换为 Rust**

  **English:** Every numerical hot path runs in the compiled `circuitopt_core`
  extension (PyO3, abi3). Python keeps what it is good at — CLI, service,
  JSON configuration, optimization strategy, SciPy orchestration (DC root
  selection, sparse periodic solves, FFT) and the external ngspice/Cadence
  oracles — and delegates all device evaluation, matrix assembly and
  time-domain/small-signal/periodic solving to Rust. The project ships as two
  locked distributions: `circuit-optimization` (pure Python) pins
  `circuitopt-core` to the exact same version; `tools/version.py` keeps them
  in lockstep and CI rejects drift. `--engine`/`CIRCUIT_ENGINE` remain but
  accept only `rust`. The OTFT parameter bundle is renamed
  `OtftParams`/`get_otft_params()`; a `NumbaParams` compatibility alias stays
  exported for v1.x imports.

  **中文：** 全部数值热路径运行于编译扩展 `circuitopt_core`（PyO3、abi3）。
  Python 保留其擅长的部分——CLI、service、JSON 配置、优化策略、SciPy 编排
  （DC 选根、周期族稀疏解、FFT）与外部 ngspice/Cadence oracle——器件求值、
  矩阵装配、时域/小信号/周期求解全部交给 Rust。发布形态为版本锁死的双发行版：
  `circuit-optimization`（纯 Python）精确 pin 同版本 `circuitopt-core`，
  `tools/version.py` 双向同步、CI 拒绝漂移。`--engine`/`CIRCUIT_ENGINE` 保留
  但仅接受 `rust`。OTFT 参数包更名 `OtftParams`/`get_otft_params()`，兼容别名
  `NumbaParams` 仍导出，v1.x import 不断。

### Added / 新增

- **Compiled solver core with proven numerical parity / 编译求解核心（数值 parity 已证）**

  **English:** The Rust workspace carries the full solver family: the OTFT
  analytic device (currents, internal Newton, charges, terminal derivatives —
  including the root-selection recovery used on sensitive circuits), the
  vendored Berkeley BSIM4.5 compiled at build time behind a safe FFI host
  (per-handle concurrency, a long-standing destroy leak fixed), MNA assembly
  with the same-pivoting dense solver, damped circuit Newton, fixed
  backward-Euler and adaptive Gear2 transient, AC/noise, and the periodic
  family (shooting PSS support, harmonic-balance blocks, PAC orbit
  linearization, cyclostationary PSD folding). Equivalence to the retired
  reference was gated phase by phase: device grids and fixed-grid waveforms
  bit-exact or within 1e-12, the Cadence calibration byte-gates unchanged,
  and the frozen golden corpus (`tests/golden/engine_parity`) is now the
  permanent regression anchor.

  **中文：** Rust workspace 承载完整求解器族：OTFT 解析器件（电流、内部
  Newton、电荷、端口导数——含敏感电路上的选根恢复）、构建期编译的 vendored
  Berkeley BSIM4.5 安全 FFI 宿主（逐 handle 并发、修复长期存在的 destroy
  泄漏）、同主元稠密解的 MNA 装配、阻尼电路 Newton、固定后向欧拉与自适应
  Gear2 瞬态、AC/噪声、周期族（打靶 PSS 支撑、谐波平衡块、PAC 轨道线性化、
  cyclostationary PSD 折叠）。与退役参考的等价性逐期过门：器件网格与固定
  网格波形逐位或 1e-12 内，Cadence 校准字节门一字未动，冻结 golden 语料
  （`tests/golden/engine_parity`）成为永久回归锚点。

- **GIL-free batch execution / 无 GIL 批处理**

  **English:** Production batch workloads no longer serialize on the
  interpreter. `CompiledCampaign.evaluate_batch` takes a candidate matrix and
  runs PDK expansion, device construction, DC/AC/noise and metric reduction
  inside a single GIL-released region on one Rayon pool, with seeded
  mismatch drawn up front and candidate-index-ordered, byte-deterministic
  write-back (identical results for 1/2/8 workers). The closed-loop SAR
  conversion runs the same way. Measured on an 8-core laptop: design-space
  sweeps 5.4× (FreePDK45) / 2.2× (TSMC28) at 8 workers; SAR mismatch MC went
  from a GIL-bound 0.13 scaling efficiency to 0.70 — with the single-thread
  path itself ~10× faster.

  **中文：** 生产批处理不再被解释器串行化。`CompiledCampaign.evaluate_batch`
  接收候选矩阵，在单个释放 GIL 的区间、一个 Rayon 池上完成 PDK 展开、器件
  构建、DC/AC/noise 与指标归约；失配按 seed 预抽、按候选索引有序写回，
  1/2/8 workers 结果逐字节一致。闭环 SAR 转换同样整体进入 Rust。8 核实测：
  设计空间扫描 8 workers 下 FreePDK45 5.4×、TSMC28 2.2×；SAR 失配 MC 扩展
  效率从 GIL 束缚的 0.13 提至 0.70——且单线程路径本身快约 10×。

- **SPICE and PDK compilation in Rust / SPICE 与 PDK 编译进 Rust**

  **English:** The HSPICE expression engine (numbers with SPICE suffixes,
  Pratt parser, lazy case-insensitive scopes, user functions), the deck
  parser and the elaborator are compiled, and the FreePDK45/SKY130/TSMC28
  adapters produce numeric model cards (corner/polarity normalization,
  geometry-bin selection, `nf`/`mult`/mismatch rules) behind an immutable,
  thread-safe `CompiledPdk` cache. Licensed content never enters tests,
  goldens or logs. Differential parity against the retired Python reference
  was bit-exact across the full real corpora — including 198,758 real TSMC28
  parameter expressions and every bundled SKY130 card.

  **中文：** HSPICE 表达式引擎（SPICE 后缀数字、Pratt 解析、惰性大小写不敏感
  作用域、用户函数）、deck 解析器与 elaborator 已编译化；FreePDK45/SKY130/
  TSMC28 适配器在 immutable、线程安全的 `CompiledPdk` 缓存后生成数值模型卡
  （corner/极性归一、几何 bin 选择、`nf`/`mult`/失配规则）。授权内容不进
  测试、golden 或日志。对退役 Python 参考的差分 parity 在全量真实语料上
  逐位一致——含 198,758 条真实 TSMC28 参数表达式与全部捆绑 SKY130 卡。

- **Prebuilt deployment / 预编译部署**

  **English:** No JIT warm-up and no compiler on the user's machine: the
  BSIM4.5 C is compiled when the `circuitopt-core` wheel is built, and cold
  start drops accordingly (first AC solve ~2% of the former JIT path). The
  release workflow publishes both distributions.

  **中文：** 免 JIT 预热、用户机器免编译器：BSIM4.5 C 在 `circuitopt-core`
  wheel 构建期编译，冷启动相应下降（首个 AC 解约为旧 JIT 路径的 2%）。
  发布工作流同时发布两个发行版。

### Removed (breaking) / 移除（破坏性）

- **numba, the Python kernels, the runtime cc backend, and OSDI / numba、Python 内核、运行时 cc 后端与 OSDI**

  **English:** The numba dependency and engine, the pure-Python `_impl`
  kernels (their OTFT root-selection duty was ported into Rust first; the
  frozen golden corpus replaces them as the oracle), the runtime cc/ctypes
  BSIM4.5 build path, and the OSDI/OpenVAF compatibility layer are all gone.
  `--no-numba`, `CIRCUIT_USE_NUMBA` and `--engine numba|python` now fail
  loudly with a pointer to this changelog instead of silently doing nothing.
  Explicit ngspice and Cadence regression oracles remain.

  **中文：** numba 依赖与引擎、纯 Python `_impl` 内核（其 OTFT 选根职责已先
  移植进 Rust；冻结 golden 语料接任 oracle）、运行时 cc/ctypes BSIM4.5 编译
  路径、OSDI/OpenVAF 兼容层全部移除。`--no-numba`、`CIRCUIT_USE_NUMBA` 与
  `--engine numba|python` 现在响亮报错并指向本 changelog，而非静默无操作。
  显式 ngspice 与 Cadence 回归 oracle 保留。

## [1.4.1] - 2026-07-17

### Fixed / 修复

- **Default test-suite runtime and native BSIM4 build robustness / 默认测试集耗时与原生 BSIM4 构建健壮性**

  **English:** Marked the complete SAR/ADC conversion regressions as
  `heavy_e2e` and excluded them from the default pytest run (on a machine with
  FreePDK45 cards they took ~20 of the suite's ~22 minutes; the default run is
  now minutes-level again, run them explicitly with `pytest -m heavy_e2e`).
  The native BSIM4.5 runtime build now tolerates compilers that promote
  implicit function declarations to errors (clang 16+ on Linux rejected the
  unmodified Berkeley sources), and a failed build is cached per process
  instead of being retried by every test — a CI run had burned 2.5 h
  re-running the same failing compile.

  **中文：** 将完整 SAR/ADC 转换回归标记为 `heavy_e2e` 并移出默认 pytest
  运行（装有 FreePDK45 卡的机器上它们占 ~22 分钟中的 ~20 分钟；默认运行
  恢复到分钟级，用 `pytest -m heavy_e2e` 显式执行）。原生 BSIM4.5 运行时
  构建现兼容将隐式函数声明视为错误的编译器（Linux 上 clang 16+ 拒绝编译
  未修改的 Berkeley 源码），且构建失败在进程内缓存、不再被每个测试重试
  ——此前一次 CI 曾因重复运行同一失败编译烧掉 2.5 小时。

## [1.4.0] - 2026-07-17

### Added / 新增

- **Native SKY130 BSIM4 adapter / 原生 SKY130 BSIM4 适配器**

  **English:** Added packaged geometry-resolved SKY130 BSIM4.5 cards and native
  `sky130.nmos` / `sky130.pmos` devices using the in-process C backend. OpenVAF,
  OSDI, and ngspice are now explicit regression/card-generation tools.

  **中文：** 新增随包分发的按几何展开 SKY130 BSIM4.5 参数卡，以及使用进程内
  C 后端的原生 `sky130.nmos` / `sky130.pmos`。OpenVAF、OSDI 与 ngspice
  现仅作为显式回归或参数卡生成工具。

- **Centralized version management / 集中式版本管理**

  **English:** Added `tools/version.py` with show, check, sync, set, and release
  commands. `pyproject.toml` is now the canonical version source for Python,
  npm, and Tauri manifests; CI and release workflows reject version drift or
  mismatched tags.

  **中文：** 新增 `tools/version.py`，提供 show、check、sync、set 和 release
  命令。`pyproject.toml` 现为 Python、npm 与 Tauri 清单的版本号唯一来源；
  CI 和发布工作流会拒绝版本漂移及不匹配的 tag。

### Changed / 变更

- **Native-only normal workflows / 正常流程统一使用原生后端**

  **English:** Migrated the FreePDK45 3-bit and 6-bit SAR examples to
  `freepdk45.*`. The package root and default model registry now expose only
  native silicon PDK keys; ngspice/OSDI aliases require an explicit oracle
  module import. Default pytest runs exclude the `ngspice_oracle` suite, and
  MDAC full-circuit campaigns are named as explicit oracle regressions.

  **中文：** 将 FreePDK45 3-bit/6-bit SAR 示例迁移到 `freepdk45.*`，移除
  正常测试对 ngspice 的前置依赖。包根接口和默认模型注册表仅暴露原生硅工艺
  模型键；ngspice/OSDI 别名需要显式导入 oracle 模块。默认 pytest 排除
  `ngspice_oracle` 测试集，MDAC 全电路 campaign 也明确命名为 oracle 回归。

### Fixed / 修复

- **Native transient initialization and source power / 原生瞬态初值与源功耗**

  **English:** The SAR workflow now passes circuit `dc_guesses` into native
  transient, avoiding unnecessary shared AC initialization in parallel runs,
  and native transient reports MOS rail/gate source currents with the
  source-power sign convention. All transient backends now reject mismatch maps
  that reference devices absent from the topology.

  **中文：** SAR 工作流现将电路 `dc_guesses` 传给原生瞬态，避免并行运行中
  不必要的共享 AC 初值求解；MOS 电源与门极驱动支路电流也统一为源功耗符号约定。
  所有瞬态后端现会拒绝引用拓扑中不存在器件的 mismatch 映射。

## [1.3.0] - 2026-07-17

### Added / 新增

- **Native FreePDK45 BSIM4 adapter / 原生 FreePDK45 BSIM4 适配器**

  **English:** Added a flat level-54 model-card loader and native
  `freepdk45.nmos` / `freepdk45.pmos` devices backed by the bundled Berkeley
  BSIM4.5 kernel. The adapter exposes four-terminal current, conductance,
  charge, capacitance, and correlated noise across `nom`, `tt`, `ss`, `ff`,
  `sf`, and `fs` corners without launching ngspice.

  **中文：** 新增平铺 level-54 模型卡加载器，以及由仓库内 Berkeley
  BSIM4.5 内核驱动的原生 `freepdk45.nmos` / `freepdk45.pmos` 器件。
  适配器在 `nom`、`tt`、`ss`、`ff`、`sf`、`fs` 工艺角下提供四端电流、
  电导、电荷、电容和相关噪声，正常仿真不再启动 ngspice。

- **FreePDK45 native regression coverage / FreePDK45 原生回归覆盖**

  **English:** Added no-ngspice single-device and 5T OTA DC, AC, noise, and
  transient tests, plus optional ngspice comparisons for device operating
  points/noise and complete OTA AC behavior.

  **中文：** 新增不依赖 ngspice 的单管与五管 OTA DC、AC、噪声、瞬态测试，
  并保留可选 ngspice 对照，用于核对器件工作点、噪声和完整 OTA AC 行为。

- **Native BSIM4 Numba bridge / 原生 BSIM4 Numba 桥**

  **English:** Added a versioned C ABI with conserved four-terminal evaluation,
  an all-`void *` runtime entry point, and a batch evaluator. Native BSIM4
  transient now calls the C compact model directly from a Numba Newton/time-step
  loop for both FreePDK45 and TSMC28HPC+, while retaining the Python reference
  path when Numba is disabled.

  **中文：** 新增带版本号的 C ABI、守恒四端求值入口、全 `void *` 运行时入口
  和批量求值器。FreePDK45 与 TSMC28HPC+ 的原生 BSIM4 瞬态现可在 Numba
  Newton/时间步循环内直接调用 C 紧凑模型；禁用 Numba 时仍保留 Python
  参考路径。

### Changed / 变更

- **FreePDK45 default backend / FreePDK45 默认后端**

  **English:** `freepdk45.*` now selects the native in-process BSIM4 path.
  The historical cached-ngspice evaluator remains available explicitly as
  `freepdk45_ngspice.*`, and complete-circuit ngspice helpers remain optional
  regression oracles.

  **中文：** `freepdk45.*` 现默认选择进程内原生 BSIM4 路径。旧的 ngspice
  缓存网格求值器以 `freepdk45_ngspice.*` 显式保留，完整电路 ngspice helper
  继续作为可选回归 oracle。

- **Historical SAR oracle binding / 历史 SAR oracle 绑定**

  **English:** Kept the existing 3-bit and 6-bit FreePDK45 SAR/StrongARM
  examples explicitly on `freepdk45_ngspice.*`. Native migration is validated
  with the 5T OTA; the dynamic SAR examples no longer change backend implicitly.

  **中文：** 现有 3-bit 与 6-bit FreePDK45 SAR/StrongARM 示例显式使用
  `freepdk45_ngspice.*`。原生迁移以五管 OTA 完成验证，动态 SAR 示例不再随
  默认模型名称隐式切换后端。

### Fixed / 修复

- **Native BSIM internal topology and charge reduction / 原生 BSIM 内部拓扑与电荷归并**

  **English:** Replaced the two-internal-node limit with pivoted reduction for
  the complete BSIM4 drain/source, distributed-gate, and body-resistance
  network. Corrected external bulk aggregation of distributed junction charge
  and normalized PMOS terminal-charge signs, making charge derivatives agree
  with the AC capacitance matrix for both polarities.

  **中文：** 将原生 host 的两个内部节点上限改为带主元消元，覆盖完整 BSIM4
  漏源、分布式栅和体电阻网络；同时修正分布式结电荷向外部 bulk 的归并及
  PMOS 端口电荷符号，使 N/P 两种极性的电荷导数与 AC 电容矩阵一致。

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

[Unreleased]: https://github.com/751K/circuit-optimization-lab/compare/v2.1.5...HEAD
[2.1.5]: https://github.com/751K/circuit-optimization-lab/compare/v2.1.0...v2.1.5
[2.1.0]: https://github.com/751K/circuit-optimization-lab/compare/v2.0.2...v2.1.0
[2.0.2]: https://github.com/751K/circuit-optimization-lab/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/751K/circuit-optimization-lab/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.4.1...v2.0.0
[1.4.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.0.5...v1.1.0
[1.0.5]: https://github.com/751K/circuit-optimization-lab/compare/v0.1.0...v1.0.5
[0.1.0]: https://github.com/751K/circuit-optimization-lab/releases/tag/v0.1.0
