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

### Changed (breaking, v2.0.0) / 破坏性变更（v2.0.0）

- **Rust is the only compute engine / Rust 成为唯一计算引擎**

  **English:** The compiled Rust core (`circuitopt_core`, `CIRCUIT_ENGINE=rust`)
  is now the sole engine. The `--engine` flag and `CIRCUIT_ENGINE` env var are
  retained (compatibility contract) but accept only `rust`; the former `numba`
  (JIT) and `python` (pure-Python) engine values now raise a clear error that
  points here. The main package (`circuit-optimization`) hard-depends on and
  **pins `circuitopt-core` exactly to its own version** — the two distributions
  are built and released together, kept in lockstep by `tools/version.py` (a
  `version.py check` failure blocks drift).

  **中文：** 编译 Rust 核（`circuitopt_core`，`CIRCUIT_ENGINE=rust`）现为唯一
  引擎。`--engine`/`CIRCUIT_ENGINE` 保留（契约）但仅接受 `rust`；旧的 `numba`
  与 `python` 引擎值会明确报错并指向本处。主包 `circuit-optimization` 硬依赖并
  **精确 pin `circuitopt-core` 到同一版本**——两个发行版一起构建发布，由
  `tools/version.py` 锁死（`version.py check` 拒绝漂移）。

### Removed (breaking, v2.0.0) / 移除（破坏性，v2.0.0）

- **numba engine and dependency / numba 引擎与依赖**

  **English:** The optional numba JIT engine and the `numba` runtime dependency
  were removed. `--no-numba` and `CIRCUIT_USE_NUMBA` (and `--engine numba` /
  `--engine python`) are now hard errors, not silent no-ops. `numba_transient.py`
  and the numba BSIM4 transient arm were deleted. The frozen golden corpus
  (`tests/golden/engine_parity`) is the reference oracle for engine parity.

  **中文：** 移除可选 numba JIT 引擎与 `numba` 运行时依赖。`--no-numba`、
  `CIRCUIT_USE_NUMBA`（以及 `--engine numba`/`--engine python`）现为明确报错，
  不再静默无操作。删除 `numba_transient.py` 与 numba BSIM4 瞬态臂。冻结 golden
  语料（`tests/golden/engine_parity`）是引擎 parity 的参考 oracle。

- **Python `_impl` reference kernels removed; OTFT root-selection recovery
  ported to the compiled core (R7) / Python `_impl` 参考内核移除；OTFT 选根恢复
  移植进编译核（R7）**

  **English:** `numba_kernels.py` (the interpreted `_impl` scalar/driver
  kernels) was deleted **whole**. Its load-bearing part — the OTFT
  root-selection recovery oracle the rust engine invokes on bifurcation-edge
  circuits (sc_lpf adaptive-gear2 orbit setup, AFE latch/mismatch screens, the
  AC extreme-point retry) — was first ported into `circuitopt_core` as
  `OtftModel(..., reference=True)`: the system-libm `pow` `Vt` square
  (bit-exact to CPython's `x ** 2`), the finite-difference-Jacobian internal
  Newton, the finite-difference terminal derivatives, and the standalone
  capacitance equation, verified **bit-exact (0 ULP)** against the retired
  `_impl` across 112,815 points × 5 geometries before deletion. The trigger is
  now `pmos_tft_model.otft_reference_mode` (renamed from
  `rust_otft_reference_mode`). Every dead non-rust dispatch arm went with it:
  the Python transient drivers and their ~700-line argument marshal, the
  Python dense AC/noise MNA assembly, `_reference_pac_linearization` and the
  numba PAC/PNoise linearization arms, and the BSIM4 per-step Python Newton.
  Result keys that could only ever be `False` were dropped
  (`numba_grid_solver`, `numba_adaptive_solver`, `gear2_python_retry_solver`,
  `pac_numba_*`, `pnoise_numba_*`, `bsim4_numba_transient`,
  `numba_newton_*`). `NumbaParams`/`get_numba_params` were renamed to
  `OtftParams`/`get_otft_params`.

  **中文：** **整文件删除** `numba_kernels.py`（解释执行的 `_impl` 标量/驱动
  内核）。其承重部分——rust 引擎在分岔边缘电路上调用的 OTFT 选根恢复 oracle
  （sc_lpf 自适应 gear2 轨道初值、AFE latch/mismatch 筛查、AC 极端点重试）——
  已先移植进 `circuitopt_core`：`OtftModel(..., reference=True)` 提供系统 libm
  `pow` 的 `Vt` 平方（与 CPython `x ** 2` 位级一致）、有限差分 Jacobian 内部
  Newton、有限差分端子导数与独立电容方程，删除前经 112,815 点 × 5 几何对照
  验证与 `_impl` **逐位一致（0 ULP）**。触发器更名为
  `pmos_tft_model.otft_reference_mode`（原 `rust_otft_reference_mode`）。全部
  死的非 rust 分派臂一并删除：Python 瞬态驱动及其约 700 行参数编组、Python
  稠密 AC/noise MNA 装配、`_reference_pac_linearization` 与 numba PAC/PNoise
  线性化臂、BSIM4 逐步 Python Newton。恒为 `False` 的结果键移除
  （`numba_grid_solver`、`numba_adaptive_solver`、`gear2_python_retry_solver`、
  `pac_numba_*`、`pnoise_numba_*`、`bsim4_numba_transient`、`numba_newton_*`）。
  `NumbaParams`/`get_numba_params` 更名为 `OtftParams`/`get_otft_params`。

- **BSIM4 cc runtime-compile backend removed (R7) / BSIM4 cc 运行时编译后端
  移除（R7）**

  **English:** `native.py` no longer compiles the vendored Berkeley C at
  runtime with the user's compiler. `CIRCUIT_BSIM4_BACKEND` defaults to and
  only accepts `rust` (the compiled `circuitopt_core` cdylib); `=cc` raises a
  loud removal error, mirroring the engine-switch removals. The vendored C
  sources are untouched — `co-bsim4` compiles them at wheel-build time. The
  engine-parity golden corpus was re-frozen under the rust backend; the shift
  was attributed A/B: the R7 code itself is **bit-identical** to base+`rust`
  env (0.0 over 809 leaves), and the backend flip moves the silicon DC ops by
  at most 2.0e-6 rel (tsmc28 `vout` +0.91 µV — both roots converge below
  `DC_FALLBACK_TOL=1e-10`; iteration-path divergence from ULP-level backend
  deltas, not an equation change). All 406 golden BSIM device grids and the
  OTFT circuit case remain bit-exact.

  **中文：** `native.py` 不再于运行时用用户编译器编译 vendor Berkeley C。
  `CIRCUIT_BSIM4_BACKEND` 默认且仅接受 `rust`（编译的 `circuitopt_core`
  cdylib）；`=cc` 明确报错（对齐引擎开关移除模式）。vendor C 源码不动——由
  `co-bsim4` 在 wheel 构建期编译。engine-parity golden 已在 rust 后端下重冻结；
  位移经 A/B 归因：R7 代码本身与基点+`rust` 环境**位级一致**（809 叶子 0.0），
  后端翻转使硅 DC 工作点至多移动 2.0e-6 相对（tsmc28 `vout` +0.91 µV——新旧根
  都收敛于 `DC_FALLBACK_TOL=1e-10` 之下；ULP 级后端差经迭代路径发散，非方程
  变更）。全部 406 个 golden BSIM 器件网格与 OTFT 电路 case 保持逐位一致。

### Added / 新增

- **Rust BSIM4.5 native backend (R2) / Rust 原生 BSIM4.5 后端（R2）**

  **English:** `co-bsim4` now compiles the *unmodified* vendored Berkeley
  BSIM4.5 C at build time (via the `cc` crate) and reimplements the `host.c`
  adapter layer in Rust — parameter binding, the internal-node Newton
  reduction, the terminal I/G/Q/C extraction and the noise combination — with
  `bindgen`-derived struct layouts for an identical ABI. `circuitopt_core`
  exports the four-terminal `co_bsim4_*` C ABI (consumed by `native.py` via
  `ctypes`, including the Numba `eval_vp` function pointer) plus a
  `Bsim4Device` class. A call-time `CIRCUIT_BSIM4_BACKEND` selector switched
  backends per evaluation (R7 made `rust` the default and only value; the
  runtime `cc` build was removed); results match
  the reference C bit-for-bit for currents/conductance/charges and to ~1 ULP
  for the complex AC solve (validated against the frozen v1.4.0 golden corpus
  at `rel <= 1e-13`). The Rust `destroy` also fixes a `host.c` leak by freeing
  the `pSizeDependParamKnot` chain that `BSIM4v5temp` allocates.

  **中文：** `co-bsim4` 现在在构建期（经 `cc` crate）编译*未修改*的 Berkeley
  BSIM4.5 vendor C，并用 Rust 重写 `host.c` 适配层——按名设参、内部节点
  Newton 消元、四端 I/G/Q/C 提取与噪声归并——以 `bindgen` 生成的结构体布局
  保持 ABI 一致。`circuitopt_core` 导出四端 `co_bsim4_*` C ABI（供 `native.py`
  经 `ctypes` 调用，含 `eval_vp` 函数指针）及 `Bsim4Device` 类。调用时读取的
  `CIRCUIT_BSIM4_BACKEND` 开关逐次评估切换后端（R7 起 `rust` 为默认且唯一值，
  运行时 `cc` 编译已移除）；电流/电导/电荷与参考 C 位级一致，复数 AC 解在 ~1 ULP 内
  （对照冻结的 v1.4.0 golden 语料 `rel <= 1e-13`）。Rust 的 `destroy` 还修复了
  `host.c` 的泄漏：释放 `BSIM4v5temp` 分配的 `pSizeDependParamKnot` 链。

- **Rust core scaffolding and engine switch / Rust 核心脚手架与引擎开关**

  **English:** Added the `rust/` workspace — `co-core` (solver kernels, R3),
  `co-bsim4` (BSIM4.5 host, R2), and the `circuitopt_core` PyO3 extension —
  plus a `CIRCUIT_ENGINE` / `--engine` switch selecting `rust`, `numba`, or
  `python` (precedence argv > env > default; a missing rust core warns once
  and falls back to numba, and `--no-numba` keeps its exact behavior as the
  `python` shorthand). `tools/version.py` now synchronizes the Rust workspace
  version; CI gates `cargo fmt`/`clippy`, installs `circuitopt_core` in the
  test matrix, and the release workflow archives per-OS wheels as artifacts.
  As of R3/R4 the numerical hot paths run in `co-core` under
  `CIRCUIT_ENGINE=rust`; the default engine remains numba until the R6 flip.

  **中文：** 新增 `rust/` workspace——`co-core`（求解内核，R3）、`co-bsim4`
  （BSIM4.5 宿主，R2）与 `circuitopt_core` PyO3 扩展——并引入
  `CIRCUIT_ENGINE` / `--engine` 开关在 `rust`、`numba`、`python` 间选择
  （优先级 argv > 环境变量 > 默认；rust 核心缺失时警告一次并回退 numba，
  `--no-numba` 行为完全不变，等价于 `python`）。`tools/version.py` 现同步
  Rust workspace 版本号；CI 新增 `cargo fmt`/`clippy` 门禁并在测试矩阵安装
  `circuitopt_core`，发布工作流归档各平台 wheel 构件。自 R3/R4 起，数值热
  路径在 `CIRCUIT_ENGINE=rust` 下运行于 `co-core`；默认引擎在 R6 翻转前仍为 numba。

- **Rust solver core (R3/R4) / Rust 求解核心（R3/R4）**

  **English:** Ported the numba-executed solver hot paths into `co-core`: the
  OTFT device cluster (currents, internal 2-D Newton, capacitances/charges,
  terminal derivatives), the MNA term/stamp kernels and the same-pivoting dense
  GEPP, the damped circuit Newton, fixed backward-Euler and adaptive gear2
  transient, the AC/noise MNA assembly, and — for R4 — the periodic-family
  kernels (HB block assembly, cyclostationary PSD fold, PAC orbit
  linearization incl. the `gate1` state). Under `CIRCUIT_ENGINE=rust` these run
  through `circuitopt_core` with the GIL released, taking zero-copy read-only
  NumPy views of the compiled-topology flat arrays and returning NumPy
  waveforms/matrices; the result-dict keys are unchanged. Fixed-grid waveforms
  match numba to `rel <= 1e-12`; calibration byte-gates hold on both engines.

  **中文：** 将 numba 执行的求解热路径移植进 `co-core`：OTFT 器件簇（电流、
  内部二维 Newton、电容/电荷、端口导数）、MNA 三元组/stamp 内核与同主元稠密
  GEPP、阻尼电路 Newton、固定后向欧拉与自适应 gear2 瞬态、AC/噪声 MNA 装配，
  以及（R4）周期族内核（HB 块装配、cyclostationary PSD fold、含 `gate1` 状态的
  PAC 轨道线性化）。在 `CIRCUIT_ENGINE=rust` 下经 `circuitopt_core` 执行并释放
  GIL，以零拷贝只读 NumPy 视图接收编译拓扑平铺数组、返回 NumPy 波形/矩阵；
  结果字典键不变。固定网格波形与 numba 达 `rel <= 1e-12`；两引擎校准字节门均绿。

- **GIL-free parallel corner/MC workers (R5-A) / 无 GIL 并行 corner/MC（R5-A）**

  **English:** The `corners`, `mc`, and SAR CLI subcommands (and the
  `corner_table` / `mismatch_mc` APIs) accept a `--workers` count that evaluates
  independent corners/samples concurrently on a thread pool, relying on the
  Rust engine releasing the GIL. Results are bit-identical to the serial path.
  The BSIM4 Rust backend gained a per-handle concurrency model (per-handle lock,
  one-time front-end init, thread-local noise callback target).

  **中文：** `corners`、`mc` 与 SAR CLI 子命令（及 `corner_table` /
  `mismatch_mc` API）新增 `--workers` 计数，在线程池上并发评估独立 corner/
  样本，依赖 Rust 引擎释放 GIL；结果与串行路径逐位一致。BSIM4 Rust 后端引入
  per-handle 并发模型（逐 handle 锁、一次性前端初始化、thread-local 噪声回调）。

- **Rust SPICE/PDK compilers (R5-B) / Rust SPICE/PDK 编译器（R5-B）**

  **English:** `co-spice` now also carries 1:1 ports of the HSPICE expression
  engine, the deck parser (logical lines, assignments, `.lib`/`.subckt`
  structure) and the elaborator (section selection, scope filling, model
  numericization). A new `co-pdk` crate compiles FreePDK45 / SKY130 / TSMC28
  cards — corner/polarity normalization, geometry-bin selection, `nf`/`mult`/
  mismatch instance rules — into an immutable, thread-safe `CompiledPdk`
  (cache keyed by canonical path + mtime/size + section + options; D12: no
  licensed content in tests, goldens or logs). Exposed through
  `circuitopt_core` for differential verification only — production stays on
  the frozen Python reference until the compiled-campaign flip. Differential
  parity against the Python reference is bit-exact across the full real-PDK
  corpora (parser canonical trees byte-identical; elaborator and all three
  PDK numeric cards worst rel 0.0).

  **中文：** `co-spice` 现同时承载 HSPICE 表达式引擎、deck 解析器（逻辑行、
  赋值、`.lib`/`.subckt` 结构）与 elaborator（section 选择、作用域填充、
  model 数值化）的 1:1 移植。新增 `co-pdk` crate 把 FreePDK45 / SKY130 /
  TSMC28 卡编译为 immutable、线程安全的 `CompiledPdk`——corner/极性归一、
  几何 bin 选择、`nf`/`mult`/mismatch 实例规则；缓存键 = 规范路径 +
  mtime/size + section + 选项（D12：授权内容不入测试/golden/日志）。经
  `circuitopt_core` 暴露仅供差分验证——生产在 compiled-campaign 翻转前仍走
  冻结的 Python 参考。对 Python 参考的差分 parity 在全量真实 PDK 语料上
  逐位一致（parser canonical 树逐字节相同；elaborator 与三 PDK numeric
  card 最差 rel 0.0）。

- **Compiled campaign / candidate executor (R5-C) / 编译式 campaign（R5-C）**

  **English:** A new device-agnostic batch executor in `co-core`
  (`campaign`) runs a candidate matrix through one Rayon pool with an adaptive
  candidate-vs-frequency parallel axis (never nested, so the single pool is
  never oversubscribed), candidate-index-ordered write-back, and atomic
  progress + cooperative cancellation kept out of every numeric reduction; the
  `bw_from_gain` / `band_rms` metric reductions are ported from the frozen
  Python path. Two device families are wired. The AFE OTFT evaluator
  (`otft_campaign`) composes the native `otft` device kernels, dense `mna`
  solver, and complex `lti` MNA; the silicon BSIM4 evaluator (freepdk45 /
  sky130 / tsmc28) composes `co-pdk` numeric cards — with the TSMC28
  `to_bsim4_cards` `mulu0 → u0` mobility fold, factored into
  `co_pdk::apply_mulu0_fold` and unit-tested — into `co-bsim4` handles,
  `bsim_transient::solve_dc`, dense 4×4 terminal linearizations, and the
  per-device BSIM4 noise matrices with the evaluate-then-noise call order the
  backend requires. Both run the whole batch under one `py.detach` through
  `circuitopt_core.CompiledCampaign` with zero per-candidate Python callback.
  Verification only — no production workflow is wired to it yet (that is
  R5-D). AFE parity against a cold-consistent Python reference (fresh
  `PMOS_TFT` small-signal params → the same `LtiProblem` → the same
  reductions) is bit-for-bit: gain/bandwidth worst rel ~1e-15, IRN ~2e-16
  (the `band_rms` naive-sum ULP), DC operating point bit-for-bit. Silicon
  parity is measured against the frozen `ac_solve`/`noise_analysis` path
  directly (BSIM4 evaluation is a pure function — no warm/cold split): across
  the three 5T OTA geometry×corner matrices, gain worst rel ~4e-16, bandwidth
  ~2e-15, IRN ~1.7e-16, and the same-seed Rust DC Newton reproduces the
  Python operating point bit-for-bit. Results are byte-identical across
  worker counts {1,2,8}. Flagged deviations: the AFE OTFT internal 2-node
  Newton stops at `tol=1e-12`, so its operating point is seed-path-dependent —
  the frozen warm-cache `corners.metrics` path and a cold evaluation of the
  same model disagree by up to ~6e-8 in `gm`/`gds`; the campaign is
  cold-seed-consistent and therefore matches the warm path only to that
  inherent floor. The SKY130 reference-width pin (`extract_w != W`) is outside
  the `CompiledPdk` surface, so silicon campaigns cover `extract_w == W`
  geometries until an R5-B surface extension lands.

  **中文：** `co-core` 新增器件无关的批处理执行器（`campaign`）：单个 Rayon
  池、候选级/频点级自适应并行轴（互斥不嵌套，单池永不过订阅）、按候选索引有序
  写回、原子进度 + 协作取消且不入任何数值归约；`bw_from_gain` / `band_rms`
  归约按冻结 Python 路径 1:1 移植。AFE OTFT evaluator（`otft_campaign`）把原生
  `otft` 器件核、稠密 `mna` 求解与复数 `lti` MNA 组合成逐候选
  器件构建 → DC → AC → noise 流水线，经 `circuitopt_core.CompiledCampaign`
  暴露（`evaluate_batch` 在单个 `py.detach` 内跑完整 batch，零 per-candidate
  Python 回调）。硅 BSIM4 evaluator（freepdk45 / sky130 / tsmc28）同批接入：
  把 `co-pdk` numeric card——含 TSMC28 `to_bsim4_cards` 的 `mulu0 → u0`
  迁移率折叠（提取为 `co_pdk::apply_mulu0_fold` 并单测）——组合进
  `co-bsim4` handle、`bsim_transient::solve_dc`、4×4 端子线性化与逐器件
  BSIM4 噪声矩阵（遵守后端要求的 evaluate-then-noise 调用序）。仅供验证——
  尚未接入任何生产工作流（那是 R5-D）。AFE 对 cold-consistent Python 参考
  （新建 `PMOS_TFT` 小信号参数 → 同一 `LtiProblem` → 同一归约）逐位一致：
  增益/带宽最差 rel ~1e-15、IRN ~2e-16（`band_rms` 朴素求和 ULP）、DC
  工作点逐位相同。硅侧直接对冻结的 `ac_solve`/`noise_analysis` 路径
  （BSIM4 求值为纯函数，无 warm/cold 之分）：三家 5T OTA 几何×corner
  矩阵上增益最差 rel ~4e-16、带宽 ~2e-15、IRN ~1.7e-16，同 seed 的 Rust DC
  Newton 逐位复现 Python 工作点。结果在 workers ∈ {1,2,8} 下逐字节相同。
  诚实标注：AFE OTFT 内部 2 节点 Newton 停在 `tol=1e-12`，工作点随 seed 路径
  漂移——冻结的 warm-cache `corners.metrics` 路径与同模型 cold 求值在
  `gm`/`gds` 上可差 ~6e-8；campaign 走 cold-seed-consistent，故与 warm 路径
  仅一致到该固有地板。SKY130 的参考宽钉扎（`extract_w != W`）不在
  `CompiledPdk` 面内，硅 campaign 现阶段覆盖 `extract_w == W` 几何，待
  R5-B 面扩展补齐。

- **Compiled campaign wired into the design-space sweep (R5-D) / 编译式 campaign 接入设计空间扫描（R5-D）**

  **English:** The compiled campaign is now the batch executor for the
  design-space sweep (`benchmarks.bench_sweep`, the `--workers` design-space
  path) under the rust engine, via a new `circuitopt._campaign_sweep` dispatch
  layer (family detection + the cold-DC safety policy; returns to the frozen
  scalar reference when the engine is not rust or the circuit is not
  campaign-able). The whole candidate matrix runs under one Rayon pool with no
  per-candidate Python callback — proven two ways (a monkeypatch trap and a
  `sys.setprofile` frame counter both read zero PDK/device frames). Throughput
  (candidates/s, 8 vs 1 worker): FreePDK45 ~5.4×, TSMC28 ~2.2× (both ≥2×);
  index-ordered results are byte-identical across workers {1,2,8}.
  Prerequisites: `CompiledPdk::numeric_card` gained SKY130 `reference_width_um`
  (the `extract_w != W` explore path), pinned bit-for-bit vs `load_sky130_card`
  over every bundled geometry. A cold-DC behaviour gate
  (`tests/test_campaign_cold_dc.py`) proves the campaign's cold circuit Newton
  reaches the same physical branch as scipy `fsolve` on the monostable silicon
  OTAs (worst node dV freepdk45 1.3e-8 V / sky130 1.3e-9 V / tsmc28 4.5e-6 V,
  << the 1e-3 V calibration DC tol), with identical convergence rate; a 0-bin
  geometry is rejected identically by both engines and never sinks its batch.
  **Flagged / out of scope:** the AFE OTFT is multistable — a cold circuit
  Newton can pick a different branch than `fsolve` (~tens of volts), and even
  seeded from the nominal op it does not reproduce `fsolve`'s basin selection on
  bifurcation-edge (latch-prone) designs, so it under-reports `latch_rate`. The
  AFE latch/metric workflows (`corners.corner_table` / `mismatch_mc` /
  `latch_screen`, and the `mc` service job that calls them) therefore stay on
  the frozen scalar reference (both engines) and keep their R5-A worker pool —
  no silent root substitution. SAR mismatch MC is a closed-loop transient (not a
  DC→AC→noise campaign) and its native-backend conversions rebuild each bit's
  problem in Python, so its 8-thread scaling stays GIL-bound (~0.13 efficiency
  on this ngspice-less host); this is a pre-existing SAR/transient-architecture
  limit, untouched here, and the byte-identity worker contract still holds.

  **中文：** 编译式 campaign 现作为设计空间扫描（`benchmarks.bench_sweep` 的
  `--workers` 路径）在 rust 引擎下的批处理执行器，经新增的
  `circuitopt._campaign_sweep` 分派层接入（器件族识别 + 冷 DC 安全策略；非
  rust 引擎或电路不可编译时回退到冻结的 Python 标量参考）。整个候选矩阵在
  单个 Rayon 池内跑完、零 per-candidate Python 回调——双重证明（monkeypatch
  陷阱 + `sys.setprofile` 帧计数器均读到零 PDK/device 帧）。吞吐（候选/秒，
  8 vs 1 线程）：FreePDK45 ~5.4×、TSMC28 ~2.2×（均 ≥2×）；按索引有序的结果在
  workers {1,2,8} 下逐字节相同。前置：`CompiledPdk::numeric_card` 新增 SKY130
  `reference_width_um`（`extract_w != W` 的 explore 路径），对 `load_sky130_card`
  在全部 bundled 几何上逐位钉死。冷 DC 行为门（`tests/test_campaign_cold_dc.py`）
  证明 campaign 冷启动电路 Newton 在单稳的硅 OTA 上与 scipy `fsolve` 落在同一
  物理分支（最差节点 dV：freepdk45 1.3e-8 V / sky130 1.3e-9 V / tsmc28
  4.5e-6 V，远小于 1e-3 V 校准 DC 容差），收敛率一致；0-bin 几何两引擎同报错
  且不拖累同批。**诚实标注/超出范围：** AFE OTFT 多稳——冷启动电路 Newton
  可选到与 `fsolve` 不同的分支（~几十伏），即便从标称工作点 seed 也无法在
  分岔边缘（易 latch）设计上复现 `fsolve` 的盆地选择，故会低报 `latch_rate`。
  因此 AFE 的 latch/metric 工作流（`corners.corner_table` / `mismatch_mc` /
  `latch_screen`，及调用它们的 `mc` service job）仍走冻结的标量参考（两引擎）
  并保留 R5-A 线程池——不做静默换根。SAR mismatch MC 是闭环 transient（非
  DC→AC→noise campaign），其原生后端逐 bit 在 Python 重建问题，故 8 线程扩展
  仍受 GIL 限制（本无 ngspice 机器上 ~0.13 效率）；这是既有 SAR/transient
  架构限制，本期未动，逐位一致的 worker 契约仍成立。

### Changed / 变更

- **Removed the OSDI/OpenVAF compatibility path / 删除 OSDI/OpenVAF 兼容路径**

  **English:** The Rust refactor now standardizes silicon simulation on the
  native BSIM4 backend. Removed the OSDI host/device/transient modules,
  OpenVAF discovery and compile tooling, SKY130 OSDI model registration and
  dispatch, OSDI-specific Numba kernels, tests, and current documentation.
  Explicit ngspice and Cadence regression oracles remain available.

  **中文：** Rust 重构现将硅工艺仿真统一到原生 BSIM4 后端。已删除 OSDI
  host/device/transient 模块、OpenVAF 路径解析与编译工具、SKY130 OSDI 模型
  注册与分派、OSDI 专用 Numba 内核、测试和当前使用文档。显式 ngspice 与
  Cadence 回归 oracle 继续保留。

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

[Unreleased]: https://github.com/751K/circuit-optimization-lab/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/751K/circuit-optimization-lab/compare/v1.0.5...v1.1.0
[1.0.5]: https://github.com/751K/circuit-optimization-lab/compare/v0.1.0...v1.0.5
[0.1.0]: https://github.com/751K/circuit-optimization-lab/releases/tag/v0.1.0
