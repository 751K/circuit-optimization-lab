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

### Added
- **FreePDK45 MDAC 第一级 OTA(14-bit 流水线 ADC)/ FreePDK45 MDAC first-stage OTA
  (14-bit pipeline ADC)** — 全差分两级 OTA(望远镜级联 + NMOS 共源二级,Miller 补偿
  + 调零三极管),面向 14-bit/100 MS/s 流水线 ADC 第一级 MDAC(残差增益 8,5 ns 保持相,
  0.90/1.00/1.10 V × −40/27/125 °C × tt/ss/ff/sf/fs)。从 ADC 规格完整推导
  (SNR 72 dB → 噪声预算 → CDAC Cs=2.6 pF → Cf=325 fF/β → UGBW/建立 → 增益/摆幅,
  `docs/mdac_ota_derivation.md`);DUT 全晶体管,单一理想量为测试台 20 µA 基准,内部偏置
  为「镜像骨干 + 四条复制-Vgs+poly-R 参考腿」(VBNC/VBPC/VREF1 二级电流复制跟踪/VCMIN
  输入虚地参考),全 PVT 跟踪。六个测试台 JSON 由 `examples/mdac_ota_gen.py` 单源生成
  (闭环残差瞬态/开环 AC/差分环增益(差分 Middlebrook,TB 侧镜像 VCVS)/两条 CMFB 环/
  闭环噪声)。11 点 mini-PVT 全过:增益 91.6–103.4 dB、DM PM 100–129°、CMFB PM>75°、
  静态 CM 误差 ≤13 mV、满幅残差建立 ≤0.07 %FS、核心管全饱和;噪声 213 µV rms(预算 452)、
  功耗 12.9 mW @1 V。CI 测试 `tests/test_mdac_ota.py`(skip-guarded)。
  A fully-differential two-stage OTA (telescopic cascode + NMOS common-source second
  stage, Miller compensation with triode nulling devices) for the first-stage MDAC of a
  14-bit / 100 MS/s pipeline ADC (residue gain 8, 5 ns hold, 0.90/1.00/1.10 V ×
  −40/27/125 °C × tt/ss/ff/sf/fs). Full spec derivation from the ADC requirements
  (72 dB SNR → noise split → CDAC Cs=2.6 pF → Cf=325 fF/β → UGBW/settling → gain/swing)
  in `docs/mdac_ota_derivation.md`; the DUT is all-transistor with a single ideal 20 µA
  testbench reference — the internal bias is a mirror backbone plus four replica-Vgs +
  poly-R reference legs (VBNC/VBPC, VREF1 replica-tracked stage-2 current, VCMIN input
  virtual-ground CM), PVT-tracking by construction. Six testbench JSONs generated from
  one source of truth (`examples/mdac_ota_gen.py`): closed-loop residue transient,
  open-loop AC, differential loop gain (differential Middlebrook with a TB-side mirror
  VCVS), both CMFB loops, closed-loop noise. All 11 mini-PVT points pass every spec:
  gain 91.6–103.4 dB, DM PM 100–129°, CMFB PMs >75°, static CM error ≤13 mV, full-scale
  residue settling ≤0.07 %FS, all core devices saturated; 213 µV rms output noise
  (452 µV budget), 12.9 mW @ 1 V. CI tests in `tests/test_mdac_ota.py` (skip-guarded).
- **ngspice 瞬态求解器容差直通 / ngspice transient solver-tolerance passthrough** —
  `transient_ngspice`/`render_freepdk45_transient_netlist` 新增可选 `extra_options`
  (如 `{"reltol":1e-7,"vntol":1e-11,"abstol":1e-15}` → 额外 `.options` 行):0.1 % 级
  建立测量必需(ngspice 默认 reltol=1e-3 在 ~0.5 V 节点留 ~百 µV 数值带,DC 解差分残差
  存入采样电容后被 1/β 放大)。缺省 None 网表逐字节不变(golden 兼容)。
  `transient_ngspice`/`render_freepdk45_transient_netlist` gain an optional
  `extra_options` mapping (e.g. `{"reltol":1e-7,...}` → an extra `.options` line):
  required for sub-0.1 % settling measurements (ngspice's default reltol=1e-3 leaves a
  ~100 µV numerical band whose DC-solve differential residue is stored on the sampling
  caps and amplified by 1/β). Default None renders byte-identical decks.
- **circuit JSON schema 补全 / schema completion** — `schemas/circuit.schema.json`
  现覆盖 `vcvs`/`cccs`/`ccvs` 顶层块(loader/渲染器一直支持,schema 此前缺失,含这三类
  元素的电路会误报 schema 违例),并补上 `adc.clock` 与 `adc.mismatch` 子对象
  (`circuitopt.sar`/`sar_mc` 已消费,schema 缺失导致 `freepdk45_sar6.json` 校验失败)。
  向后兼容。`schemas/circuit.schema.json` now validates the `vcvs`/`cccs`/`ccvs`
  top-level blocks (always supported by the loader/renderers, previously missing) and
  the `adc.clock` / `adc.mismatch` sub-objects (consumed by `circuitopt.sar`/`sar_mc`;
  their absence failed validation of `freepdk45_sar6.json`). Backward compatible.
- **PVT 机理：混合角 sf/fs + 全电路 ngspice oracle / PVT machinery: mixed sf/fs corners
  + full-circuit ngspice oracles** — FreePDK45 角名新增混合角 `sf`(NMOS 慢 + PMOS 快)、
  `fs`(反之) 与 `tt`(= `nom` 别名)：角名现按极性各自选卡目录
  (`freepdk45_model.corner_card_dir`)，特征化栅格路径与全电路 ngspice 渲染两条路径都支持；
  栅格缓存按角**名**建键(`sf` NMOS 用 `ss` 卡但独立缓存，永不与 `ss` 撞键)；两极性不同 (`sf`/`fs`)
  时网表 `.include` **两个**卡文件；nom/ss/ff 网表与改动前逐字节一致(golden 锁定)。新模块
  `circuitopt/ngspice_ac.py` 提供四个全电路 oracle:`ac_ngspice`(`.ac dec`，复数传函 + `ac_response`/
  `dc_gain_db`/`peak_gain_db`/`unity_gain_freq`/`phase_margin`/`gain_margin_db` 辅助;FD-OTA 例
  58.9 dB/119.9 MHz/84°)、`noise_ngspice`(`.noise` 输出/输入折合 PSD + 频带 rms;裸电阻 4kTR)、
  `op_ngspice`(`.op` 逐器件 vds/vgs/vdsat/id/gm/gds + `region_ok=|vds|>=|vdsat|+margin` 饱和判定)、
  `loop_gain_ngspice`(Middlebrook 单电压注入环路增益 + PM/GM)。四者共享渲染层
  `circuitopt/ngspice_render.py`(`.tran` 后端一并复用),温度/角(含 sf/fs)/电源均生效。角词表
  `sf/fs/tt` 在 `FREEPDK45_CORNERS`、`binding.at_corner`、CLI `--corner`、service capabilities 全线接受。
  Mixed corners `sf` (NMOS slow + PMOS fast), `fs` (reverse) and the `tt`↔`nom` alias for
  FreePDK45: a corner name now selects the `models_<dir>` card directory PER POLARITY
  (`freepdk45_model.corner_card_dir`), honored on both the characterisation-grid path and
  the full-circuit ngspice render; the grid cache keys on the corner NAME (so an `sf` NMOS
  grid, built from the `ss` card, is cached separately and never collides with `ss`); when
  the polarities differ (`sf`/`fs`) the deck `.include`s BOTH cards; nom/ss/ff decks are
  byte-identical to before (golden-locked). New `circuitopt/ngspice_ac.py` adds four
  full-circuit ngspice oracles — `ac_ngspice` (`.ac dec`, complex node transfer + the
  `ac_response`/`dc_gain_db`/`peak_gain_db`/`unity_gain_freq`/`phase_margin`/`gain_margin_db`
  helpers; FD-OTA reads 58.9 dB / 119.9 MHz / 84°), `noise_ngspice` (`.noise` output &
  input-referred PSD + integrated-band rms; a bare resistor reads 4kTR), `op_ngspice`
  (`.op` per-device vds/vgs/vdsat/id/gm/gds with `region_ok = |vds| >= |vdsat| + margin`
  saturation check), and `loop_gain_ngspice` (Middlebrook single-voltage-injection loop
  gain + PM/GM). All share the deck renderer `circuitopt/ngspice_render.py` (also reused by
  the `.tran` backend) and honor temperature / corner (incl. sf/fs) / supply. The `sf/fs/tt`
  vocabulary is accepted across `FREEPDK45_CORNERS`, `binding.at_corner`, the CLI `--corner`
  flag and the service capabilities. Docs: `docs/ngspice_oracles.md`.
  角名校验收紧(campaign 安全)：FreePDK45 角名大小写不敏感(`"SF"` 即 `sf`)，`None`/`""`
  仍为 `nom`；**未知角名(如 `"sx"`)在栅格路径上现在抛 `ValueError`**(列出合法集)，不再
  静默回退 nom——与 ngspice 渲染路径的硬错误一致(`freepdk45_model.normalize_corner`)。
  Corner-name validation hardened (campaign safety): FreePDK45 corner names are
  case-insensitive (`"SF"` behaves as `sf`), `None`/`""` still mean `nom`, and an
  **unknown name (e.g. `"sx"`) now raises `ValueError`** naming the valid set on the
  grid path instead of silently falling back to nom — matching the ngspice render
  path's hard error (`freepdk45_model.normalize_corner`).
- **SAR ADC 出图 + `adc --plot`/`--mc` / SAR ADC figures + `adc --plot`/`--mc`** — 新增
  `examples/plot_adc.py`，四类 SAR ADC 图（沿用 `plot_bode`/`plot_transient` 约定，headless
  Agg、返回保存路径）：`plot_sar_static`（传输阶梯 + 逐码 DNL + 逐 transition INL，±0.5 LSB
  参考线、缺码标记、max|DNL|/max|INL| 标注）、`plot_sar_spectrum`（归一化 dBc 功率谱、线性
  频率轴、基波 + 2..5 次谐波标注、SNDR/SNR/SFDR/ENOB 文本框）、`plot_sar_conversion`（单次
  转换逻辑分析视图：采样/逐位 CDAC 驱动/顶板/比较器节点，逐位物理判决时刻标注 kept·cleared，
  节点/键名从 spec 的 `adc` 块推导，3-bit 无 clk 与 6-bit 有 clk 同一套代码渲染）、
  `plot_sar_mc`（max|DNL|/max|INL|/offset 直方图、阈值线 + yield 标注，非单调 trial 的 ∞ 落入
  显式溢出桶）。`adc` 子命令新增 `--plot [DIR]`（默认 `results/`，按模式渲染对应图；缺 matplotlib
  时与 `plot` 子命令一致地干净退出）与第四种运行模式 `--mc N`（逐器件 mismatch MC，复用电路
  `adc.mismatch` 配置与 `--seed`/`--workers`/`--corner`）。文档 `docs/cli_reference.md` 同步。
  New `examples/plot_adc.py` with four SAR ADC figure kinds (following the
  `plot_bode`/`plot_transient` conventions — headless Agg, each returns its saved
  path): `plot_sar_static` (transfer staircase + per-code DNL + per-transition INL with
  ±0.5 LSB reference lines, missing-code markers, max|DNL|/max|INL| annotations),
  `plot_sar_spectrum` (fundamental-normalized dBc power spectrum on a linear frequency
  axis, fundamental + harmonics 2..5 marked, SNDR/SNR/SFDR/ENOB text box),
  `plot_sar_conversion` (logic-analyzer view of one conversion: sample controls, per-bit
  CDAC drives, top plates and the comparator node, with each bit's physical decision
  instant marked kept/cleared; node/key names are derived from the spec's `adc` block so
  the 3-bit no-clk and 6-bit clocked cases render from one code path), and `plot_sar_mc`
  (max|DNL|/max|INL|/offset histograms with threshold lines + yield annotation, non-
  monotonic `inf` trials clipped into a labeled overflow bin). The `adc` subcommand gains
  `--plot [DIR]` (default `results/`, renders the figure matching the run mode; degrades
  with the same clean SystemExit as `plot` when matplotlib is absent) and a fourth run
  mode `--mc N` (per-instance mismatch MC reusing the circuit's `adc.mismatch` config and
  `--seed`/`--workers`/`--corner`). `docs/cli_reference.md` updated.
- **6-bit 差分 SAR ADC 设计案例 + 时钟同步动态比较器 / 6-bit differential SAR ADC design
  case + clocked dynamic comparator** — 新增完整设计案例 `examples/freepdk45_sar6.json`：
  6-bit 全差分共模翻转 CDAC(32/16/8/4/2/1C + dummy，单位电容 2 fF/边总 128 fF)配一个
  **时钟同步 StrongARM 动态锁存比较器**。为在既有判决抽取约束(在 `decision_time` 处对
  `comparator_node` 电压插值、与静态 `comparator_threshold` 比较)下驱动真正的动态锁存,
  给 SAR 机制新增了**可选、向后兼容**的 `adc.clock` 选通块:`sar_input_waveforms` 生成一路
  选通波形——CDAC 建立期间保持复位电平、每个 bit 的 `decision_time` 附近脉冲到评估电平,
  锁存器复位期预充、判决时刻再生到轨。因为每次转换只在被试 bit 的 `decision_time` 读比较器、
  且每个 bit 从 t=0 重放,一条固定的逐 bit 选通模式(与 `trial_index`、判决无关)即可服务所有
  重放。无 `clock` 块时不生成选通波形,故静态比较器的 `freepdk45_sar3` 渲染出字节级一致的网表。
  头条结果(经本地 ngspice 实跑):nom/ss/ff 三角全部 64 个码心正确转换、`max|DNL|`/`max|INL|`
  均为 0(理想电容+理想驱动);128 样本 13 周期相干正弦 **SNDR 36.9 dB / ENOB 5.84 bit /
  SFDR 44.1 dB**;每次转换功耗约 **7.2 µW**、能量约 **0.54 pJ**;失配 MC 见设计报告。新增
  探索配置 `examples/freepdk45_sar6_explore.json`、设计报告
  `docs/freepdk45_sar_design.md`、测试 `tests/test_freepdk45_sar6.py`;schema 与 JSON 格式
  文档同步 `adc.clock`。
  A complete design case `examples/freepdk45_sar6.json`: a 6-bit fully-differential
  common-mode-switching SAR (32/16/8/4/2/1C + dummy CDAC, 2 fF unit → 128 fF/side)
  with a **clocked StrongARM dynamic latched comparator**. To drive a real dynamic
  latch under the existing decision-extraction constraint (interpolate
  `comparator_node` at `decision_time`, compare to a static `comparator_threshold`),
  the SAR machinery gains an **optional, backward-compatible** `adc.clock` strobe
  block: `sar_input_waveforms` emits one strobe waveform that rests at the reset
  level during CDAC settling and pulses to the evaluate level around every bit's
  `decision_time`, so the latch precharges during reset and regenerates to the rails
  at the decision instant. Because each conversion reads the comparator only at the
  trial bit's `decision_time` and replays every bit from t=0, one fixed per-bit
  strobe pattern (independent of `trial_index` and the decisions) serves every
  replay; with no `clock` block no strobe is emitted, so the static-comparator
  `freepdk45_sar3` renders a byte-identical netlist. Headline results (measured on
  the bundled ngspice): all 64 code centers convert correctly at nom/ss/ff with
  `max|DNL|`/`max|INL|` = 0 (ideal caps + ideal CDAC drivers); a 128-sample,
  13-cycle coherent sine gives **SNDR 36.9 dB / ENOB 5.84 bit / SFDR 44.1 dB**;
  power ≈ **7.2 µW**/conversion, energy ≈ **0.54 pJ**; mismatch MC in the report.
  Adds the explore config `examples/freepdk45_sar6_explore.json`, the design report
  `docs/freepdk45_sar_design.md`, and tests `tests/test_freepdk45_sar6.py`; schema
  and the JSON format doc document `adc.clock`.
- **SAR ADC 逐器件失配蒙特卡洛 / Per-instance mismatch Monte-Carlo for the SAR ADC** —
  新增 `circuitopt.sar_mismatch_mc`（`circuitopt/sar_mc.py`），为 FreePDK45/ngspice
  的闭环 SAR 工作流补上硅工艺侧的失配 MC（此前 `circuitopt.corners` 的逐器件失配只作用于
  本地 OTFT 求解器，不覆盖 ngspice 路径）。两类失配：晶体管阈值电压偏移作为 BSIM4 实例
  参数 `delvto` 注入（经本地 ngspice 验证会真实移动漏极电流），sigma 按 Pelgrom 面积律
  `sigma_vth0 / sqrt(W*L / (w0*l0))` 缩放，可分 N/P 管；CDAC 单位电容相对扰动
  `sigma_cu / sqrt(C / c_unit)`，通过复制 spec 拓扑逐次施加、不改动已加载的 spec。每次
  试验跑码心扫描并汇总 `max_abs_dnl` / `max_abs_inl` / `missing_codes` / 首次跳变偏移，
  给出 mean/std/worst 与对 ±0.5 LSB 阈值的良率。为 `transient()` →
  `transient_ngspice` → `render_freepdk45_transient_netlist` 及
  `run_sar_conversion` / `run_sar_sweep` 增加 keyword-only 的 `mismatch` 入参
  （默认 `None`，字节级复现旧网表）。新增可选 `adc.mismatch` JSON 配置块（schema + 中英
  文档同步）。新增 `tests/test_sar_mc.py`。
  A new `circuitopt.sar_mismatch_mc` (`circuitopt/sar_mc.py`) adds silicon-side
  mismatch MC to the FreePDK45/ngspice closed-loop SAR workflow — the gap left by
  `circuitopt.corners`, whose per-device mismatch only reaches the local OTFT
  solver. Two families: transistor Vth offsets injected as the BSIM4 instance
  parameter `delvto` (verified against the bundled ngspice to actually shift drain
  current), sigma area-scaled by Pelgrom's law `sigma_vth0 / sqrt(W*L / (w0*l0))`
  with optional per-polarity split; and CDAC unit-capacitor perturbations
  `sigma_cu / sqrt(C / c_unit)`, applied on a per-trial copy of the spec topology so
  the loaded spec is never mutated. Each trial runs the code-center sweep and
  reports `max_abs_dnl` / `max_abs_inl` / `missing_codes` / first-transition offset,
  plus mean/std/worst and a yield against ±0.5 LSB limits. A keyword-only `mismatch`
  arg was threaded through `transient()` → `transient_ngspice` →
  `render_freepdk45_transient_netlist` and `run_sar_conversion` / `run_sar_sweep`
  (default `None` renders the byte-identical legacy netlist). A new optional
  `adc.mismatch` JSON block (schema + bilingual docs) configures it. New tests in
  `tests/test_sar_mc.py`.
- **SAR 转换并行化 / Parallel independent SAR conversions** — `run_sar_sweep`、
  `run_sar_signal` 与 `sar_mismatch_mc` 新增 keyword-only 的 `workers`（默认 1）。
  由于每次转换独立、无共享可变状态（`run_sar_conversion` 每次重建
  `spec.binding().at_corner()`，每个 ngspice `.tran` 在独立临时目录里以子进程运行、
  释放 GIL），用 `ThreadPoolExecutor` 跨转换（扫描/信号）与跨试验（MC）并行；转换内部
  的逐位判决仍严格串行。`workers=1` 保持原串行代码路径，任意 worker 数结果保序且与串行
  逐字节一致。MC 为保证与完成顺序无关的可复现性，将所有试验的随机抽样在开始前按试验序一次
  性抽完，再并行评估（`progress` 仍在主线程按单调递增的完成计数触发，运行摘要聚合已完成
  的试验，完成顺序在 `workers>1` 时不确定但最终结果确定）。CLI `circuit-opt adc` 的
  sweep/sine 路径新增 `--workers`。3-bit 例子上 8 点扫描 `workers=4` 相对 `workers=1`
  实测约 3.3× 加速。
- **SAR ADC 设计空间探索 / SAR ADC design-space exploration** — 新增
  `circuitopt/sar_explore.py`：把 SAR 工作流接入探索回路（`explore.py` 的 ADC 版）。
  复用 `explore` 的 `Variable`/`sample`/`pareto_front`/`is_feasible`/`write_csv`/
  `write_jsonl`（不重复实现），并新增两类目标：`"C:<name>"` 设置 CDAC 电容值 [F]、
  `"W:M1"`/`"L:M1"`/`"NF:M1"` 冒号前缀的尺寸目标（原点号形式 `"M1.W"` 也兼容）；电容改动
  施加在拓扑的浅拷贝上（同 `sar_mc.perturb_capacitors`，抽取为共享的
  `_copy_with_capacitors`），从不改动已加载的 spec。`evaluate_sar` 跑码心静态扫描
  （可用 `sweep_points` 子采样提速）并产出 `max_abs_dnl`/`max_abs_inl`/`missing_codes`/
  `monotonic`/`power_uw`/`conv_time_ns`/`energy_per_conv_pj`，可选（配置 `"dynamic"` 时）
  相干正弦的 `enob`/`sndr_db`/`sfdr_db`。驱动 `sar_explore` 用 `workers` 跨候选并行
  （候选内扫描仍串行，不嵌套线程池）。独立配置文件（含 `"circuit"` 指向电路 JSON）由
  `load_sar_explore_json`/`sar_explore_from_dict` 加载。CLI：
  `circuit-opt adc <circuit.json> --explore <config.json> [-n N] [--seed S]
  [--workers W] [--csv out.csv] [--jsonl out.jsonl]`（与 `--vin/--sweep/--sine`
  互斥，位置电路参数为准）。新增示例 `examples/freepdk45_sar3_explore.json` 与测试
  `tests/test_sar_parallel.py`、`tests/test_sar_explore.py`。
  Added keyword-only `workers` (default 1) to `run_sar_sweep`, `run_sar_signal` and
  `sar_mismatch_mc`. Independent conversions carry no shared mutable state
  (`run_sar_conversion` rebuilds `spec.binding().at_corner()` per call and each
  ngspice `.tran` runs as a GIL-releasing subprocess in its own temp dir), so a
  `ThreadPoolExecutor` parallelises across conversions (sweep/signal) and across
  trials (MC) while the per-bit decisions inside one conversion stay strictly
  serial. `workers=1` keeps the exact serial path; any worker count is
  order-preserving and byte-identical to it. The MC draws every trial's random
  offsets/perturbations up front, in trial order, from the single seeded RNG before
  evaluating in parallel, so results are seed-deterministic regardless of completion
  order (the `progress` callback still fires from the main thread with a monotonic
  completed count; its running summary aggregates whichever trials have finished —
  completion order is non-deterministic under `workers>1`, the final result is not).
  A new `circuitopt/sar_explore.py` wires the SAR workflow into the exploration loop
  — the ADC-metric sibling of `explore.py` — reusing explore's `Variable` / `sample`
  / `pareto_front` / `is_feasible` / `write_csv` / `write_jsonl` (imported, not
  duplicated). Two new target kinds: `"C:<name>"` sets a CDAC capacitor value [F] and
  colon-prefixed `"W:M1"`/`"L:M1"`/`"NF:M1"` size targets (the native dotted `"M1.W"`
  form still works); capacitor edits land on a shallow topology copy (the
  `sar_mc.perturb_capacitors` pattern, factored into a shared `_copy_with_capacitors`)
  so the loaded spec is never mutated. `evaluate_sar` runs a code-center static sweep
  (optionally subsampled via `sweep_points`) and reports `max_abs_dnl` /
  `max_abs_inl` / `missing_codes` / `monotonic` / `power_uw` / `conv_time_ns` /
  `energy_per_conv_pj`, plus optional coherent-sine `enob` / `sndr_db` / `sfdr_db`
  when a `"dynamic"` block is configured. The `sar_explore` driver parallelises
  across candidates with `workers` (each candidate's sweep stays serial — no nested
  pools). A standalone config file (with a `"circuit"` pointer) loads via
  `load_sar_explore_json` / `sar_explore_from_dict`. CLI:
  `circuit-opt adc <circuit.json> --explore <config.json> [-n N] [--seed S]
  [--workers W] [--csv out.csv] [--jsonl out.jsonl]` (mutually exclusive with
  `--vin/--sweep/--sine`; the positional circuit is authoritative). New example
  `examples/freepdk45_sar3_explore.json` and tests `tests/test_sar_parallel.py`,
  `tests/test_sar_explore.py`. A workers=4 vs workers=1 8-point sweep on the 3-bit
  example measured ~3.3x speedup.

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

[Unreleased]: https://github.com/751K/circuit-optimization-lab/compare/v1.0.5...HEAD
[1.0.5]: https://github.com/751K/circuit-optimization-lab/compare/v0.1.0...v1.0.5
[0.1.0]: https://github.com/751K/circuit-optimization-lab/releases/tag/v0.1.0
