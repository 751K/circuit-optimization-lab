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

### Added / 新增

- **Analog design-loop tooling / 模拟设计环路工具**

  **English:** Four gaps found while taking a 14-bit pipeline MDAC OTA to a
  45/45 PVT signoff, each previously covered by a throwaway script.

  `tools/design_iterate.py` drives a full PVT campaign from a generator
  module, overriding constants in memory and staging decks in a temp
  directory. `run` prints per-spec pass counts **plus the corner list behind
  every failing constraint** — the stock campaign reports one global
  `worst_case`, which is the wrong summary for a design decision because the
  variant with the better worst margin routinely has fewer passing points.
  `map` reports one measurement's distribution over the grid with its mean
  grouped along each PVT axis (a temperature-dominated spread is a device
  offset; a supply-dominated one is a reference mismatch) and the optimal
  common trim for signed errors. `trace` dumps node trajectories at one point.

  `circuit-opt signoff --margins` prints a per-constraint margin table ordered
  tightest-first; `--tolerance R=20,C=10` re-runs with passives perturbed by
  class or by element and reports which constraints break. A signoff that
  holds only at nominal component values is not a signoff.

  `circuit-opt verify-engine` runs one signoff case through both the native
  BSIM4 engine and the ngspice model-card path and diffs the trajectories,
  keying the verdict on the settled deviation. The defect this exists for
  drifted a common mode to +0.42 V, failed every residue case at every PVT
  point, and left the golden corpus bit-exact because no golden case
  exercised that path.

  `circuit-opt passive-bom` inventories every resistor and capacitor with a
  silicon-area estimate, deriving DUT-vs-testbench membership from which of a
  manifest's decks an element appears in, and prints the passive/active area
  ratio. On the MDAC design it shows one 40 pF compensation capacitor holding
  64% of the passive area and the passives outweighing every transistor by
  21x — true since the compensation was tuned, and invisible until something
  added it up.

  **中文：** 把 14-bit 流水线 MDAC OTA 推到 45/45 PVT 签核的过程中暴露出四个
  工具缺口，此前都靠临时脚本顶着。

  `tools/design_iterate.py` 用生成器模块驱动整个 PVT campaign，在内存里覆盖常量、
  把网表暂存到临时目录。`run` 打印逐规格通过数**以及每条失败约束背后的角点清单**
  ——内置 campaign 只报一个全局 `worst_case`，那对设计决策是错误的汇总，因为最差
  余量更好的变体往往通过点更少。`map` 报告某测量量在网格上的分布及按每条 PVT 轴
  分组的均值（跨度落在温度轴＝器件失调，落在电源轴＝参考不匹配），并给出有符号
  误差的最优公共修调。`trace` 输出单点节点轨迹。

  `circuit-opt signoff --margins` 打印按最紧优先排序的逐约束余量表；
  `--tolerance R=20,C=10` 按器件类或单个器件扰动无源器件后重跑，报告哪条约束先破。
  只在标称值下成立的签核不算签核。

  `circuit-opt verify-engine` 把同一个 signoff case 交给内部 BSIM4 引擎与 ngspice
  模型卡路径分别求解并对比轨迹，判定基于稳定后偏差。它要防的那个缺陷曾把共模推到
  +0.42 V、让全部 45 点的 residue case 全灭，而 golden 语料始终逐位通过——因为没有
  任何 golden case 走那条路径。

  `circuit-opt passive-bom` 枚举全部电阻电容并估算硅面积，由"元件出现在 manifest 的
  哪些 deck 中"推导 DUT 与测试台件，给出无源/有源面积比。在 MDAC 设计上它显示单个 40 pF 补偿
  电容占了无源面积的 64%，无源总面积是全部晶体管的 21 倍——自补偿整定以来一直如此，
  只是没有任何视图把它加起来。

## [2.5.0] - 2026-07-27

### Added / 新增

- **`tools/workers.py`: per-machine `--workers` advice / `tools/workers.py`：按机器给 `--workers` 建议**

  **English:** The worker counts quoted in the docs were measured on one
  4P+6E Apple M4 and do not transfer. The new tool detects the machine's
  topology (macOS `hw.perflevel*`, Intel-hybrid Linux sysfs, uniform
  fallback elsewhere; stdlib-only) and prints per-workload recommendations
  from the measured scheduling rules: conversion-parallel modes
  (ramp/sine/explore/transitions) use every logical core — E cores carry
  real load — while a mismatch MC with up to `4x` cores' worth of trials
  runs one trial per task (`workers = trials`; 16 trials on 10 cores are
  faster at 16 workers than at 10, whose 2-vs-1 trial split leaves a tail)
  and larger MCs use the core count. `--mc-trials N` answers for a concrete
  run, `--json` is machine-readable, and `--calibrate` measures the real
  saturation knee with the compiled SAR ramp (~8 s), preferring the
  smallest worker count within 5% of the fastest and recording everything
  to `results/workers_calibration.json`; on the reference machine it
  resolved the knee to 16 workers at 6.1x over serial. Unit tests pin the
  topology contract, the recommendation rules and the knee picker
  (mutation-verified); a CLI smoke pins the JSON shape.

  **中文：** 文档里引用的 worker 数是在一台 4P+6E Apple M4 上测的，换机器不成立。
  新工具用纯标准库侦测本机拓扑（macOS `hw.perflevel*`、Intel 混合架构 Linux 走
  sysfs，其余按均匀核回退），并按实测调度规则打印逐工作负载建议：转换级并行模式
  （ramp/sine/explore/transitions）用全部逻辑核——E 核承载真实负载；失配 MC 在
  trial 数不超过 4 倍核数时一 trial 一任务（`workers = trials`；10 核上 16 个
  trial 用 16 worker 反而比 10 快——后者 2/1 分摊必留尾巴），更大的 MC 用核数。
  `--mc-trials N` 回答具体某次运行，`--json` 机器可读，`--calibrate` 用编译 SAR
  ramp 实测饱和拐点（约 8 s），在最快值 5% 容差内取最小 worker 数并把全部数据写进
  `results/workers_calibration.json`；参考机上它把拐点定在 16 worker、对串行
  6.1 倍。单元测试钉死拓扑契约、建议规则与拐点选取（经变异验证）；CLI smoke 钉死
  JSON 形状。

- **Transition bisection: final-verification DNL at search cost / 码界二分：以搜索代价做终验 DNL**

  **English:** `adc --transitions` (API `run_sar_transitions`) locates
  physical code-transition voltages by lockstep bisection — the converter
  itself is the search oracle, `T(k)` being the smallest input whose
  conversion reaches code `k`. Every round batches all pending probes
  through one compiled SAR call, so `--workers` parallelises across
  transitions and the serial depth is the round count (7 with the default
  2-LSB bracket; a bracket that screening mispredicted widens to the full
  range automatically, and a transition outside the input range reports NaN,
  never a guess). The default target set covers both DNL bins around every
  binary major carry plus the offset transition — where a binary-weighted
  CDAC concentrates its worst DNL — and `transition_dnl_inl` reduces the
  measured boundaries to DNL/INL, reporting only bins whose both edges were
  measured. This closes the loop the subsampled screening mode opens: sparse
  ramps bound |INL| but cannot measure DNL, and even a full code-center ramp
  quantizes every transition to ±0.5 LSB, while bisection reaches `--tol-lsb`
  (default 0.05) directly. On the nominal 6-bit example: 14 transitions in
  125 conversions and 1.75 s at 8 workers, resolving a real, uniform
  −0.17 LSB INL (a global offset the ideal-staircase ramp could not see) and
  DNL within 0.031 LSB of zero; results are bit-identical across worker
  counts. At 12 bits the carry set is ~34 transitions ≈ 310 conversions
  against a 4096-conversion ramp. Pure-numpy unit tests pin the bisection
  invariant, the bracket-recovery path, the unmeasured contract and the
  reducer (mutation-verified); simulator tests pin the nominal carries,
  mismatch sensitivity, worker-count identity, and the CLI guards.

  **中文：** `adc --transitions`（API `run_sar_transitions`）用锁步二分定位物理
  码界电压——转换器自己就是搜索的 oracle，`T(k)` 即转换结果首次达到码 `k` 的最小
  输入。每轮把所有未收敛探针合成一次编译 SAR batch 调用，`--workers` 跨跃变并
  行，串行深度只是轮数（默认 2 LSB 括号下 7 轮；筛查失真导致括号不套住跃变时自动
  放宽到全量程，跃变落在输入范围之外则报 NaN，绝不编造）。默认目标集覆盖每个二进
  制 major carry 两侧的 DNL bin 加失调跃变——二进制加权 CDAC 的最差 DNL 就集中在
  那里——`transition_dnl_inl` 把实测边界归约为 DNL/INL，只报两侧边界都测到的
  bin。这补上了子采样筛查模式的闭环：稀疏 ramp 能约束 |INL| 但测不了 DNL，而全码
  中心 ramp 也只能把每个跃变量化到 ±0.5 LSB，二分则直接收敛到 `--tol-lsb`（默认
  0.05）。名义 6-bit 示例：14 个跃变、125 次转换、8 worker 下 1.75 s，分辨出真实
  存在的均匀 −0.17 LSB INL（全局失调，理想阶梯 ramp 根本看不见）与 0.031 LSB 内
  的 DNL；各 worker 数结果逐位一致。12-bit 的 carry 集约 34 个跃变 ≈ 310 次转换，
  对照 4096 次的全码 ramp。纯 numpy 单元测试钉死二分不变量、括号恢复路径、不可测
  契约与归约器（经变异验证）；仿真测试钉死名义 carry、失配敏感性、worker 数一致
  性与 CLI 守卫。

- **Subsampled SAR sweeps with honest code-error metrics / SAR 子采样扫描与诚实的码误差指标**

  **English:** A 12-bit static ramp is 4096 conversions and the full-density
  cost scales as `2**n * (n+1)`, so sparse ramps are the only affordable
  screening mode at that resolution — but the toolchain's only subsampling
  hook (`sar_explore`'s `sweep_points`) scored garbage: a **perfect** 6-bit
  converter subsampled at 16 of 64 centers reported `missing_codes = 48` (the
  unsampled codes counted as missing — 4032 at 12 bits) and `max_abs_dnl =
  3.5` (adjacent samples alias several code boundaries onto one transition
  midpoint, so the transition metrics read wrong, not merely incomplete), so
  any constraint or objective on them failed or misled every candidate.

  The new `adc.sampled_transfer_metrics` measures what a sparse ramp actually
  determines: the signed **code error** at each sample (`|INL|` at a sample is
  within half an LSB of `|code error|`), plus monotonicity. Everything below
  full density now reports transition DNL/INL — and missing codes, which a
  sparse ramp cannot prove present or absent, and whose "expected vs produced"
  bookkeeping misfires on a plain +1-code offset — as unmeasured NaN, never as
  aliased numbers. Wired end to end: `run_sar_sweep` returns the code-error
  metric family (with `subsampled: True`) below `2**n_bits` samples and gains
  `max_abs_code_err` at full density too; the CLI `--sweep` floor drops from
  `2**n_bits` to 2; `--plot` renders a staircase + signed-code-error figure
  for sparse sweeps instead of the aliased DNL/INL panels the showcase used
  to stamp "sub-sampled" onto; explore's `sweep_points` scores clean and
  `max_abs_code_err` joins the constraint/objective vocabulary; and mismatch
  MC accepts `adc.mismatch.sweep_points` (CLI `--sweep-points`), converts the
  subsampled inputs through the same compiled Rust batch, and gates yield on a
  new `code_err_threshold` (default 0.5 LSB — every sampled center must read
  its own code). Measured on the 6-bit example at 16 of 64 points: sweep
  compute 0.69 s -> 0.19 s and an 8-trial MC 5.24 s -> 1.37 s (both about
  3.7x of the 4x conversion-count bound), identical across worker counts;
  MC's full-density rows, gates and summaries are unchanged. One deliberate
  behavior change: a `run_sar_sweep` call with fewer samples than `2**n_bits`
  used to return the full transition-metric schema computed on aliased data
  and now returns the code-error schema — the parity tests that probed those
  fields now probe the code-error family.

  **中文：** 12-bit 静态 ramp 是 4096 次转换，全密度代价按 `2**n * (n+1)` 增长，
  稀疏 ramp 是该分辨率下唯一负担得起的筛查模式——但工具链里唯一的子采样钩子
  （`sar_explore` 的 `sweep_points`）打出来的是垃圾分数：**完美的** 6-bit 转换器
  在 64 取 16 的子采样下报 `missing_codes = 48`（未采样的码全算 missing——12-bit
  下是 4032）、`max_abs_dnl = 3.5`（相邻样本把多个码界混叠到同一个跃变中点上，
  跃变指标读出来是**错的**，不只是不全），因此任何相关约束或目标要么全军覆没、
  要么误导。

  新增 `adc.sampled_transfer_metrics`，只测稀疏 ramp 真能确定的东西：每个样本处
  带符号的**码误差**（样本处 `|INL|` 与 `|码误差|` 相差不超过半 LSB），外加单调
  性。低于全密度时，跃变 DNL/INL——以及 missing codes（稀疏 ramp 无法证明某码存
  在或缺失，"期望 vs 产出"的记账在整体 +1 码失调下必然误报）——一律报为未测量的
  NaN，绝不给混叠数字。全链打通：`run_sar_sweep` 在样本数低于 `2**n_bits` 时返回
  码误差指标族（带 `subsampled: True`），全密度下也新增 `max_abs_code_err`；CLI
  `--sweep` 下限由 `2**n_bits` 放宽到 2；`--plot` 对稀疏扫描渲染"阶梯 + 带符号
  码误差"图，取代 showcase 过去盖着"sub-sampled"字样输出的混叠 DNL/INL 面板；
  explore 的 `sweep_points` 打分归正、`max_abs_code_err` 进入约束/目标词汇表；
  失配 MC 接受 `adc.mismatch.sweep_points`（CLI `--sweep-points`），子采样输入走
  同一条编译 Rust batch，良率改由新的 `code_err_threshold` 判定（默认 0.5 LSB，
  即每个被采样码中心必须读出自己的码）。6-bit 示例 64 取 16 实测：sweep 计算
  0.69 s -> 0.19 s，8-trial MC 5.24 s -> 1.37 s（都约为 4 倍转换数上限的 3.7
  倍），各 worker 数结果一致；MC 全密度的行、门与汇总不变。一处有意的行为变化：
  样本数少于 `2**n_bits` 的 `run_sar_sweep` 调用过去返回在混叠数据上算出的完整
  跃变指标 schema，现在返回码误差 schema——原先探测那些字段的 parity 测试改为
  探测码误差族。

### Changed / 变更

- **ADC explore parallelises inside each candidate's sweep / ADC explore 的并行下沉到每个候选自己的 sweep**

  **English:** `sar_explore` used to fan candidates out on a Python thread pool
  with each candidate's 64-conversion sweep pinned to `workers=1`. Measured on
  the machine that motivated the change, that outer parallelism topped out at
  1.8x on 8 workers — concurrent serial sweeps contend inside one process —
  while the compiled kernel's own Rayon parallelism runs the same sweep at
  4.4-5.3x. Candidates now run serially and `workers` reaches each candidate's
  own conversions (`evaluate_sar` grew a `workers` keyword; the coherent-sine
  dynamic probe also stops recording trajectories it never read). The SAR6
  explore CLI moved from 9.08 s to 3.59 s for 4 candidates on 8 workers
  (scaling 1.84x -> 4.74x) and 17.39 s to 7.02 s for 8; serial runtime is
  unchanged, and per-candidate wall at 8 workers is now 0.83 s against the
  0.68 s of a bare 64-code ramp. Results are bit-identical across worker
  counts on the real 6-bit config, including `power_uw`; earlier profiling
  attributed the gap to per-candidate template recompilation, which measurement
  refuted — `build_sar_batch` costs 11 ms cold and 1 ms warm, and trajectory
  finalization costs 10 ms per sweep, so scheduling was the whole story. A
  regression test pins the worker count each candidate's conversion batch
  receives.

  **中文：** `sar_explore` 原先把候选摊到 Python 线程池上、每个候选的 64 次转换
  sweep 固定 `workers=1`。在触发本次改动的机器上实测，这种外层并行 8 worker 只有
  1.8x——同进程内并发的串行 sweep 相互争抢——而编译内核自己的 Rayon 并行跑同一
  sweep 有 4.4-5.3x。现在候选串行执行，`workers` 直达每个候选自己的转换
  （`evaluate_sar` 新增 `workers` 关键字；相干正弦动态探针也不再录制它从未读过的
  轨迹）。SAR6 explore CLI 在 8 worker 下 4 候选由 9.08 s 降至 3.59 s（扩展比
  1.84x -> 4.74x），8 候选由 17.39 s 降至 7.02 s；串行耗时不变，8 worker 下每
  候选墙钟 0.83 s，对照裸 64 码 ramp 的 0.68 s。真 6-bit 配置下各 worker 数结果
  逐位一致（含 `power_uw`）。此前的画像把差距归因于每候选重编译模板，实测证伪
  ——`build_sar_batch` 冷 11 ms、热 1 ms，轨迹 finalize 每 sweep 10 ms，问题
  全部在调度。回归测试钉死每个候选转换批次实际收到的 worker 数。

- **`adc -o` writes codes and metrics by default; `--waveforms` restores the full payload / `adc -o` 默认只落码流与指标；`--waveforms` 恢复完整载荷**

  **English:** `circuit-opt adc ... -o` used to force full trajectory recording
  and serialize every conversion's waveforms: a 64-code SAR6 sweep wrote an
  82 MB JSON, and of its 6.10 s wall time 2.75 s (45%) was recording (1.04 s)
  plus serializing (1.71 s) — more than the codes cost to solve. The output now
  defaults to the slim tier — codes, bits, decision traces, and power/metric
  scalars — for all three conversion modes (`--vin`/`--sweep`/`--sine`), and a
  `--sweep` run no longer records trajectories at all unless asked (the sweep
  figure never read them). The same sweep with `-o` now runs in 0.95 s and
  writes 20 KB. The new `--waveforms` flag restores the previous behavior
  bit-for-bit (verified against a pre-change capture); it requires `-o` and is
  rejected for `--mc`/`--explore`, whose rows never carried waveforms. Anyone
  parsing waveforms out of `adc -o` files must pass `--waveforms` from now on.
  The library API is unchanged (`run_sar_sweep(include_transients=...)` keeps
  its default); regression tests pin the slim payload shape, the full-payload
  round-trip, both rejection guards, and — via a spy on `run_sar_sweep` — the
  fact that a bare `-o` stays on the codes-only path.

  **中文：** `circuit-opt adc ... -o` 原先会强制录制全部轨迹并序列化每次转换的
  波形：64 码 SAR6 sweep 要写 82 MB JSON，6.10 s 墙钟里有 2.75 s（45%）花在
  录制（1.04 s）与序列化（1.71 s）上——比解码流本身还贵。现在三种转换模式
  （`--vin`/`--sweep`/`--sine`）的 `-o` 默认只落精简档——码流、bits、判决轨迹
  与功耗/指标标量——且 `--sweep` 不再录制轨迹（sweep 图本就不读它们）。同一
  sweep 加 `-o` 现在 0.95 s、写 20 KB。新增 `--waveforms` 恢复旧行为（与改动
  前抓取的文件逐位一致已验证）；它必须与 `-o` 同用，`--mc`/`--explore` 会拒绝
  （其行本就不含波形）。此前从 `adc -o` 文件解析波形的用法，今后必须加
  `--waveforms`。库 API 不变（`run_sar_sweep(include_transients=...)` 默认值
  保持）；回归测试钉死精简载荷形状、完整载荷往返、两个拒绝守卫，以及（通过
  对 `run_sar_sweep` 的探针）裸 `-o` 停留在 codes-only 路径这一事实。

## [2.4.0] - 2026-07-26

### Added / 新增

- **FreePDK45 low-threshold flavors and a complemented SAR strobe / FreePDK45 低阈值型号与 SAR 反相选通**

  **English:** `freepdk45.nmos_vtl` / `freepdk45.pmos_vtl` bind the kit's
  `NMOS_VTL`/`PMOS_VTL` cards (nominal vth0 0.322 V against VTG's 0.4106 V);
  flavored models fold onto their base polarity for corner directories, BSIM
  polarity, and the per-polarity mismatch sigmas. `adc.clock` accepts an
  optional `bar_input`, emitting the complemented strobe `high + low - clock`
  as a second waveform row in both the Python builder and the compiled
  `co_core::sar` roles, for latches whose reset devices need the opposite
  phase. Both are additive; existing circuits render identical waveforms.

  **中文：** 新增 `freepdk45.nmos_vtl` / `freepdk45.pmos_vtl`，绑定套件自带的
  `NMOS_VTL`/`PMOS_VTL` 卡（名义 vth0 0.322 V，对 VTG 的 0.4106 V）；带型号后缀的
  model 在 corner 目录、BSIM 极性与逐极性失配 sigma 上折叠回基础极性。
  `adc.clock` 接受可选的 `bar_input`，在 Python 波形生成器与编译版
  `co_core::sar` 角色中同步生成反相选通 `high + low - clock`，供复位管需要相反
  相位的锁存器使用。两者均为增量特性；既有电路的波形逐位不变。

### Changed / 变更

- **Whole-card BSIM handle construction / 整卡构建 BSIM handle**

  **English:** `co_bsim4_set_card` applies a model or instance card in one FFI
  crossing, and the vendor keyword tables are now indexed into a map once per
  process rather than scanned linearly for every parameter. A TSMC28 model card
  names 339 parameters, so building one handle cost 321 us of which the model
  card was 97%; it is now 23.6 us, about 13.6x. The linear scan, not the
  crossings, was the larger half: replacing 339 crossings with one took it to
  244 us, and indexing the table took the rest. `co_bsim4_setup` itself is
  0.9 us, so handle construction was never dominated by the model setup it
  prepares. Because that construction runs under the backend's process-wide
  lock, it was a serial floor on every parallel campaign: an 8-worker 45-point
  signoff spent 3.50 s of its 14.16 s wall there, now 1.09 s, and the campaign
  moved from 14.64 s to 13.05 s with all 45 points reproducing every signoff
  field bit-for-bit. The single-value setters remain; a parity test pins the
  whole-card path to them on a real card, and another checks that a rejected
  parameter is still named. The C ABI version is now 2.

  The keyword indexes are built once for the two vendor tables and read
  lock-free afterwards. The first version kept them behind a mutex, which
  concurrent compiled-campaign workers hit on every parameter while
  constructing handles: the silicon campaign's 8-worker speedup sat at 1.3-1.6x
  with occasional runs slower than one worker. Lock-free lookup takes the same
  benchmark to a steady 5.2-5.4x.

  **中文：** `co_bsim4_set_card` 在一次 FFI 穿越中应用整张 model 或 instance
  card；vendor 关键字表也改为每进程建一次索引，不再为每个参数线性扫描。TSMC28
  的 model card 有 339 个参数，构建单个 handle 原本需要 321 us，其中 model card
  占 97%；现在是 23.6 us，约 13.6 倍。其中线性扫描才是较大的一半：把 339 次穿越
  合成一次只降到 244 us，建索引才拿下其余部分。`co_bsim4_setup` 本身仅
  0.9 us——handle 构造从来就不是被它所准备的 model setup 主导的。由于该构造在
  backend 的进程级锁下执行，它是所有并行 campaign 的串行地板：8 worker、45 点
  signoff 曾在其中花掉 14.16 s wall 里的 3.50 s，现为 1.09 s，campaign 由
  14.64 s 降至 13.05 s，45 个点的全部 signoff 字段逐位不变。逐参数 setter 仍然
  保留；一个对拍测试用真实 card 把整卡路径钉死到它们上，另一个检查被拒参数仍会
  被具名报出。C ABI 版本升至 2。

  两张 vendor 表的关键字索引只在首次使用时构建，之后的读取无锁。最初的版本把
  索引放在一把 mutex 后面，编译式 campaign 的并发 worker 在构造 handle 时每个
  参数都要过这把锁：硅 campaign 的 8-worker 加速比只有 1.3-1.6 倍，偶发比单
  worker 还慢。改为无锁读取后，同一基准稳定在 5.2-5.4 倍。

- **Newton converges against each node's own step budget / Newton 按各节点自己的步长预算收敛**

  **English:** Adaptive native BSIM transients accept a new
  `newton_error_fraction`. When positive, each node's Newton update must fall
  below that fraction of its own step-error budget
  `adaptive_reltol*|V| + adaptive_vabstol` instead of the single absolute
  `newton_vtol`. The absolute rule holds a rail node and a node at zero to the
  same number, so on a 0.9 V circuit with `reltol=1e-5` it converged the large
  nodes to 0.33% of what the step controller was about to accept. The bound
  stays per node rather than root-mean-square, so nodes near zero are held
  *tighter* than before. The exact default is 0, which keeps the absolute
  criterion; the value may not exceed 1, and a fixed grid rejects it because it
  has no per-node budget to take a fraction of. The six TSMC28 MDAC signoff
  cases now declare `0.1`: one 5 ns residue moved from 78.1 ms to 58.8 ms
  pooled and 109.2 ms to 82.7 ms inline (both about 1.33x) as Newton dropped
  from 6.61 to 4.58 iterations per step and 89,490 to 61,864 model
  evaluations, with accepted and rejected step counts unchanged. Across all
  45 PVT points no case or point status changed and the campaign moved from
  15.89 s to 14.62 s; the smaller campaign share is expected, because
  serialized native handle construction alone accounts for 3.50 s of that
  wall. Against a reference run with 100x tighter step tolerances, the
  settling-tail deviation grew from 87.6 uV to 118.9 uV while the final output
  difference stayed at a few uV. Tightening `newton_vtol` from 3e-8 to 1e-8
  instead changed that deviation by 0.6 uV for 19% more evaluations, which is
  the measurement this option acts on.

  **中文：** 自适应原生 BSIM transient 新增 `newton_error_fraction`。取正值时，
  每个节点的 Newton 更新必须低于该节点自身步长误差预算
  `adaptive_reltol*|V| + adaptive_vabstol` 的这一比例，而不再是单一的绝对
  `newton_vtol`。绝对判据对轨电位节点和零电位节点用同一个数，因此在 0.9 V、
  `reltol=1e-5` 的电路上，大电压节点被收敛到步长控制器即将接受的误差的 0.33%。
  该界限按节点逐个判定而非取均方根，所以零附近的节点反而被收得**更紧**。默认值
  严格为 0，即保持原绝对判据；配置值不得超过 1；固定网格会拒绝该选项，因为它没有
  逐节点预算可供取比例。六个 TSMC28 MDAC signoff case 现声明 `0.1`：单个 5 ns
  residue 在池调度下由 78.1 ms 降至 58.8 ms、inline 下由 109.2 ms 降至
  82.7 ms（均约 1.33 倍），Newton 由每步 6.61 次迭代降至 4.58 次、模型求值由
  89,490 降至 61,864，接受与拒绝步数均不变。完整 45 个 PVT 点没有任何 case 或
  点的状态变化，campaign 由 15.89 s 降至 14.62 s；campaign 侧占比更小是预期的，
  因为仅被串行化的原生 handle 构造就占去其中 3.50 s。相对步长容差收紧 100 倍的
  参考解，建立尾段偏差由 87.6 uV 增至 118.9 uV，最终输出差仍保持在几 uV。作为
  对照，把 `newton_vtol` 由 3e-8 收紧到 1e-8 只把该偏差改变 0.6 uV，却多花 19%
  的模型求值——这正是本选项针对的测量结果。

- **One matrix-clear path for every schedule / 所有调度共用一条矩阵清零路径**

  **English:** Clearing only the active `CKTmaxEqNum` block of the BSIM matrix
  frame is now the default everywhere, not only inside outer-parallel workers.
  A same-machine A/B of all three candidates under both schedules showed the
  contiguous full-frame write is the slowest under *both*, so the previous
  schedule-dependent branch was choosing the wrong side for isolated solves;
  clearing whole active rows in one memset landed inside run-to-run noise. A
  single TSMC28 residue moved from 83.9 ms to 79.0 ms and the FreePDK45 SAR6
  64-code sweep from 3.217 s to 3.134 s with an unchanged code stream, while
  the 45-point campaign kept its runtime because it already took this path. All
  45 points reproduce every signoff field bit-for-bit, the golden corpus now
  exercises the block clear on its isolated AC, noise and transient cases, and
  a subprocess test pins the block clear to the full-frame clear across
  transient, AC and noise. `CIRCUITOPT_BSIM_FULL_FRAME_CLEAR=1` still restores
  the full-frame write.

  **中文：** 只清零 BSIM 矩阵帧中 `CKTmaxEqNum` 有效块现在是所有路径的默认，不再
  限于外层并行 worker。同机对三种清零方式、两种调度做的 A/B 显示连续整帧写入在
  **两种**调度下都最慢，因此原先按调度分岔的实现给单次求解选错了分支；而"整行块
  一次连续 memset"落在运行间噪声内。单个 TSMC28 residue 由 83.9 ms 降至
  79.0 ms，FreePDK45 SAR6 64-code sweep 由 3.217 s 降至 3.134 s 且码流不变；
  45 点 campaign 耗时不变，因为它本就走这条路径。45 个 PVT 点的全部 signoff
  字段与改动前逐位相同，golden 语料的孤立 AC/noise/transient 用例现在也真正覆盖
  该路径，另有子进程测试在 transient、AC 和 noise 上把块清零与整帧清零逐位钉死。
  `CIRCUITOPT_BSIM_FULL_FRAME_CLEAR=1` 仍可恢复整帧写入。

- **Faster adaptive MDAC Newton and LTE solves / 更快的自适应 MDAC Newton 与 LTE**

  **English:** Native BSIM adaptive Gear2 now uses a guarded variable-step
  quadratic state predictor once three accepted states are available, retaining
  the linear predictor after restarts and disabling prediction at input edges.
  The charge-defect LTE projection also solves its second right-hand side with
  the converged Newton LU factors instead of stamping and factoring the same
  Jacobian again. On the local 5 ns residue benchmark, warm median runtime moved
  from 55.6 ms to 50.4 ms for FreePDK45 and from 139.2 ms to 106.4 ms for
  TSMC28; Newton iterations moved from 873 to 801 and 2,644 to 1,995. Regridding
  the new trajectories onto the previous accepted grids gave maximum all-node
  differences of 0.249 mV and 0.137 mV respectively, with final-output changes
  of 1.5 uV and 12.0 uV.

  **中文：** 原生 BSIM 自适应 Gear2 在获得三个已接受状态后，现使用带保护的变步长
  二阶状态外推；重启后仍先使用线性 predictor，输入边沿处继续禁用预测。基于电荷
  defect 的 LTE 投影也会复用收敛 Newton 的 LU 因子，仅求解第二个右端项，不再重新
  stamp 并分解同一个 Jacobian。本机 5 ns residue 基准中，FreePDK45 热运行中位数
  由 55.6 ms 降至 50.4 ms，TSMC28 由 139.2 ms 降至 106.4 ms；Newton 次数分别由
  873 降至 801、2,644 降至 1,995。将新轨迹重采样到旧 accepted grid 后，全节点
  最大差异分别为 0.249 mV 和 0.137 mV，最终输出变化为 1.5 uV 和 12.0 uV。

- **Exact BSIM internal-node and small-signal reuse / 精确复用 BSIM 内部节点与小信号状态**

  **English:** Native BSIM evaluation now factors each internal-node Schur
  matrix once and solves its four external-terminal right-hand sides together.
  The previous row/column loop rebuilt and solved the same column four times,
  performing 16 factorizations per real or complex reduction. A dedicated
  bit-pattern test pins the shared-factor result to the former independent
  solves. The TSMC28 5 ns MDAC warm median moved from 113.1 ms to 97.4 ms in a
  same-session A/B (about 14% faster), with identical accepted grid, all 41 node
  histories, all five branch-current histories, output, and final state.
  A full evaluation also reuses the charge and capacitance fields already
  produced by its final floating-point BSIM load instead of repeating that
  operating point in `MODEINITSMSIG` before `acLoad`. This moved TSMC28 from
  97.1 ms to 90.1 ms and FreePDK45 from 38.5 ms to 36.1 ms; transient histories
  and AC/noise arrays remained bit-identical for both PDKs. The raw `dc` entry
  retains the reload as an oracle, and
  `CIRCUITOPT_BSIM_REUSE_SMSIG_LOAD=0` restores it for external A/B checks.
  An opt-in `CIRCUITOPT_BSIM_REUSE_FINAL_LOAD=1` mode also consumes the
  converged private-node linearization directly once its Newton update is below
  1 pV. TSMC28 moved from 84.8 ms to 77.7 ms and FreePDK45 from 33.7 ms to
  31.1 ms (about another 8%). Maximum MDAC node differences were 0.36 nV and
  42 pV, but isolated PMOS current/noise points exceed the public bit-level
  golden contract, so this mode remains disabled by default.

  **中文：** 原生 BSIM 求值现在只分解一次内部节点 Schur 矩阵，并一起求解四个外部
  端子的右端项。旧的行列循环会把同一列重复构建、求解四次，使每次实数或复数消元
  执行 16 次分解；新增的位模式测试将共享分解结果锁定为与原独立求解逐位一致。
  同一进程内 A/B 中，TSMC28 的 5 ns MDAC 热运行中位数由 113.1 ms 降至
  97.4 ms，约快 14%；accepted grid、全部 41 条节点历史、5 条支路电流历史、输出
  和最终状态均完全相同。完整求值还会直接复用最终 floating-point BSIM load 已生成
  的电荷与电容字段，不再在 `acLoad` 前以 `MODEINITSMSIG` 重复同一工作点。
  TSMC28 因此进一步由 97.1 ms 降至 90.1 ms，FreePDK45 由 38.5 ms 降至
  36.1 ms；两个 PDK 的瞬态历史和 AC/noise 数组均逐位相同。原始 `dc` 入口保留
  reload 作为 oracle，也可设置 `CIRCUITOPT_BSIM_REUSE_SMSIG_LOAD=0` 恢复旧路径
  做外部 A/B。还可显式设置 `CIRCUITOPT_BSIM_REUSE_FINAL_LOAD=1`：私有内部节点
  Newton 更新低于 1 pV 后直接使用收敛线性化，不再额外执行一次完整 model load。
  TSMC28 由 84.8 ms 降至 77.7 ms，FreePDK45 由 33.7 ms 降至 31.1 ms，均再快约
  8%；MDAC 最大节点差只有 0.36 nV 和 42 pV，但个别 PMOS 电流/噪声点会超出公共
  API 的位级 golden 契约，因此该模式默认关闭。

- **Bit-exact BSIM host and SAR throughput / BSIM 宿主与 SAR 吞吐量逐位精确优化**

  **English:** The BSIM host now caches each setup-complete terminal/internal
  node layout. A converged private-node load is skipped automatically only when
  the solved internal voltages are bit-identical to those used by the current
  load (or no private nodes exist); this preserves exact scalar, SAR, and MDAC
  trajectories while avoiding provably redundant work. Isolated solves retain
  the faster contiguous full-frame clear, whereas outer-parallel campaign/SAR
  workers clear only the active matrix block to reduce aggregate memory
  traffic. `CIRCUITOPT_BSIM_FULL_FRAME_CLEAR=1` restores full-frame clearing for
  A/B runs. Transient batches also bypass the per-call handle mutex after the
  backend or campaign worker has granted exclusive ownership for the whole
  solve. Public scalar/C ABI calls remain locked, and repeated handles inside a
  batch remain serialized. A dedicated NMOS/PMOS trajectory test pins the
  exclusive entry to the locked entry bit-for-bit; the local 64-code SAR6 warm
  median improved by about 2.5% (`574.3 ms -> 559.8 ms`).

  The compiled SAR single-trial sweep now creates exactly
  `min(workers, inputs)` balanced contiguous chunks. This removes an Auto-axis
  cliff where 64 inputs with 12 workers formed 11 chunks and ran serially
  (`3.44 s -> 0.59 s` locally); 10-worker runtime moved from about 0.74 s to
  0.57 s. Continuation also rebuilds only the newly active bit and the previous
  cleared bit waveform instead of all input rows. The 64-code stream and full
  eight-conversion node/branch/final-state histories remain bit-identical to the
  previous exact path.

  **中文：** BSIM host 现在会缓存 setup 完成后的外部/内部节点布局。只有在内部节点
  解与当前 model load 使用的电压在位模式上完全相同（或器件没有私有节点）时，才会
  自动省略最终 load；因此标量、SAR 与 MDAC 轨迹保持逐位一致，同时去掉可证明冗余
  的工作。单次求解继续使用更快的一次性整帧清零；外层并行的 campaign/SAR worker
  仅清零有效矩阵块，以降低汇总内存流量。设置
  `CIRCUITOPT_BSIM_FULL_FRAME_CLEAR=1` 可强制恢复整帧清零做 A/B。backend 或
  campaign worker 为整个 solve 授予 handle 独占所有权后，transient batch 还会
  跳过逐调用 mutex；公共标量/C ABI 入口仍保留锁，batch 内重复 handle 也仍然串行。
  专门的 NMOS/PMOS 轨迹测试将独占入口锁定为与加锁入口逐位一致；本机 64-code
  SAR6 热运行中位数约提升 2.5%，由 `574.3 ms` 降至 `559.8 ms`。

  编译式 SAR 的单 trial sweep 现在严格生成 `min(workers, inputs)` 个均衡连续分块。
  这消除了 64 输入、12 worker 被错误切成 11 块后触发串行 Auto 轴的性能悬崖，本机
  由 `3.44 s` 降至 `0.59 s`；10 worker 由约 `0.74 s` 降至 `0.57 s`。
  continuation 也只重建新激活 bit 与上一清零 bit 的波形，不再重建全部输入行。
  64-code 码流以及 8 次完整转换的节点、支路和最终状态历史均与旧精确路径逐位一致。

- **Pooled setup-complete BSIM handles / 池化复用 setup 完成的 BSIM handle**

  **English:** Native transient now leases its per-MOS handles from the common
  BSIM backend instead of rebuilding every C instance for every solve. The
  backend cache is a real per-card pool: matched devices with the same model and
  W/L/NF receive separate mutable handles during a trajectory, then all idle
  handles can be reused by the next trajectory. The configured
  `BSIM4_DEVICE_CACHE_SIZE` remains a global handle-count bound per PDK
  namespace, and active handles are never evicted or shared. On the local
  TSMC28 `residue_plus_fs16` benchmark the warm median moved from 124.6 ms to
  101.9 ms (1.22x); the six residue/code-transition cases at one PVT point moved
  from 0.815 s to 0.692 s (1.18x). FreePDK45 moved from 42.7 ms to 38.7 ms
  (1.10x). Maximum interpolated output differences against freshly constructed
  handles were 0.15 uV and 8.6 pV respectively, with identical signoff status;
  compiled regenerative SAR continues to own independent Rust handles and its
  64-code regression remains unchanged. When no explicit transient initial
  state is supplied, native BSIM also attempts the declared DC guesses directly
  through the already-built Rust topology and leased handles. It accepts only a
  finite solution within the configured residual and voltage-box contracts,
  otherwise falling back to the full AC/DC path. This removes duplicate device
  construction and unused 1 Hz AC reduction: the same TSMC28 residue moved from
  102.3 ms to 94.4 ms and the six-case point from 0.699 s to 0.645 s, with a
  maximum output difference of 38 pV.

  **中文：** 原生 transient 现在从公共 BSIM backend 租用逐 MOS handle，不再为
  每次求解重新构造全部 C 实例。backend cache 已提升为真正的逐 card handle 池：
  相同 model 与 W/L/NF 的匹配器件在一条轨迹中仍获得彼此独立的可变 handle，归还
  后则可由下一条轨迹整体复用。`BSIM4_DEVICE_CACHE_SIZE` 仍按每个 PDK namespace
  的 handle 总数设上限；活动 handle 永不驱逐、永不共享。本机 TSMC28
  `residue_plus_fs16` 热运行中位数由 124.6 ms 降至 101.9 ms（1.22 倍），单个
  PVT 点的六个 residue/code-transition case 由 0.815 s 降至 0.692 s
  （1.18 倍）；FreePDK45 由 42.7 ms 降至 38.7 ms（1.10 倍）。相对每次新建
  handle 的旧路径，两者插值后最大输出差分别为 0.15 uV 与 8.6 pV，signoff 状态
  不变；再生式编译 SAR 仍独占 Rust handle，64-code 回归保持不变。未显式提供
  transient 初始状态时，原生 BSIM 还会直接复用已构造的 Rust topology 和租用
  handle 尝试声明的 DC guesses；只有有限、满足残差及 voltage-box 契约的解才会
  接受，否则回退到完整 AC/DC 路径。由此去掉重复 device 构造和无用的 1 Hz AC
  reduction：同一 TSMC28 residue 由 102.3 ms 降至 94.4 ms，六 case 单点由
  0.699 s 降至 0.645 s，最大输出差约 38 pV。

- **Scoped MDAC Newton tolerance and cached card keys / 限定 MDAC Newton 容差并缓存 card key**

  **English:** The six TSMC28 MDAC residue/code-transition signoff cases now
  declare `newton_vtol=3e-8 V`, while the global solver default remains
  `1e-8 V`. Across all 45 PVT points this reduced the 270 transient runs from
  13.848 s to 10.247 s (1.35x) without changing any case or point status.
  Against the original tolerance, the maximum output-trajectory and final-output
  differences were 0.116 mV and 18.3 uV; direct PWL-driven nodes were excluded
  from the regridded trajectory comparison because adaptive grids sample their
  discontinuities on opposite sides. Immutable model and instance cards also
  precompute their sorted parameter tuples once. A 600-parameter key
  microbenchmark reduced 20,000 key builds from 350.9 ms to 1.35 ms; this exact
  control-plane optimization applies to every native BSIM PDK.

  **中文：** 六个 TSMC28 MDAC residue/code-transition signoff case 现显式声明
  `newton_vtol=3e-8 V`，全局求解器默认值仍为 `1e-8 V`。完整 45 点、270 次
  transient 由 13.848 s 降至 10.247 s（1.35 倍），所有 case 和 PVT 点的状态均
  未改变。相对原容差，输出轨迹最大差异为 0.116 mV，最终输出最大差异为
  18.3 uV；直接由 PWL 驱动的节点未纳入重采样轨迹比较，因为不同 adaptive grid
  会落在不连续边沿的两侧。不可变 model/instance card 还会在构造时一次性缓存已
  排序参数元组。600 参数 card 的微基准中，20,000 次 key 构建由 350.9 ms 降至
  1.35 ms；该逐位精确的控制面优化适用于所有原生 BSIM PDK。

- **Bounded BSIM Newton bypass / 有界 BSIM Newton bypass**

  **English:** Native transient can now opt into the vendored BSIM4 model's
  standard device-bypass path with `bsim_model_bypass_tolerance`. The default is
  exactly zero, the configured value may not exceed `newton_vtol`, relative
  voltage tolerance is disabled during bypass, and each exclusive call restores
  the handle's public settings before returning. The six validated TSMC28 MDAC
  transient cases use `3e-9 V` (one tenth of their Newton tolerance). Across
  45 PVT points, the 270-run 8-worker campaign moved from 12.025 s to 8.377 s
  (1.43x) with no case or point status changes. The maximum/P99 regridded output
  differences were 8.77/6.25 uV and the maximum final-output difference was
  2.74 uV. Other transient, scalar, DC, AC, noise, and regenerative SAR paths
  retain exact bypass-off behavior unless they explicitly opt in.

  **中文：** 原生 transient 现可通过 `bsim_model_bypass_tolerance` 显式启用
  vendored BSIM4 的标准 device-bypass 路径。默认值严格为 0，配置值不得超过
  `newton_vtol`；bypass 期间关闭相对电压容差，并在每次独占调用返回前恢复 handle
  的公共设置。六个已验证 TSMC28 MDAC transient case 使用 `3e-9 V`，即其 Newton
  容差的十分之一。完整 45 PVT 点、270 次、8 worker campaign 由 12.025 s 降至
  8.377 s（1.43 倍），没有 case 或 PVT 点状态变化。重采样输出轨迹的最大/P99
  差异为 8.77/6.25 uV，最终输出最大差异为 2.74 uV。其他 transient、scalar、
  DC、AC、noise 与再生式 SAR 路径除非显式启用，否则继续保持精确的 bypass-off
  行为。

### Fixed / 修复

- **The 6-bit SAR example now converts correctly at nominal / 6-bit SAR 示例在名义点正确转换**

  **English:** `freepdk45_sar6.json`'s nominal 64-code ramp was deformed and
  nothing pinned it: the stream was non-monotonic with wild codes (inputs 1
  and 2 read 41 and 53) and a deterministic LSB inversion (`code = i XOR 1`
  over half the range). The engine was exonerated -- the compiled continuation,
  the production path, and the frozen full replay agree bit-for-bit, and the
  CDAC top-plate differentials measure exactly `trial - vin` -- the deformity
  was the comparator design. Its StrongARM input pair sat at the CDAC common
  mode of about 0.47 V, at the VTG threshold: with millivolt overdrives the
  latch resolved through charge-share races between the stacked internal nodes
  and the outputs rather than through the input signal. No single flavor
  rescues that topology on this kit -- VTG inverts small differentials, VTL
  fixes them but its subthreshold slide inverts large ones, and a double-tail
  second stage either parks metastably on the 100 ps grid or locks capacitive
  feedthrough of the wrong sign; each regime was measured, not assumed.

  The comparator is now fully static: two diode-loaded differential preamp
  stages (VTL input pairs, zero systematic offset by symmetry, soft-clipped
  range compression) into the same five-transistor mirror stage the 3-bit
  example has always used. The nominal ramp is the ideal staircase -- all 64
  code centers resolve to their ideal codes, strictly monotone -- and is now
  pinned absolutely by a new regression test, closing the gap that let the
  deformity ship: the previous SAR tests were self-consistency checks only.
  Pinned conversions updated to the ideal codes (0.7109375 now reads 45, not
  44; 0.2890625 reads 18, not 19), the strobe-machinery tests inject their
  clock block through the config override, and the explore config retargets
  the new preamp pair. The mismatch hook still reaches the comparator: +50 mV
  on one preamp input shifts a three-point sweep by a coherent four codes.

  **中文：** `freepdk45_sar6.json` 的名义 64 码斜坡是畸形的，且没有任何测试钉住它：
  码流非单调、带狂码（输入 1、2 读出 41、53），一半量程上还有确定性的 LSB 反相
  （`code = i XOR 1`）。引擎已洗清——编译续算、生产路径与冻结全重放逐位一致，
  CDAC 顶板差分实测恰为 `trial - vin`——畸形出自比较器设计本身。其 StrongARM
  输入对栅极共模约 0.47 V，正贴 VTG 阈值：毫伏级过驱动下，锁存靠栈接内部节点与
  输出间的电荷分享竞争而非输入信号定胜负。该拓扑在这套模型上无论换哪种阈值都
  救不回——VTG 反转小差分，VTL 治好小差分却让亚阈值滑移反转大差分，double-tail
  二级要么在 100 ps 网格上亚稳停车、要么锁住符号相反的容性馈通；每个 regime
  都经实测而非推断。

  比较器现改为全静态：两级二极管负载全差分前放（VTL 输入对，对称结构系统失调
  为零，软限幅自带量程压缩）接 3-bit 示例一直使用的同款五管镜像级。名义斜坡
  成为理想阶梯——64 个码中心全部落到理想码、严格单调——并由新增回归测试绝对
  钉死，补上让畸形溜过的缺口：此前的 SAR 测试全部只是自洽性检验。已钉转换更新
  为理想码（0.7109375 现读 45 而非 44；0.2890625 读 18 而非 19），选通机制测试
  改为经 config 注入时钟块，explore 配置改指新前放对。失配钩子仍能到达比较器：
  单侧前放输入 +50 mV 使三点扫码相干偏移 4 码。

- **Device evaluation waits for the model's own convergence signal / 器件求值等待模型自己的收敛信号**

  **English:** The vendored BSIM4 load limits the terminal voltages it was
  asked for against the ones the previous load settled on, walking each large
  bias step over several loads. The host's internal-node loop stopped as soon
  as the internal nodes settled, and nothing checked whether that walk had
  finished: `DEVfetlim` limits silently, and the rbodyMod junction branch
  overwrites the core vbs/vbd `DEVpnjlim` flag with the body-network flags
  alone, so even `CKTnoncon` stays quiet. On devices whose internal nodes are
  insensitive to the walking voltage the loop could therefore exit mid-walk
  and report an operating point nobody requested. A cold FreePDK45 PMOS
  evaluation at Vg=0.66/Vd=0.88 -- forward-biased drain-body junction --
  exited 72.7 mV short on the junction voltage and returned a source current
  9.9% away from the converged point; approaching the same bias gradually gave
  the converged one, so the answer depended on handle history at 8 of that
  device's 36 grid biases.

  Two exit conditions were added to the loop: `CKTnoncon` is cleared before
  every load and must stay clear, and the limited-voltage block the load
  stores in `CKTstate0` (`vbd..vdes`) must not have moved by more than 1 uV --
  a walking limiter always moves it by at least one limiting step, tens of mV.
  Every evaluation now lands on the fixed point that repeated and gradual
  approaches reach; a regression test pins cold-equals-walked at the bias
  above, and a subprocess test keeps it bit-identical between block and
  full-frame matrix clears. Runtime is unchanged (45-point campaign 13.1 s,
  single residue 55.9 ms). Numbers move where evaluations used to stop early:
  no signoff status changes and at most 8.8e-6 relative on non-degenerate
  campaign fields, 6 of 64 codes in the nominal FreePDK45 SAR6 ramp probe as it
  stood at the time -- all on conversions that were already non-monotonic before
  the fix, on the StrongARM comparator this release also replaces -- and 181
  device grids across all three PDKs in the engine-parity corpus, re-frozen
  after verifying magnitudes: at most 5.4e-16 A absolute on currents
  (6e-5 relative, subthreshold points), with the five circuit-level golden
  cases bit-identical throughout.

  **中文：** vendored BSIM4 的 load 会把调用者要求的端电压对上一次 load 落定的
  电压做限幅，大的偏置跳变要分好几次 load 才走完。host 的内部节点循环只要内部
  节点稳定就退出，没有任何东西检查这段"走步"是否完成：`DEVfetlim` 静默限幅，
  rbodyMod 的结限幅分支又用体网络两个结的标志覆盖了核心 vbs/vbd 的 `DEVpnjlim`
  标志，连 `CKTnoncon` 也保持沉默。在内部节点对该电压不敏感的器件上，循环因此
  可能在半路退出，返回一个没人要求过的工作点。FreePDK45 PMOS 在 Vg=0.66/
  Vd=0.88（漏-体结正偏）的冷求值就在结电压差 72.7 mV 时提前退出，源电流偏离
  收敛点 9.9%；而逐步逼近同一偏置得到的是收敛值——该器件 36 个网格偏置中有 8 个
  的答案取决于 handle 历史。

  循环新增两个退出条件：每次 load 前清零 `CKTnoncon` 且必须保持为零；load 写进
  `CKTstate0` 的限幅电压块（`vbd..vdes`）移动不得超过 1 uV——还在走步的限幅器
  每次至少移动一个限幅步长，数十 mV。现在每次求值都落到重复求值与逐步逼近共同
  到达的不动点上；回归测试在上述偏置钉死"冷启动 == 走过去"，子进程测试保持块
  清零与整帧清零逐位一致。运行时间不变（45 点 campaign 13.1 s，单 residue
  55.9 ms）。数值只在原先提前退出的地方移动：signoff 状态零变化、非退化
  campaign 字段最大相对差 8.8e-6；当时的名义 FreePDK45 SAR6 斜坡探针 64 码中
  6 码变化（全部落在修复前就已非单调的转换上，即本次同时被替换掉的 StrongARM
  比较器）；engine-parity 语料中三个 PDK 共
  181 个器件网格变化，核对幅度后已重新冻结：电流最大绝对差 5.4e-16 A
  （相对 6e-5，亚阈值点），五个电路级 golden 用例全程逐位不变。

- **Lifecycle, payload, and SAR input guards / 生命周期、输出与 SAR 输入校验**

  **English:** Native BSIM backend shutdown now becomes terminal before native
  handles are destroyed and continues cleaning remaining handles and stores if
  one teardown fails. CLI payload filtering applies recursively to arbitrary
  mappings, sequences, sets, and object arrays, including opaque mapping keys.
  The compiled SAR path now rejects unknown or non-finite per-device mismatch
  offsets before simulation, restoring the input contract that the reference
  transient path already enforced.

  **中文：** 原生 BSIM 后端现在会先原子地进入永久关闭状态，再销毁 native handle；
  即使某个 handle 清理失败，也会继续清理同一 store 内的其余 handle 和其他 store。
  CLI 输出过滤现会递归处理通用 mapping、sequence、set 和 object array，并过滤
  opaque mapping key。编译式 SAR 路径也会在仿真前拒绝未知器件或非有限的逐器件
  失配偏移，恢复 reference transient 路径原有的输入契约。

## [2.3.0] - 2026-07-26

### Changed / 变更

- **Much faster SAR conversions and Monte-Carlo / SAR 转换与蒙特卡洛大幅加速**

  **English:** A closed-loop SAR conversion used to re-simulate the whole time
  grid once per bit. It now carries the simulation forward instead: deciding a
  bit only changes the stimulus after that bit's decision instant, so the run
  resumes from there rather than restarting at t=0. An eight-trial FreePDK45
  SAR6 ramp went from 235.1 s to 34.7 s and the SAR3 ramp from 4.3 s to 1.2 s;
  one mismatch trial went from about 32 s to 4.5 s, which puts a 200-trial
  Monte-Carlo at roughly 8.6 minutes on eight workers instead of about 106.
  Results are unchanged — 576 ramp codes and a mismatch Monte-Carlo were checked
  bit-for-bit against the previous kernel.

  **中文：** 闭环 SAR 转换过去每定一个 bit 就把整条时间网格重跑一遍，现在改为继续
  往前推进：一个 bit 定下来只会改变其判决时刻之后的激励，因此从该时刻续算，而不是
  回到 t=0 重来。8 trial 的 FreePDK45 SAR6 斜坡由 235.1 s 降至 34.7 s，SAR3 斜坡
  由 4.3 s 降至 1.2 s；单次失配 trial 由约 32 s 降至 4.5 s，因此 200 trial 蒙特
  卡洛在 8 worker 下由约 106 分钟降至约 8.6 分钟。结果不变——576 个斜坡码和一次
  失配蒙特卡洛与改前内核逐位比对一致。

- **Parallel runs now actually use the extra cores / 并行运行现在真的用得上多余的核**

  **English:** Anything that runs many simulations at once — signoff PVT points,
  SAR conversions and Monte-Carlo trials, corner and PVT sweeps, exploration
  candidates — used to hand each simulation's device evaluation to one shared
  thread pool. That pool filled the machine on its own, leaving the outer
  workers nothing to run on. Measured with sixteen TSMC28 MDAC transients: the
  old scheme took 4.90 s on one thread while keeping 7.8 of ten cores busy, and
  eight threads only reached 4.03 s; the new one takes 3.19 s on eight threads
  using 6.0 cores. Drivers that manage their own parallelism now say so, and
  their workers evaluate inline. A single simulation, or a batch too small to
  fill the machine, still uses the pool. Only scheduling changes, never results.
  Set `CIRCUITOPT_BSIM_NESTED_POOL=1` for the old behaviour.

  **中文：** 所有同时跑多个仿真的场景——signoff PVT 点、SAR 转换与蒙特卡洛 trial、
  corner 与 PVT 扫描、探索候选——过去都把每个仿真的器件求值交给同一个共享线程池。
  那个池自己就占满了机器，外层 worker 无核可用。以 16 次 TSMC28 MDAC 瞬态实测：
  旧方案单线程 4.90 s 却占满十核中的 7.8 核，八线程也只到 4.03 s；新方案八线程
  3.19 s，占 6.0 核。自己管理并行的驱动现在会显式声明，其 worker 内联求值；单次
  仿真或不足以填满机器的小批次仍然使用线程池。改变的只有调度，结果不受影响。设置
  `CIRCUITOPT_BSIM_NESTED_POOL=1` 可恢复旧行为。

- **Faster transient post-processing / 瞬态后处理加速**

  **English:** Rebuilding branch currents after a native BSIM transient walked
  the time grid in Python, once per device and per capacitor. It is now array
  arithmetic over the whole run and consumes the integration coefficients
  selected by the Rust solver. A FreePDK45 SAR6 conversion went from 754.5 ms
  to 503.2 ms, and the Python share of one TSMC28 MDAC transient from 72 ms
  to 27 ms.

  **中文：** 原生 BSIM 瞬态之后重建支路电流时，过去按每个器件、每个电容分别用
  Python 遍历时间网格，现在整段用数组运算完成，并直接使用 Rust 求解器实际选择的
  积分系数。FreePDK45 SAR6 单次转换由 754.5 ms 降至 503.2 ms，单次 TSMC28 MDAC
  瞬态中 Python 占用由 72 ms 降至 27 ms。

- **MDAC signoff runs its transients adaptively / MDAC 签核瞬态改用自适应步长**

  **English:** The TSMC28HPC+ MDAC signoff manifest now asks for adaptive Gear2
  on its six transient cases instead of a fixed 10 ps grid, at a tolerance
  (`reltol=1e-5`) chosen where tightening further stops improving accuracy. The
  45-point campaign takes 78.9 s instead of the fixed grid's 99.3 s. Nothing
  about the verdicts moves: across 45 PVT points x 11 cases no case changes
  status and the global worst case stays the same case at the same corner,
  temperature and supply.

  **中文：** TSMC28HPC+ MDAC 签核配置的六个瞬态 case 现在请求自适应 Gear2，不再使用
  固定 10 ps 网格；容差取 `reltol=1e-5`，即再收紧也不再提升精度的拐点。45 点
  campaign 为 78.9 s，固定网格为 99.3 s。判定完全不变：45 个 PVT 点 × 11 个 case
  无任何状态变化，全局最差点仍是同一 case、同一 corner、温度与电压。

- **Every engine now reads its integration formula from one place / 所有引擎的积分公式收归一处**

  **English:** The rule for stepping a transient forward — start on backward
  Euler, use variable-step BDF2 afterwards, drop back to backward Euler when a
  step more than doubles, and estimate the local error from the BDF3 defect —
  used to be written out by hand in four places: the two BSIM solvers, the OTFT
  solver, and the Python PSS monodromy. The guard constant appeared once per
  copy, and they had begun to drift; a fix made in one copy did not reach the
  others. All four now call `co-core`'s new integrator module, and the two
  identical copies of the stimulus sampling live in one module as well. The two
  adaptive drivers had likewise each copied the step arithmetic — the startup
  step, the growth clamp, the clamp onto the next breakpoint, the give-up test,
  and the restart detection after landing on a breakpoint — so every decision
  about step *size* now comes from one shared planner too. Each driver keeps
  only its state machine and its local-error estimator; those stay separate
  deliberately, because comparing a step against two half steps and projecting
  a BDF3 charge defect are different cost/robustness trades. Two scaling
  conventions also remain on purpose (whether the `1/h` factor is folded in),
  because converting between them reorders floating-point operations and would
  move every frozen result; property tests pin both closed forms to the same
  Lagrange generator instead. Every golden stays bit-exact.

  **中文：** 瞬态推进的规则——首步用后向欧拉、之后用变步长 BDF2、步长翻倍以上时退回
  后向欧拉、用 BDF3 缺陷估计局部误差——过去在四处分别手写：两个 BSIM 求解器、OTFT
  求解器，以及 Python 的 PSS monodromy。判据常数每份抄一遍，且已经开始漂移：在一份
  里修好的问题传不到其余几份。现在四处统一调用 `co-core` 新增的 integrator 模块，
  两份逐字相同的激励采样也合并为一个模块。两个自适应驱动同样各抄了一份步长算法——
  启动步、增长钳制、钳到下一个断点、放弃判据，以及落到断点后的重启判定——因此所有关于
  步长*大小*的决策现在也统一出自一个共享 planner。各驱动只保留自己的状态机与局部误差
  估计器；后者有意不合并，因为"整步对两个半步"与"投影 BDF3 电荷缺陷"是两种不同的
  代价/健壮性取舍。两种缩放约定（是否已并入 `1/h` 因子）同样有意保留，因为互相转换会
  重排浮点运算顺序、移动所有已冻结结果；改由 property test 把两条闭式同时钉在同一个
  Lagrange 生成器上。所有 golden 保持位一致。

- **The backward-Euler rerun option has a name that says what it does / 后向欧拉重跑选项改用能说明其作用的名字**

  **English:** `gear2_be_fallback` named two different things at once: the
  whole-run rerun on backward Euler after a gear2 solve fails too many steps,
  and the per-sample order drop a gear2 solve already performs when one step
  more than doubles. The option is now `be_rerun_on_step_failures`, and the
  result keys are `be_rerun_used` / `be_rerun_step_failures`. Decks and Python
  callers using the old spelling keep working — it is accepted as an alias and
  the old result keys are still emitted — and the canonical spelling wins if a
  deck carries both.

  **中文：** `gear2_be_fallback` 同时指代了两件不同的事：gear2 求解失败步过多后
  在后向欧拉上重跑整场，以及 gear2 求解本来就会在某一步长翻倍以上时对该采样点降阶。
  该选项现名为 `be_rerun_on_step_failures`，结果键为 `be_rerun_used` /
  `be_rerun_step_failures`。使用旧拼写的电路 JSON 与 Python 调用方不受影响——旧名作为
  别名继续接受，旧结果键继续输出——若同一份配置里两种拼写并存，以新名为准。

### Fixed / 修复

- **Adaptive restarts are error-controlled instead of accepted blind / 自适应重启步改为受误差控制，不再无条件接受**

  **English:** Whenever the adaptive solver had no step history — at t=0 and
  again after every input breakpoint it restarts on — its local-error estimate
  was undefined, so the next two steps were accepted without any check, at
  whatever size the startup heuristic picked. Those steps land right after a
  clock edge, where the circuit moves fastest, and their error was invisible to
  `reltol`: on the TSMC28 MDAC deck, tightening `reltol` tenfold (115% more
  steps) barely moved the result while shrinking the startup step 25-fold cut
  the error by four. Such a trial is now solved twice, once whole and once as
  two halves, and the difference between the two is the error estimate;
  accepting keeps both halves, so the following step already has real history
  and full error control. No step of a solve escapes error control any more.
  Measured against a 40x-finer reference on the same stimulus, the largest
  deviation over a 5 ns MDAC window falls from 3.69 mV to 92 uV on the
  major-carry case and from 308 uV to 68 uV on a residue case; the adaptive run
  is now 6 to 43 times closer to the reference than the 501-point fixed grid it
  replaced, where before it was worse. It costs about 35% more steps: the
  45-point campaign goes from 66.7 s to 78.9 s, still under the fixed grid's
  99.3 s. No signoff verdict changes anywhere, and of 50760 reported values the
  largest move is 0.1 mV.

  **中文：** 自适应求解器只要没有步长历史——t=0 时，以及每次在输入断点处重启之后——
  局部误差估计就无从计算，于是接下来两步会按启动启发式给出的尺寸被无条件接受。这两步
  恰好落在时钟沿之后电路变化最快的位置，而它们的误差对 `reltol` 不可见：在 TSMC28
  MDAC 电路上，把 `reltol` 收紧十倍（步数增加 115%）几乎不改变结果，而把启动步缩小
  25 倍则使误差降为四分之一。现在这种试探步会求解两次，一次整步、一次拆成两个半步，
  两者之差即误差估计；接受时两个半步都保留，因此紧接的下一步就已具备真实历史与完整
  误差控制。求解过程中不再有任何一步脱离误差控制。以同一激励下 40 倍密网格为基准，
  5 ns MDAC 窗口内的最大偏差：major carry case 由 3.69 mV 降至 92 µV，某个 residue
  case 由 308 µV 降至 68 µV；自适应结果现在比它取代的 501 点固定网格更接近基准 6 至
  43 倍，而此前是更差。代价是步数约增加 35%：45 点 campaign 由 66.7 s 增至 78.9 s，
  仍低于固定网格的 99.3 s。所有签核判定无一变化，50760 个上报数值中最大变动 0.1 mV。

- **Near-duplicate breakpoint times are merged / 近重复的断点时刻会被合并**

  **English:** The same near-coincidence left a sub-ULP interval in the merged
  grid itself. The solver tolerates it, but a zero-rise clock edge sampled
  across such a pair swings full scale over a gap of ~3e-27 s, and the implied
  slope then dominates the scale used to decide which input breakpoints deserve
  a solver restart — leaving every genuine edge below the threshold and
  undetected. Whether this happened at all came down to which way one sample
  rounded. Breakpoints within a tolerance of an existing sample now collapse
  onto the earliest time of their cluster. Keeping such a pair out of the grid
  is not sufficient on its own, because a caller can hand the adaptive solver
  any time grid it likes, so the breakpoint search itself now ignores intervals
  narrower than the smallest step the solver would ever take and compares each
  sample against its nearest steppable neighbour on either side. A degenerate
  pair the stimulus actually moves across is a real discontinuity and is
  reported as a breakpoint; one it does not move across says nothing and is
  dropped. On a grid without such a pair this is exactly the previous test.

  **中文：** 同一种近重合还会在合并后的网格里留下小于 1 ULP 的间隔。求解器本身能容忍
  它，但零上升沿的时钟在这样一对采样点之间会在约 3e-27 s 内完成整个摆幅，由此得出的
  斜率会主导"哪些输入断点值得让求解器重启"的判据尺度——真正的边沿全部落到阈值之下而
  被漏检。此事是否发生，仅取决于某个采样点向哪个方向舍入。现在与已有采样点相差在容差
  以内的断点会合并到该簇最早的时刻上。仅让网格不再携带这种采样对还不够，因为调用方
  可以把任意时间网格直接交给自适应求解器，所以断点检测本身现在也会忽略比求解器可能
  采用的最小步长还窄的间隔，并改用两侧最近的"可步进"邻居计算斜率。若激励确实在这种
  退化采样对之间发生了跳变，那是真实的不连续点，会被作为断点上报；若没有跳变则不含
  信息，直接丢弃。在不含此类采样对的网格上，结果与此前完全一致。

- **Native backend shutdown and reproducible CLI payloads / 原生后端关闭与 CLI 输出可复现**

  **English:** `NativeBsim4Backend.close()` now manages the current shared and
  scoped handle stores instead of referring to fields removed by the card-cache
  refactor. Shutdown is idempotent, rejects active leases, releases every cached
  handle, and prevents later evaluation. CLI result serialization now removes
  opaque objects and callables recursively from mappings, sequences, and object
  arrays, so internal sentinels and process-specific memory addresses cannot
  leak into output files.

  **中文：** `NativeBsim4Backend.close()` 现按当前 card cache 架构管理 shared 与
  scoped handle store，不再引用重构中已删除的字段。关闭操作可重复调用，会拒绝仍有
  活动租约的后端、释放全部缓存 handle，并阻止后续求值。CLI 结果序列化也会从映射、
  序列和 object array 中递归移除 opaque 对象与 callable，内部哨兵和进程相关内存
  地址不会再泄漏到输出文件。

- **SAR continuation now serves the public ADC workflow / SAR 续算现已接入公开 ADC 工作流**

  **English:** The Rust SAR kernel already continued one trajectory across bit
  decisions, but only the mismatch Monte-Carlo batch called it. The public
  `run_sar_conversion`, sweep, signal, CLI `adc`, and SAR exploration paths
  still replayed a complete transient for every bit. These entry points now
  obtain all decisions from the compiled continuation kernel and run exactly
  one final transient per input to retain the existing waveform, comparator
  trace, and power result contract. Sweeps compile their complete input vector
  once. Unsupported topologies retain an explicit Python replay fallback.

  **中文：** Rust SAR 内核早已能在各 bit 判决之间续算同一条轨迹，但此前只有失配
  蒙特卡洛 batch 调用了它。公开的 `run_sar_conversion`、sweep、signal、CLI `adc`
  和 SAR 探索路径仍然每个 bit 重跑一次完整瞬态。现在这些入口统一从编译式续算内核
  获得全部判决，每个输入只再运行一次最终瞬态，以保留原有波形、比较器 trace 与功耗
  返回契约；sweep 会一次编译完整输入向量。不支持的拓扑仍显式回退 Python replay。

- **Fixed-grid Gear2 solves and reconstructs with the same discretization / 固定步长 Gear2 的求解与重构改用同一离散式**

  **English:** The Rust Newton solve previously used variable-step BDF2 at
  every fixed-grid Gear2 sample, while Python branch-current reconstruction
  independently fell back to backward Euler when the step ratio exceeded two.
  The Rust solver is now the single source of truth: the first step uses
  backward Euler, later steps use variable-step BDF2 for
  `h_n / h_(n-1) <= 2`, and larger growth falls back to backward Euler. The
  exact selected coefficients are returned to Python for device-charge,
  capacitor-current, and source-current reconstruction.

  **中文：** 此前固定网格 Gear2 的 Rust Newton 求解在每个后续采样点都使用变步长
  BDF2，而 Python 支路电流重构会在步长比大于 2 时自行回退后向欧拉，导致状态求解
  与电流重构使用不同离散式。现在 Rust 求解器是唯一规则来源：首步使用后向欧拉，
  后续在 `h_n / h_(n-1) <= 2` 时使用变步长 BDF2，增长更快时回退后向欧拉；实际
  选中的系数原样返回 Python，用于器件电荷、电容电流和电源支路电流重构。

- **Native handles are isolated across concurrent solvers / 原生 handle 在并发求解器之间隔离**

  **English:** Native BSIM handles carry internal-node and voltage-limiting
  history. A per-call mutex prevents memory races but does not make it valid for
  two solvers to alternately advance the same handle. The backend now treats an
  active lease as exclusive and gives overlapping callers independent temporary
  handles, while sequential callers retain normal cache reuse. Signoff PVT
  points additionally keep private cache namespaces so history cannot leak
  between consecutive independent points. Two eight-worker runs of the
  45-point TSMC28HPC+ MDAC campaign are now identical, and eight workers agree
  with one. The guarantee no longer depends on every concurrent driver
  remembering a signoff-specific scope.

  **中文：** 原生 BSIM handle 会携带内部节点与电压限幅历史。单次调用互斥锁只能
  防止内存数据竞争，不能允许两个求解器轮流推进同一 handle。现在 backend 将 active
  lease 视为独占租约：重叠调用者获得独立的临时 handle，顺序调用仍正常复用缓存。
  signoff PVT 点另外保留私有缓存命名空间，防止连续的独立点继承历史。45 点
  TSMC28HPC+ MDAC campaign 的两次 8 worker 运行现在完全一致，且 8 worker 与单
  worker 结果相同；其他并发驱动也不再需要记住 signoff 专用 scope 才有基本隔离。

- **Adaptive transients no longer fail at t=0 / 自适应瞬态不再在 t=0 失败**

  **English:** `transient(..., adaptive=True)` could abort immediately with
  "failed at t=0", having solved nothing, on any grid fine enough for an input
  event to nearly coincide with a time sample. Those two nearly-equal times left
  a gap of about 3e-27 s, which was mistaken for the smallest step the stimulus
  demanded and shrank the starting step below the solver's own minimum. Only
  intervals the solver could actually step over are considered now. Both
  transient engines were affected and both are fixed: the BSIM engine first,
  and then the OTFT engine, which carried its own copy of the same routine and
  so did not inherit the first fix — the two now share one. Each engine keeps
  its own startup step size, which was never a shared choice. Runs that already
  worked are untouched: 552 result arrays across the SC-LPF, periodic-RC,
  FreePDK45 and TSMC28 MDAC adaptive paths are bit-identical.

  **中文：** 只要网格细到某个输入事件与时间采样几乎重合，`transient(...,
  adaptive=True)` 就可能立即以 "failed at t=0" 中止且未解出任何步。那两个几乎相等
  的时刻之间留下约 3e-27 s 的间隔，被误当作激励所要求的最小步长，把起始步长压到
  低于求解器自身下限。现在只考虑求解器真正可以跨越的间隔。两个瞬态引擎都受此影响，
  现均已修复：先是 BSIM 引擎，之后是 OTFT 引擎——后者持有同一段例程的另一份拷贝，
  因而没有继承前一次修复，现在两者共用同一份。各引擎仍保留自己的启动步长，那从来
  不是共用的选择。原本可用的运行不受影响：SC-LPF、periodic-RC、FreePDK45 与
  TSMC28 MDAC 自适应路径共 552 个结果数组逐位一致。

- **Adaptive transients stay inside the requested time window / 自适应瞬态不再超出请求的时间窗**

  **English:** An adaptive transient driven by a periodic stimulus solved to the
  end of the period rather than to its own `tstop`, and returned the extra
  samples. The MDAC signoff cases ask for 5 ns of a 10 ns period and got results
  out to 10 ns, which looked like a gross disagreement with the fixed grid when
  it was only a different window. The breakpoint merger also forced the first
  sample to zero and generated edges only in the first period, so a requested
  nonzero start was lost. The transient path now preserves both absolute
  endpoints and generates every repeated edge inside that window; a PSS orbit
  still closes exactly one period. Those MDAC cases also stopped simulating
  twice as far as asked, which is most of why adaptive is now under half the
  cost of the fixed grid.

  **中文：** 由周期性激励驱动的自适应瞬态会一直解到周期末尾而不是自己的 `tstop`，
  并把多出来的采样一并返回。MDAC 签核用例请求 10 ns 周期中的 5 ns，却拿回直到
  10 ns 的结果，看起来像与固定网格严重不符，实际只是窗口不同。断点合并器还会把
  首点强制改为零，并且只生成第一个周期内的边沿，因此非零起始时间会丢失。现在瞬态
  路径保持两个绝对时间端点，并生成窗口内每一次重复边沿；PSS 轨道仍然精确闭合一个
  周期。这些 MDAC 用例也因此不再多算一倍时长——这正是自适应现在能降到固定网格
  一半以下的主要原因。

- **Settling time reports zero when nothing had to settle / 无需建立时 settling time 报零**

  **English:** When a signal is already inside its tolerance band as the
  measurement window opens, the reported settling time was the floating-point
  gap between the declared `start_time` and the first sample at or after it —
  the TSMC28 MDAC `residue_zero` case reported 3.2e-27 s. It now reports 0.0 s,
  which is what "already settled" means. No constraint verdict changes; both
  values pass any settling limit.

  **中文：** 当信号在测量窗口开启时就已处于容差带内，报告的建立时间是声明的
  `start_time` 与其后第一个采样之间的浮点间隔——TSMC28 MDAC `residue_zero` 用例
  因此报出 3.2e-27 s。现在报告 0.0 s，这才是"已经建立"的含义。约束判定不变，两个
  值都能通过任何建立时间上限。

- **A circuit's other DC guesses are tried when the first one fails / 首个 DC 初值失败时会尝试其余初值**

  **English:** A circuit may declare several DC guesses. Loading one through
  `CircuitSpec.binding()` promoted the first of them to an authoritative seed,
  which made the solver trust it and skip its ordinary search — so a wrong first
  guess failed the whole analysis while the remaining declared guesses were
  never tried. The AFE example is exactly that: its first guess does not
  converge and its second does. A seed that only came from the binding default
  now falls back to the declared guesses. A seed the caller passed explicitly is
  still authoritative and gets no fallback, because the latch screen probes a
  specific operating region on purpose and reads a failure to converge as
  evidence about the design. Of the 25 example circuits that declare DC guesses,
  24 operating points are bit-identical; the 25th is the AFE, which changes from
  `invalid` to converged on the same branch the unseeded path finds. This also
  restores the engine-parity golden for `afe_explore`, which v2.1.5 had frozen
  as an empty payload because it silently dropped the non-converged analysis.

  **中文：** 一个电路可以声明多个 DC 初值。经 `CircuitSpec.binding()` 加载时，其中
  第一个会被提升为权威种子，求解器于是信任它并跳过常规搜索——首个初值写错就会让
  整个分析失败，而电路声明的其余初值从未被尝试。AFE 示例正是如此：第一个初值不
  收敛，第二个可以。现在仅来自 binding 默认值的种子会回退到声明的初值；调用方显式
  传入的种子仍然权威、不做回退，因为 latch 筛选正是刻意探测特定工作区，并把不收敛
  本身当作对设计的判据。在声明了 DC 初值的 25 个示例电路中，24 个工作点逐位不变；
  第 25 个即 AFE，从 `invalid` 变为收敛，且落在无种子路径找到的同一条支路上。这也
  恢复了 `afe_explore` 的 engine-parity golden——v2.1.5 曾因静默丢弃不收敛的分析
  而把它冻结成空载荷。

## [2.2.0] - 2026-07-25

### Added / 新增

- **Rust-compiled analog exploration / Rust 编译式模拟电路探索**

  **English:** Wired eligible `explore()` runs into `CompiledCampaign`.
  The BSIM compiler is topology-driven rather than a 5T macro: arbitrary MOS
  connectivity plus resistors, capacitors, independent sources, controlled
  sources, and augmented MNA branches are compiled from `CircuitSpec`.
  Candidate geometry, NF, process corner, and arbitrary named bias values are
  marshalled in cancellable chunks and evaluated under one GIL-free Rayon pool.
  Lazy noise now uses an explicit reusable `PreparedCampaign`: DC, operating
  point device linearization, the assembled MNA system, and the forward AC
  response are retained, then only survivor indices run the noise continuation.
  No native C handle crosses calls or worker threads. Campaign results expose
  low-frequency gain, UGF, integrated input/output noise, full four-terminal DC
  currents, and the augmented DC state, allowing exact topology-level source
  power and area reductions. CLI, HTTP jobs, and MCP exploration accept
  `workers`. Monostable BSIM4 circuits run cold; multistable OTFT AFE circuits
  require an explicit `seed_fn`, otherwise they retain the scalar reference
  path to prevent silent DC-root changes.

  **中文：** 将满足条件的 `explore()` 接入 `CompiledCampaign`。BSIM 编译器由
  拓扑驱动，不再是 5T 宏：可从 `CircuitSpec` 编译任意 MOS 连接、电阻、电容、
  独立源、受控源和增广 MNA 支路。候选 geometry、NF、工艺角和任意命名 bias
  以可取消分块送入同一个释放 GIL 的 Rayon 线程池。lazy noise 现在使用显式可复用
  `PreparedCampaign`：保留 DC、工作点器件线性化、已组装 MNA system 和正向 AC
  响应，再只对幸存候选索引续跑 noise；native C handle 不跨调用或 worker 线程。
  Campaign 结果包含低频增益、UGF、输入/输出积分噪声、完整四端 DC 电流和增广
  DC 状态，因此可精确归约拓扑源功耗与面积。CLI、HTTP job 和 MCP exploration
  均支持 `workers`。单稳态 BSIM4 电路可冷启动；多稳态 OTFT AFE 必须显式提供
  `seed_fn`，否则继续使用标量参考路径，避免静默改变 DC 根。

- **Per-candidate bias in compiled campaigns / compiled campaign 候选级 bias**

  **English:** Extended the Rust `CompiledCampaign` template and candidate
  schemas with stable named bias slots. OTFT and FreePDK45/SKY130/TSMC28 BSIM4
  batches can now evaluate different arbitrary bias values in one detached
  batch. Candidate mappings may partially override template defaults; Python
  regenerates topology DC guesses for the merged bias, while Rust uses the same
  normalized vector for passive MNA terms, MOS DC terminals, and operating-point
  linearization. Unknown names, non-finite values, wrong vector lengths, and
  malformed candidate guesses fail explicitly. Existing candidates that omit
  `bias` retain the template defaults.

  **中文：** 扩展 Rust `CompiledCampaign` 模板与候选 schema，加入稳定的命名
  bias 槽。OTFT 以及 FreePDK45/SKY130/TSMC28 BSIM4 批处理现在可在同一次
  detached batch 中评估各自不同的任意 bias 值。候选字典可只覆盖模板默认值的一部分；
  Python 会按合并后的 bias 重新生成拓扑 DC guesses，Rust 则让无源 MNA 端子、
  MOS DC 端子和工作点线性化统一使用同一归一化向量。未知名称、非有限值、错误向量
  长度和非法候选 guesses 都会显式失败；未提供 `bias` 的旧候选继续使用模板默认值。

- **Model Context Protocol server / Model Context Protocol 服务**

  **English:** Added the optional `mcp>=1.27,<2` adapter with
  `circuit-opt mcp`, `circuit-opt-mcp`, and `python -m circuitopt.mcp`
  entry points. The stdio and loopback-only Streamable HTTP transports expose
  capability discovery, strict circuit validation, bounded analysis summaries,
  asynchronous exploration/mismatch/signoff jobs, cooperative cancellation,
  and filtered signoff inspection. All file paths are confined to an explicit
  workspace; full outputs are written under `results/mcp`. FastAPI and MCP now
  share transport-neutral application operations rather than duplicating parse,
  solve, signoff, or serialization behavior.

  **中文：** 新增可选 `mcp>=1.27,<2` 适配层及 `circuit-opt mcp`、
  `circuit-opt-mcp`、`python -m circuitopt.mcp` 三个入口。stdio 与只允许
  loopback 的 Streamable HTTP transport 提供能力发现、严格电路校验、有界分析摘要、
  异步探索/失配/signoff、协作式取消和 signoff 条件筛选。所有文件路径都被限制在
  显式工作区内，完整结果写入 `results/mcp`。FastAPI 与 MCP 现在共用与 transport
  无关的 application operations，不重复实现解析、求解、signoff 或序列化行为。

- **Multi-testbench PVT signoff campaign / 多测试台 PVT 签核 campaign**

  **English:** Added `circuit-opt signoff`, the strict
  `schemas/signoff_campaign.schema.json` manifest, deterministic parallel PVT
  execution, and case/point/global worst-case aggregation. Relative circuit
  paths are resolved from the manifest and cannot escape its directory; affine
  `$pvt` expressions bind supply-dependent stimuli without machine-specific
  paths or generated netlists. Model failures, non-convergence, non-finite
  results, and invalid signoff configurations remain explicit `invalid` cases.
  Transient saturation can now declare named time checkpoints; every checkpoint
  reconstructs the node state and re-evaluates the exact PDK-bound MOS operating
  regions. Resistor thermal noise now follows the shared PVT-bound MOS
  temperature instead of remaining fixed at 300.15 K. The TSMC28HPC+ MDAC
  example supplies an 11-case, 45-point manifest
  covering open-loop gain, differential/CMFB loops, wideband input/output noise,
  five residue levels, and the 0111-to-1000 major-carry transition.

  **中文：** 新增 `circuit-opt signoff`、严格
  `schemas/signoff_campaign.schema.json` 配置、顺序确定的并行 PVT 执行，以及
  case/逐点/全局最差点汇总。电路路径从 manifest 相对解析且不能逃出其目录；
  仿射 `$pvt` 表达式可绑定随电源变化的激励，不需要机器相关路径或生成网表。
  模型失败、不收敛、非有限结果和无效 signoff 配置都会保留为显式 `invalid`。
  瞬态饱和检查现在可声明命名时刻；每个检查点都会重建节点状态，并用精确 PDK
  绑定重新计算 MOS 工作区。电阻热噪声现在跟随 PVT 绑定的统一 MOS 温度，不再固定
  为 300.15 K。TSMC28HPC+ MDAC 示例提供 11-case、45 点 manifest，
  覆盖开环增益、差模/CMFB 环、宽带输入/输出噪声、五档 residue 和
  0111→1000 主进位切换。

### Changed / 变更

- **Native BSIM LTE-adaptive Gear2 / 原生 BSIM LTE 自适应 Gear2**

  **English:** Native SKY130, FreePDK45, and TSMC28 BSIM transient dispatch now
  honors `adaptive=True` instead of expanding and solving a fixed grid. The Rust
  core performs one nonlinear Gear2 solve per trial, forms a variable-step
  BDF3-minus-BDF2 defect directly from the accepted BSIM terminal-charge
  history, and projects that defect through the converged BDF2 Jacobian with one
  linear solve. A PI controller rejects over-tolerance trials, limits step
  growth, suppresses growth immediately after a rejection, and restarts
  conservatively at detected input-slope breakpoints. LTE controls dynamic
  node-voltage states; algebraic MNA ideal-source branch currents are excluded.
  Profiles distinguish LTE estimates, LTE linear solves, LTE rejections, and
  Newton rejections. On the local FreePDK45 MDAC case, default tolerances reduced
  500 fixed steps to 146 accepted steps with one rejection and warm end-to-end
  time from about 167 ms to 56.6 ms. Against a 0.5 ps reference with the same
  10 ps input edge, peak/final differential-output errors were about 2.04 mV /
  50 uV; tightening `reltol/vabstol` from `1e-4/1e-6 V` to `1e-5/1e-7 V`
  reduced them to about 0.42 mV / 12 uV.

  **中文：** SKY130、FreePDK45 和 TSMC28 的原生 BSIM transient 派发现在会真正
  执行 `adaptive=True`，不再展开后按固定网格求解。每个 trial 只做一次非线性
  Gear2 求解；Rust 核从已接受的 BSIM 端电荷历史直接构造变步长
  BDF3−BDF2 defect，再通过收敛的 BDF2 Jacobian 做一次线性投影。PI controller
  负责拒绝 LTE 超差试算、限制增长、拒步后禁止立即增步，并在输入斜率断点保守
  重启。LTE 只控制动态节点电压，理想源的 MNA 代数支路电流不进入误差范数。
  profile 会分别报告 LTE 估计/线性求解/LTE 拒步/Newton 拒步。在本地
  FreePDK45 MDAC 用例上，默认容差把 500 个固定步降为 146 个接受步和 1 次拒步，
  热端到端时间约由 167 ms 降至 56.6 ms。相对具有相同 10 ps 输入边沿的 0.5 ps
  固定网格参考，差分输出峰值/最终误差约为 2.04 mV / 50 uV；把
  `reltol/vabstol` 从 `1e-4/1e-6 V` 收紧到 `1e-5/1e-7 V` 后，误差降为约
  0.42 mV / 12 uV。

- **Converged-state BSIM result reuse / 收敛状态 BSIM 结果复用**

  **English:** Native BSIM transient now records accepted-state device-current
  and charge history directly from the final converged Newton batch. Because a
  sub-tolerance correction is not applied, that batch already corresponds
  exactly to the accepted state; only failed rounds refresh I/G/Q/C after their
  last state update. On the reference TSMC28 MDAC `residue_plus_fs16` case this
  removed 500 batches and 19,000 MOS evaluations (`5,360 -> 4,860` batches,
  `203,680 -> 184,680` evaluations), reducing warm solver median from about
  0.2558 s to 0.2333 s while preserving 4,859 Newton iterations, 497 predictor
  steps, and zero failures. Final differential output changed by only 2.3 pV.

  **中文：** 原生 BSIM transient 现在直接使用最后一轮收敛 Newton batch 记录
  接受状态的器件电流和电荷历史。小于容差的 correction 不会写回状态，因此该 batch
  已与接受状态精确对应；只有失败轮次在最后一次状态更新后重新计算 I/G/Q/C。在参考
  TSMC28 MDAC `residue_plus_fs16` case 上，这消除了 500 次 batch 和 19,000 次
  MOS 求值（batch `5,360 -> 4,860`，求值 `203,680 -> 184,680`），热 solver
  中位时间由约 0.2558 s 降至 0.2333 s，同时保持 4,859 次 Newton、497 个
  predictor 步及零失败；最终差分输出仅变化 2.3 pV。

- **Guarded Gear2 state predictor / 受保护的 Gear2 状态预测器**

  **English:** Native BSIM Gear2 transient now seeds Newton with the
  variable-step linear extrapolation
  `x[n] + h[n+1]/h[n] * (x[n] - x[n-1])`. Prediction starts only after two
  consecutive converged states, rejects step growth above 4x, and detects input
  slope discontinuities so clock and DAC-code edges still start from the last
  accepted state. Set `CIRCUITOPT_BSIM_GEAR2_PREDICTOR=0` for regression A/B;
  profiles report `gear2_predictor_steps`. On the reference TSMC28 MDAC
  `residue_plus_fs16` case, predictor on/off reduced Newton iterations from
  9,970 to 4,859 and warm solver time from about 0.4945 s to 0.2558 s, with
  zero failed steps in both runs. The maximum A/B node-waveform difference was
  75.9 uV and the differential-output difference was 63.8 uV.

  **中文：** 原生 BSIM Gear2 transient 现在使用变步长线性外推
  `x[n] + h[n+1]/h[n] * (x[n] - x[n-1])` 作为 Newton 初值。仅在连续两个状态
  收敛后启用；步长增长超过 4 倍时禁用；输入斜率出现突变时也会禁用，因此时钟和
  DAC 码型边沿仍从上一接受状态启动。回归 A/B 可设置
  `CIRCUITOPT_BSIM_GEAR2_PREDICTOR=0`，profile 会报告
  `gear2_predictor_steps`。在参考 TSMC28 MDAC `residue_plus_fs16` case 上，
  predictor on/off 将 Newton 次数从 9,970 降至 4,859，热 solver 时间由约
  0.4945 s 降至 0.2558 s，两边均无失败步；A/B 全节点波形最大差 75.9 uV，
  差分输出最大差 63.8 uV。

- **Batched native BSIM transient Newton / 原生 BSIM 瞬态 Newton 批量求值**

  **English:** Each native BSIM transient Newton iteration now resolves all MOS
  terminal voltages first and submits them through a persistent
  `EvalBatchWorkspace` with `co_bsim4::eval_batch_into`. Handles, terminal rows,
  I/G/Q/C result slots, and status storage are allocated once per transient;
  Rayon workers write directly to disjoint result slots without a temporary
  `BatchResult` vector or second internal copy. The dedicated pool follows the
  requested machine parallelism but caps its default at 10 workers to avoid the
  measured 12-thread scheduling regression. Set
  `CIRCUITOPT_BSIM_BATCH_THREADS=1..10` to tune it. Cached devices can repeat a
  native handle: those slots are grouped onto one worker and evaluated serially,
  while distinct handles remain parallel, avoiding concurrent mutation of one
  BSIM instance. Residual/Jacobian stamping retains deterministic device order;
  initial history and failed-state refreshes use the same batch route, while a
  converged state reuses its final Newton result. The existing `eval_batch` ABI
  remains available as a compatibility wrapper.
  Native transient profiles expose `bsim_batch_calls` so scalar fallback cannot
  remain hidden. On the reference TSMC28 MDAC `residue_plus_fs16` case, the
  500-step Rust solver time fell from about 1.377 s to about 0.493 s while preserving
  9,970 Newton iterations, 397,898 device evaluations, and zero failed steps.

  **中文：** 原生 BSIM 瞬态的每轮 Newton 现在会先解析全部 MOS 端口电压，再通过
  持久的 `EvalBatchWorkspace` 和 `co_bsim4::eval_batch_into` 提交。handle、端口行、
  I/G/Q/C 结果槽及 status 存储在每次 transient 中只分配一次；Rayon worker 直接写入
  互不重叠的结果槽，不再创建临时 `BatchResult` 向量或执行内部二次复制。专用线程池
  会跟随请求的机器并行度，但默认最多使用 10 个 worker，以避开实测的 12 线程调度
  反噬；可用 `CIRCUITOPT_BSIM_BATCH_THREADS=1..10` 调节。缓存设备可能重复使用同一
  native handle：这些槽会归到同一 worker 内串行求值，不同 handle 仍可并行，从而
  避免并发修改同一 BSIM 实例。residual/Jacobian 仍按确定的器件顺序盖章；初始历史
  和失败状态刷新走相同批量路径，收敛状态则复用最后一轮 Newton 结果；既有
  `eval_batch` ABI 保留为兼容包装。原生
  transient profile 新增 `bsim_batch_calls`，避免标量回退被静默隐藏。在参考 TSMC28
  MDAC `residue_plus_fs16` case 上，500 步 Rust solver 时间由约 1.377 s 降至约
  0.493 s，同时保持 9,970 次 Newton、
  397,898 次器件求值及零失败步。

- **Shared native BSIM4 card cache / 公共原生 BSIM4 card 缓存**

  **English:** TSMC28, FreePDK45, and SKY130 now reuse immutable
  PDK/model/instance card bundles through one bounded, thread-safe BSIM4 LRU.
  Its normalized key covers the complete explicit PDK/model/section/bin
  binding, geometry, finger count, multiplicity, temperature, process corner,
  mismatch offset, source-file fingerprint, and PDK-specific instance fields.
  Per-key single-flight construction avoids duplicate cold builds under
  parallel PVT execution. Replacing a model source invalidates derived entries
  through its path/mtime/size fingerprint. Repeated 121-point TSMC28 5T OTA AC
  runs on the reference development machine improved from 27.91 ms to 1.33 ms
  median.

  **中文：** TSMC28、FreePDK45 与 SKY130 现在通过同一个有界、线程安全的
  BSIM4 LRU 复用只读 PDK/model/instance card 三件套。规范化缓存键覆盖完整显式
  PDK/model/section/bin 绑定、几何尺寸、指数、倍乘、温度、工艺角、失配偏移、
  源文件指纹及 PDK 特有实例字段；按键 single-flight 构建可避免并行 PVT 的重复
  冷启动。模型源的路径、mtime 和大小变化会自动使派生项失效。在参考开发机上，
  重复执行 121 点 TSMC28 5T OTA AC 的中位时间由 27.91 ms 降至 1.33 ms。

- **Strict simulation contract / 严格仿真契约**

  **English:** Every MOS now requires an explicit `pdk`/`model`/`section`/`bin`
  binding. Compact-model failures, non-convergence, and non-finite outputs mark a
  candidate invalid instead of substituting currents, noise, or waveforms. Solve
  outputs expose unit-bearing measurements; power is computed from solved source
  branches and complete terminal currents. The new explicit `signoff` contract
  requires PM to come from a declared voltage-source loop injection and return
  ratio, settling to use a declared signal/target/window/tolerance, noise to name
  its integration band and input/output reference, and saturation to name its MOS
  set and minimum headroom. `circuit-opt run --output` and the solve service now
  share `{status, results, signoff}`, where signoff always contains
  `status/measurements/constraints/passed/worst_case`. Ordinary AC responses and
  final transient samples are no longer silently interpreted as PM or targets.

  **中文：** 每个 MOS 现在都必须显式绑定 `pdk`/`model`/`section`/`bin`。紧凑模型
  求值失败、不收敛或非有限输出都会把候选标为 invalid，不再替代电流、噪声或波形。
  输出提供带单位的测量值；功耗由已求解的真实源支路和完整端口电流计算。新增显式
  `signoff` 契约：PM 必须来自声明的电压源环路注入与返回比，建立时间必须声明
  signal/target/window/tolerance，噪声必须声明积分频带及输入/输出引用，饱和检查必须
  声明 MOS 集合与最小余量。`circuit-opt run --output` 与求解服务统一使用
  `{status, results, signoff}`，其中 signoff 固定包含
  `status/measurements/constraints/passed/worst_case`。普通 AC 响应和瞬态末点不再被
  静默解释为 PM 或建立目标。

### Fixed / 修复

- **Cross-platform calibration gates / 跨平台校准门**

  **English:** Added two-stage PSS with `final_tgrid`: adaptive Gear2 performs
  stabilization without freezing its state-dependent accepted grid, then
  shooting, analytic monodromy, profiling, and the returned PAC/PNoise orbit run
  on a deterministic event-aligned grid. JSON analysis dispatch exposes the
  same path through `pss.final_n_points`. SC-LPF calibration now uses a
  201-point event-aware warmup grid and a 3201-point base final grid augmented
  with every clock edge; changing the adaptive initial step by +/-10% changes
  integrated output noise by less than 2e-5 relative. The latch-screen
  regression also accepts the documented conservative `+inf` result when an
  adversarial solve does not converge on non-reference platforms.

  **中文：** 新增带 `final_tgrid` 的两阶段 PSS：adaptive Gear2 只负责
  stabilization，且不冻结状态相关的 accepted grid；随后 shooting、解析
  monodromy、profile 和返回给 PAC/PNoise 的轨迹全部运行在确定性的事件对齐
  网格上。JSON analysis dispatch 通过 `pss.final_n_points` 提供同一入口。
  SC-LPF 校准使用 201 点事件感知 warmup 网格，以及并入全部时钟边沿的 3201 点
  基础 final grid；adaptive 初始步长变化 +/-10% 时，积分输出噪声的相对变化
  小于 2e-5。latch-screen 回归测试也同步接受已定义的保守 `+inf` 语义：
  非参考平台上的对抗求解不收敛时不再误判 CI 失败。

- **Native BSIM transient profiling / 原生 BSIM 瞬态性能统计**

  **English:** `transient(..., profile=True)` now reaches the native BSIM
  execution path instead of being dropped during dispatch. The Rust fixed-grid
  kernel directly counts Newton iterations, native BSIM evaluations, and failed
  expanded-grid steps; Python exposes those counters, step indices, solver work,
  and wall time through `result["transient_profile"]`. Profiling disabled keeps
  the counters and result field off.

  **中文：** `transient(..., profile=True)` 现在会真正传入原生 BSIM 执行路径，
  不再在分派时丢失。Rust 固定网格内核直接统计 Newton 迭代次数、原生 BSIM
  求值次数及失败的展开网格步；Python 通过 `result["transient_profile"]` 返回
  这些计数、失败步索引、solver 工作量和 wall time。关闭 profile 时不启用计数，
  也不返回该结果字段。

- **Exact `max_step` grid subdivision / 精确 `max_step` 网格分段**

  **English:** Native BSIM4 and compiled SAR grid expansion now snap an
  interval-to-`max_step` ratio to its nearest integer within a `1e-12` relative
  tolerance before applying `ceil`. This prevents decimal `linspace` roundoff
  from splitting intervals that already equal the requested maximum step, while
  genuinely oversized intervals still subdivide. The 501-point, 10 ps TSMC28
  MDAC transient no longer creates 207 spurious internal steps; its warm runtime
  on the reference development machine fell from about 1.92 s to 1.42 s.

  **中文：** 原生 BSIM4 与编译 SAR 的网格展开现在会先在 `1e-12` 相对容差内，
  将“区间长度/`max_step`”吸附到最近整数，再执行 `ceil`。这避免十进制
  `linspace` 舍入误差把已经等于最大步长的区间再次切分，同时仍会切分真正超限
  的区间。TSMC28 MDAC 的 501 点、10 ps 瞬态不再产生 207 个伪内部步；在参考
  开发机上，热运行时间由约 1.92 s 降至 1.42 s。

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

[Unreleased]: https://github.com/751K/circuit-optimization-lab/compare/v2.5.0...HEAD
[2.5.0]: https://github.com/751K/circuit-optimization-lab/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/751K/circuit-optimization-lab/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/751K/circuit-optimization-lab/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/751K/circuit-optimization-lab/compare/v2.1.5...v2.2.0
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
