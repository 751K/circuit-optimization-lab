# 后续开发计划

[English README](README.md) | [中文说明](README_zh.md) | [核心求解器概览](core_overview_zh.md)

## 当前状态（2026-06-28）

项目是一个成熟的本地模拟电路仿真与设计探索框架，首个应用场景为 AT4000TG PMOS-OTFT ECG AFE，
已对 Cadence Spectre 24.1 完成校准。核心能力全部落地，正在从"功能开发"转向"生态完善"阶段。

### 已完成的能力矩阵

| 领域              | 交付内容                                                                                                             | 状态       |
| --------------- | ---------------------------------------------------------------------------------------------------------------- | -------- |
| **电路描述**        | JSON 格式 + schema 校验，多器件/多输出/仿真参数内嵌                                                                               | ✅ 成熟     |
| **DC/AC/Noise** | 工作点求解、小信号增益/带宽、热噪声+闪烁噪声、等价输入噪声                                                                                   | ✅ 成熟     |
| **瞬态**          | 后向欧拉（默认）+ gear2/BDF2（可选），BE 与裸 gear2 maxstep/retry/subdivision 均走 Numba grid；Python/LS 仅作兜底                                  | ✅ 成熟     |
| **周期分析**        | 通用 shooting PSS（解析 monodromy + Broyden 复用）、通用 PAC（通用默认解析伴随 HB，chopper 默认 time-domain Floquet + PMOS gate1 内部状态）、通用 PNoise（HB 对照/兜底 + chopper 默认 TD-adjoint，PMOS gate1 扩维，第一性原理，无标定常数） | ✅ 成熟     |
| **周期验证**        | SC 低通（2-PMOS 开关电容，201 点/周期，f_clk=1kHz），PSS/PAC/PNoise 全路径跑通。**2026-06-22 修了 3 个 bug**（见下文 A1/A3/A4），修复后 PAC BW 17.2 Hz vs 解析 15.9 Hz（Δ=+8%，修复前 Δ=−28%），PNoise IRN 4.5 µVrms（修复前 20.6 µVrms，flicker 过估计已消除） | ✅ 新增     |
| **Stiff PSS**    | Levenberg-Marquardt trust-region Newton（mu=0 → 精确 Newton；stiff τ≫T 时自动 regularize）+ 物理边界 runaway 检测 + best-physical 回退，SC 低通 residual 从 0.07V 降至 0.007V（10× 改善），chopper 路径逐字节不变 | ✅ 完成     |
| **Flicker噪声修正** | PNoise 闪烁噪声用 FFT(√PWR) 调制幅值向量替代 FFT(PWR) 功率谱构建 cyclostationary 折叠矩阵，消除强调制器件（硬开关）的过估计（<PWR>/<√PWR>² 倍），恒偏置器件（AFE amp）不变 | ✅ 完成     |
| **Chopper**     | 5 层分析：理想 LPTV → PMOS 静态相位 → 有限边沿谐波 → quasi-LPTV 边带折叠 → hard-switched PSS/PAC/PNoise                              | ✅ 成熟     |
| **元件类型**        | PMOS_TFT、电阻、电容、理想直流电流源、VCCS、VCVS、CCCS、CCVS、理想时变电流源（charge injection）、理想电压源（真·MNA）。受控源 DC/AC/Noise/Transient/PSS/PAC/PNoise 全覆盖（2026-06-22 补齐 transient + compiled_topology 路径） | ✅ 成熟     |
| **器件模型接口**      | `TransistorModel` ABC + `NumbaParams` + 工厂/注册表，求解器全部通过接口调用，支持新增模型类型而不改 solver 代码                                            | ✅ 完成     |
| **设计探索**        | JSON 配置层、LHS/随机采样、约束过滤、Pareto 选择、CSV/JSONL 导出、CLI                                                                | ✅ 成熟     |
| **Surrogate 基础** | 已具备可信 teacher solver、JSON 设计空间、批量 sweep/export、Cadence 校准数据；尚未实现 ML 训练/推理层                                      | 🟡 待扩展    |
| **工艺角/鲁棒性**     | 全局 corner（typ/slow/fast）、逐器件 mismatch MC、确定性 latch 筛查                                                            | ✅ 成熟     |
| **Numba 加速**    | PMOS 电流、内部节点 Newton、偏置电容、terminal derivative、BE/gear2 transient grid（含 maxstep/retry）、PAC 周期线性化/time-domain PAC、PMOS gate1 PAC 转换线性化、PNoise HB block 组装和噪声折叠 | ✅ 全覆盖    |
| **gear2/BDF2**  | 变步长 BDF2、Numba grid、解析 monodromy、裸 transient retry/subdivision。PAC baseband 三 corner 全部 <1%（BE 时 −2.5%）。PSS/PAC/PNoise 默认 gear2 | ✅ 完成     |
| **CLI**         | `python -m core <circuit.json>` 全分析 dispatch + exploration 模式 + 结果导出 + CLI 参考手册（`docs/cli_reference.md`）             | ✅ 完成     |
| **Demo**        | Flask Web 前端 + REST API（`demo/server.py`）                                                                        | ✅ 可用     |
| **测试**          | 16 个测试文件；最近全量回归 `146 passed, 9 skipped`                                                        | ✅ 覆盖核心路径 |
| **文档**          | 英/中双语：README、core_overview、JSON 格式参考、gear2 完成报告、CLI 参考手册                                                     | ✅ 完善     |

### 代码规模

```
core/                     ~13,000 行  (22 个 .py 文件)
tests/                     ~2,800 行  (16 个 test_*.py 文件)
benchmarks/                  ~500 行  (4 个 benchmark)
examples/                    ~400 行  (sc_lpf.py + sc_lpf.json + vcvs_amplifier.json)
docs/                      ~4,800 行  (12 个 .md 文件)
calibration/                ~450 行  (5 个 case 目录, 含 PSFASCII 参考文件)
```

### 对标状态

| 指标                                      | 对标结果                                              |
| --------------------------------------- | ------------------------------------------------- |
| DC 工作点 / AC 增益                          | 与 Spectre 误差 ~0.01 dB                             |
| AC 带宽                                   | 对齐 Spectre                                        |
| 等价输入噪声（非 chopper）                       | 百分之几以内                                            |
| Chopper PSS/PAC/PNoise（原生，无标定常数）        | 默认 time-domain PAC 和 TD-adjoint PNoise 已对齐 D3 slow：PAC +0.03%、IRN +0.02%；三 corner TD PNoise IRN 误差 +0.02% / −0.00% / +0.57% |
| Chopper transient（8-PMOS hard-switched） | 输出均值 −10.76 mV vs Spectre −10.62 mV，nfail=0       |
| Mismatch MC mean/std                    | 与 Cadence 趋势一致                                    |

---

## 路线图

### 优先级排序

1. ~~Cadence 校准闭环~~ ✅ **已完成** — `calibration/` + `core/psf.py` + `core/calibration.py` + `core/cadence_netlist.py` + CI 回归测试
2. **扩展验证覆盖** 🟡 — SC 低通本地调试 ✅ + Cadence 对标 ✅（2026-06-22，`calibration/sc_lpf/` 全 PASS）；剩余:扩展对标覆盖（更多 switch 尺寸 / f_chop）
3. ~~器件模型抽象~~ ✅ **已完成** — `TransistorModel` ABC + `NumbaParams` + 工厂/注册表
4. ~~扩展元件类型~~ ✅ **已完成** — VCCS、理想电压源、VCVS、CCCS、CCVS 全覆盖（2026-06-22 补齐 transient + compiled_topology 路径）；仅互感未实现
5. ~~transient 性能深化：BE/gear2 grid 化~~ ✅ **已完成** — 默认 BE 全 Numba；raw gear2 maxstep/retry/subdivision 也留在 Numba grid
6. **搜索策略扩展** — 贝叶斯优化、进化算法
7. **ML surrogate 建模路线** — 把现有 solver/explore/calibration 升级成 surrogate 数据生成、训练、验证和优化闭环
8. **编译后端评估** — Rust/Cython 承担千级/万级 sweep

---

## 1. Cadence/Spectre 校准闭环 ✅ 已完成（2026-06-21）

### 交付内容

一键式自动校准流水线，四层组件协同工作：

```
core/cadence_netlist.py    从仓库拓扑+sizes/bias 生成 Spectre 网表（与 solver 同参）
        ↓
Spectre on flex            跑 DC/AC/Noise/PSS/PAC/PNoise → PSFASCII 参考文件归档到 calibration/<case>/
        ↓
core/psf.py                通用 PSFASCII 解析器（7 种分析类型），从 HEADER 自动提取 provenance
        ↓
core/calibration.py         load_reference → run_local → compare_* → format_report，逐指标 + per-case 容差
        ↓
tests/test_calibration.py   回归守卫：5 个 case 全部 PASS 才算通过
```

### 已完成的全部任务

- [x] **`calibration/` 目录 + 格式约定** — 每用例一子目录 + `metadata.json`（provenance/circuit/analyses/tolerances/solver），PSFASCII 参考文件就地存放
- [x] **`core/psf.py`（224 行）** — 通用 PSFASCII 解析：`parse_dc` / `parse_ac` / `parse_noise` / `parse_tran` / `parse_pac` / `parse_pnoise`，`provenance()` 从 PSF HEADER 自动提取 Spectre 版本/日期/fundamental
- [x] **`core/calibration.py`（360 行）** — `load_reference`→`run_local`→`compare_dc/ac/noise/pac/pnoise`→报告；CLI `python -m core.calibration [--all --analyses --json --relaxed]`
- [x] **`core/cadence_netlist.py`（203 行）** — `gen_amp_netlist` / `gen_chopper_netlist` 从仓库拓扑+sizes 生成网表，与 solver 同参；含 `gen_runner` bash 模板
- [x] **`tests/test_calibration.py`（59 行）** — amp 精确 PASS + chopper 三 corner 慢测守卫（`RUN_SLOW_CHOPPER=1`）；参考数据随码入库
- [x] **首次全量闭环 PASS**（Spectre 24.1.0.078，2026-06-21）：amp DC/AC/noise 精确到机器精度（gain +0.00dB / IRN +0.0%）；chopper PAC/PNoise 首版三 corner 均在 ~1–2%。后续 TD PAC/TD PNoise 已把 D3 slow 收到 PAC +0.03%、IRN +0.02%。

### 对标结果

| case | 指标 | local | Cadence | Δ |
|------|------|------:|--------:|----:|
| amp_design3_typical | gain / IRN | 22.90 dB / 38.31 µV | 22.89 dB / 38.31 µV | **+0.00 dB / +0.0%** |
| chopper_design3_typical | PAC gain / IRN | TD PAC / TD PNoise | Spectre PSS/PAC/PNoise | **PAC <1% / IRN −0.00%** |
| chopper_design3_slow | PAC gain / IRN | TD PAC / TD PNoise | Spectre PSS/PAC/PNoise | **PAC +0.03% / IRN +0.02%** |
| chopper_design3_fast | PAC gain / IRN | TD PAC / TD PNoise | Spectre PSS/PAC/PNoise | **PAC <1% / IRN +0.57%** |

PNoise IRN 的旧 HB-K32 → 新 TD-adjoint 三角误差：

| corner | IRN 修前（HB-K32） | IRN 修后（TD） |
|--------|-------------------:|---------------:|
| slow | +1.81% | +0.02% |
| typical | +1.05% | −0.00% |
| fast | +0.66% | +0.57% |

这一路出现过三次“假舒适”：HB-K64 gain 看似收敛、IRN 折算看似接近、PNoise K 截断看似够用。最终闭合方式都是避开经验截断/塌缩：PAC 用 time-domain Floquet + gate1 + average orbit，PNoise 用截断无关的 TD adjoint。

### 剩余跟进（低优先级）

- [ ] **扩展对标用例** — 更多 switch 尺寸（5000/30）、更多 f_chop（100/300/1k Hz）、stb/xf 分析
- [x] **替换经验常数**（2026-06-22）— `_CADENCE_PMOS_CHOPPER_CONVERSION_PHASE_RAD`=24.93° 和 `_PERIODIC_NOISE_PSD_SCALE`=1.0355 已 **retire**。它们只是给快速一阶 quasi-static `pmos_chopper_lptv_analysis` 打的补丁；无常数的一等公民是谐波平衡路径（`pmos_chopper_pss`→`pmos_chopper_pac`/`pmos_chopper_pnoise`，`core/calibration.py` 校验它）。`lptv_analysis` 现诚实返回一阶估计（增益偏低 ~10%）。

---

## 2. 扩展周期分析验证 🟡 高优先级（进行中）

### 现状

PSS/PAC/PNoise 三件套已做成通用拓扑级求解器（`pss_solve` / `pac_solve` / `pnoise_solve`），
已验证的周期拓扑：

- `examples/periodic_rc.json` — 无源 RC 低通（trivial 用例，无真实时变元件）
- PMOS chopper 八开关拓扑（通过 `pmos_chopper_pss/pac/pnoise` 包装器）
- ✅ **新增** `examples/sc_lpf.json` + `examples/sc_lpf.py` — 两相开关电容低通，PMOS 开关 + 理想 vsource 时钟，PSS/PAC/PNoise 全路径验证

### SC LPF 验证结果（2026-06-22，bug 修复后）

| 指标 | 结果 | 解析值 | 评价 |
|------|------|--------|------|
| PSS 收敛 | residual=0.0068V（0.034% of 20V），pss_status=converged_stabilization | 0 | ✓ 10× better than pre-fix (0.07V)；LM Newton 让 stiff 轨道不跑飞 |
| PAC 增益 | 0.987 (−0.11 dB) | 1.0 (0 dB) | ✓ 高度一致 |
| PAC 带宽 | 17.2 Hz | 15.9 Hz | Δ=+8%（PMOS Ron 增加等效电阻；修复前 Δ=−28%） |
| PNoise IRN | 4.46 µVrms | — | ✓ 量级合理（修复前 20.6 µVrms 因 flicker 过估计） |
| JSON dispatch | `python -m core run sc_lpf.json -a pac,pnoise` 跑通 | — | ✓ |

### 2026-06-22 Bug 修复详情

SC 低通最初（2026-06-21）的本地结果不可靠，PSS 收敛勉强且 PNoise IRN 明显异常。排查后找到并修复了三个根因：

#### A1. Transient: signed device current（根因 #1 — SC LPF runaway）

**症状**：SC 低通 VOUT 从 20V 漂移至 ~333V（rail-clipped），PSS shooting 无法收敛。

**根因**：`transient_solver.py` 中 PMOS 电流使用 `abs(Idc)`。对正向 PMOS（源端电位更高，`I_d1_d>0`），`abs==signed`，AFE amp/chopper 不变。但对反向偏置的 pass-gate 开关（漏端电位高于源端），`abs(Idc)` 翻转了电流符号 → 反恢复泵（anti-restoring pump），将轨道推向 spurious 固定点。

**修复**：`signed_devices` 不再可选 —— `dev_meta` 中所有器件强制 `signed=True`，始终使用带符号的 Verilog-A 漏电流。Chopper 原有的 `signed_devices` 参数保留（无操作）以向后兼容。`transient_solver.py:169-173`。

#### A3. PSS: Levenberg–Marquardt + physical runaway detection（根因 #2 — shooting 在 stiff basin 中 overshoot）

**症状**：即使 transient 修复后，PSS shooting 在 τ/T≈10 的 stiff 轨道上仍容易 overshoot。旧的 line-search damping（`alpha *= 0.5`）在近乎奇异的 I−M（Floquet multiplier ≈ 0.9）面前不够 —— 一步就可能跨出物理 basin。

**修复**（`pss_solver.py` 全部重写 step-acceptance 路径，共 ~240 行变更）：

1. **Levenberg–Marquardt trust-region**（默认 on，`levenberg_marquardt=True`）：`mu=0` → 精确 Newton step（well-conditioned 电路逐字节不变）；只在 rejection 时 `mu` 增长，regularize 近奇异的 I−M。`mu` 通过 `lm_up=8×`/`lm_down=1/3` 跨迭代自适应携带。
2. **Physical bounds**（`_physical_span` / `_within`）：从 rail 电压计算物理边界 `[lo − 2×span, hi + 2×span]`。超出边界的 trial step 触发 LM `mu` 提升而不是浪费一次 transient 评估。
3. **`_stabilize()` 子程序**：伪瞬态 stabilization 不再做 rail-clip —— 剪辑 runaway 会在边界上锻造出虚假的零残差固定点。现在每个 period 追踪 best-physical-orbit；检测 runaway（越界 OR 残差回头上升超过 best 的 3×），立即回退到 best-physical。
4. **A2 adaptive-stabilization fallback**：shooting 不收敛 but best orbit 物理 → 从 best orbit 扩展伪瞬态 stabilization。
5. **A4 honest status**：结果新增 `pss_status` 字段 —— `"converged_shooting"` | `"converged_stabilization"` | `"best_physical"` | `"diverged"`。物理越界的最终轨道标记为 diverged，不谎报收敛。
6. **`_branch_incidence` 扩展**：包含 VCVS+CCVS 支路（不再只是 vsources），PSS monodromy 和 PAC/PNoise HB 的 bordered 块现在对受控电压源正确。

**结果**：SC 低通 PSS residual 从 0.07V → 0.0068V（10× 改善）；chopper 路径在 `levenberg_marquardt=True`（默认）下逐字节不变。

#### A4. PNoise: flicker noise modulation（根因 #3 — 闪烁噪声强调制过估计）

**症状**：SC 低通 PNoise IRN = 20.6 µVrms，量级可疑偏高。

**根因**：`pnoise_solver.py` 和 `numba_kernels.py` 中原闪烁噪声折叠用 `FFT(PWR)` 构建 cyclostationary 谐波矩阵，再加可分离的 `1/√(νₖνₗ)` 权重。这对恒偏置器件（AFE amp 的 `PWR(t)` ≈ 常数 → `FFT(PWR)` 仅 DC 分量非零）正确，但对硬开关器件（PWR(t) ∝ Ich(t)² 在导通期间尖峰）过估计 `<PWR>/<√PWR>²` 倍。

**修复**：闪烁噪声是调制稳态源 `i(t) = m(t)·n(t)`，调制幅值 `m(t) = √PWR(t)`。现在对 `√PWR` 做 FFT 得到调制谐波向量 `M_{-2K..2K}`，在 `pnoise_fold_psd` 中按 `∑_a |∑_r Z_r M_{r-a}|² / ν_a` 构建 cyclostationary 折叠。Thermal（白噪声）路径不变 —— FFT(PWR) 对它仍正确。

**影响**：SC 低通 IRN 从 20.6 → 4.46 µVrms；AFE amp/chopper 不变（恒偏置）。`pnoise_solver.py` 和 `numba_kernels.py` 两条路径同步修改。

关键经验：
- PMOS source/drain 方向必须正确（source 在更高电位端）
- `dc_guesses` 被 AC 分析绕过（波形 key 的 vsource 在 AC 中 E=0），需靠 tstab 周期自然充电
- `fd_input_step` 默认 1e-4 是正确的（大值引入非线性），SC 类电路不需修改
- PAC 在 ω→0 时病态（τ/T 大→ phi≈I→ 边界值矩阵奇异），f_min≥1Hz 回避
- **不要 `abs(Idc)`** — 带符号电流是物理刚需，`abs` 会在反向偏置器件上制造反恢复泵
- **不要 rail-clip runaway** — 剪辑会在边界锻造虚假固定点，让检测失效；让真实轨迹跑飞才能发现 runaway
- **强调制 flicker 要用 √PWR 建模** — FFT(PWR) + separable weight 是错误的 cyclostationary 折叠

### 剩余任务

- [x] ~~SC 低通本地 bug 修复~~ ✅ 已完成（2026-06-22）— A1 signed current + A3 LM Newton + A4 flicker modulation 三个修复，PSS residual 10× 改善，PAC BW 从 Δ=−28% 提升到 Δ=+8%，PNoise IRN 从 20.6→4.46 µVrms
- [x] ~~受控源 (VCVS/CCCS/CCVS) transient + compiled_topology 路径补齐~~ ✅ 已完成（2026-06-22）— 所有分析全覆盖
- [x] ~~编写 SC 低通的 Cadence 对标~~ ✅ **已完成（2026-06-22；2026-06-28 切换 adaptive 默认）** — Spectre 24.1.0.078 跑了相同 SC 低通的 PSS/PAC/PNoise，参考数据（`pss.td.pss`/`pac.0.pac`/`pnoise.pnoise` + `metadata.json`）入库 `calibration/sc_lpf/`。SC-LPF calibration 当前默认使用 `gear2 + adaptive + cap_mode="average"`，本地 PASS：PAC 增益 −0.32% / BW +1.07% / 输出噪声 +2.82% vs Cadence。回归测试 `tests/test_calibration.py::test_calibration_sc_lpf_matches_cadence`。
- [ ] **扩展对标覆盖** — 更多 switch 尺寸、更多 f_chop、三 corner × 多频率组合
- [x] ~~评估 gear2 vs BE 对周期分析精度的影响~~ ✅ **已评估并修复（2026-06-22）** — gear2-vs-BE 全 case 扫描发现一处 silent landmine:JSON dispatch 默认 gear2，刚性 τ≫T 开关电容（SC-LPF）的 PAC 旧版退回 x0-敏感的 FD shooting → gear2 增益 **24×**（vs BE/Cadence ~1）。根因:解析伴随 PAC 对 true-MNA vsource drive 会 bail。已修(把 vsource 小信号驱动耦合进 bordered HB 支路约束行)→ PAC 现**与积分阶数无关**（gear2==BE==1.006，且比旧 FD-BE 更准）。chopper 逐字节不变。守卫 `test_sc_lpf_pac_is_integration_method_independent`
- [x] ~~给 dispatch 的 `_PSS_KWARGS` 加 `integration_method`~~ ✅ 已完成（2026-06-22）— `_PSS_KWARGS`/`_TRANSIENT_KWARGS` 均加上，`analyses.pss/transient.integration_method` 可选 gear2/be（PAC/PNoise 经共享 PSS 轨道继承）；默认不变（pss gear2、transient be）。回归 `test_dispatch_forwards_integration_method`

---

## 3. 器件模型接口抽象 ✅ 已完成（2026-06-21）

### 交付内容

- **`core/device_model.py`**（新增，~260 行）
  - `TransistorModel` ABC：`get_Idc`、`get_op`、`get_capacitances`、`get_capacitance_charges_from_op`、
    `get_capacitance_branch_terms_from_op`、`get_noise_psd`、`get_numba_params` 7 个抽象方法；
    `get_ss_params` 提供有限差分默认实现，子类可覆盖；`g_area`、`estimate_channel_charge` 提供默认值
  - `NumbaParams` frozen dataclass：16 个标量参数，瞬态求解器一次提取，循环中不再触碰模型对象
  - `register_model()` / `create_device()` 工厂 + 注册表
- **`PMOS_TFT`** 直接继承 `TransistorModel`
  - 新增 `get_numba_params()` → `NumbaParams(...)`
  - 新增 `get_ss_params()` 覆盖（含 numba 优化路径，从 `ac_solver.py` 移入）
  - 新增 `get_capacitance_charges_from_op()`、`get_capacitance_branch_terms_from_op()` 公开接口方法
  - 模块末尾 `register_model("pmos_tft", PMOS_TFT)`
- **8 个求解器文件**全部改用 `create_device("pmos_tft", ...)` 工厂创建，不再直接 `import PMOS_TFT`
  - `ac_solver`、`noise_solver`、`transient_solver`、`pss_solver`、`pac_solver`、
    `pnoise_solver`、`chopper`、`explore`
- **transient solver** 中 15 个裸属性提取改为 `dev.get_numba_params().field`
- **`core/__init__.py`** 导出 `TransistorModel`、`NumbaParams`、`create_device`、`register_model`
- 测试：91 passed, 0 failed — 全部数值结果与原实现逐位一致

### PDK + 极性分层（2026-06-22）

注册表现在是 **PDK + 极性感知**的:一个 *PDK*（工艺）把极性（pmos/nmos）映射到各自的紧凑模型类。
每个 `(pdk, polarity)` 以结构化键 `"<pdk>.<polarity>"`（如 `"at4000tg.pmos"`）注册进扁平模型表，
`create_device` 直接解析。当前 AT4000TG（`PDK/veriloga.va` 的 `pmos_TFT`）是唯一、也是默认 PDK；
`"pmos_tft"` 保留为向后兼容别名。新增工艺/极性只需一次 `register_pdk`，**不改任何 solver**:

```python
class NMOS_TFT(TransistorModel):
    def get_Idc(self, Vs, Vd, Vg): ...        # 实现其余抽象方法
register_pdk("myproc", {"pmos": MyPMOS, "nmos": NMOS_TFT})

create_device("myproc.nmos", W=100, L=10)             # 结构化键
create_transistor("nmos", pdk="myproc", W=100, L=10)  # 便捷创建
```

通用元件（电阻/电容/理想 V/I 源/受控源）是与工艺无关的拓扑原语，**不进**模型注册表 —— 新工艺零改动
复用全部源原语。8 个求解器不再硬编码 `"pmos_tft"`，改为 `create_device(get_default_model_type(), …)`，
默认极性/工艺由注册表单点决定。`tests/test_device_model.py`（6 例）守卫可区分性 + 默认路径逐字节不变。

### 尚未完成（低优先级，按需启动）

- [ ] **JSON `"model"` 字段**（消费层）：让电路文件按器件声明 `pdk.polarity`，loader/dispatch 把每器件
  模型线程到 solver 调用点。注册/区分机制已就绪;当前所有器件走注册表默认 `at4000tg.pmos`，单模型场景够用
- [ ] 实现一个最小 NMOS 模型作为验证用例（注册机制已就绪，缺的是物理）
- [ ] JSON schema 更新（随 `"model"` 字段一起）

---

## 4. 扩展元件类型 ✅ 已完成（2026-06-22）

### 现状

已支持：电阻、电容、理想直流电流源、VCCS（压控电流源）、理想时变电流源（用于 charge injection）、理想电压源、VCVS（压控电压源）、CCCS（流控电流源）、CCVS（流控电压源）。**全部 7 种非晶体管元件的 DC/AC/Noise/Transient/PSS/PAC/PNoise 路径已打通。**

### 已完成

- [x] VCCS — AC/DC/Noise/Transient 全覆盖
- [x] Ideal voltage source — 真·MNA，全分析覆盖
- [x] **VCVS / CCCS / CCVS** ✅ **全分析路径补齐（2026-06-22）**
  - 2026-06-21：DC 残差 + AC/Noise MNA stamp + JSON 加载器
  - 2026-06-22 补齐：
    - `compiled_topology.py`：`VcvsPlan` / `CccsPlan` / `CcvsPlan` dataclass + DC residual + AC token 方法
    - `transient_solver.py`：纯 Python n_aug 路径 VCVS/CCCS/CCVS 的 RHS 和 Jacobian
    - `ac_mna.py`：`_stamp_vcvs` / `_stamp_cccs` / `_stamp_ccvs` 原语
    - PSS/PAC/PNoise：`_branch_incidence` 扩展为 vsource+VCVS+CCVS 并集
  - JSON schema + `tests/test_controlled_sources.py`（22 例）+ `examples/vcvs_amplifier.json`
  - 级联支持：CCCS/CCVS 可控制任何已有支路电流源
- [ ] **互感和耦合电感** — 较低优先级，视需求

### 依赖

- ✅ 电压源已完成（真·MNA），为 VCVS/CCVS 提供了支路电流基础
- ✅ VCCS 已完成，为 CCCS 提供了电流注入模式
- ✅ VCVS / CCCS / CCVS 引用已有支路电流索引（`vsource_index`），无需额外基础设施

---

## 5. 深化 transient / 周期小信号性能 🟢 中优先级

### 现状

- Numba 内核已覆盖 PMOS 电流、内部节点 Newton、transient Newton 内循环、BE grid solver、gear2 grid solver
- gear2/BDF2 已上线，PAC 精度从 BE 的 −2.5% 提升到 <1%
- 裸 `transient()` 默认 BE；默认 BE chopper transient 已从“Numba + Python tail + 1 次 LS”修成全 Numba（0 LS）
- `integration_method="gear2"` 在请求 `max_step` / `flat_max_step` / `max_retry_subdivisions` 时也走 Numba grid，按 accepted substep 维护 rolling 两步 BDF2 历史；Python solve_chunk 仅作为 Numba 拒绝 robust step 时的兜底
- 当前 chopper 周期全流程旧瓶颈是通用 HB PAC frequency solve（UI 尺寸，f_chop=225Hz，switch=5000/30，edge=20us）：显式 `PSS+PAC(HB)+PNoise`（`time_domain=False`）61 点约 25.6s，其中 PAC≈24.7s；121 点约 48.9s，其中 PAC≈47.6s。默认 chopper time-domain PAC 已加入 PMOS `gate1` 内部小信号状态并启用 gate1 Numba 转换装配，修复了 slow −1.89% 误差；同一 PSS 轨道上 61 点 PAC 约 1.4s，并已作为 `pmos_chopper_pac` 默认路径。PSS 轨道生成对 chopper 默认使用 `cap_mode="average"`，用于匹配 Cadence commutation feedthrough；通用 transient/PSS 仍保持 charge 默认。

### 后续方向

- [x] ~~**compiled step plan / raw gear2 subdivision**~~ ✅ 已完成：raw gear2 maxstep/retry/subdivision 已移入 Numba grid
- [ ] **小矩阵特化**：对 6×6（chopper）级别矩阵做专用 dense solve，跳过通用 LU 开销
- [x] ~~**chopper transient 深度编译化（默认 BE 热路径）**~~ ✅ 已完成：默认 BE UI chopper transient 全 Numba，`nfail=0`、`least_squares_calls=0`
- [x] ~~**PAC time-domain Floquet 加速入口**~~ ✅ 已完成：`pac_solve(..., time_domain=True)` / `pmos_chopper_pac(...)` 默认 time-domain / JSON `analyses.pac.time_domain` 均可启用，HB 保留为通用兜底。
- [x] ~~**修 time-domain PAC slow-corner 误差**~~ ✅ 已完成：根因是 PMOS_TFT 周期转换中把内部 `gate1` 节点塌缩成静态端口 `{gm,gds,Cgs,Cgd}`，丢失 Cadence PAC 保留的 `R_cap/R_cap2/Cgs/Cgd` 内部小信号状态。现已在 PAC 周期线性化中为每个 PMOS 扩展 `gate1` state，D3 slow 默认 PAC baseband/200Hz 进入 <1% 门限。
- [x] ~~**time-domain PAC 默认化评估**~~ ✅ 已完成：`pmos_chopper_pac` 默认切到 time-domain；`analytic=False` 仍强制走原有限差分 shooting；显式 `time_domain=False` 可跑 HB 对照。
- [x] ~~**PNoise gate1 扩维**~~ ✅ 已完成：PNoise HB 复用 PAC 的 PMOS `gate1` 扩维线性化，噪声源仍按 drain/source 电流源注入，但传播矩阵保留内部 `gate1` 状态；slow PNoise guard 通过，并新增 `pnoise_internal_gate1_states` 回归断言。
- [x] ~~**monodromy 缩放/平衡**~~ ✅ 已完成：PSS LM normal equation 改为 lazy 计算并按有限分量缩放，`mu=0` Newton 步不再提前形成 `J.T@J`；time-domain PAC gear2 companion 连乘加入增长检测，超过安全尺度时切到分块 multiple-shooting 周期边界求解，消除了 overflow/invalid warning。
- [x] ~~**PAC gate1 转换 Numba 化**~~ ✅ 已完成：新增 `pac_linearize_orbit_gate1_numba`，全 PMOS gate1 拓扑的 Verilog-A `C(V)*ddt(V)` 周期转换装配走编译内核；测试守卫 numba 与 Python 装配在 D3 slow chopper PAC 上 <0.05%。
- [x] ~~**PNoise TD adjoint 去截断**~~ ✅ 已完成：chopper PNoise 默认 `time_domain=True`，用稀疏周期伴随 BVP 替代 K 截断 HB adjoint；slow chopper IRN 从 HB-K32 的 +1.81% 降到 +0.02%，并新增 K16/K64 截断无关慢测守卫。
- [ ] **HB PAC frequency solve 优化**：作为通用兜底路径继续保留；若需要 HB（bordered/vsource 驱动或 time-domain 不适用），再 profile/优化每频率 HB 求解、factorization 复用或批量线性解。
- [ ] **batch transient / MC 并行化**：多个瞬态仿真并行（thread-level 或 process-level）
- [x] ~~**gear2 grid subdivision/retry 硬化**~~ ✅ 已完成：裸 transient 的 `integration_method="gear2"`
  在 `max_retry_subdivisions` / `max_step` 请求下进入 Numba gear2 grid，维护 rolling 两步历史并做固定二分 retry；
  stiff chopper 边沿不再触发 BE clean rerun，且不再走 Python gear2 retry 热路径。
- [x] ~~**LTE adaptive gear2 transient/PSS**~~ ✅ 已完成：`transient/pss_solve/pmos_chopper_pss`
  增加 opt-in `adaptive=True`，使用 step-doubling LTE 控制非均匀 accepted grid；PSS
  接近收敛后冻结 grid 再生成最终 orbit/monodromy。`n_aug == n` 的 adaptive gear2
  有 Numba kernel，含 vsource branch 的拓扑走 Python adaptive fallback。SC-LPF
  calibration 默认已切到 `gear2 + adaptive + cap_mode="average"`；adaptive driver
  在 clock slope discontinuity 后重启两步历史，避免 BDF2 history 跨边沿污染。
  当前 `calibration/sc_lpf` PASS：PAC gain −0.32%、BW +1.07%、PNoise output +2.82%
  vs Spectre，并用 `pnoise_n_period_samples=512` / `pnoise_max_sideband=20` 守住噪声采样。

### 原则

不牺牲精度。任何近似 Jacobian 都要用波形回归验证。

---

## 6. 扩展优化搜索策略 🟢 中优先级

### 现状

`core/explore.py` 已提供完整的搜索流水线：LHS/随机采样 → 逐候选 AC-first 评估 → 约束过滤 → Pareto 选择 → CSV/JSONL 导出。

### 后续方向

- [ ] **贝叶斯优化** — 对昂贵的目标函数（如含 transient 或 PNoise 的约束）用 GP 代理模型引导采样，减少评估次数
- [ ] **进化算法** — NSGA-II 或类似多目标进化算法，适合非凸 Pareto 前沿
- [ ] **与 Cadence 验证闭环集成** — explore 产出的 Pareto 最优候选自动送入校准对比脚本（依赖第 1 步）

---

## 7. ML Surrogate 建模路线 🟡 中高优先级

### 定位

当前代码已经适合做 **Machine Learning based Surrogate Modeling for Fast Amplifier Design Optimization** 的底座，
但还不是 surrogate 本身。现有 solver/explore/calibration 应承担三个角色：

- **Teacher simulator**：用已对齐 Cadence 的本地 DC/AC/Noise/Transient/PSS/PAC/PNoise 生成标签
- **Design-space engine**：用 JSON/explore 定义固定拓扑的尺寸、bias、corner、load、clock 参数空间
- **Validation oracle**：用 `calibration/` 和少量 Spectre 回归检查 surrogate 没有学到错误物理趋势

Surrogate 层的目标不是替代 sign-off SPICE，而是在固定拓扑的 refinement 阶段快速筛选和优化候选点。

### 后续方向

- [ ] **Dataset builder**：新增 `core/dataset.py` 或 `tools/build_dataset.py`，从 JSON explore 配置生成训练集。
  输入包含 `W/L/NF`、bias、load、corner、clock 参数；输出包含 gain、BW、IRN、power、area、DC 成功标志、
  以及可选 transient/PSS 波形。导出格式优先用 Parquet/NPZ，保留 JSONL/CSV 便于调试。
- [ ] **标签规格标准化**：定义稳定的 label schema，例如
  `gain_dB`、`bw_Hz`、`phase_margin_deg`（待实现）、`slew_rate`（待实现）、
  `settling_time`（待实现）、`irn_uV`、`power_uW`、`area`、`dc_converged`、`pss_converged`。
  失败样本不能简单丢弃，应作为分类标签或约束边界样本保留。
- [ ] **Metric surrogate baseline**：先实现低风险的指标模型，而不是一开始做 Neural ODE。
  基线模型建议：standardized MLP / gradient boosted trees / Gaussian Process。输出均值和误差估计，
  先覆盖 AC/noise/power/area，再扩展到 PSS/PAC/PNoise。
- [ ] **Transient waveform surrogate**：在 metric surrogate 稳定后，再做波形模型。
  可选路线：TCN/RNN/operator-style 模型，或 Neural ODE。输入是设计参数和激励参数，输出固定时间网格波形；
  误差指标包括波形 RMSE、峰值/均值误差、settling/slew 派生指标误差。
- [ ] **Differentiable optimization loop**：对 PyTorch/JAX surrogate 接入梯度优化。
  目标函数支持多目标和约束惩罚，例如最大化 gain/BW、最小化 noise/power/area，同时约束 DC/PSS 收敛和输出共模范围。
- [ ] **Active learning 闭环**：surrogate 只负责提出候选；不确定度高、靠近约束边界、或 Pareto 前沿附近的点回灌本地 solver/Cadence。
  这样避免单次离线训练覆盖不足导致错误外推。
- [ ] **泛化边界管理**：记录训练数据的参数范围、corner、拓扑 hash、PDK 版本、solver commit。
  Surrogate 只能声明在固定拓扑/固定 PDK/已覆盖参数域内有效；换拓扑必须重新生成数据或迁移学习。
- [ ] **面向演示的 notebook/CLI**：提供一条完整命令链：
  `build-dataset -> train-surrogate -> validate -> optimize -> verify-with-solver`。
  面试/演示时重点展示 surrogate 比本地 solver/SPICE 快多少，以及最终候选回到 solver/Cadence 后误差多少。

### 精度目标

- AC/noise/power 指标 surrogate：验证集 median error <1%，P95 <5%
- Transient 派生指标：settling/slew/peak-to-peak P95 <5%，波形 RMSE 用输出满量程归一化 <1%
- 最终优化候选：必须回到本地 solver 校验；关键候选再用 Cadence 抽样校验
- 对训练域外样本必须拒绝或降级为 solver 评估，不能静默外推

### 与现有代码的关系

- `core/explore.py` 提供采样、约束、Pareto 基础，可复用为数据生成入口
- `core/analysis_dispatch.py` 提供 JSON 分析执行入口，可复用为 dataset job runner
- `benchmarks/bench_sweep.py` 可扩展成 dataset throughput benchmark
- `calibration/` 和 `core/calibration.py` 作为 surrogate teacher 的可信度守卫
- 现有 Numba/compiled topology 优化继续有价值，因为训练数据的瓶颈首先是标签生成速度

---

## 8. 编译后端路线评估 🟢 低优先级

### 现状

Python + Numba 对百级候选扫描已足够（200 候选 AC+noise ~0.5s）。当前没有性能瓶颈。

### 触发条件

下列场景出现时启动评估：

- 千级/万级 sweep 需求（explore 规模扩大 10–100×）
- 大规模 MC（>1000 样本 × 多 corner）
- 瞬态成为批量评估主要瓶颈且 Numba 优化已到天花板

### 备选方案

- **Rust**（PyO3）：单器件模型批量评估、transient Newton/Jacobian 热路径。Python 层保留 JSON 配置、拓扑编排
- **Cython**：更渐进式的迁移路径，可逐函数替换
- **JAX**：如果模型可微，自动获得梯度（利于基于梯度的优化），但需要重写模型

---

## 低优先级 / 技术债

| 项目                      | 说明                                                                                           |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| ~~gear2 subdivision/retry~~ ✅ 已修复 | 裸 transient 的 gear2 maxstep/retry/subdivision 已移入 Numba grid；Python solve_chunk 只保留为异常兜底 |
| ~~`_rail_clip` 在 PSS stabilization~~ ✅ 已修复 | 旧的 line-search damping + rail-clip 在 stiff 电路上锻造虚假的零残差固定点，已被 LM + physical bounds + best-physical 回退替代（2026-06-22） |
| ~~闪烁噪声 cyclostationary 折叠~~ ✅ 已修复 | 从 FFT(PWR) 改为 FFT(√PWR) 调制幅值向量构建折叠矩阵（2026-06-22） |
| PNoise HB solver 扩展     | 已有 dense/sparse/iterative 三条路径。若 HB 规模继续增长（数十+谐波），再评估 matrix-free matvec 或低秩边带截断             |
| `results/` 目录           | 含历史 benchmark 和 explore 输出，已在 .gitignore 中。考虑移到独立数据仓库或加 README                               |

---

## 不做的事项

| 事项                      | 原因                                                 |
| ----------------------- | -------------------------------------------------- |
| 全局切换到 Verilog-A/average 电容模式 | 不做。通用 stiff 电路仍以 charge Q-stamp 为默认；只在 PMOS chopper PSS 轨道生成中用 `cap_mode="average"` 匹配 Cadence feedthrough。PAC/PNoise conversion 另行使用 Spectre PAC 折叠的 `C(V)*ddt(V)` 小信号算子。 |
| 大规模 CI/CD               | 项目目前为研究型单人开发，手动 pytest + benchmark 足够。有协作者再加入      |
| GPU 加速                  | PMOS 模型和 MNA 矩阵规模（≤20×20）对 GPU 无优势。如未来处理大规模阵列电路再评估 |
| Sign-off 级仿真器认证         | 项目定位是设计探索工具，不做 Spectre 替代品                         |

---

## 执行建议

**第 1/3/4 步已于 2026-06-21 全部完成。** 第 2 步（扩展验证覆盖）SC 低通已于 2026-06-22 完整闭环，
并在 2026-06-28 把 calibration 默认切到 `gear2 + adaptive + cap_mode="average"`:
三个 bug 修复（PSS residual 10× 改善、PAC BW、flicker 过估计修正）+ Cadence 对标入库
`calibration/sc_lpf/`，SC-LPF 当前 PASS（PAC 增益 −0.32% / BW +1.07% /
输出噪声 +2.82% vs Spectre 24.1）。周期分析验证现已形成完整闭环。

当前最有价值的投入：

- **Surrogate dataset builder** — 先把现有 JSON/explore/analysis dispatch 串成稳定的数据生成管线，
  批量产出 `(design parameters -> metrics/waveforms)` 数据集，并记录 topology hash、corner、solver commit、
  收敛状态和失败样本；这是后续 ML surrogate、主动学习和梯度优化的前置条件。
- **扩展对标覆盖** — 更多 switch 尺寸（5000/30）、更多 f_chop（100/300/1k Hz）、三 corner × 多频率组合，
  趁校准基础设施（`core/cadence_netlist.py` + `core/calibration.py` + `calibration/`）正热扩大回归网。
- **gear2 vs BE 已评估并修复（2026-06-22）** — 全 case 扫描发现并修复了 SC-LPF PAC 的 gear2 silent
  landmine（解析伴随 PAC 现支持 vsource drive、与积分阶数无关，gear2==BE）；周期分析精度 chopper/sc_lpf 均 calibration 内 <2%。

测试套件和校准回归是持续维护项，每次改动确认无回归。
