# 后续开发计划

[English README](README.md) | [中文说明](README_zh.md) | [核心求解器概览](core_overview_zh.md)

## 当前状态（2026-07-05）

项目是一个成熟的本地模拟电路仿真与设计探索框架，首个应用场景为 AT4000TG PMOS-OTFT ECG AFE，
已对 Cadence Spectre 24.1 完成校准；现已扩展支持两个硅 CMOS 工艺——**SKY130**（130nm，OSDI/BSIM4，
DC/AC/noise/瞬态）与 **FreePDK45**（45nm，ngspice-C 求值器，DC/AC/noise），且 ML surrogate 全链路
（dataset→surrogate→optimize→PVT）在两个工艺上都跑通了完整的全差分 OTA 设计案例。核心能力全部落地，
正在从"功能开发"转向"生态完善"阶段。

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
| **ML Surrogate**  | `dataset`→`surrogate`（GBT，可选 sklearn）/`surrogate_torch`（可微，torch/MPS）→`optimize`（筛选→Pareto→solver 校验）全链路闭环；`--filter` 训练感兴趣区域；`<Res>.R` 结构化设计轴（optimize 校验阶段也生效） | ✅ 完成     |
| **硅 PDK（SKY130）** | OpenVAF 编译 BSIM4 Verilog-A → `.osdi`，ctypes 原生宿主（`core/osdi_host.py`）+ `TransistorModel` 适配器（`core/osdi_device.py`）+ SKY130 参数卡（ngspice 解析，`core/sky130_model.py`）；DC/AC/noise/瞬态全走同一求解器引擎（**model==oracle** 校验）；配置 `models` 块绑定，`--corner tt/ss/ff/sf/fs`；全差分望远镜 OTA 设计案例（`docs/sky130_fd_ota_design.md`，63dB/103MHz/60.5µW）已跑通 | ✅ 完成（DC/AC/noise/瞬态；斩波器待做） |
| **硅 PDK（FreePDK45）** | 45nm/1.0V，用户目标工艺；求值器是 **ngspice-C 本身**（非 OSDI VA——卡为 ngspice 内置 BSIM4 调）：批量 `.dc`/`.noise` 表征成缓存 `(Vsb,Vds,Vgs)` 网格再插值（`core/ngspice_char.py` / `core/ngspice_device.py` / `core/freepdk45_model.py`），节点处 exact ngspice-C；支持 `extract_w`（<0.7% 线性 W 缩放）+ `temperature`（PVT）；全差分 OTA 设计案例（`docs/freepdk45_fd_ota_design.md`，58.9dB/119.9MHz/17µW，含整机对 ngspice `.ac` 交叉核对）已跑通 | ✅ 完成（DC/AC/noise；无瞬态/PSS） |
| **工艺角/鲁棒性**     | 全局 corner（typ/slow/fast）、逐器件 mismatch MC、确定性 latch 筛查                                                            | ✅ 成熟     |
| **Numba 加速**    | PMOS 电流、内部节点 Newton、偏置电容、terminal derivative、BE/gear2 transient grid（含 maxstep/retry）、PAC 周期线性化/time-domain PAC、PMOS gate1 PAC 转换线性化、PNoise HB block 组装和噪声折叠 | ✅ 全覆盖    |
| **gear2/BDF2**  | 变步长 BDF2、Numba grid、解析 monodromy、裸 transient retry/subdivision。PAC baseband 三 corner 全部 <1%（BE 时 −2.5%）。PSS/PAC/PNoise 默认 gear2 | ✅ 完成     |
| **CLI**         | `python -m core <circuit.json>` 全分析 dispatch + exploration 模式 + 结果导出 + CLI 参考手册（`docs/cli_reference.md`）             | ✅ 完成     |
| **Demo**        | Flask Web 前端 + REST API（`demo/server.py`）                                                                        | ✅ 可用     |
| **测试**          | 31 个测试文件；最近全量回归 `283 passed, 1 skipped`（硅工具链缺失时的预期 skip）；OTFT Cadence byte-gate 5/5 逐字节稳定                                    | ✅ 覆盖核心路径 |
| **文档**          | 英/中双语：README、core_overview、JSON 格式参考、CLI 参考手册、两个全差分 OTA 设计案例（SKY130 + FreePDK45）             | ✅ 完善     |

### 代码规模

```
core/                     ~19,200 行  (34 个 .py 文件)
tests/                      ~4,900 行  (28 个 test_*.py 文件)
benchmarks/                  ~1,100 行  (5 个 benchmark)
examples/                     ~900 行  (8 个 JSON + 7 个 Python 脚本)
docs/                       ~5,900 行  (14 个 .md 文件)
calibration/                 ~450 行  (5 个 case 目录, 含 PSFASCII 参考文件)
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
7. ~~ML surrogate 建模路线~~ ✅ **v1 全链路已完成** — `dataset` → `surrogate`/`surrogate_torch` → `optimize`（筛选→Pareto→solver 校验），OTFT 与硅（SKY130）均验证；剩余是精度/覆盖面的持续打磨，见 §7
8. ~~硅 CMOS PDK（SKY130）~~ ✅ **DC/AC/noise/瞬态已完成并验证** — OpenVAF 编译 BSIM4 通过 OSDI ctypes 宿主接入现有求解器引擎；硅设计闭环（含跨工艺角复验）已跑通；斩波器（PSS/PAC/PNoise）待做，见 §9
9. **编译后端评估** — Rust/Cython 承担千级/万级 sweep

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
- [x] ~~**LTE adaptive gear2 transient/PSS**~~ ✅ 已完成：`transient/pss_solve`
  增加 opt-in `adaptive=True`，使用 step-doubling LTE 控制非均匀 accepted grid；PSS
  接近收敛后冻结 grid 再生成最终 orbit/monodromy。`n_aug == n` 的 adaptive gear2
  有 Numba kernel，含 vsource branch 的拓扑走 Python adaptive fallback。SC-LPF
  calibration 默认已切到 `gear2 + adaptive + cap_mode="average"`；adaptive driver
  在 clock slope discontinuity 后重启两步历史，避免 BDF2 history 跨边沿污染。
  当前 `calibration/sc_lpf` PASS：PAC gain −0.32%、BW +1.07%、PNoise output +2.82%
  vs Spectre，并用 `pnoise_n_period_samples=512` / `pnoise_max_sideband=20` 守住噪声采样。
  `pmos_chopper_pss(adaptive=True)` 目前明确抛 `ValueError`，继续使用已验证的固定
  edge-refined grid；避免调用方误以为 adaptive 已在 hard-switched chopper 上生效。

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

- [x] ~~**Dataset builder**~~ ✅ **完成** — `core/dataset.py` + `python -m core dataset` 子命令，
  复用 explore 采样/评估但**不做约束/Pareto 过滤**（每个样本都保留，DC 失败样本作为分类/边界标签），
  **总是评估噪声**（每个收敛点带完整标签 `gain_dB/gain_peak_dB/bw_Hz/irn_uV/power_uW/area`），并写 manifest
  记录 schema 版本、solver commit(+dirty)、拓扑 hash、PDK、`models` 绑定、corner、变量范围、counts（供泛化边界判断）。
  产出 `<prefix>.jsonl`（可读）+ `.manifest.json`（provenance）+ `.npz`（稠密 X/Y + 掩码）+ 可选 `.parquet`。
  **schema 1.2** label 组：`ac_noise`（默认）/ `transient`（激励无关波形特征）/ `pss`（周期稳态质量 + 轨道输出）
  / `pac`（基带转换增益 `pac_gain(_dB)` + `pac_bw_Hz`）/ `pnoise`（带内积分输出/等效输入噪声
  `pnoise_out_uV`/`pnoise_irn_uV`）。`pss`/`pac`/`pnoise` 三组走 `run_analysis_suite` 每候选一次调用：
  config `analyses` 块里**已验证的求解设置**（gear2/tstab/残差容差、`time_domain`、drive、band）原样生效，
  PSS 轨道只算一次共享，PNoise 复用 PAC 增益做输入折算；`pac`/`pnoise` 组要求配置带对应 `analyses` 块
  （硬开关电路用错 PAC 设置会收敛到静默错误的增益，绝不代造激励）。硅斩波即用此链产标签
  （`examples/sky130_chopper.json` 自带 explore 块，`--labels pss,pac,pnoise` 直接可跑；
  单候选全链约 2s，见 tests/test_sky130_chopper.py::test_dataset_chopper_labels）。
  **结构化设计轴**：`<Cap>.C` / `<Res>.R`（无源器件值）+ `periodic.frequency`（clock）+ `pvt0`/`pbeta0`（连续 PVT）
  ——逐候选重建电路，`dataset` **和** `optimize` 校验阶段共享同一 `candidate_circuit()` 构造器（此前 `optimize`
  会忽略/崩在结构化变量上，现已修复）。
- [x] ~~**标签规格标准化**~~ ✅ **完成** — `AC_NOISE_LABELS`/`TRANSIENT_LABELS`/`PSS_LABELS`/`PAC_LABELS`/
  `PNOISE_LABELS` 五组固定 schema；DC 失败样本保留 `dc_converged=False` + null 标签，不丢弃。相位裕度/settling
  暂不属于 PSS 组（前者是 AC 环路指标，后者是阶跃响应指标），维持独立议题。
- [x] ~~**Metric surrogate baseline**~~ ✅ **完成** — `core/surrogate.py`：`HistGradientBoostingRegressor`
  （可选 sklearn 依赖），每标签独立回归 + 自动 log-space 拟合（跨数量级的标签，如 IRN）+ `filter_rows`/
  `--filter label:lo:hi` 训练感兴趣区域（剔除甩轨/collapse 的极端点，避免污染回归又不必丢弃——被约束
  自然筛掉的样本，精细拟合是浪费容量）。`score()` 报告 median/P95 相对误差 + R²。
- [ ] **Transient waveform surrogate**：指标 surrogate 稳定后的下一步，尚未做。
  可选路线：TCN/RNN/operator-style 模型，或 Neural ODE。输入是设计参数和激励参数，输出固定时间网格波形；
  误差指标包括波形 RMSE、峰值/均值误差、settling/slew 派生指标误差。
- [x] ~~**Differentiable optimization loop**~~ ✅ **完成** — `core/surrogate_torch.py`（torch/MPS，可微
  MLP surrogate）+ `python -m core.surrogate_torch optimize --verify`：梯度上升在约束惩罚下多目标优化，
  `--verify` 收尾接回本地 solver。GBT 路线（`core/optimize.py`）用筛选-验证代替梯度：大池子交给
  surrogate 筛（µs/candidate），Pareto 前沿的候选送回 solver 校验，两条路线互补而非取代。
- [ ] **Active learning 闭环**：surrogate 只负责提出候选；不确定度高、靠近约束边界、或 Pareto 前沿附近的点回灌本地 solver/Cadence。
  这样避免单次离线训练覆盖不足导致错误外推。目前用「screen-and-verify」代替：不确定度不显式建模，而是让
  solver-verify 兜底所有最终候选（见下方硅闭环的诚实教训）。
- [x] ~~**泛化边界管理**~~ ✅ **完成** — manifest 记录 schema 版本、solver commit(+dirty)、拓扑 hash、PDK、
  `models` 绑定、corner、采样 seed/method、变量范围、counts；`surrogate` 训练 metadata 记 train_npz/filter/
  solver_commit/corner/topology_hash。跨拓扑/跨 PDK 不自动迁移，换拓扑需重新生成数据。
- [x] ~~**面向演示的完整命令链**~~ ✅ **完成** — `python -m core dataset` → `python -m core.surrogate train`
  → `python -m core.optimize`（筛选→Pareto→solver 校验），一条命令链跑通；`--corner` 支持跨工艺角复验。
  在 OTFT AFE 和硅 SKY130 OTA 上都验证过，见 §9。

**一个训练时才浮现的诚实教训**（硅闭环验证时发现，具有普遍性）：**没有一个 surrogate 能同时是"工作区
精确模型"又是"甩轨/失败区感知的好筛选器"**——在感兴趣区域（`--filter`）训练精度最高，但对失败区域一无所知，
筛选阶段会把它当作全可行区（screen 效果差）；不筛选训练则精度被极端点拖累。**screen-and-verify 架构本来
就是为容忍这个矛盾设计的**：筛选阶段用全量数据训练的 surrogate（哪怕对工作区精度一般）粗筛,solver-verify
兜底保证最终候选的可行性判断永远来自真实求解器，不依赖 surrogate 的失败区判断力。

### 精度目标

- AC/noise/power 指标 surrogate：验证集 median error <1%，P95 <5%
- Transient 派生指标：settling/slew/peak-to-peak P95 <5%，波形 RMSE 用输出满量程归一化 <1%
- 最终优化候选：必须回到本地 solver 校验；关键候选再用 Cadence 抽样校验
- 对训练域外样本必须拒绝或降级为 solver 评估，不能静默外推

**已验证达标的具体例子**（硅 SKY130 5T OTA，工作区 `--filter gain_dB:0:60` 训练，见 §9）：held-out
median error — gain 0.11%、power 0.58%、bw 1.35%、irn 0.92%、area 1.12%，全部 <1.4%，达到上述目标。

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

## 9. 硅 CMOS PDK（SKY130） ✅ DC/AC/noise/瞬态已完成（2026-07-02 起）

### 定位

项目原来只焊死一个有机 PMOS-only OTFT 工艺（AT4000TG）。为了让优化器/surrogate 论点在**正常硅
CMOS** 上也成立，接入 **SKY130**（SkyWater 130nm，Apache-2.0，开源、真实硅、模拟模型+corner 齐全）。
其晶体管用行业标准 **BSIM4** 模型。

### 方案决策（否决路径见记忆 `silicon-pdk-openvaf`）

- 否决 **per-PDK LUT**（每加一个 PDK 都要建一套表，难维护）
- 否决 **手写转录 BSIM4**（~15000 行 = 重复造轮子）
- 否决 **移植成 Rust**（同样是重复造轮子，且失去和 ngspice 交叉验证的"同一个 .osdi"优势）
- 采用：**复用行业标准模型** —— OpenVAF 把标准 BSIM4 Verilog-A 编译成原生 `.osdi`，通过现有
  `device_model.TransistorModel` ABC 在同一个 solver 引擎里调用。一个编译好的模型服务所有 bulk-BSIM4
  工艺（SKY130/GF180/…）；新增 PDK = 读一份参数卡，不改 solver 代码。Oracle = 本地 ngspice（同一个
  `bsim4.osdi` 既是我们的模型又是 oracle 的模型 → **model==oracle**，正确性与 BSIM4.5-vs-4.8 版本差异无关）。

### 已交付并验证

- **`core/osdi_host.py`**：OSDI 0.4 ABI 的 ctypes 原生宿主 —— 加载 `.osdi`、内部节点 Newton、Schur
  补 Jacobian，返回 Id/gm/gds/gmb/电容/噪声。vs ngspice 校验：Id 精确、gm/gds <0.5%、Cgg 精确、噪声
  1/f 正确。
- **`core/osdi_device.py`**：`OsdiDevice(TransistorModel)` 适配器，把编译模型接进现有 ABC。
- **`core/sky130_model.py`**：SKY130 的 63-bin BSIM4 子电路参数卡**让 ngspice 解析**（`showmod` 拿到
  完全展开的 731 个参数），缓存到 `data/pdk/sky130/*.json`；注册为 `"sky130"` PDK（`default=False`，
  纯增量，AT4000TG 数值不受影响）。`extract_w` 参数：在参考 W 处解析一次卡片，实际 W 交给 bsim4va
  缩放 → ~2ms/eval，扫 W 不必逐点跑 ngspice。
- **求解器接入**：`ac_solve`/`noise_analysis`/`build_devices` 新增 `model_types`/`device_kwargs`（None
  时行为不变，byte-safe）；`TransistorModel.kcl_sign` 修正 NMOS（source-low）与 PMOS/OTFT（source-high）
  的 DC KCL 符号差异。互补 5T OTA（NMOS 差分对+PMOS 镜像负载+NMOS 尾电流）验证增益 == `gm1*(ro2‖ro4)`
  到 0.3%。
- **`core/osdi_transient.py`**：纯 Python 后向欧拉瞬态（`.osdi` 不能进 numba 紧循环），验证收敛到 DC
  工作点 0.009%、RC τ 匹配 2.3%。
- **硅设计闭环**：`models` 配置块把某个器件绑到硅模型（其余器件仍走默认 PDK，纯增量）；`explore`/
  `dataset`/`optimize` 全线接通 `model_types`/`device_kwargs`；`apply_silicon_corner()` 把 SKY130 的
  离散 corner（`tt/ss/ff/sf/fs`）路由到硅器件卡片，与 OTFT 的连续 PVT shift（`pvt0`/`pbeta0`）分开处理。
  `<Res>.R` 结构化设计轴补齐（此前只有 `<Cap>.C`）。

### 验证结果（`examples/sky130_5t_ota.json`，互补 5T OTA，见 CLI 参考 §1.8）

数据集 n=400（400/400 DC 收敛，~80s）；surrogate 在工作区（`--filter gain_dB:0:60`，剔除掉约 30% 甩轨
角）held-out median error：gain 0.11%、power 0.58%、bw 1.35%、irn 0.92%、area 1.12%。`optimize` 筛选
50000 候选 ~0.9s（**~6000× vs solver**），solver 校验 **9/10 可行**（真实 OTA：~35dB / ~135µW / ~1.1MHz）；
`--corner ss` 跨慢角复验，同一批候选 **9/10 仍可行**（慢角尾流 106→75µA、bw 1.58→1.07MHz，物理正确）。

### 一个电路拓扑上的教训

单级放大器负载"标配"是**有源电流源负载（定 DC 工作点 + ro）+ 电容负载（定带宽）**——纯电阻负载在
片上不现实（占面积、匹配差），而且开环电阻负载 CS 的 `vout=I·RL` 没有钳位机制，扫描尺寸/偏置会大范围
甩轨（这在早期尝试 `sky130_cs_amp.json` 时踩到，后已删除该示例）。现在唯一的硅设计示例
（`sky130_5t_ota.json`）用的是正统的电流镜有源负载 + `load_caps` 电容负载，DC 失败样本干净地返回
`None`/落在约束外，不会污染 surrogate 回归。

### 剩余工作

- ~~硅瞬态（全保真）~~ ✅ **已完成（2026-07-04，Phase B）**："`.osdi` 不能进 numba 紧循环"的旧假设
  已被实验推翻（numba nopython 直呼 OSDI 函数指针，0.45 µs/eval，见
  `docs/environment_performance.md`）。`core/osdi_transient.py::transient_osdi` +
  `numba_kernels._osdi_transient_grid_impl`：**全局折叠**形式（未知量 = 外部节点 + 各器件内部
  节点，一个电路级 Newton，内部节点电荷动力学精确积分，无准静态近似），BE + 变步长 BDF2，经标准
  `transient(model_types=...)` 入口自动路由，OTFT 核零接触（byte-gate 5/5 不变）。四重验证
  （`tests/test_osdi_transient.py`）：DC-hold（漂移 <1e-9 V）、与独立纯 Python 参考逐点一致
  （<0.1 mV）、解析 τ=(RL‖ro)·CL 0.4% 内、**ngspice 同卡同 `.osdi` `.tran` oracle**
  （终值差 1 µV、τ63 一致）。性能：5T OTA 2000 步 gear2 **24.8 ms**（12.4 µs/步）。
- ~~硅 PSS + 斩波 testbench~~ ✅ **已完成（2026-07-04）**：`pss_solve`/`analysis_dispatch`/
  `_marshal_transient` 全部透传 `model_types`/`device_kwargs`——PSS 打靶编排原样复用（经
  `transient()` 路由自动走硅核；解析单值矩阵为 OTFT 专属，硅电路自动降级 FD 打靶并记录
  diagnostics）。**硅斩波 testbench**（`examples/sky130_chopper.json`）：nmos 输入斩波开关 →
  nmos 差分对 + pmos 二极管负载 → nmos 输出斩波开关 → 保持电容，250 kHz 方波时钟经波形
  vsource 驱动。三方验证（`tests/test_sky130_chopper.py`）：本地瞬态增益 −1.766、PSS 轨道
  增益 −1.766（收敛残差 1.3e-9，稳定化 2 周期即达不动点，0.18 s）、**ngspice 同卡同
  `.osdi` `.tran` 增益 −1.766**（解调差分输出差 ~2 µV）；PSS 轨道与稳定瞬态末周期逐点
  一致（<5e-5 V）。
- ~~硅 PAC 扫频 / PNoise~~ ✅ **已完成（2026-07-04）**：`_assemble_pac_linearization_python`
  的器件循环按 `isinstance(OsdiDevice)` 分支到 4×4 端口准静态稠密 stamp
  （`Device.terminal_linearization`：内部节点一阶消元 G=Jtt−Jti·X、
  C=Ctt−Cti·X−Jti(W−V·X)，按偏置 memo；`ac_mna._stamp_dense_lti`），噪声采样直接走 ABC 的
  `get_noise_psd`（OsdiDevice 已实现）；`model_types` 随 PSS 结果携带,PAC/PNoise 全路径
  （解析伴随/时域/FD 兜底/LTI 快路径）自动穿透。**验证**：冻结时钟 LTI oracle——HB 折叠 PAC ==
  平稳 `ac_solve`、HB 折叠 PNoise == 平稳 `noise_analysis`,全频点 **0.000%**;斩波 TD PAC
  基带 1.7251 vs 大信号 δ→0 真值 1.7319（**0.4%**）。**两个关键教训**：(1) PAC/PNoise 线性化
  只认 `node_inputs`,不认 `transient_inputs`——斩波时钟必须经 node_inputs 驱动栅节点
  （否则栅被冻结在常量上线性化,PAC 悄悄给出冻结相位增益）;(2) 硬开关电路的 HB 路径受边带
  截断限制（方波电导谐波 ~1/K 收敛,K=40 仍差 %级）,**必须用 `time_domain: true`**
  （无截断,OTFT 斩波 PNoise 默认 TD 同理）,testbench 时钟给 ~100ns 有限沿。
- ~~硅瞬态三个守卫项~~ ✅ **已完成（2026-07-04）**：
  - **受控源**（VCCS/VCVS/CCCS/CCVS）：核内全量 stamp（按 `dc_residuals` 的 F=−res 约定），
    无源件+受控源+gmin 抽成单源 `_osdi_stamp_elements_impl`（网格核/自适应核共用）；
    测试用一个电路同时验证四类（VBUF=2·VOUT、VX=VY=VZ=VOUT 逐点精确）。
  - **自适应步长**（`_osdi_transient_adaptive_impl`）：样本区间内误差控制子步进
    （变步长 BDF2、BE 起步、线性外推预测子 LTE、reltol/vabstol/iabstol、拒绝折半/
    接受最多×2、min-h 兜底计 nfail、输入列间线性插值），`transient(adaptive=True)` 直达;
    验证：粗网格自适应 vs 同激励细网格固定参考 <1e-3 V、稳态 2e-5;DC-hold 下步长自动放大。
  - **混合电路 = 多 `.osdi` 库**：核携带两组函数指针 + 每器件库索引（>2 库干净报错）,
    每器件 n_nodes/n_jac 填充到 max 宽度;验证：**BSIM4 + BSIM3** 同一电路瞬态 ==
    各自单库运行逐点一致（<1e-6,独立子电路精确 oracle）。**字面的 OTFT-numba+硅混合：
    非需求**（用户确认 2026-07-04：不会有同一电路两个工艺的场景,不投入）。技术记录备查：
    干净路径（`PDK/veriloga.va` 过 OpenVAF）会触发 OpenVAF-Reloaded 编译期 ICE
    （解析错误已排除：删 5 行从未使用的复合 branch 声明后 panic,`ddx()` 替换无效）。
- 硅 corner 目前只有 `tt`/`ss` 缓存过卡片，`ff`/`sf`/`fs` 首次使用时会自动提取（需外置盘工具链）。

### 9b. 第二个硅工艺：FreePDK45（45nm/1.0V，ngspice-C 求值器）✅ DC/AC/noise（2026-07-05 起）

FreePDK45 是用户的目标工艺，但**不能走 SKY130 的 OSDI 路径**：它的 BSIM4 卡声明 `version = 4.0`,
而我们 OpenVAF 编的 BSIM4.8 VA 无版本开关、在这些激进的 45nm 卡上算出 ~30% 不同的 I-V（已用改卡
验证与版本无关）。所以 FreePDK45 的 **oracle 是 ngspice-C 本身**,求值器换成"表征网格 + 插值":

- **`core/ngspice_char.py`**：每个 `(model, W, L, corner, temp)` 用一次批量 `.dc vg vd` 扫（每 Vsb 一
  切片,~0.03s/1000 点）成 Id/gm/gds/Cgs/Cgd 网格;噪声用逐偏置 `.noise`（CCVS 跨阻 → 漏噪 PSD,拟合
  S_id=A+B/f）。缓存到 `data/pdk/freepdk45/*.npz`,`temp_c` 进缓存键（27°C 无标签,老缓存不失效）。
- **`core/ngspice_device.py`**：`NgspiceDevice(TransistorModel)` 插值网格（scipy 线性）。`extract_w`：
  参考 W 表征一次 + 线性缩放实际 W（BSIM4 近似正比 W,<0.7% vs 逐 W 真卡）→ dataset/优化器扫 W 变纯插
  值;`temperature`（开尔文 kwarg）→ 按该温度重表征（PVT 温度轴）。
- **`core/freepdk45_model.py`**：`Fp45Nfet/Fp45Pfet` + `register_pdk("freepdk45", …)`,`corner` 选
  `models_<nom/ss/ff>/` 卡目录。工具链：`PDK_ROOT/freepdk45/`（卡）+ ngspice(经 `run-ngspice.sh`);无
  OpenVAF 依赖。
- **已验证**：单器件 Id/gm/gds 逐位对 ngspice `.op`;5T OTA 过 `ac_solve` 对 ngspice `.ac` <0.05dB/0.3%;
  输出噪声 <5% vs ngspice `.noise`。**整机全差分 OTA 对 ngspice 自己的 `.ac` 交叉核对**（`test_fd_ota_ac
  _matches_ngspice`）:增益/PM <0.2dB/<8°,UGBW 偏高 ~8%——网格 AC 只带 Cgs/Cgd、缺漏/源结电容 Cdb/Csb,
  头条数字取 ngspice 值。设计案例见 `docs/freepdk45_fd_ota_design.md`。
- **范围/未做**：仅 DC+AC+noise(网格无电荷伴随 → 无瞬态/PSS,相应钩子 `NotImplementedError`)。
- **`run_analysis_suite` 修复（2026-07-05）**：AC/noise 分支此前**未透传** `model_types`/`device_kwargs`
  也未播种 DC——任何硅配置经 `python -m core run` 都会静默退回 OTFT 模型 + 多稳态 DC 落错分支(SKY130 也
  中招)。已修:两分支绑定 `models` 并用第一个字典型 `dc_guesses` 播种;OTFT（回调式种子）守卫为 None,
  byte-gate 5/5 不变。

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

- ~~**ML surrogate 全链路**~~ ✅ **完成（2026-07-02）** — `dataset` → `surrogate`/`surrogate_torch` →
  `optimize` 一条命令链跑通（筛选→Pareto→solver 校验），OTFT 和硅两条工艺都验证过精度目标（median <1.4%）。
- ~~**硅 CMOS PDK（SKY130）+ 硅设计闭环**~~ ✅ **DC/AC/noise/瞬态完成（2026-07-04）** —
  OpenVAF 编译 BSIM4 通过 OSDI ctypes 宿主接入现有求解器引擎（model==oracle）；`models` 配置块 + 硅
  corner 路由打通 `dataset`/`optimize`/`explore` 全链路；互补 5T OTA 闭环验证（筛选 ~6000×、solver 校验
  9/10 可行、跨工艺角复验仍 9/10）。详见 §9。剩余：硅斩波器（PSS/PAC/PNoise）。
- **扩展对标覆盖** — 更多 switch 尺寸（5000/30）、更多 f_chop（100/300/1k Hz）、三 corner × 多频率组合，
  趁校准基础设施（`core/cadence_netlist.py` + `core/calibration.py` + `calibration/`）正热扩大回归网。
- **gear2 vs BE 已评估并修复（2026-06-22）** — 全 case 扫描发现并修复了 SC-LPF PAC 的 gear2 silent
  landmine（解析伴随 PAC 现支持 vsource drive、与积分阶数无关，gear2==BE）；周期分析精度 chopper/sc_lpf 均 calibration 内 <2%。

测试套件和校准回归是持续维护项，每次改动确认无回归（当前 283 passed, 1 skipped；Cadence byte-gate 5/5 保持）。
