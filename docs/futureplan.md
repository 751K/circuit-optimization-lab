# 后续开发计划

[English README](README.md) | [中文说明](README_zh.md) | [核心求解器概览](core_overview_zh.md)

## 当前状态（2026-06-21）

项目是一个成熟的本地模拟电路仿真与设计探索框架，首个应用场景为 AT4000TG PMOS-OTFT ECG AFE，
已对 Cadence Spectre 24.1 完成校准。核心能力全部落地，正在从"功能开发"转向"生态完善"阶段。

### 已完成的能力矩阵

| 领域              | 交付内容                                                                                                             | 状态       |
| --------------- | ---------------------------------------------------------------------------------------------------------------- | -------- |
| **电路描述**        | JSON 格式 + schema 校验，多器件/多输出/仿真参数内嵌                                                                               | ✅ 成熟     |
| **DC/AC/Noise** | 工作点求解、小信号增益/带宽、热噪声+闪烁噪声、等价输入噪声                                                                                   | ✅ 成熟     |
| **瞬态**          | 后向欧拉（默认）+ gear2/BDF2（可选），Numba grid solver，stiff 电路自动 BE 回退                                                      | ✅ 成熟     |
| **周期分析**        | 通用 shooting PSS（解析 monodromy + Broyden 复用）、通用 PAC（解析伴随 HB，O(1) 每频点）、通用 PNoise（harmonic balance，第一性原理，无标定常数）      | ✅ 成熟     |
| **Chopper**     | 5 层分析：理想 LPTV → PMOS 静态相位 → 有限边沿谐波 → quasi-LPTV 边带折叠 → hard-switched PSS/PAC/PNoise                              | ✅ 成熟     |
| **元件类型**        | PMOS_TFT、电阻、电容、理想直流电流源、理想时变电流源（charge injection）                                                                 | ✅ 成熟     |
| **设计探索**        | JSON 配置层、LHS/随机采样、约束过滤、Pareto 选择、CSV/JSONL 导出、CLI                                                                | ✅ 成熟     |
| **工艺角/鲁棒性**     | 全局 corner（typ/slow/fast）、逐器件 mismatch MC、确定性 latch 筛查                                                            | ✅ 成熟     |
| **Numba 加速**    | PMOS 电流、内部节点 Newton、偏置电容、terminal derivative、transient Newton 内循环、gear2 grid solver、PNoise HB block 组装和噪声折叠      | ✅ 全覆盖    |
| **gear2/BDF2**  | 变步长 BDF2、Numba grid、解析 monodromy、graceful BE 回退。PAC baseband 三 corner 全部 <1%（BE 时 −2.5%）。PSS/PAC/PNoise 默认 gear2 | ✅ 完成     |
| **CLI**         | `python -m core <circuit.json>` 全分析 dispatch + exploration 模式 + 结果导出                                             | ✅ 完成     |
| **Demo**        | Flask Web 前端 + REST API（`demo/server.py`）                                                                        | ✅ 可用     |
| **测试**          | 12 个测试文件、94 个测试函数（含 RUN_SLOW_CHOPPER）、Numba 环境下 93 passed                                                        | ✅ 覆盖核心路径 |
| **文档**          | 英/中双语：README、core_overview、JSON 格式参考、gear2 完成报告                                                                  | ✅ 完善     |

### 代码规模

```
core/                     ~10,600 行  (17 个 .py 文件)
tests/                     ~2,200 行  (12 个 .py 文件)
benchmarks/                  ~500 行  (4 个 benchmark)
docs/                      ~3,400 行  (6 个 .md 文件)
```

### 对标状态

| 指标                                      | 对标结果                                              |
| --------------------------------------- | ------------------------------------------------- |
| DC 工作点 / AC 增益                          | 与 Spectre 误差 ~0.01 dB                             |
| AC 带宽                                   | 对齐 Spectre                                        |
| 等价输入噪声（非 chopper）                       | 百分之几以内                                            |
| Chopper PSS/PAC/PNoise（原生，无标定常数）        | PAC baseband + 200 Hz <1%，IRN <1%（D3 slow corner） |
| Chopper transient（8-PMOS hard-switched） | 输出均值 −10.76 mV vs Spectre −10.62 mV，nfail=0       |
| Mismatch MC mean/std                    | 与 Cadence 趋势一致                                    |

---

## 路线图

### 优先级排序

1. **Cadence 校准闭环** — 每次改动后量化验证"快了多少、偏了多少"，最高 ROI
2. **扩展验证覆盖** — 更多非 chopper 周期拓扑、更多 corner/频率对标
3. **器件模型抽象** — 支持 NMOS 等新器件类型，解锁更广泛电路
4. **transient 性能深化** — 编译化 substep 调度、batch 并行、gear2 硬化（低优先级）
5. **搜索策略扩展** — 贝叶斯优化、进化算法
6. **编译后端评估** — Rust/Cython 承担千级/万级 sweep

---

## 1. Cadence/Spectre 校准闭环 🔴 最高优先级

### 现状

当前校准是手动的——每次修改模型或求解器后，需要在 chopper 或 solver 代码里嵌入对比常数，
或者手动跑 Spectre 导出 PSF 再 Python 对比。这套流程容易出错、不可重复、且跟不上代码迭代速度。

### 目标

建立一键式自动校准流水线：

```
本地 solver 跑 DC/AC/Noise/Tran/PSS/PAC/PNoise
        ↓
自动对比 Cadence PSF/CSV 参考数据
        ↓
输出：最大/相对误差、gain dB 误差、BW 误差、noise RMS 误差、transient 波形差
        ↓
CI 可集成：每次 commit 自动跑校准套件
```

### 具体任务

- [ ] 建立 `calibration/` 目录和参考数据格式约定
  - Cadence 导出的 DC OP、AC 曲线、noise contribution、transient 波形
  - 支持 PSF（通过 `cadence-server-verify` skill）和 CSV 两种格式
  - 每个参考用例一个子目录，含 `metadata.json`（f_chop、corner、switch 尺寸等）
- [ ] 编写 `core/calibration.py` — 自动对比脚本
  - 加载参考数据 + 跑本地 solver
  - 按指标输出差异表：gain、BW、IRN、transient 波形 max|Δ| 和 RMS 差
  - 支持 corner（typ/slow/fast）批量对比
  - CLI 入口：`python -m core.calibration <case_dir>`
- [ ] 集成到现有 benchmark 或 CI
  - 每次改动后一键判断回归状态
  - 对标数据随代码版本管理
- [ ] 扩展对标用例
  - 非 chopper 周期拓扑（RC、RLC、放大器）
  - 更多 switch 尺寸 / f_chop / corner 组合
  - 更多 Spectre 分析类型（stb、xf 等，视需要）

### 预期收益

- 改动模型/求解器后无需手动对比
- 新人或协作者可立即判断改动是否破坏精度
- 为未来的 PDK 迁移或新拓扑提供可复用的对标框架

---

## 2. 扩展周期分析验证 🟡 高优先级

### 现状

PSS/PAC/PNoise 三件套已做成通用拓扑级求解器（`pss_solve` / `pac_solve` / `pnoise_solve`），
但目前实际验证的周期拓扑只有两个：

- `examples/periodic_rc.json` — 无源 RC 低通（trivial 用例）
- PMOS chopper 八开关拓扑（通过 `pmos_chopper_pss/pac/pnoise` 包装器）

### 目标

让通用周期求解器有更多独立验证案例，证明它确实可以"不经专用 wrapper 即处理任意周期拓扑"。

### 具体任务

- [ ] 新增 1–2 个非 chopper 周期 JSON 示例
  - 例如：开关电容低通、周期驱动放大器
  - 每个示例在 JSON 里直接配置 `periodic` + `analyses` 块
  - 无需写任何 chopper 专用 wrapper
- [ ] 编写对应的 Cadence 对标
  - 在 Spectre 里跑相同拓扑的 pss/pac/pnoise
  - 导出参考数据到 `calibration/`
- [ ] 扩展自动化对标覆盖
  - 更多 switch 尺寸（5000/30、20000/80、极端值）
  - 更多 f_chop（100 Hz、225 Hz、300 Hz、1 kHz）
  - 三 corner × 多频率组合

### 预期收益

- 验证通用 PSS/PAC/PNoise 求解器的拓扑无关性
- 积累对标用例库，防止回归
- 为未来非 AFE 电路提供可参考的起点

---

## 3. 器件模型接口抽象 🟡 高优先级

### 现状

所有 solver 硬编码使用 `PMOS_TFT` 模型。要支持 NMOS、BJT 或其他 PDK 器件，需要改 solver 内部逻辑。

### 目标

定义统一的器件模型接口，让 solver 通过接口调用而不关心具体模型实现。
JSON 电路可以声明每个器件使用哪个模型。

### 具体任务

- [ ] 定义 `DeviceModel` 抽象接口（protocol / ABC）
  
  ```python
  class DeviceModel(Protocol):
      def get_op(self, Vds, Vgs, Vs, sizes, ...) -> dict: ...
      def get_Idc(self, Vds, Vgs, Vs, sizes, ...) -> float: ...
      def get_Idc_and_capacitances(self, ...) -> tuple: ...
      def get_noise_psd(self, freqs, Vds, Vgs, ...) -> np.ndarray: ...
      def g_area(self, sizes) -> float: ...
  ```
- [ ] `PMOS_TFT` 实现该接口（重构现有代码，不改行为）
- [ ] JSON 格式扩展：`"models"` 块 + 每个 device 的 `"model"` 字段
  
  ```json
  {
    "models": {"pmos_tft": {"type": "PMOS_TFT"}},
    "devices": [
      {"name": "M1", "model": "pmos_tft", "drain": "OUT", "gate": "IN", "source": "VDD", "W": 2000, "L": 80}
    ]
  }
  ```
- [ ] Solver 层改调接口方法而非 `PMOS_TFT` 类方法
- [ ] 实现一个最小 NMOS 模型（简化版，用于验证接口抽象）并编写验证用例

### 预期收益

- 解锁 NMOS、BJT 等新器件类型
- 新 PDK 只需实现一个 model 类即可接入全部 solver
- 不影响现有 PMOS_TFT 行为和性能

---

## 4. 扩展元件类型 🟢 中优先级

### 现状

已支持：电阻、电容、理想直流电流源、理想时变电流源（用于 charge injection）。

### 待补充

- [ ] **理想电压源** — DC 和瞬态分析中最基础的激励方式
  - DC：固定节点电压（与 rail 不同，电压源节点仍在 solved 集合中）
  - AC/Noise：短路（零内阻），视为 AC 地
  - Transient：时变电压源约束
- [ ] **受控源**（VCCS / VCVS / CCCS / CCVS）— 支持经典放大器/运放宏模型
  - 优先 VCCS（已有 stamp 原语，只是缺 JSON 表达）
- [ ] **互感和耦合电感** — 较低优先级，视需求

### 依赖

- 电压源需要改动 DC 求解器（新增约束方程），影响面较大
- 受控源相对独立，可先行

---

## 5. 深化 transient 性能 🟢 中优先级

### 现状

- Numba 内核已覆盖 PMOS 电流、内部节点 Newton、transient Newton 内循环、gear2 grid solver
- gear2/BDF2 已上线，PAC 精度从 BE 的 −2.5% 提升到 <1%
- 裸 `transient()` 默认 BE，`integration_method="gear2"` 带 safe BE fallback

### 后续方向

- [ ] **compiled step plan**：把每个 interval/substep 的 retry 拆步和输出采样做成预编译 plan，减少 Python 调度开销
- [ ] **小矩阵特化**：对 6×6（chopper）级别矩阵做专用 dense solve，跳过通用 LU 开销
- [ ] **chopper transient 深度编译化**：固定 8 开关拓扑的整条 transient 链路留在 Numba 内
- [ ] **batch transient / MC 并行化**：多个瞬态仿真并行（thread-level 或 process-level）
- [ ] **gear2 grid subdivision/retry 硬化（低优先级）**：真正的 robust 2 阶裸 transient，
  需要 gear2 版 solve_chunk + 平滑开关模型。当前 graceful BE 回退足以保证安全性。
  详见 `docs/gear2_integration_plan.md`。

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

## 7. 编译后端路线评估 🟢 低优先级

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
| gear2 subdivision/retry | 单步 grid 对 stiff 电路通过 graceful BE 回退保证安全。真正 robust 2 阶裸 transient 需要 gear2 solve_chunk，低价值高成本 |
| 理想电压源                   | DC/瞬态求解器需要新增约束方程，影响面大，暂时 rails 可覆盖大部分场景                                                      |
| PNoise HB solver 扩展     | 已有 dense/sparse/iterative 三条路径。若 HB 规模继续增长（数十+谐波），再评估 matrix-free matvec 或低秩边带截断             |
| `results/` 目录           | 含历史 benchmark 和 explore 输出，已在 .gitignore 中。考虑移到独立数据仓库或加 README                               |

---

## 不做的事项

| 事项                      | 原因                                                 |
| ----------------------- | -------------------------------------------------- |
| Verilog-A 电容模式（C·dV/dt） | 已证伪数值不稳定（slow 不收敛、fast 收敛到错误轨道），charge 模式是最优解      |
| 大规模 CI/CD               | 项目目前为研究型单人开发，手动 pytest + benchmark 足够。有协作者再加入      |
| GPU 加速                  | PMOS 模型和 MNA 矩阵规模（≤20×20）对 GPU 无优势。如未来处理大规模阵列电路再评估 |
| Sign-off 级仿真器认证         | 项目定位是设计探索工具，不做 Spectre 替代品                         |

---

## 执行建议

当前阶段最有价值的投入是 **校准闭环（第 1 步）** 和 **验证扩展（第 2 步）**——
两者直接提升代码改动的信心和速度。**模型抽象（第 3 步）** 是支持新电路类型的架构前提，
适合在有明确需求时启动。

其余步骤按需推进。测试套件和 benchmark 是持续维护项，每次改动确认无回归。
