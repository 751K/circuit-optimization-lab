# 后续开发计划

[English README](README.md) | [中文说明](README_zh.md) | [核心求解器概览](core_overview_zh.md)

## 当前状态

项目已从单一 AFE 原型推进为通用本地电路仿真框架。已完成的能力：

- **JSON 电路描述**：schema 校验、多器件、多输出、仿真参数内嵌（`schemas/circuit.schema.json`）
- **求解器栈**：DC 工作点、AC 小信号、噪声（热噪声+闪烁噪声）、瞬态（BE 默认，可选 gear2/BDF2）
- **元件类型**：PMOS_TFT、电阻、电容、理想直流电流源
- **周期分析**：通用 shooting PSS（`pss_solve`，默认解析 monodromy Jacobian，gear2 支持增广 2n 态 monodromy）、通用 PSS 辅助 PAC（`pac_solve`，默认解析伴随谐波平衡，`O(1)` 每频点）、通用 harmonic-balance PNoise（`pnoise_solve`）；chopper 侧保留 `pmos_chopper_pss/pac/pnoise` 兼容包装器
- **Chopper 分析**：理想 LPTV 频域、PMOS 静态相位、有限边沿谐波、charge injection、PMOS quasi-LPTV 边带折叠（需 Cadence 标定常数）、hard-switched PMOS transient（已对齐 Spectre `tran`）、PMOS 八开关 PSS/PAC/PNoise（无需标定常数，已对齐 Spectre PNoise）
- **设计空间探索**：JSON 配置层、LHS 采样、约束过滤、Pareto 选择、CSV/JSONL 导出（`core/explore.py`）
- **工艺角与鲁棒性**：全局 corner、逐器件 mismatch MC、latch 筛查（`core/corners.py`）
- **交互式 demo**：Web 前端 + REST API（`demo/server.py`）
- **性能基准**：4 个固定 benchmark（AFE / model / chopper / sweep），支持 Numba 对比
- **测试覆盖**：12 个测试文件覆盖核心路径
- **Numba 加速**：已覆盖 PMOS 电流、内部节点 Newton、transient 热路径、gear2 grid 求解器

PSS/PAC/PNoise 三件套已落地，并已把 PAC/PNoise 抽成通用拓扑级周期小信号/噪声求解器。
✅ PAC 已从有限差分升级为解析伴随谐波平衡（`analytic=True`，默认），PSS shooting
Jacobian 也已有解析 monodromy 路径（`analytic_jacobian=True`，默认）。下一步重点：
完善 Cadence 校准闭环，深化 transient/PNoise 性能，以及扩展更多周期拓扑验证。

### gear2/BDF2 积分器升级（2026-06 完成主体）

chopper PSS 已默认使用 gear2（二阶 BDF2），三 corner PAC baseband 全部 <1%（BE 时 typ/fast 差 −2.5%）。关键交付：
- ✅ 变步长 BDF2（charge 模式 + 步长比限制 ρ≤2 保证零稳定），gear2 二阶收敛已验证
- ✅ Numba gear2 grid 求解器（单步/区间，BE 自启动，warm 0.89s）
- ✅ 解析 gear2 monodromy（增广 2n 态，warm 0.24s，比 FD shooting 快 4×）
- ✅ PSS/PAC/PNoise/chopper 全部默认 gear2；裸 `transient()` 保留 BE 默认
- ⚠️ **待硬化**：gear2 grid 的 subdivision/retry（像 BE grid 一样 pieces + rolling 两步历史）。试过一次重构但引入了不可接受的 PAC 回归（−3.5%，收敛到了另一个有效周期解），已回退到可靠的单步版。裸 `transient()` 要在 gear2 默认下健壮运行需要这个硬化。
- 全套 93 passed（含 RUN_SLOW_CHOPPER）；详见 `docs/gear2_integration_plan.md`。

---

## 1. PSS/PAC/PNoise — 验证与泛化

PSS/PAC/PNoise 第一版已完成并通过 Cadence 对标：

1. **PSS**（`core/pss_solver.py`）：通用 transient shooting 求解器，`Φ_T(x0)-x0=0`。默认解析 monodromy Jacobian（沿轨道采样 G(t)/C(t)，`Φ=∏A_m`，一次遍历替代 `n_state` 次有限差分瞬态），失败自动回退到 FD。后续默认 Broyden secant 复用，带真实 residual 校验、阻尼 Newton、best-so-far 追踪和 period-run 性能计数器。`pmos_chopper_pss()` 复用八 PMOS chopper 波形/拓扑，支持 pulse/phase 两种时钟风格。
2. **PAC**（`core/pac_solver.py::pac_solve()`）：✅ 默认解析伴随谐波平衡（`analytic=True`）：沿 PSS 轨道采样 G(t)/C(t) 和输入耦合列，FFT 到谐波系数，构建转换矩阵 Y_HB(f)，每频率一次伴随线性求解得到 sideband-0 增益 — O(1) 求解，零额外瞬态运行。`analytic=False` 可回退到原有限差分 shooting 路径。静态 PSS 仍走 LTI `ac_solve` fast path。`pmos_chopper_pac()` 现在只是 chopper 差分输入包装器。
3. **PNoise**（`core/pnoise_solver.py::pnoise_solve()`）：PSS 轨道上的 harmonic-balance 转换矩阵法。沿 PSS 轨道 N 点采样 → 时变小信号 G(t)/C(t) → FFT 到频域 → 组装 `nb×nb` 块对角系统 `Y[kr,kc] = G_{kr-kc} + jω·C_{kr-kc}` → 伴随求解传递阻抗 Z_{j,k} → 循环平稳器件/电阻噪声折叠 `S_out = Σ_j Σ_k |Z_{j,k}|² S_j`。无需 Cadence 标定常数，已是第一性原理 LPTV 噪声解。静态轨道自动走 LTI noise fast path；LPTV 轨道在 `pss_result` 上复用采样 `G(t)/C(t)`、HB block 和相同频点 adjoint 解。HB 求解支持 dense / sparse direct / block-Jacobi 预条件 GMRES；默认 auto 在矩阵大且非常稀疏时切 sparse。`pmos_chopper_pnoise()` 现在只是 chopper 包装器。
4. **对标**：UI 锁定尺寸、`f_chop=225 Hz`（或 300 Hz）、switch `5000/30`、`rise/fall=20 us` 下，PNoise IRN 经输出 RC 滤波后与 Spectre PNoise 吻合（`test_pmos_chopper_pnoise_matches_cadence_band`）。

后续工作：

- ~~将 PAC 从有限差分升级为解析伴随~~ ✅ 已完成（`analytic=True`，默认）。
- ~~PSS 解析 monodromy Jacobian~~ ✅ 已完成（`analytic_jacobian=True`，默认），shooting Jacobian 构建从 `O(n_state)` 降为 `O(1)` 轨道遍历。
- 继续扩展 JSON 周期分析示例，让更多非 chopper 周期拓扑无需写专用 wrapper 即可调用。
- 扩展更多 switch 尺寸、频率和 corner 的自动化对标。
- ~~重复 DC/seed 求解复用~~ ✅ 已完成：chopper PSS 缓存 bare-AFE DC seed；JSON
  dispatch 在 AC/noise 同频点时复用 AC 结果，不同频点时复用 AC `dc_op` warm seed。
- PNoise HB 已有 dense / sparse direct / block-Jacobi 预条件 GMRES；后续若 HB 规模继续增长，再评估矩阵自由 matvec、更强 block preconditioner 或低秩边带截断。
- PSS 周期 transient 已保持在 Numba grid 内运行；继续提升 PSS 速度可进一步优化 Broyden 复用策略或减少 shooting 迭代数。

## 2. Cadence/Spectre 校准闭环

本地模型需要持续对齐仿真器。当前校准是手动的——每次改模型后在 chopper 里嵌入对比常数。需要系统化：

- 建立校准目录，收集 Cadence 导出的 DC OP、AC 曲线、noise contribution、transient 波形 CSV/PSF
- 编写本地 solver 对比脚本，自动输出：最大绝对/相对误差、gain dB 误差、BW 误差、noise RMS 误差、transient 波形最大差和 RMS 差
- 每次改模型或加速内核后，一键判断"快了多少、偏了多少"

## 3. 深化 transient 性能

第一阶段 Numba 内核优化已完成（PMOS 电流、内部节点 Newton、偏置电容、terminal derivative、transient Newton 内循环）。gear2/BDF2 积分器已上线（chopper PSS/PAC/PNoise 默认 gear2），chopper PAC baseband 从 BE 的 −2.5% 改善到 <1%。后续方向：

- **gear2 grid subdivision/retry 硬化**：当前 gear2 grid 是单步版（无 pieces + retry），裸 `transient()` 仍默认 BE。需要把 BE grid 的 subdivision/retry/rolling 两步历史正确复刻到 gear2 grid，才能让裸 `transient()` 安全默认 gear2。上次尝试因轨道漂移（−3.5% PAC）已回退，需更仔细地保持与单步版的轨道一致。
- 把每个 interval/substep 的重试拆步和输出采样做成 compiled step plan，减少 Python 调度开销
- 对常用小矩阵（6x6 / chopper 拓扑规模）做更专门的 solve 路径
- chopper transient 固定开关拓扑的深度编译化
- batch transient / Monte Carlo 并行化

原则不变：不牺牲精度，任何近似 Jacobian 都要用波形回归验证。

## 4. 抽象器件模型接口

当前 solver 默认使用 `PMOS_TFT`。要支持更多器件或 PDK，需要显式化模型接口：

```json
{
  "models": {
    "pmos_tft": {"type": "PMOS_TFT"}
  },
  "devices": [
    {"name": "M1", "model": "pmos_tft", "drain": "OUT", "gate": "IN", "source": "VDD", "W": 2000, "L": 80}
  ]
}
```

统一接口方法：`get_op()`, `get_Idc()`, `get_Idc_and_capacitances()`, `get_noise_psd()`，可选 small-signal derivative。拓扑和 JSON 支持指定 device model，solver 不硬编码具体模型类。

## 5. 扩展元件类型（剩余项）

已支持：电阻、电容、理想直流电流源、PMOS chopper 全链路。待补充：

- 理想电压源 / 受控源
- 更多 active device 类型（NMOS 等，依赖第 4 步的模型注册）

## 6. 扩展优化搜索策略

`core/explore.py` 首版已落地：随机/LHS 采样 + 约束过滤 + Pareto。后续可扩展：

- 贝叶斯优化
- 进化算法
- 将推荐候选接入 Cadence 验证闭环（依赖第 2 步）

## 7. 编译后端路线评估

当前 Python + Numba 的性能对百级候选扫描已足够。如果后续需要千级/万级 sweep 或大规模 MC：

- 评估 Rust 或 Cython 内核承担单器件模型批量评估、transient Newton/Jacobian 热路径、MC 并行
- Python 层保留 JSON 配置、拓扑编排、实验管理和报告

## 执行顺序建议

1. **第 1 步**（PSS/PAC/PNoise 验证与泛化）——第一版已完成；后续泛化与性能优化优先级高
2. **第 2 步**（Cadence 校准闭环）——与第 1 步并行，每次改动后需要量化验证
3. **第 3 步**（transient 性能深化）——transient 仍是 sweep/MC 的主要耗时项
4. **第 4 步**（模型注册）——支持新 PDK 和更多电路类型的前提
5. **第 5-7 步**——按需推进

测试套件和性能基准是持续维护项，每次改动都应确认没有回归。
