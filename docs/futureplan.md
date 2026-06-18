# 后续开发计划

[English README](README.md) | [中文说明](README_zh.md) | [核心求解器概览](core_overview_zh.md)

## 目标

当前代码已经从单一 AFE 原型，推进到支持 JSON 电路描述、通用拓扑求解、可选 Numba 加速的本地电路仿真框架。下一步的核心目标不是继续堆功能，而是把“可换电路、可验证、可优化”这三件事做扎实。

优先级按实际收益排序：

1. 让新电路接入流程稳定，不需要改 solver 源码。
2. 建立回归测试，避免后续优化破坏 DC/AC/noise/transient 精度。
3. 把性能热点继续模块化，为 Numba、Cython 或 Rust 后端留接口。
4. 将尺寸/偏置搜索重新接到新的 JSON 拓扑层上。
5. 建立 Cadence/Spectre 对齐流程，用仿真数据校准模型误差。

## 近期任务

### 1. 完善 JSON 电路格式

当前 `examples/single_stage.json` 已经验证了基础字段，但还需要让格式更适合真实电路。

需要补充：

- JSON schema 或轻量校验文档。
- 更清晰的错误信息，例如未知节点、缺失 W/L、输出节点不是 solved node。
- 支持在 JSON 中指定仿真任务，例如 AC 频率范围、noise 积分频段、transient 时间步和输入波形类型。
- 支持多个 named input，例如 `vip`、`vin`、`clk`、`reset`。
- 支持多个输出观测量，而不只是单端或差分一个输出。

建议先不要引入复杂 netlist parser。当前阶段用显式 JSON 更可控，也更容易和 solver 的拓扑对象一一对应。

### 2. 增加回归测试

目前主要靠手动运行脚本。下一步应该加 `tests/`，至少覆盖以下场景：

- 当前 AFE 的 DC/AC/noise/transient smoke test。
- `examples/single_stage.json` 的通用拓扑 smoke test。
- Numba 关闭和开启时 `_eval_currents()` 数值一致。
- JSON loader 对缺失字段和错误节点能给出明确异常。
- transient 在固定输入下 `nfail=0`，输出末值在容差范围内。

建议使用 `pytest`，但先保持测试规模小。目标是每次改模型或 solver 后，几秒内能确认没有明显回归。

### 3. 整理性能基准

现在已有手动 benchmark 结果，但还没有固定脚本。建议新增 `benchmarks/`：

- `bench_afe.py`: 跑当前 AFE 的 `ac121`、`noise121`、`tran200`。
- `bench_model.py`: 单器件 PMOS_TFT 热路径微基准。
- 支持 `CIRCUIT_USE_NUMBA=1` 对比。
- 输出固定格式，便于复制到报告或 future optimization notes。

当前可参考的性能目标：

- `ac121`: 约 8-12 ms。
- `noise121`: 约 12-16 ms。
- `tran200`: 默认 Python 路径约 100-130 ms。
- `tran200`: Numba 预热后约 55-60 ms。

这些数字会随机器、conda 环境和首次 JIT 编译变化，基准脚本应区分 cold run 和 warm run。

## 中期任务

### 4. 抽象器件模型接口

当前 solver 仍默认使用 `PMOS_TFT`。如果后续要支持更多器件或 PDK，需要把器件模型接口显式化。

建议定义统一接口：

- `get_op(Vs, Vd, Vg)`
- `get_Idc(Vs, Vd, Vg)`
- `get_Idc_and_capacitances(Vs, Vd, Vg)`
- `get_noise_psd(Vs, Vd, Vg, frequency)`
- 可选 small-signal derivative 接口

然后让 topology 或 JSON 支持指定 device model，例如：

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

这样 solver 就不需要知道具体模型类，后续可以加入 NMOS、resistor、capacitor、ideal current source 等元件。

### 5. 扩展元件类型

> 状态（首批已落地）：电阻 / 电容 / 理想直流电流源已作为两端元件接入 Topology
> 与四类分析 —— DC KCL、AC（电阻 `1/R`、电容 `jωC`）、noise（电阻热噪声 `4kT/R`）、
> transient（电导 + 恒流 + 电容伴随）。JSON 字段 `resistors` / `capacitors` /
> `current_sources` 见 `docs/json_circuit_format_zh.md`，示例
> `examples/resistor_load_stage.json`，测试 `tests/test_elements.py`。

现在拓扑里所有 active device 都按 PMOS_TFT 三端器件处理。真实电路需要更多基础元件：

- 固定电容。✅
- 电阻。✅
- 理想电流源。✅
- 理想电压源或受控源。（待做）
- 理想同步 chopper 频域分析。✅（`core/chopper.py`，按八开关差分换向器的理想
  +/-1 方波乘法模型，用边带折叠计算 gain/BW/noise）
- PMOS 八开关 chopper 拓扑。✅（`build_afe_pmos_chopper()` /
  `pmos_chopper_analysis()`，用真实 `PMOS_TFT` pass switch 估算 Ron、寄生电容
  与 switch 噪声对静态相位 gain/BW/noise 的影响）
- 有限边沿 / dead time chopper 谐波权重。✅（`finite_edge_chopper_harmonics()`）
- clock feedthrough transient。✅（有限边沿 clock 驱动 PMOS gate，已有 PDK
  `Cgss/Cgdd * ddt()` 电容伴随 stamp）
- charge injection 一阶模型。✅（`PMOS_TFT.estimate_channel_charge()` +
  `transient(current_inputs=...)`，由 PDK 电容公式估算 turn-off 注入脉冲）
- PMOS quasi-LPTV sideband folding。✅（`pmos_chopper_lptv_analysis()`，用 PMOS
  静态相位 response/noise 和有限边沿谐波权重做边带折叠）
- hard-switched PMOS chopper transient 收敛优化。✅（`refine_chopper_tgrid()` +
  `transient(fallback_least_squares=True)`，clock edge 附近自动细分，并用电源轨
  有界 fallback solve 处理硬开关 DAE 步）
- 完整 PSS/PNoise 对齐，用于 correlated periodic noise、clock feedthrough /
  charge injection 的 Cadence 级验证、有限边沿时间、时变工作点边带折叠等
  非理想效应的精确标定。（待做）

resistor/capacitor/current source 已完成（它们对 DC、AC、noise、transient 的 stamp 都比较清晰）；
chopper 的理想 LPTV 频域分析、PMOS 静态相位拓扑、有限边沿谐波、charge injection
一阶 transient stamp、PMOS quasi-LPTV folding 与 hard-switched transient 基础收敛
优化已完成；下一步是理想电压源 / 受控源，以及与 Spectre PSS/PNoise 对齐的周期
时变验证路径。

### 6. 优化 transient 内核

Transient 仍是最重的部分。当前 Numba 只加速了 `_eval_currents()`，后续可以继续推进：

- 把 `terminal_derivatives()` 的标量计算迁到 Numba。
- 对固定拓扑预编译 device metadata，减少 Python tuple/list 解包。
- 对常用小矩阵 6x6 solve 做专门路径或缓存结构。
- 对 batch transient 或 Monte Carlo 并行化。

原则仍然是不牺牲精度：残差方程必须保持完整模型，任何近似 Jacobian 都要用波形回归验证。

## 长期任务

### 7. 重新接入优化搜索

> 状态（首版已落地）：`core/explore.py` 已把设计空间探索放到配置层 —— `explore`
> 配置块（变量范围 / 约束 / 目标）+ 随机或 LHS 采样 + 本地 solver 评估 + 约束过滤
> + Pareto 选择 + CSV/JSONL 导出 + `python -m core.explore` CLI。`evaluate()` 和
> `explore()` 已支持可选的 `corner` 参数，可在指定工艺角下进行搜索。示例见
> `examples/afe_explore.json` 与 `examples/single_stage.json` 的 `explore` 块，
> 回归测试见 `tests/test_explore.py`。下一步剩：把推荐候选接入 Cadence 验证闭环
> （第 8 节）、补更多采样/搜索策略、随新元件类型扩展面积/功耗定义。
>
> 工艺角与鲁棒性工具已整合到 `core/corners.py`（本地求解器侧）：
> - 全局工艺角 `CORNERS`（typ/slow/fast）。
> - 逐器件 mismatch MC（`mismatch_mc`）。
> - 确定性 latch 筛查（`latch_screen`），用于替代搜索中的完整 MC。
> - 驱动脚本 `examples/mc_mismatch.py`，测试 `tests/test_corners.py`。
> 另一文件 `core/mc_corners.py` 是 Cadence PSF 后处理侧，与本地求解器侧的 `core/corners.py` 功能对等但相互独立。

通用 JSON 拓扑稳定后，可以把设计空间探索放到配置层：

- 尺寸变量范围。
- 偏置变量范围。
- 约束，例如 gain、BW、IRN、功耗、面积。
- 优化目标，例如最小面积、最小功耗、最小噪声。

建议先实现简单但可靠的流程：

1. JSON 读取电路和设计变量。
2. 随机采样或 Latin hypercube 生成候选。
3. 本地 solver 快速评估。
4. 保存 CSV/JSONL 结果。
5. Pareto 过滤。
6. 输出推荐候选给 Cadence 验证。

不要过早引入复杂机器学习。当前 physics-based surrogate 的速度已经足够支持大量候选筛选。

### 8. Cadence/Spectre 校准闭环

本地模型最终需要持续对齐仿真器。建议建立一个校准目录：

- Cadence 导出的 DC operating point。
- AC gain/BW 曲线。
- Noise contribution 和 IRN。
- Transient 波形 CSV。
- 本地 solver 对比脚本。

校准结果应该输出：

- 最大绝对误差。
- 最大相对误差。
- gain dB 误差。
- BW 误差。
- noise RMS 误差。
- transient 波形最大差和 RMS 差。

这样每次改模型或加速内核，都能明确判断“快了多少、偏了多少”。

### 9. 编译后端路线

目前最合理的路线是：

1. Python 负责 JSON、拓扑、实验编排和报告。
2. Numba 加速 PMOS_TFT 和 transient 热路径。
3. 如果大规模 sweep 仍不够快，再考虑 Rust 或 Cython 内核。

不建议现在全量重写 Rust。Rust 更适合在接口稳定后承担内核层，例如：

- 单器件模型批量评估。
- transient Newton/Jacobian 热路径。
- Monte Carlo 并行仿真。

Python 层仍然保留，因为电路配置、数据分析和优化实验迭代速度更重要。

## 建议执行顺序

推荐接下来按这个顺序做：

1. 维护并扩展 `tests/`，当前已覆盖 AFE、JSON 示例电路、loader 错误处理和模型内核一致性。
2. 维护 `benchmarks/`，当前已固定 AFE 的 `ac121`、`noise121`、`tran200` cold/warm 基准。
3. 维护 JSON schema/说明，当前已新增 `schemas/circuit.schema.json` 和格式文档。
4. 把 optimizer 接到配置层（已落地首版 `core/explore.py`，见第 7 节；后续补搜索策略）。
5. 支持 resistor/capacitor/current source（已落地，见第 5 节；下一步理想电压源 / 受控源 / 开关）。
6. 抽象 device model registry。
7. 建立 Cadence CSV 对比脚本。
8. 继续 Numba 化 transient derivative 内核。

其中第 1 和第 2 步最重要。没有测试和基准，后续继续优化会很难判断是否真的变好。
