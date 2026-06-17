# 电路本地建模与优化流程

[English](README.md) | [中文说明](README_zh.md)

## 项目概述

这个项目的目标是搭建一个本地电路建模、仿真与优化流程，用于模拟电路设计空间探索。核心动机是减少早期尺寸和偏置迭代阶段对 Cadence/Spectre 大规模参数扫描的依赖，同时仍然以 Cadence/Spectre 结果作为最终验证参考。

第一个应用场景来自 AT4000TG 薄膜晶体管放大器设计。在这个项目中，本地 Python 模型被用于复现和分析关键电路行为，包括 DC 工作点、小信号响应、瞬态响应、噪声以及设计约束。这个仓库后续不希望局限在当前 Python 实现或单一 PDK 上，而是希望逐步扩展成更通用的电路探索和优化框架。

## 当前范围

当前流程已经覆盖或计划覆盖以下内容：

- 器件紧凑模型计算。
- DC 工作点求解。
- AC 小信号增益和带宽估计。
- 瞬态响应仿真。
- 噪声分析，包括热噪声和 flicker noise。
- 与 Cadence/Spectre 结果对比和校准。
- 对增益、带宽、输入参考噪声、功耗和面积进行约束检查。
- 在本地快速探索设计空间，而不是每个候选点都直接跑 Cadence。
- 使用搜索、greedy shrink 和 Pareto selection 进行尺寸和偏置优化。
- 工艺角和 mismatch Monte Carlo 类型的鲁棒性检查。
- 为报告和科研汇报生成设计图表。

当前核心代码结构见 [核心求解器概览](core_overview_zh.md)。

## 环境安装

建议使用 Python 3.10 或更新版本：

```bash
python3 -m pip install -r requirements.txt
```

`requirements.txt` 会以 editable 模式安装本项目，因此外部脚本和 notebook 可以直接 `from core...` 导入，不需要手动修改导入路径。

可选的 Numba 加速后端用于高频标量模型内核。普通短任务默认不启用，避免首次 JIT 编译开销；`core.explore` 和 `core.corners` 这类 sweep/MC 入口会默认启用 Numba。需要关闭时可显式设置 `CIRCUIT_USE_NUMBA=0`。

```bash
python3 -m pip install -r requirements-numba.txt
CIRCUIT_USE_NUMBA=1 python3 -m benchmarks.bench_afe --skip-noise --warm-runs 3
```

固定性能基准可用：

```bash
python3 -m benchmarks.bench_afe --warm-runs 3
CIRCUIT_USE_NUMBA=1 python3 -m benchmarks.bench_afe --warm-runs 3
```

## JSON 电路描述

求解器现在支持从 JSON 加载通用电路描述，避免在 `core/*.py` 里硬编码节点名和器件名。格式说明见 [JSON 电路描述格式](json_circuit_format_zh.md)，示例见 `examples/single_stage.json`。

最小调用方式：

```python
import numpy as np

from core.circuit_loader import load_circuit_json
from core.ac_solver import ac_solve
from core.noise_solver import noise_analysis
from core.transient_solver import transient

spec = load_circuit_json("examples/single_stage.json")
freqs = np.logspace(0, 4, 121)

ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
noise = noise_analysis(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)

t = np.linspace(0, 1e-3, 100)
vin = np.full_like(t, spec.bias["VIN"])
tran = transient(spec.sizes, spec.bias, t, topo=spec.topology,
                 nf=spec.nf, inputs={"vin": vin})
```

JSON 主要字段包括 `solved`、`rails`、`devices`、`bias`、`outputs`、`input_drives`、`load_caps`、`dc_guesses` 和 `transient_inputs`。其中 `devices` 可直接包含 `W/L/NF`，也可以用单独的 `sizes`/`nf` 字段统一指定。

## 项目动机

模拟电路设计通常需要反复运行仿真器来调整晶体管尺寸、偏置电流和补偿元件。直接扫描的结果可靠，但速度较慢，尤其是在候选设计数量很多，或者需要检查工艺角和 mismatch 时。

这个项目采用互补的方式：

1. 使用 Cadence/Spectre 作为可信参考。
2. 建立能够匹配关键仿真行为的本地模型。
3. 使用本地求解器快速探索和优化。
4. 只把筛选后的候选设计送回 Cadence 验证。

这样可以更快地理解设计 trade-off，在正式仿真前缩小搜索空间，并获得更合理的候选设计。

## 优化方向

当前优化方法更适合称为 model-based design-space exploration，而不是严格意义上的 machine learning。它使用经过校准的 physics-based surrogate model 来快速评估候选设计。

已有和计划中的优化方法包括：

- 对尺寸和偏置变量进行随机全局搜索。
- 基于约束进行可行解筛选。
- 使用 greedy per-device shrink 在保持指标通过的同时减小面积。
- 使用 Pareto selection 分析面积-功耗或噪声-功耗 trade-off。
- 后续扩展到可微分 surrogate model 或 machine-learning surrogate model。

## 后续计划

后续计划包括：

- 支持更多 PDK 和晶体管紧凑模型。
- 支持更通用的电路拓扑描述。
- 拓展更高级的 DC、AC、瞬态和噪声求解器。
- 改进与仿真器数据的校准流程。
- 自动生成验证报告。
- 搭建交互式图形界面，用于查看设计 trade-off。
- 集成机器学习 surrogate model，用于更快的优化。

## 使用定位

这个仓库面向科研和早期模拟电路设计探索，不用于替代 sign-off 级别的电路仿真器。它的作用是在本地快速理解趋势、缩小搜索空间，并为 Cadence/Spectre 验证准备更好的候选设计。
