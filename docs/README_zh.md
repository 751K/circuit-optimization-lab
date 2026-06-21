# 电路本地建模与优化流程

[English](README.md) | [中文说明](README_zh.md)

## 项目概述

本地 Python 电路求解器，用于模拟电路设计空间探索，已对 Cadence/Spectre 完成标定。首个应用场景是 **AT4000TG PMOS 薄膜晶体管 ECG AFE**（带 chopper 的心电模拟前端放大器）。

你能用它做什么：

- **DC / AC / Noise / Transient** — 标准电路分析，无需仿真器 license。
- **PSS / PAC / PNoise** — 周期稳态、周期 AC、周期噪声分析（对标 Spectre RF 分析）。
- **设计空间探索** — 扫描器件尺寸和偏置电压，按约束过滤（增益、带宽、噪声、功耗、面积），找到 Pareto 最优设计。
- **工艺角与失配** — 全局工艺角、逐器件 mismatch Monte Carlo、latch 筛查。

求解器内部实现见 [核心求解器概览](core_overview_zh.md)。

---

## 快速上手

```bash
# 1. 安装
python3 -m pip install -r requirements.txt

# 2. 可选：Numba 加速（transient 可提速 10–50 倍）
python3 -m pip install -r requirements-numba.txt

# 3. 运行第一个电路 — 一条命令
python3 -m core examples/periodic_rc.json

# 4. 验证安装 — 跑 AFE 基准
python3 -m benchmarks.bench_afe --warm-runs 1 --skip-noise
```

上面第三条命令会对一个无源 RC 低通电路运行 AC、Noise、PSS、PAC、PNoise 分析并输出摘要。
无需编写任何 Python 代码。打印出数字即说明一切正常。之后可替换任意电路 JSON 文件，
或用 `-a ac,noise` 选择特定分析。

### 一分钟搞懂代码结构

在看具体工作流之前，先搞清楚几个核心概念：

| 概念               | 是什么                                  | 在哪定义                                             |
| ---------------- | ------------------------------------ | ------------------------------------------------ |
| **Topology（拓扑）** | 电路结构——有哪些节点、器件如何连接、输入输出在哪            | `core/topology.py`，或从 JSON 自动生成                  |
| **Sizes（尺寸）**    | `{器件名: (W_µm, L_µm)}`——晶体管宽长         | JSON `sizes` 字段                                  |
| **NF**           | Number of fingers（晶体管并联数，等比例放大电流）    | JSON `nf` 字段，或每个器件单独指定 `devices[].NF`            |
| **Bias（偏置）**     | `{节点名: 电压}`——各 rail 节点的 DC 工作电压      | JSON `bias` 字段                                   |
| **Solver（求解器）**  | 接收 拓扑 + 尺寸 + 偏置 → 输出结果（增益、噪声、波形…）的函数 | `core/ac_solver.py`、`core/transient_solver.py` 等 |

所有求解器的调用模式都一样：

```python
result = solver(sizes, bias, ..., topo=topology, nf=nf)
```

JSON 文件把这些输入打包在一起；`load_circuit_json()` 解包成 `CircuitSpec` 对象，
包含 `.topology`、`.sizes`、`.bias`、`.nf` 以及可选的 `.explore`。

---

## 常用工作流

以下所有示例均可直接复制运行。它们使用的都是 `examples/afe_explore.json`
中已与 Cadence 对标验证的锁定 AFE 设计。

### 1. 加载电路 + DC / AC / Noise 分析

```python
import numpy as np
from core.circuit_loader import load_circuit_json
from core.ac_solver import ac_solve
from core.noise_solver import noise_analysis, band_rms

# 从 JSON 加载电路 —— 求解器代码里不硬编码任何节点名
spec = load_circuit_json("examples/afe_explore.json")
freqs = np.logspace(-2, 4, 121)   # 0.01 Hz 到 10 kHz

# DC 工作点 + AC 增益 / 带宽
ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
print(f"增益: {ac['Av_dc_dB']:.2f} dB,  带宽: {ac['bw_Hz']:.1f} Hz")
# → 增益: 22.89 dB,  带宽: 549.3 Hz

# 噪声分析（热噪声 + 闪烁噪声）
noise = noise_analysis(spec.sizes, spec.bias, freqs,
                       topo=spec.topology, nf=spec.nf)
irn_uv = band_rms(freqs, noise["irn_psd"], 0.05, 100.0) * 1e6
print(f"IRN (0.05–100 Hz): {irn_uv:.2f} µVrms")
# → IRN (0.05–100 Hz): 36.97 µVrms
```

### 2. 瞬态仿真

```python
from core.transient_solver import transient

# 4 ms 仿真，在 t=0.5 ms 处施加 0.5 mV 差分阶跃
t = np.linspace(0, 4e-3, 400)
vip = np.where(t >= 0.5e-3, 30.65 + 0.5e-3, 30.65)
vin = np.where(t >= 0.5e-3, 30.65 - 0.5e-3, 30.65)

tran = transient(spec.sizes, spec.bias, t, vip, vin,
                 topo=spec.topology, nf=spec.nf)
print(f"瞬态步数: {len(t)},  失败步数: {tran['nfail']}")
# → 瞬态步数: 400,  失败步数: 0
```

### 3. Chopper 分析（三种精度层级）

#### 层级 1 — 理想 LPTV（最快，方波乘法器模型）

```python
from core.chopper import chopper_analysis

chop_ideal = chopper_analysis(
    spec.sizes, spec.bias, freqs, f_chop=225.0,
    topo=spec.topology, nf=spec.nf, max_harmonic=31,
    band=(0.05, 100.0))
print(f"理想 chopper: {chop_ideal['peak_dB']:.2f} dB,  "
      f"IRN: {chop_ideal['irn_uV_band']:.2f} µVrms")
```

#### 层级 2 — PMOS 开关（静态两相分析，无需 PSS）

```python
from core.chopper import pmos_chopper_analysis

pmos = pmos_chopper_analysis(
    spec.sizes, spec.bias, freqs,
    switch_size=(20000, 80), band=(0.05, 100.0))
print(f"PMOS 静态 chopper: {pmos['peak_dB']:.2f} dB,  "
      f"IRN: {pmos['irn_uV_band']:.2f} µVrms")
```

#### 层级 3 — 完整 PSS / PAC / PNoise（第一性原理，对标 Spectre）

```python
from core.chopper import (pmos_chopper_pss, pmos_chopper_pac,
                           pmos_chopper_pnoise)

# 第一步：PSS — 求解周期稳态轨道
pss = pmos_chopper_pss(
    spec.sizes, spec.bias, f_chop=225.0,
    switch_size=(5000, 30), edge_time=20e-6,
    tstab_periods=2, n_points=121)
print(f"PSS 收敛: {pss['converged']},  "
      f"残差: {pss['residual_norm']:.2e}")

# 第二步：PAC — 在 PSS 轨道上做周期 AC 增益分析
pac = pmos_chopper_pac(
    spec.sizes, spec.bias, freqs, f_chop=225.0,
    pss_result=pss)
print(f"PAC 增益: {pac['Av_dc_dB']:.2f} dB,  带宽: {pac['bw_Hz']:.1f} Hz")

# 第三步：PNoise — 周期噪声（谐波平衡法，无需标定常数）
pnoise = pmos_chopper_pnoise(
    spec.sizes, spec.bias, freqs, f_chop=225.0,
    pss_result=pss, pac_result=pac, max_sideband=10,
    band=(0.05, 100.0))
print(f"PNoise IRN: {pnoise['irn_uV_band']:.2f} µVrms")
```

PSS→PAC→PNoise 三件套是 Cadence Spectre `pss` + `pac` + `pnoise` 的本地等价实现。
PAC 默认使用解析伴随谐波平衡内核：在 PSS 轨道转换矩阵上每频率一次伴随线性求解，
零额外瞬态运行。设置 `analytic=False` 可回退到原有限差分 shooting 路径（精度优先，
每频点需 `n_state+2` 次瞬态周期）。PNoise 在 PSS 轨道上做谐波平衡——这是第一性
原理的 LPTV 噪声解，**不需要任何 Cadence 标定常数**。对 D3 / `chop_tb_d3`
官方 `slow` corner Spectre 参考，在 `f_chop=200 Hz`、
PNoise `maxsideband=10`、dec=10 噪声网格下，原生 PAC 增益和 PNoise IRN 误差均 <1%。
`pmos_chopper_pac` / `pmos_chopper_pnoise` 是 chopper 兼容包装器；任意周期拓扑可直接使用
`core.pac_solver.pac_solve` 和 `core.pnoise_solver.pnoise_solve`，输入为通用 `pss_solve`
返回的周期轨道和 `input_drive` 映射。

**JSON dispatch** — 当电路 JSON 包含 `periodic` 和 `analyses` 块时，一个调用即可运行所有分析：

```python
from core.analysis_dispatch import run_analysis_suite
from core.circuit_loader import load_circuit_json

spec = load_circuit_json("examples/periodic_rc.json")
results = run_analysis_suite(spec)
# results["pss"]、results["pac"]、results["pnoise"] 全部就绪
```

### 4. 设计空间探索 / 优化

```python
from core.explore import explore
from core.circuit_loader import load_circuit_json

spec = load_circuit_json("examples/afe_explore.json")

# JSON 中的 "explore" 块定义了设计变量、约束和目标。
# explore() 采样候选点，通过求解器逐个评估，按约束过滤，返回 Pareto 前沿。
result = explore(spec.topology, spec.sizes, spec.bias, spec.nf,
                 spec.explore, n=500, method="lhs", seed=42)

print(f"候选总数: {result['n_total']},  "
      f"可行解: {result['n_feasible']},  "
      f"Pareto 最优: {len(result['pareto'])}")
# → 候选总数: 500,  可行解: 87,  Pareto 最优: 12
```

也支持命令行：

```bash
python -m core.explore examples/afe_explore.json --n 500 --seed 42
```

结果导出为 CSV 和 JSONL。JSON 中的 explore 配置指定了扫描哪些变量（器件 W/L、偏置电压）、
什么约束条件（增益 > X、IRN < Y 等），以及优化什么目标。

### 5. 工艺角与失配分析

```python
from core.corners import CORNERS, corner_table, mismatch_mc, latch_screen
import numpy as np

# 工艺角扫描 — 一个设计在 typ/slow/fast 下的指标
table = corner_table(spec.sizes, spec.bias, np.logspace(-2, 4, 121),
                     topo=spec.topology, nf=spec.nf)
for row in table:
    print(f"{row['corner']:>6s}:  增益={row['gain_peak_dB']:.2f} dB,  "
          f"BW={row['bw_Hz']:.0f} Hz,  IRN={row['irn_uV']:.2f} µVrms")
# → typical:  增益=22.89 dB,  BW=549 Hz,  IRN=36.97 µVrms
# →   slow:  增益=20.81 dB,  BW=328 Hz,  IRN=45.72 µVrms
# →   fast:  增益=24.41 dB,  BW=846 Hz,  IRN=28.40 µVrms

# 快速 latch 筛查（确定性方法，速度足够快，可嵌入搜索内循环）
rng = np.random.default_rng(0)
latch = latch_screen(spec.sizes, spec.bias, topo=spec.topology,
                     nf=spec.nf, rng=rng, k_sigma=3.0)
print(f"Latch dV: {latch['latch_dV']*1e3:.2f} mV  "
      f"({'已 latch' if latch['latched'] else '正常'})")

# 完整 mismatch Monte Carlo（较慢，用于最终验证）
mc = mismatch_mc(spec.sizes, spec.bias, np.logspace(-2, 4, 61),
                 topo=spec.topology, nf=spec.nf, n=200,
                 corner=CORNERS["typical"], seed=1)
print(f"Latch 率: {mc['latch_rate']*100:.1f}%,  "
      f"IRN: {mc['irn_mean']:.2f} ± {mc['irn_std']:.2f} µVrms")
```

---

## JSON 电路格式

新电路通过 JSON 定义，无需修改求解器源码。完整字段参考见
[JSON 电路描述格式](json_circuit_format_zh.md)。

快速示例 (`examples/single_stage.json`)：

```json
{
  "solved": ["OUT"],
  "rails": {"VDD": 40.0, "GND": 0.0},
  "devices": [
    {"name": "M1", "drain": "OUT", "gate": "IN", "source": "VDD",
     "W": 2000, "L": 80, "NF": 1}
  ],
  "bias": {"VDD": 40.0, "VIN": 30.0, "VB": 10.0},
  "outputs": ["OUT"],
  "input_drives": {"IN": 1.0},
  "load_caps": {"OUT": 1e-12}
}
```

主要字段说明：

| 字段                 | 必填  | 用途                                                                                                    |
| ------------------ | --- | ----------------------------------------------------------------------------------------------------- |
| `solved`           | 是   | 求解器需要求解电压的节点列表                                                                                        |
| `rails`            | 是   | 固定电压节点：`{"VDD": 40.0, "GND": 0.0, ...}`                                                               |
| `devices`          | 是   | PMOS 晶体管；无源电路可写空数组 `[]`                                                                              |
| `bias`             | 是   | 每个 rail 节点的 DC 电压：`{"VDD": 40.0, "VIN": 30.0, ...}`                                                   |
| `outputs`          | 是   | 观测增益/噪声的输出节点                                                                                          |
| `input_drives`     | —   | AC 小信号激励注入位置（驱动器件栅极）                                                                                  |
| `load_caps`        | —   | 各输出节点的负载电容 (F)：`{"OUT": 1e-12}`                                                                       |
| `resistors`        | —   | `[名称, 节点A, 节点B, 阻值]`                                                                                  |
| `capacitors`       | —   | `[名称, 节点A, 节点B, 容值]`                                                                                  |
| `current_sources`  | —   | 理想直流电流源：`[名称, nplus, nminus, 电流]`                                                                     |
| `nf`               | —   | 全局 NF（fingers），作用于所有器件；可被器件自身的 `NF` 覆盖                                                                |
| `dc_guesses`       | —   | DC 收敛的初始电压猜测，复杂电路需要此字段帮助收敛                                                                            |
| `transient_inputs` | —   | 瞬态输入波形名到驱动节点的映射                                                                                       |
| `ac_drives`        | —   | 类似 `input_drives`，但驱动的是*节点*而非器件栅极（用于 testbench 前端网络）                                                  |
| `periodic`         | —   | PSS/PAC/PNoise 和周期 transient 使用的大信号周期输入描述                                                                |
| `analyses`         | —   | `run_analysis_suite()` 的分析 dispatch 配置：`ac/noise/transient/pss/pac/pnoise`                                     |
| `aliases`          | —   | 节点别名，方便工具/扫描按名称找到关键节点（如 `"VOP"`、`"VON"`）                                                              |
| `explore`          | —   | 设计空间探索配置（变量范围、约束条件、优化目标）                                                                              |

---

## 交互式 AFE Tuner

基于 Web 的实时调参工具：

```bash
python3 -m pip install -r requirements-demo.txt
python3 demo/server.py
# 浏览器打开 http://localhost:5100
```

在浏览器中调整器件 W/L 和偏置电压，实时查看增益、带宽和等价输入噪声变化。
内置预设设计（Base、Final Locked、Min Area、First Feasible）。

---

## 性能基准

四个固定性能基准，用于性能回归跟踪：

```bash
python3 -m benchmarks.bench_afe --warm-runs 3         # AC+noise+transient
python3 -m benchmarks.bench_model --warm-runs 3       # 单管微基准
python3 -m benchmarks.bench_chopper --warm-runs 3     # Chopper: 5 个分析层级
python3 -m benchmarks.bench_sweep --n-candidates 200  # 批量 explore 负载
```

设置 `CIRCUIT_USE_NUMBA=0` 可对比纯 Python 性能。Numba 内核默认使用磁盘缓存，
后续新的 Python 进程可避免大部分重复冷启动 JIT 开销；设置 `CIRCUIT_NUMBA_CACHE=0`
可关闭该缓存。MacMini M4 上 Numba 预热后的典型耗时：

| 基准                                      | 耗时      |
| --------------------------------------- | ------- |
| AC 121 点                                | ~1.5 ms |
| Noise 121 点                             | ~1.7 ms |
| Transient 200 步                         | ~5 ms   |
| 理想 chopper（31 次谐波）                      | ~5 ms   |
| PMOS chopper LPTV                       | ~22 ms  |
| Chopper transient（8-PMOS, 225 Hz, 8 周期） | ~0.6 s  |
| 批量 sweep（200 候选, AC+noise）              | ~0.5 s  |

---

## 示例文件

| 文件                                  | 说明                                                |
| ----------------------------------- | ------------------------------------------------- |
| `examples/afe_explore.json`         | 锁定 10 管 AFE 设计，含尺寸、偏置、NF 和 explore 扫描配置           |
| `examples/single_stage.json`        | 最小单管共源级——新建电路的最佳起点                                |
| `examples/resistor_load_stage.json` | 带电阻负载的单管电路，演示 `resistors` 和 `current_sources` 字段         |
| `examples/periodic_rc.json`         | 无源 RC 低通，带 PSS/PAC/PNoise dispatch——最简单的端到端周期示例           |
| `examples/afe_testbench.py`         | 完整 testbench：干电极前端（R∥C 网络）→ AFE 核心 → AC + 噪声 + 瞬态 |
| `examples/mc_mismatch.py`           | Monte Carlo mismatch 驱动：工艺角表 + 3-corner MC 图      |

---

## 常见问题

**DC 求解不收敛。**
先用 `examples/single_stage.json`（单管，必然收敛）。对复杂电路，在 JSON
中加上 `dc_guesses`——一个近似节点电压的字典。锁定 AFE 的 JSON 里就包含了这些猜测值。

**Transient 出现 `nfail > 0`。**
部分 Newton 步失败。尝试：(a) 增加时间点 `np.linspace(0, T, more_steps)`；
(b) 收紧 `newton_vtol`（默认 `1e-8`）；(c) 启用 `fallback_least_squares=True`。
对开关电路，确保 `max_step` 小于最快的边沿时间。

**PSS 不收敛（`converged=False`）。**
增加 `tstab_periods`（shooting 前的额外稳定周期），或降低 `max_shooting_iters`。
检查 `pss['shooting_history']`，看残差是否在下降。如果停滞，可能是轨道本身非周期——
检查所有输入波形是否周期相同、周期一致。

**PNoise 太慢。**
减少 `max_sideband`（奇次谐波主导折叠，5–7 通常够了），或降低 `n_period_samples`
（用时域分辨率换速度）。扫不同输出带宽或重复频点时复用同一个 `pss_result`：
PNoise 现在会缓存 LPTV 线性化、HB block 和相同频点的 adjoint 解。安装
Numba 时，大规模 HB block 组装、噪声折叠和 gm/gds 线性化也会走编译内核。

**PSS / 周期 transient 太慢。**
首先确保使用默认的 `analytic_jacobian=True` — 它将 shooting Jacobian 构建
从 `n_state` 次有限差分瞬态降为一次轨道遍历。Chopper PSS 现在默认使用
`fallback_least_squares=False`，这样完整周期会留在 Numba grid solver 内，
失败 interval 会被记录，但不会把整个周期退回 Python 重跑。只有在排查困难
收敛问题时才建议手动打开 `fallback_least_squares=True`。PMOS chopper wrapper
通常 1 个 stabilization 周期就够；继续增加主要是吞吐量取舍。

**PAC 太慢。**
普通运行不要设置 `compute_condition`。PAC condition 诊断只会在 `profile=True`、
`debug=True` 或显式 `compute_condition=True` 时计算，因为这个诊断每个频点都要
对 HB 矩阵做一次 SVD，不影响 gain/BW/noise 结果。

---

## 延伸阅读

| 文档                                       | 作用                     |
| ---------------------------------------- | ---------------------- |
| [核心求解器概览](core_overview_zh.md)           | 理解每个求解器的原理、导入依赖关系和标定数据 |
| [JSON 电路描述格式](json_circuit_format_zh.md) | JSON 的字段级参考            |
| [后续开发计划](futureplan.md)                  | 了解已完成、待做事项和执行路线图       |
| `tests/` 目录                              | 每个 API 调用的可运行示例，带预期输出  |
| `benchmarks/` 目录                         | 性能基线及 Numba 加速对比       |

---

## 项目动机

模拟电路设计需要反复跑仿真来调整晶体管尺寸和偏置。Cadence/Spectre 精确但慢，
尤其是有大量候选设计或需要检查工艺角和 mismatch 时。

本项目的工作流：

1. **Cadence/Spectre** = 可信参考。
2. **本仓库** = 经标定匹配 Spectre 行为的本地快速模型。
3. **本地探索** — 扫描尺寸、偏置、工艺角；用约束过滤。
4. **Cadence 验证** — 只把最优候选送回 Spectre 最终确认。

---

## 参与贡献

欢迎提 Issue 和 PR。

---

## 使用定位

面向科研和早期模拟电路设计探索。**不是** sign-off 级仿真器替代品。
用于在本地快速理解设计趋势、缩小搜索空间、为 Cadence/Spectre 验证准备更优候选。
