# 核心求解器概览

[项目概览](README.md) | [中文说明](README_zh.md)

本文介绍当前 `core/` 求解器栈。代码是 AT4000TG OTFT ECG AFE 求解器的紧凑本地实现，已针对 Cadence/Spectre 行为进行校准。它是更广泛的本地电路优化流程的第一个具体后端。

## 覆盖范围

当前求解器栈覆盖：

- DC 工作点求解。
- AC 小信号增益与带宽分析。
- 噪声分析，包括闪烁噪声和热噪声。
- 瞬态响应仿真。
- 工艺角与逐器件 mismatch 扰动。
- 面向 Cadence/Spectre 的验证，涵盖工作点、AC、噪声和瞬态行为。

实现刻意保持小而自包含。目前由 `core/` 下的十三个 Python 源文件组成（不含 `__init__.py`）。

## 文件结构

```text
core/
  topology.py          电路拓扑单一事实来源。
  compiled_topology.py 运行态拓扑/index/stamp 元数据编译层。
  circuit_loader.py    JSON 电路描述加载器。
  pmos_tft_model.py    AT4000TG PMOS-OTFT 紧凑模型实现。
  numba_kernels.py     可选 Numba 加速标量内核。
  ac_mna.py            MNA stamp 原语。
  ac_solver.py         DC 工作点与 AC 小信号求解器。
  noise_solver.py      噪声传播与等价输入噪声分析。
  chopper.py           理想与 PMOS 开关差分 chopper 分析。
  transient_solver.py  时域瞬态求解器。
  explore.py           设计空间探索 / 优化驱动。
  corners.py           工艺角、mismatch MC 与 latch 检测。
  mc_corners.py        Cadence PSF Monte-Carlo 后处理辅助脚本。
```

## 导入关系

```text
topology.py          <- 无内部依赖
compiled_topology.py <- 无内部依赖；运行时消费 Topology 风格对象
circuit_loader.py    <- topology
numba_kernels.py     <- 无内部依赖；运行时可选 numba
pmos_tft_model.py    <- 可选 numba_kernels
ac_mna.py            <- 无内部依赖
ac_solver.py         <- topology, compiled_topology, ac_mna, pmos_tft_model
noise_solver.py      <- ac_solver, compiled_topology, topology, ac_mna, pmos_tft_model
chopper.py           <- noise_solver, topology
transient_solver.py  <- ac_solver, compiled_topology, topology, pmos_tft_model
explore.py           <- ac_solver, noise_solver, pmos_tft_model, topology, circuit_loader
corners.py           <- ac_solver, noise_solver, topology
mc_corners.py        <- 仿真器侧 PSF 解析辅助逻辑
```

## 主要组件

### `pmos_tft_model.py`

实现了 AT4000TG PMOS-OTFT 紧凑模型的 Python 版本。提供：

- 通过 `get_Idc` 计算端电流。
- 通过 `get_noise_psd` 计算漏极电流噪声 PSD。
- 通过 `get_capacitances` 计算偏置相关的端电容。
- 通过 `g_area` 计算几何面积。
- 工艺和 mismatch 参数，如 `pvt0`、`mvt0`、`pbeta0` 和 `mbeta0`。
- 带热启动的内部节点工作点求解。
- 安装 Numba 时自动对热点标量内核启用加速；设置 `CIRCUIT_USE_NUMBA=0`
  可强制关闭。

AC 和噪声分析时，求解器通过有限差分 `get_Idc` 提取端 `gm` 和 `gds`，与电路求解器使用的端行为保持一致。

### `topology.py`

将电路拓扑定义为单一事实来源。拓扑包含晶体管列表、被求解节点列表、rail/bias 节点、输出、AC 输入驱动、负载电容、瞬态输入映射、DC 初值猜测和 DC 别名。求解器运行态元数据均从这个拓扑派生，而不是在各个求解器中分别手写。

除了 PMOS_TFT 晶体管之外，还承载两端无源/源元件——`resistors`（a-b，阻值 R 欧姆）、`capacitors`（a-b，容值 C 法拉）和 `isources`（理想直流电流源，I 从 nplus 流向 nminus）。这些通用于全部四种分析：电阻支路电流和电流源注入进入 DC KCL；电阻在 AC/噪声中按 `1/R` stamp，电容按 `jωC` stamp；电阻贡献热噪声 `4kT/R`；瞬态加入电导、电容伴随模型及恒定源电流。电流源在小信号 AC 系统中视为开路（且无噪声）。这些都不影响 PMOS_TFT 相关逻辑。

默认拓扑是 `AFE_TOPO`，一个 10 管全差分 AFE 核心，包含尾电流器件、输入对、输出级和交叉耦合正反馈电平移位器件。

### `compiled_topology.py`

从声明式 `Topology` 以及当前 bias/input 上下文构建运行态 plan。它会把节点名一次性解析成紧凑 terminal token，并为 DC、AC/噪声和 transient 暴露共享元数据：

- solved-node index 和 rail 数值；
- 每个器件的 drain/gate/source terminal token；
- 电阻、电容和电流源的 stamp 元数据；
- AC/噪声使用的 `("n", idx)` / `("v", value)` 端表；
- transient input 与 `node_inputs` 映射。

这样 AC、noise 和 transient 使用同一套 indexing/stamping 约定，同时仍能保持 JSON 电路替换能力。

### `circuit_loader.py`

加载 JSON 电路描述并返回 `CircuitSpec`，包含：

- `topology`
- `sizes`
- `bias`
- `nf`

这使得可以通过 JSON 文件（如 `examples/single_stage.json`）添加新电路，而无需修改求解器源码。

### `numba_kernels.py`

为纯标量热点路径提供可选 Numba 内核。该模块可在未安装 Numba 时安全导入。安装
Numba 时默认自动启用；如需强制走纯 Python 路径，设置：

```bash
CIRCUIT_USE_NUMBA=0
```

`core.explore` 和 `core.corners` 仍会默认把该变量设为 `1`，因为设计空间探索、
corner sweep 和 mismatch MC 都是长任务；普通 solver 路径现在也会在 Numba 可用时
自动使用加速内核。

目前加速路径包括 PMOS 电流计算、内部节点 Newton 迭代、偏置相关电容计算、端导数，
以及 transient Newton 内循环：拓扑 token 查值、PMOS 工作点求解、residual/Jacobian
stamp 和小规模稠密 Newton 线性求解。如果 compiled 路径处理不了某一步，
`transient_solver.py` 会回退到原 Python Newton / full-Jacobian / least-squares 路径。

### `ac_mna.py`

提供小信号求解器使用的底层 MNA stamp 原语：

- 导纳 stamp。
- VCCS stamp。
- MOS 小信号 stamp。

### `ac_solver.py`

求解 DC 工作点和 AC 响应：

- `ac_solve(sizes, bias, freqs, corner=None, x0_guess=None, topo=AFE_TOPO, nf=None)`
- 使用 `scipy.fsolve` 求解 DC 节点方程。
- 返回增益、带宽、节点工作点以及提取的小信号参数。
- 同时支持全局工艺角和逐器件 mismatch 映射。
- 使用拓扑元数据确定输出、负载电容和 AC 输入驱动。

DC 求解包含物理支路选择、对称工作点和 rail 有界节点解的鲁棒性处理。

### `noise_solver.py`

在与 AC 分析相同的拓扑派生 MNA 系统上执行噪声传播。每个晶体管漏极电流噪声源注入到漏源之间，传播到配置的输出，并除以信号增益得到等价输入噪声。

噪声流程支持与 AC 求解器相同的拓扑派生端映射和 corner/mismatch 参数传递。

### `chopper.py`

计算 AFE 周围不同 chopper 版本的 gain、带宽和基带噪声：

- `chopper_analysis(...)` 是理想同步差分 chopper 模型，把八开关换向器看作输入
  和输出端的 +/-1 方波乘法器，再用奇次谐波系数把边带 gain/noise 折回基带。
  这是描述理想 chopping 与 flicker noise 搬移的 LPTV 频域路径。
- `build_afe_pmos_chopper(...)` 会在 AFE 输入/输出端口周围插入 8 个真实
  `PMOS_TFT` pass switch。
- `pmos_chopper_analysis(...)` 对这个 PMOS 开关拓扑分别运行静态 A/B 相 AC
  和 noise，并对两相平均；结果包含 switch Ron 负载、非线性电容和 PMOS
  switch 自身噪声。
- `finite_edge_clock_pair(...)` 与 `finite_edge_chopper_harmonics(...)` 建模有限
  clock edge 和 break-before-make dead time 对 chopper 谱线权重的影响。
- `pmos_chopper_lptv_analysis(...)` 用这些有限边沿谐波权重折叠 PMOS-switch
  sideband response/noise，是时变开关工作点的 quasi-static LPTV 近似。
- `pmos_chopper_transient(...)` 用有限边沿 clock 驱动八 PMOS 拓扑。默认 clock
  采用 Spectre `type=pulse` 语义（`delay=T/2`、`width=T/2`、有限 `rise/fall`）；
  旧的居中相位波形仍可通过 `clock_style="phase"` 使用，适合 dead-time 实验。
  clock feedthrough 来自 PDK `Cgss/Cgdd * ddt()` 项以及 PDK Verilog-A 中长期有效的
  `R_cap2` gate-leak 分支，均由 transient solver stamp；可选 charge injection
  脉冲由同一套 PDK 电容公式估算，并作为时变电流源注入。这个 helper 会在 clock
  边沿附近自动加密内部时间网格，对 8 个双向 pass switch 使用 signed terminal
  current，并收紧残差容差，避免慢 common-mode 电荷平衡被忽略。

PMOS-switch sideband 路径仍是 quasi-static 近似，不是完整相关 periodic-noise
求解器。瞬态 finite-edge 已与 Spectre `tran` 对齐；周期噪声仍应继续与 Spectre
PSS/PNoise 的 clocked testbench 对齐验证。

### `transient_solver.py`

使用后向欧拉积分求解拓扑定义系统的时域响应：

- `transient(sizes, bias, tgrid, vip=None, vin=None, nf=None, V0=None, topo=AFE_TOPO, inputs=None, node_inputs=None)`
- 支持传统的 AFE `vip/vin` 输入，也支持通过 `topo.transient_inputs` 驱动的通用 `inputs={name: waveform}`。
- `node_inputs={node: input_key}` 在某个（rail）节点上驱动波形——用于前端 testbench，其激励在源节点注入并通过无源网络传播，而非直接驱动器件栅极。
- `current_inputs=[{"p": node_a, "q": node_b, "input": key}]` stamp 一个时变
  理想电流源，方向为 `p -> q`；PMOS chopper helper 用它注入 charge-injection 脉冲。
- `max_step`、`max_retry_subdivisions`、`fallback_full_jacobian` 和
  `fallback_least_squares` 用于 switched
  transient 步的受控细分和有界 fallback 求解。
- 包含拓扑定义的负载电容（及电容元件），加上电阻和理想电流源支路。
- 在牛顿迭代期间重新计算非线性电容，并包含 PDK Verilog-A 使用的 PMOS
  `R_cap2` 源/漏到 gate 的泄漏支路。
- 支持 `signed_devices`，用于双向 pass switch。默认 AFE 路径保持与已校准
  DC/AC/noise 求解器一致的 `abs(Idc)` 约定；开关器件则可在源漏电压反向时保留
  物理 drain-current 符号。
- 使用来自 `ac_solve` 的 DC 工作点作为默认初始条件。
- Numba 可用时使用 transient Newton compiled kernel。该路径在一个内循环里完成
  PMOS 工作点/电容计算、residual/Jacobian stamp 和稠密 Newton step 求解；如果
  某个器件内部解或矩阵求解无法处理，会回退到 Python 路径。
- 使用 PMOS 内部节点的隐式微分加快瞬态 Jacobian 计算，并回退到有限差分。

### 前端激励（`ac_drives`）

对于 testbench，小信号 AC 激励可以通过 `Topology.ac_drives`（如 `{"VINP": +0.5, "VINN": -0.5}`）施加在节点上，而非器件栅极。驱动通过前端无源网络传播到（现在作为被求解节点的）放大器输入端，增益按差分激励归一化。噪声分析中这些驱动被视为 AC 地（输入端无信号）。`examples/afe_testbench.py` 在 AFE 核心之前构造了干电极 + AC 耦合前端（R_EL∥C_EL、C_AC 串联、R_AC 到 VCM），并运行 AC（带通约 0.05 Hz–几百 Hz）、等价输入噪声（含 R_EL/R_AC 热噪声）和带内瞬态。由于 AC 耦合输入使裸 AFE DC 多稳态，testbench 从鲁棒的裸 AFE 工作点（`dc_seed`）作为种子启动 DC 求解。

### `explore.py`

建立在 AC 和噪声求解器之上的设计空间探索/优化驱动——即项目名称所指的"优化"。给定一个电路及 `explore` 配置（带范围的设计变量、可行性约束和一个或多个目标），它对候选方案进行采样，通过求解器评估每个候选，按约束过滤，并 Pareto 选择权衡前沿。

- `explore(topo, base_sizes, base_bias, nf, cfg, n=, seed=, method=, corner=)`——运行一次扫描。
  `corner` 对每次评估施加工艺偏移（如 `CORNERS["slow"]`），实现在不修改配置的情况下进行 corner 感知搜索。
- `evaluate(topo, sizes, bias, nf, freqs, band, x0_guess=None, corner=None)`——单候选求解器评估，
  新增可选的 corner/mismatch 参数。在 `explore` 中评价流程为 AC-first：先计算
  gain/BW/power/area，非噪声约束失败的候选会立即淘汰；只有幸存候选的约束或目标
  需要 `irn_uV` 时才运行 `noise_analysis`。
- `load_explore_json(path)`——从完整电路 JSON 中读取 `explore` 块，或者从指定 `builtin_topology`（如 `AFE_TOPO`）加上基线 sizes/bias 的文件中读取。
- 采样方式为 `lhs`（拉丁超立方）或 `random`，使用带种子的 RNG 保证可重复性。
- 指标：`gain_dB`、`bw_Hz`、`irn_uV`、`power_uW`（顶 rail 供电电流 × rail 电压）和 `area`（各器件 `g_area` 之和）。
- 变量的 `targets` 可以同时驱动多个键值，保持匹配对（M7=M8, …）一致，使 AFE 的对称 DC 续流保持在物理支路上。
- 结果导出为 CSV 和 JSONL；CLI 运行 `python -m core.explore <config.json>`。

示例配置：`examples/afe_explore.json`（内置 AFE 拓扑）和 `examples/single_stage.json` 中的 `explore` 块（通用 JSON 路径）。

### `corners.py`

工艺角和鲁棒性工作的单一事实来源——这些内容原本会在每次扫描中重复推导：

- `CORNERS`——全局工艺偏移（`typical` / `slow` / `fast`，按 `pvt0`/`pbeta0` 表示），来源于 PDK 的 monte.scs 段落；如 slow = `{"pvt0": -0.2259, "pbeta0": -0.54}`。
- `mismatch_corner(rng, devices, base)`——在工艺角基础上叠加逐器件随机 `mvt0`/`mbeta0`。
- `metrics(...)`——单设计单 corner → `gain_peak_dB`、`bw_Hz`、`irn_uV` 和 `latch_dV`（DC 工作点的 `|out+ - out-|`；大值 ⇒ 交叉耦合正反馈已 latch）。
- `corner_table(...)`——typ/slow/fast 三个 corner 的指标汇总。
- `latch_screen(...)`——确定性最坏情况 latch 筛查：对每个对称对在所有符号组合上施加 ±kσ 推开，返回最大输出失衡。单次固定 kick 存在假阴性（latch 的符号模式依赖于设计），因此筛查遍历所有模式；计算开销足够低，可在搜索内部代替完整 MC 使用。它只需要 DC/AC 工作点和 latch 失衡，因此会跳过噪声。
- `mismatch_mc(...)`——单个 corner 上的逐器件 mismatch MC，从名义工作点播种；返回各指标数组、latch 掩码以及汇总（latch 率 + 未 latch 样本的 mean/std/P5/P95）。每个样本先跑 AC/latch，只有进入最终噪声统计的未 latch 样本才计算 IRN。

`ac_solve` / `noise_analysis` 接受相同的 `corner` 参数（扁平的工艺 dict 或逐器件 mismatch 映射）。驱动脚本 `examples/mc_mismatch.py` 将其封装为 corner 表 + 3-corner MC 图。（与 `core/mc_corners.py` 不同，后者是 Cadence PSF 输出的后处理——那是仿真器侧的流程，这是本地求解器侧的。）

## 快速示例

```python
import numpy as np

from core.ac_solver import ac_solve
from core.noise_solver import noise_analysis, band_rms
from core.transient_solver import transient

sizes = {
    "M6": (2264, 78),
    "M7": (61365, 61),
    "M8": (61365, 61),
    "M9": (3175, 468),
    "M10": (3175, 468),
    "M11": (465, 66),
    "M12": (894, 85),
    "M13": (894, 85),
    "M14": (5224, 46),
    "M15": (5224, 46),
}

bias = {
    "VDD": 40.0,
    "VCM": 30.65,
    "VB": 9.84,
    "VC": 16.0,
}

freqs = np.logspace(-2, 4, 121)

ac = ac_solve(sizes, bias, freqs)
noise = noise_analysis(sizes, bias, freqs)
irn_uv = band_rms(freqs, noise["irn_psd"], 0.05, 100) * 1e6

t = np.linspace(0, 4e-3, 400)
vip = np.where(t >= 0.5e-3, bias["VCM"] + 0.5e-3, bias["VCM"])
vin = np.where(t >= 0.5e-3, bias["VCM"] - 0.5e-3, bias["VCM"])
tran = transient(sizes, bias, t, vip, vin)
```

## JSON 电路示例

新电路可以从 JSON 加载。字段级格式见 [JSON 电路描述格式](json_circuit_format_zh.md)。

```python
import numpy as np

from core.circuit_loader import load_circuit_json
from core.ac_solver import ac_solve
from core.transient_solver import transient

spec = load_circuit_json("examples/single_stage.json")
freqs = np.logspace(0, 4, 121)

ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)

t = np.linspace(0, 1e-3, 100)
vin = np.full_like(t, spec.bias["VIN"])
tran = transient(spec.sizes, spec.bias, t, topo=spec.topology,
                 nf=spec.nf, inputs={"vin": vin})
```

## 基准测试

固定 AFE 基准位于 `benchmarks/` 下：

```bash
python3 -m benchmarks.bench_afe --warm-runs 3
CIRCUIT_USE_NUMBA=0 python3 -m benchmarks.bench_afe --warm-runs 3
```

脚本分别报告 cold 和 warm 的 `ac121`、`noise121`、`tran200` 计时。默认运行在
Numba 可用时启用加速；`CIRCUIT_USE_NUMBA=0` 可用于纯 Python 对比。

## 校准状态

当前核心已针对 AT4000TG AFE 用例在 Cadence Spectre 24.1 上完成校准。原始项目中观察到的吻合度包括：

- 典型和 corner AC 行为增益误差约 0.01 dB 以内。
- 已验证场景中等价输入噪声误差在百分之几以内。
- 逐器件 mismatch Monte Carlo 的均值和标准差与 Cadence 趋势一致。
- 瞬态阶跃和正弦响应与 Cadence `tran` 行为高度吻合。
- PMOS 八开关 chopper finite-edge transient 已按 UI 锁定尺寸、`f_chop=225 Hz`、
  switch `W/L=5000/30`、`rise/fall=20 us` 与 Spectre `tran` 对齐：最后一周期
  输出均值约 `-11.05 mV`（Spectre `-10.62 mV`），输出 `21.28 mVpp`
  （Spectre `21.46 mVpp`），输入 common-mode 摆幅 `5.18 Vpp`
  （Spectre `5.43 Vpp`），且 `nfail=0`。
- 最终锁定设计约 22.9 dB 增益、549 Hz 带宽、37 µVrms 等价输入噪声。

上述数据描述当前的 AT4000TG 验证案例。后续 PDK 或拓扑应针对其各自的仿真器参考重新进行校准。
