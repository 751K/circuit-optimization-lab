# JSON 电路描述格式

[English](json_circuit_format.md) | [中文说明](README_zh.md) | [核心求解器概览](module_overview_zh.md)

> **文档状态：持续维护。** Loader 和 `schemas/circuit.schema.json` 是单一事实来源；
> 修改任一字段时必须同步更新本文。

## 目的

JSON 电路描述用于把电路拓扑、尺寸、偏置和分析元数据从 Python 源码中抽出来。这样换电路时优先修改 JSON，而不是修改 `circuitopt/ac_solver.py`、`circuitopt/noise_solver.py` 或 `circuitopt/transient_solver.py`。

当前 schema 文件位于：

```text
schemas/circuit.schema.json
```

当前示例文件位于：

```text
examples/single_stage.json
examples/resistor_load_stage.json
examples/afe_explore.json
examples/periodic_rc.json
```

## 最小结构

一个电路 JSON 至少需要：

```json
{
  "solved": ["OUT"],
  "rails": {
    "VDD": "VDD",
    "GND": 0.0,
    "IN": "VIN"
  },
  "devices": [
    {"name": "M1", "drain": "OUT", "gate": "IN", "source": "VDD", "W": 2000, "L": 80}
  ],
  "bias": {
    "VDD": 40.0,
    "VIN": 25.0
  },
  "outputs": ["OUT"]
}
```

其中：

- `solved` 是求解器要求解的未知节点。
- `rails` 是已知节点。值可以是数字常量，也可以是 `bias` 里的键。
- `devices` 是晶体管器件列表（当前为 PMOS_TFT；模型类型由 ``device_model`` 工厂决定，默认 ``"pmos_tft"``）。
- `bias` 给出 rail 引用的偏置电压。
- `outputs` 指定 AC/noise/transient 的输出观测节点。

## 字段说明

### `name`

可选。电路名称，仅用于显示和记录。

```json
"name": "single_stage_pmos_load"
```

### `solved`

必填。未知节点列表，顺序就是 MNA/DAE 求解向量的节点顺序。

```json
"solved": ["VOP", "VON", "VFBP", "VFBN", "NET20", "NET2"]
```

要求：

- 至少一个节点。
- 节点名不能重复。
- `outputs` 中的节点必须来自 `solved`。

### `rails`

必填。已知节点映射。

```json
"rails": {
  "VDD": "VDD",
  "GND": 0.0,
  "VB": "VB"
}
```

含义：

- `"GND": 0.0` 表示 GND 恒为 0 V。
- `"VDD": "VDD"` 表示节点 VDD 的电压从 `bias["VDD"]` 读取。
- 器件端口引用的节点必须在 `solved` 或 `rails` 中出现。

### `devices`

必填。每个 active device 都是三端晶体管（drain/gate/source）。模型实现由 ``device_model`` 工厂选择，默认为 ``"pmos_tft"``。

推荐对象写法：

```json
{
  "name": "M7",
  "drain": "VOP",
  "gate": "VCM",
  "source": "NET2",
  "W": 61365,
  "L": 61,
  "NF": 1
}
```

也支持简写数组：

```json
["M7", "VOP", "VCM", "NET2"]
```

如果使用数组写法，必须在 `sizes` 中提供 W/L。

### `sizes`

可选。器件尺寸映射，适合把 topology 和 sizing 分开。

```json
"sizes": {
  "M7": [61365, 61],
  "M8": [61365, 61]
}
```

规则：

- 如果 `devices` 对象里已经写了 `W` 和 `L`，可以不写 `sizes`。
- 如果两处都写，`sizes` 中的值会覆盖 `devices` 内嵌的 W/L。
- 所有器件最终都必须有 W/L。

### `nf`

可选。指定 finger 数。

全局写法：

```json
"nf": 2
```

逐器件写法：

```json
"nf": {
  "M7": 4,
  "M8": 4
}
```

也可以在 device 对象里写 `NF`。如果同时存在，顶层 `nf` 会覆盖同名器件的内嵌 `NF`。

### `models`

可选。把特定器件绑定到非默认 PDK 模型（例如硅 SKY130），而不是默认的 `"pmos_tft"`
（AT4000TG OTFT）。这里没列到的器件仍用默认 PDK——纯增量，纯 OTFT 配置完全不需要这个字段。

```json
"models": {
  "M1": {"type": "sky130.nmos", "extract_w": 24.0},
  "M3": {"type": "sky130.pmos", "vb": 1.8, "extract_w": 12.0}
}
```

- `type` ——模型注册键，格式 `"<pdk>.<极性>"`（如 `"sky130.nmos"`、`"sky130.pmos"`、
  `"freepdk45.nmos"`、`"tsmc28hpcp.nmos"`、`"at4000tg.pmos"`）。见 `circuitopt.device_model.register_pdk`。
- 其余键透传给器件构造函数。对 SKY130 器件：`vb`（衬底偏置，伏特；默认 0）、
  `corner`（SKY130 工艺角——`tt`/`ss`/`ff`/`sf`/`fs`；默认 `tt`）、`extract_w`
  （µm——选择一个随包参考宽度参数卡，原生 BSIM 实例仍使用实际 `W`）、
  `temperature`（开尔文；默认 300.15）、`NF`（整数）。
- **FreePDK45**（`"freepdk45.nmos"` / `"freepdk45.pmos"`）直接解析平铺的
  BSIM4 level-54 模型卡，并使用进程内 Berkeley BSIM4.5 后端求值。模型卡声明
  `version=4.0`；该元数据字段不会在内置内核中切换另一套方程，原生单管与五管 OTA
  结果已用 ngspice 回归核对。器件键包括 `vb`（NMOS 为 0，PMOS 通常为 1.0V）、
  `corner`（`nom`/`tt`/`ss`/`ff`/`sf`/`fs`；默认 `nom`）、开尔文
  `temperature`、`NF`、`M` 以及内核支持的数值 BSIM4 实例参数。后端向 DC、AC、
  noise、transient、PSS、PAC、PNoise 提供完整端口电流、电导、电荷、电容和相关噪声。
  `extract_w` 仅作为旧配置兼容参数接受，原生器件始终使用实际几何尺寸。可选的
  `freepdk45_ngspice.nmos` / `.pmos` 保留旧缓存网格求值器，完整电路 ngspice helper
  则作为外部 oracle。模型卡位于 `PDK_ROOT/freepdk45/`；见
  `examples/freepdk45_fd_ota.json`（全差分 OTA 设计案例,[docs/freepdk45_fd_ota_design.md](freepdk45_fd_ota_design.md)）。
- **TSMC28HPC+**（`"tsmc28hpcp.nmos"` / `"tsmc28hpcp.pmos"`）绑定 licensed 1d8
  HSPICE deck 中的 0.9V `nch_mac` / `pch_mac` core wrapper。PMOS bulk 接 core 电源时使用
  `vb=0.9`。支持 `tt`/`ss`/`ff`/`sf`/`fs`（`nom` 是 `tt` 别名）、开尔文 `temperature`
  和原生传给 foundry macro 的 `NF`。默认可迁移入口是
  `PDK/tsmc28hpcp/models/hspice/cln28hpcp_1d8_elk_v1d0_2p2.l`，可依次用
  `TSMC28_MODEL_DIR`、`TSMC28_PDK_ROOT` 覆盖。默认模型类型由内部 HSPICE 解析器和
  原生 Berkeley BSIM4.5 后端完成 DC、AC、noise、transient、PSS、PAC、PNoise，
  不启动 ngspice。需要回归交叉核对时，显式使用
  `tsmc28hpcp_ngspice.nmos` / `.pmos` oracle 别名。详见
  [TSMC28HPC+ 适配说明](tsmc28hpcp.md)。
- 一个电路里部分器件是 OTFT、部分是硅是合法的——例如互补硅 OTA 独立绑定 NMOS/PMOS
  器件。见 `examples/sky130_5t_ota.json`。
- SKY130 正常仿真使用随包解析卡和原生 C BSIM4；FreePDK45 与 TSMC28HPC+ 使用
  各自本地模型文件和同一原生后端。首次构建需要 C 编译器，ngspice
  仅用于 oracle 或生成新卡。缺少前置条件时会清晰报错。详见
  [PDK 支持矩阵](pdk_support_zh.md)。

### `bias`

可选但通常需要。提供 `rails` 中字符串引用的数值。

```json
"bias": {
  "VDD": 40.0,
  "VCM": 30.65,
  "VB": 9.84,
  "VC": 16.0
}
```

如果某个 rail 写成 `"VDD": "VDD"`，但 `bias` 里没有 `"VDD"`，求解时会失败。

### `adc`

可选。定义闭环 SAR 转换工作流；电路本体仍由普通 `devices`/`capacitors`/`vsources` 描述。
`bit_inputs`/`bit_inputs_bar` 是差分 CDAC 的 MSB→LSB PWL 源 key，比较器判决来自真实 transient 节点。

```json
"adc": {
  "type": "sar", "n_bits": 3, "vref": 1.0, "input_common_mode": 0.5,
  "bit_inputs": ["b2p", "b1p", "b0p"],
  "bit_inputs_bar": ["b2n", "b1n", "b0n"],
  "dummy_input": "bdp", "dummy_input_bar": "bdn",
  "sample_input": "sample", "sample_bar_input": "sample_b",
  "comparator_node": "vout", "comparator_threshold": 0.5,
  "sample_end": 1e-8, "bit_period": 2e-8, "edge_time": 2e-10
}
```

执行入口是 `circuit-opt adc`；`--vin` 做单次转换，`--sweep` 计算 DNL/INL，`--sine`
计算 SNDR/SFDR/ENOB。完整示例见 `examples/freepdk45_sar3.json`（静态 5T 比较器）与
`examples/freepdk45_sar6.json`（6-bit，时钟同步 StrongARM 比较器）。

#### `adc.clock`

可选。为**时钟同步动态（StrongARM）比较器**生成选通波形。配置后
`sar_input_waveforms` 会生成名为 `input` 的波形 key：静息在 `low`（复位），在每个
bit 的 `decision_time` 附近脉冲到 `high`（评估）——锁存器在 CDAC 建立期间预充、判决
时刻再生。经 `transient_inputs` 用该 key 驱动时钟尾管/输出复位管的栅。省略该块即完全
复现静态比较器行为（不生成任何时钟波形，`examples/freepdk45_sar3.json` 渲染出的网表
逐字节不变）。

```json
"clock": {"input": "clk", "high": 1.0, "low": 0.0,
          "eval_before": 3e-9, "reset_hold": 1e-9}
```

- `input`——必填，选通波形的 transient key。
- `high` / `low`——评估/复位电平 [V]；默认 `high = adc.vref`、`low = 0`。
- `eval_before`——每个 `decision_time` 前多少秒拉高（默认 `0.3 * bit_period`）；必须
  小于 `bit_period/2 - edge_time`，保证被试 CDAC 电容切换完成后锁存器才采样。
- `reset_hold`——每个 `decision_time` 后多少秒复位（默认 `0.1 * bit_period`）。

#### `adc.mismatch`

可选。FreePDK45 SAR 的逐器件失配蒙特卡洛配置，由 `circuitopt.sar_mismatch_mc`
使用。所有 sigma 默认 `0.0`，因此省略该块（或全部置零）即复现标称转换。

```json
"mismatch": {
  "sigma_vth0": 5e-3, "w0": 1.0, "l0": 0.05,
  "sigma_cu": 0.01, "c_unit": 1e-14,
  "dnl_threshold": 0.5, "inl_threshold": 0.5
}
```

- `sigma_vth0`——晶体管阈值电压 sigma [V]，定义在参考面积 `w0*l0` 上；逐器件按
  `sigma_vth0 / sqrt(W*L / (w0*l0))`（Pelgrom 面积律）缩放，作为 BSIM4 实例参数
  `delvto` 注入。`sigma_vth0_nmos`/`sigma_vth0_pmos` 可分别覆盖 N/P 管。
- `sigma_cu`——CDAC 单位电容相对 sigma（定义在 `c_unit`）；容值 `C` 的电容相对 sigma 为
  `sigma_cu / sqrt(C / c_unit)`（二进制加权电容由多个单位电容并联，匹配更好）。
- `dnl_threshold` / `inl_threshold`——良率判定的 |DNL|/|INL| 上限，单位 LSB（默认 0.5）。

### `outputs`

可选但 AC/noise/transient 通常需要。支持单端或差分。

单端：

```json
"outputs": ["OUT"]
```

差分：

```json
"outputs": ["VOP", "VON"]
```

差分输出按第一个减第二个计算，即 `VOP - VON`。

### `input_drives`

可选。AC 分析中的小信号 gate drive，按器件名指定。

```json
"input_drives": {
  "M7": 0.5,
  "M8": -0.5
}
```

说明：

- 只对 gate 在 rail 上的器件有意义。
- 未列出的 gate 在 AC 中视为小信号地。
- 差分输入推荐使用 `+0.5/-0.5`，这样输入差分幅度为 1。

### `load_caps`

可选。固定电容列表，用于 AC/noise/transient stamping。

数组写法：

```json
"load_caps": [
  ["VOP", "GND", 5e-12],
  ["VON", "GND", 5e-12]
]
```

对象写法：

```json
"load_caps": [
  {"a": "OUT", "b": "GND", "C": 2e-12}
]
```

### `resistors`

可选。两端电阻列表，连接节点 `a`、`b`，阻值 `R`（欧姆，必须为正）。DC 走 KCL 支路电流 `(Va-Vb)/R`，AC/noise 按电导 `1/R` stamp，transient 按电导参与，并贡献热噪声 PSD `4kT/R`（计入 `dev_psd`，按电阻名索引）。

```json
"resistors": [
  {"name": "RL", "a": "OUT", "b": "GND", "R": 4e6}
]
```

数组写法：`["RL", "OUT", "GND", 4e6]`。

### `capacitors`

可选。两端电容列表，连接节点 `a`、`b`，容值 `C`（法拉，必须为正）。DC 视为开路；AC 按导纳 `jωC` stamp；transient 用后向欧拉伴随模型。与 `load_caps` 等价，二者会合并处理；区别只是 `capacitors` 带名字、更适合通用 netlist。

```json
"capacitors": [
  {"name": "CL", "a": "OUT", "b": "GND", "C": 2e-12}
]
```

数组写法：`["CL", "OUT", "GND", 2e-12]`。

### `current_sources`

可选。理想直流电流源列表。电流 `I`（安培，可正可负）在源内部从 `nplus` 流向 `nminus`——即从 `nplus` 抽取 `I`、向 `nminus` 注入 `I`。DC 参与 KCL；在小信号 AC/noise 中视为开路（无贡献，亦无噪声）；transient 为恒定电流。

```json
"current_sources": [
  {"name": "IB", "nplus": "VDD", "nminus": "OUT", "I": 1e-6}
]
```

数组写法：`["IB", "VDD", "OUT", 1e-6]`。

### `vccs`

可选。压控电流源（VCCS）。输出电流 ``p → q``：``I = gm * (Vctrl_p - Vctrl_n)``。DC 进入 KCL；AC 中 stamp 进 G 矩阵；理想无噪声；transient 为瞬时电流，含完整 Jacobian 贡献。

```json
"vccs": [
  {"name": "G1", "p": "OUT", "q": "GND",
   "ctrl_p": "IN", "ctrl_n": "GND", "gm": 1e-4}
]
```

数组写法：`["G1", "OUT", "GND", "IN", "GND", 1e-4]`。

### `vsources`

可选。理想电压源，采用**真·MNA**求解：每个源新增一个支路电流未知量和一行约束
``V_p − V_q = value``，系统从 `n` 个节点扩到 `n_aug = n + m`。`value` 为常数 EMF（数字）
或瞬态输入波形 key（字符串，表示时变 ``E(t)``）。

```json
"vsources": [
  {"name": "V1", "p": "IN", "q": "GND", "value": 2.0}
]
```

数组写法：`["V1", "IN", "GND", 2.0]`。`p`、`q` 至少有一个是 solved 节点（两端皆 rail 的源会被拒绝）。

- **DC**：精确固定节点电压（节点仍在 solved 集合）；`ac_solve` 在 `branch_currents` 里附带报告支路电流（方向：源内部 `p → q`）。
- **AC / Noise**：DC 源视为短路（AC 地）；理想源无热噪声。源名出现在 `ac_drives` 中时作为 AC 激励。
- **Transient**：支持常数或波形 key 的 `E(t)`。编译 Rust 定网格内核作用于 `n` 节点，故含电压源的电路在扩展 `n_aug` 系统上回落到纯 Python `_impl` 参考路径（v2.0.0 起 numba 引擎已移除；此路径下 `rust_grid_solver` 与已弃用的 `numba_grid_solver` 均为 `False`）。
- **PSS / PAC / PNoise** 同样支持：shooting monodromy 与 harmonic-balance 矩阵都用支路电流未知量做 bordered 扩展（PNoise 在有电压源时走 dense 路径）。

### `vcvs`

可选。压控电压源（VCVS）。输出电压 ``V_p − V_q = mu * (V_cp − V_cn)``。每个 VCVS 新增一个支路电流未知量（继承 ideal voltage source），约束行包含控制节点项。理想/无噪声。

```json
"vcvs": [
  {"name": "E1", "p": "OUT", "q": "GND",
   "cp": "INP", "cn": "INN", "mu": 100.0}
]
```

数组写法：`["E1", "OUT", "GND", "INP", "INN", 100.0]`。`p`、`q` 至少有一个是 solved 节点。

### `cccs`

可选。流控电流源（CCCS）。输出电流 ``I_out = beta * I_ctrl``，方向 ``p → q``。控制电流 ``I_ctrl`` 来自名为 `ctrl_name` 的电压源（vsource / VCVS / CCVS）的支路电流。理想/无噪声。不增加新支路电流——引用已有支路电流未知量。

```json
"cccs": [
  {"name": "F1", "p": "OUT", "q": "GND",
   "ctrl_name": "V1", "beta": 2.0}
]
```

数组写法：`["F1", "OUT", "GND", "V1", 2.0]`。`ctrl_name` 必须引用同拓扑中的 vsource、VCVS 或 CCVS。

### `ccvs`

可选。流控电压源（CCVS）。输出电压 ``V_p − V_q = gamma * I_ctrl``。控制电流 ``I_ctrl`` 来自名为 `ctrl_name` 的电压源的支路电流。每个 CCVS 新增一个支路电流未知量。理想/无噪声。

```json
"ccvs": [
  {"name": "H1", "p": "OUT", "q": "GND",
   "ctrl_name": "V1", "gamma": 100.0}
]
```

数组写法：`["H1", "OUT", "GND", "V1", 100.0]`。`p`、`q` 至少有一个是 solved 节点。`ctrl_name` 必须引用已有支路电流的源（vsource / VCVS / CCVS）。CCCS 和 CCVS 支持级联：CCCS 可控制 CCVS 的支路电流。

### `dc_guesses`

可选。DC 初值列表。每个对象只需要写部分或全部 solved 节点。

```json
"dc_guesses": [
  {"OUT": 20.0},
  {"OUT": 5.0},
  {"OUT": 35.0}
]
```

建议为容易多稳态或正反馈的电路提供几个物理合理初值。

### `aliases`

可选。给 DC operating point 增加别名，方便兼容旧代码或报告字段。

```json
"aliases": {
  "vfb": "VFBP",
  "net2": "NET2"
}
```

求解返回的 `dc_op` 会同时包含原始 solved 节点和 alias。

### `transient_inputs`

可选。把 transient 输入波形 key 连接到某些器件 gate。

```json
"transient_inputs": {
  "M7": "vip",
  "M8": "vin"
}
```

调用 transient 时传：

```python
tran = transient(sizes, bias, t, topo=topology,
                 inputs={"vip": vip_waveform, "vin": vin_waveform})
```

### `ac_drives`

可选。类似 `input_drives`，但驱动的是节点而不是器件 gate。适合输入先经过电阻、
电容或 testbench 前端网络，再到达有源器件的情况。

```json
"ac_drives": {
  "VINP": 0.5,
  "VINN": -0.5
}
```

### `periodic`

可选。给 PSS/PAC/PNoise 和周期 transient 使用的默认大信号周期激励。

```json
"periodic": {
  "frequency": 1000.0,
  "n_points": 101,
  "inputs": {
    "vin": {"type": "constant", "value": "VIN"},
    "clk": {"type": "pulse", "low": 0.0, "high": "VDD", "duty": 0.5,
            "rise": 20e-6, "fall": 20e-6}
  },
  "node_inputs": {"VIN": "vin", "CLK": "clk"},
  "current_inputs": [{"p": "VDD", "q": "OUT", "input": "iqinj"}],
  "signed_devices": ["SW1", "SW2"]
}
```

支持的波形：

- 数字或 bias key：常量波形，例如 `"VIN"`。
- `constant` / `dc`：常量。
- `sine` / `sin` / `cosine` / `cos`：正弦/余弦，字段包括 `dc`、`amplitude`、`phase`、`frequency` 或 `harmonic`。
- `square`：理想方波，字段包括 `low`、`high`、`duty`、`delay`。
- `pulse`：有限边沿周期 pulse，额外支持 `rise`、`fall`。
- `pwl`：周期 PWL，字段为 `times` 和 `values`。

### `analyses`

可选。统一分析 dispatch 配置。调用 `circuitopt.analysis_dispatch.run_analysis_suite(spec)`
会按 `ac -> noise -> transient -> pss -> pac -> pnoise` 顺序运行已配置的分析；
PAC/PNoise 需要 PSS 时会自动复用或先运行 PSS。

`transient` / `pss` / `pac` / `pnoise` 的权威 option registry 位于
`circuitopt.analysis_options`。`analysis_dispatch.py` 从这个 registry 派生转发到
solver 的 kwargs 和默认值；JSON schema 也用测试和同一 registry 对齐，避免新增
solver 参数后 dispatch/schema/docs 继续漂移。
`analyses` 块中的未知选项会直接报错（例如把 `max_sideband` 拼成 `max_sidebands`
不会被静默忽略）。

```json
"analyses": {
  "pss": {
    "corner": "slow",
    "residual_tol": 1e-12,
    "max_shooting_iters": 2,
    "jacobian_reuse": true,
    "analytic_jacobian": true
  },
  "pac": {
    "freqs": [100.0, 1000.0],
    "input_drive": {"vin": 1.0},
    "analytic": true,
    "max_sideband": 10,
    "n_period_samples": 384,
    "time_domain": false,
    "td_integration": "gear2",
    "td_n_period_samples": 768,
    "lti_fast_path": true,
    "cache_linearization": true,
    "cache_forcing": true
  },
  "pnoise": {
    "freqs": [100.0, 1000.0],
    "input_drive": {"vin": 1.0},
    "max_sideband": 0,
    "n_period_samples": 32,
    "lti_fast_path": true,
    "cache_linearization": true,
    "band": [100.0, 1000.0]
  }
}
```

`freqs` 可以是频点数组，也可以是 `{"start": 1.0, "stop": 1e4, "num": 41, "scale": "log"}`。
`input_drive` 是 PAC/PNoise 小信号输入复幅值映射；JSON 中复数可写成数字、
`[real, imag]` 或 `{"real": ..., "imag": ...}`。
每个 analysis 都可以设置 `corner` 为 `"typical"`、`"slow"`、`"fast"` 或显式
模型偏移 map。PAC/PNoise 必须和它们复用的 PSS 轨道保持同一 corner；当 PSS
没有写 corner 时，dispatch 会继承唯一的 PAC/PNoise corner；如果依赖分析请求
了和已有 PSS 不同的 corner，会直接报错，避免 typical/slow 混用。
PSS 默认使用解析 monodromy Jacobian（`"analytic_jacobian": true`）：在收敛轨迹上
一次性采样 G(t)/C(t) 小信号矩阵构建 Φ，替代 `n_state` 次有限差分瞬态。设置为
`false` 可回退到原有限差分路径。Jacobian 构建后用 Broyden 更新复用；疑难收敛或
极高精度对比时可设置 `"jacobian_reuse": false`，或用 `"jacobian_rebuild_interval": 2`
周期性重建。
Gear2 PSS/transient 可设置 `"adaptive": true` 启用 LTE-controlled adaptive timestepping。
dispatch 会转发 `"adaptive_reltol"`、`"adaptive_vabstol"`、`"adaptive_iabstol"`、
`"adaptive_max_steps"`、`"adaptive_h0"` 和 `"cap_mode"`；pulse/square 周期输入会在
adaptive run 前自动补入边沿断点。`cap_mode` 只支持 `"charge"`（id 0）和
`"average"`（id 1）及其文档别名。
PAC 默认使用解析伴随谐波平衡（`"analytic": true`）：在 PSS 轨道转换矩阵上每频率
一次伴随线性求解，零额外瞬态运行。`"max_sideband"` 和 `"n_period_samples"` 控制
HB 分辨率。对 rail-driven chopper 类电路，可设置 `"time_domain": true` 优先尝试
加速的 time-domain Floquet PAC；`"td_integration"` 和 `"td_n_period_samples"` 控制
这条路径的 BDF/grid 设置。不支持的拓扑会在 `"analytic": true` 时回退到 HB。
只有需要原有限差分 shooting 时才设置 `"analytic": false`。
PAC/PNoise 默认启用静态轨道 LTI fast path 和 PSS 结果缓存；如需逐次强制重算，
可设置 `"lti_fast_path": false`、`"cache_linearization": false`、
`"cache_forcing": false`。PNoise 会复用 `pss_result` 上的采样 `G(t)/C(t)`、
HB block 和相同频点的 adjoint 解。PNoise HB 系统变大时，可设置
`"hb_solver": "sparse"` 或 `"iterative"` 强制 sparse direct 或 block-Jacobi
预条件 GMRES；默认 `"auto"` 会让小矩阵继续走 dense，只在矩阵足够大且非常稀疏时切换。
PAC 边界矩阵 condition 诊断默认关闭，因为它每个频点都需要一次 SVD；需要排查数值
病态时可设置 `"profile": true`、`"debug": true`，或显式设置
`"compute_condition": true`。

JSON dispatch 的 `pnoise` 入口目前是通用 HB 路径。Chopper 专用包装器
`pmos_chopper_pnoise(...)` 已默认使用 TD-adjoint PNoise 来对齐 Cadence；如果需要
无 HB 截断的 chopper PNoise，应使用该包装器，或直接调用
`circuitopt.pnoise_solver.pnoise_solve(..., time_domain=True)`。

### `explore`

可选。设计空间探索配置——待扫描的变量及范围、可行性约束（gain/BW/IRN/power/area）、
优化目标。被 `circuitopt.explore`（采样→评估→约束→Pareto 选择）、`circuitopt.dataset`
（采样→评估每个候选、不过滤——产出带标签的训练集）、`circuitopt.optimize`（用训练好的
surrogate 筛选→校验入围候选）三处消费。

```json
"explore": {
  "variables": {
    "in_pair_W": {"min": 40000, "max": 90000, "targets": ["M7.W", "M8.W"]},
    "VCM":       {"min": 28.0,  "max": 33.0}
  },
  "constraints": {"gain_dB": {"min": 20}, "bw_Hz": {"min": 100},
                  "irn_uV": {"max": 44.5}},
  "objectives":  {"area": "min", "power_uW": "min"},
  "band":  [0.05, 100.0],
  "freqs": {"start": -2, "stop": 3, "num": 81}
}
```

- `variables` ——每项需要数值 `min`/`max`。`targets` 让一个变量同时驱动多个 key
  （匹配/对称器件对）；默认为 `[<变量名>]`。`round`（小数位）把采样值吸附到网格；
  `int` 取整（对 W/L/NF 很有用）。
- `constraints` ——每个指标需要 `min` 和/或 `max` 边界。已知指标：`gain_dB`、
  `gain_peak_dB`、`bw_Hz`、`irn_uV`、`power_uW`、`area`。
- `objectives` ——`{指标: "min" | "max"}`；至少要有一个。
- `band` ——`[f_lo, f_hi]`（Hz），用于频带积分的 `irn_uV` 指标。
- `freqs` ——AC/noise 分析网格：`{"start": <log10 Hz>, "stop": <log10 Hz>,
  "num": <点数>}`（对数间隔）。

**target 语法**——除了 `"DEV.W"` / `"DEV.L"` / `"DEV.NF"`（器件尺寸）和裸 bias key，
`targets` 还支持结构化设计轴（每个都会逐候选重建电路；由 `circuitopt.dataset`/
`circuitopt.optimize` 消费，`circuitopt.explore` 不支持）：

| Target | 轴 | 说明 |
|--------|------|-------|
| `"<CapName>.C"` | 具名电容的容值（F） | 对应 `capacitors` 条目需要带 `"name"` |
| `"<ResName>.R"` | 具名电阻的阻值（Ω） | 对应 `resistors` 条目需要带 `"name"` |
| `"periodic.frequency"` | 周期激励的时钟频率 | 需要 `periodic` 块 |
| `"pvt0"` / `"pbeta0"` | 连续全局工艺偏移 | 路由进 `evaluate(corner=...)`；采样它就把离散 corner 扫描变成连续 PVT 训练数据 |

上面 `models` 字段用于把某个变量目标的器件绑到非默认 PDK（比如扫描一个 SKY130 器件
的 `W`）；`explore` 块本身不用变——`models` 和 `explore.variables` 可以自由组合。

## 完整示例

见：

```text
examples/single_stage.json        # 单管共源级（PMOS_TFT）
examples/resistor_load_stage.json # PMOS + 电阻负载 + 输出电容 + 电流源
examples/voltage_divider.json     # 理想电压源（真·MNA）— 电阻分压器
examples/vcvs_amplifier.json      # VCVS 放大器 — 线性增益 100×
examples/sc_lpf.json              # 开关电容低通（两相，PMOS 开关 + vsource 时钟）
examples/afe_explore.json         # 10 管 AFE 含 explore 配置
examples/periodic_rc.json         # 纯 RC 周期 PSS/PAC/PNoise dispatch
examples/sky130_5t_ota.json       # 硅 SKY130 互补 5T OTA —— `models` 块 + explore/dataset/optimize
examples/freepdk45_sar3.json      # FreePDK45 全差分 3-bit SAR ADC —— 全电荷 .tran + DNL/INL/ENOB
```

可以这样加载并运行：

```python
import numpy as np

from circuitopt.circuit_loader import load_circuit_json
from circuitopt.ac_solver import ac_solve
from circuitopt.noise_solver import noise_analysis
from circuitopt.transient_solver import transient

spec = load_circuit_json("examples/single_stage.json")
freqs = np.logspace(0, 4, 121)

ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
noise = noise_analysis(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)

t = np.linspace(0, 1e-3, 100)
vin = np.full_like(t, spec.bias["VIN"])
tran = transient(spec.sizes, spec.bias, t, topo=spec.topology,
                 nf=spec.nf, inputs={"vin": vin})
```

或直接运行 JSON 内配置的分析：

```python
from circuitopt.analysis_dispatch import run_analysis_suite
from circuitopt.circuit_loader import load_circuit_json

spec = load_circuit_json("examples/periodic_rc.json")
results = run_analysis_suite(spec)
pac_gain = results["pac"]["gains"]
pnoise_irn = results["pnoise"]["irn_uV_band"]
```

## 当前限制

当前 JSON 格式仍是本地求解器的电路描述，不是完整 SPICE netlist。

已支持：

- 三端晶体管器件（PMOS_TFT，通过 ``TransistorModel`` 接口）。
- 电阻、电容、理想直流电流源、VCCS（压控电流源）、VCVS（压控电压源）、CCCS（流控电流源）、CCVS（流控电压源）、理想电压源（真·MNA）。
- DC/AC/noise/transient 共享拓扑（电阻含热噪声；受控源与理想电压源为理想/无噪声）。
- 单端或差分输出。
- 固定 load capacitance。
- AC gate drive 和 node drive。
- transient gate waveform 和 node waveform。
- 从 JSON dispatch 周期 PSS/PAC/PNoise。
- DC 初值。
- 器件模型抽象（``circuitopt/device_model.py``）——支持新增模型类型而不改求解器代码。
- NMOS 和 PMOS 覆盖 AT4000TG OTFT（仅 PMOS）以及 SKY130、FreePDK45、
  TSMC28HPC+ 三套硅 PDK。
- 通过 `models` 字段做逐器件模型绑定——混合 OTFT/硅电路（不覆盖时默认 PDK 仍是
  ``"at4000tg.pmos"``）。
- 硅 DC/AC/noise/transient；SKY130、FreePDK45 与 TSMC28HPC+ 均走项目内部
  原生 BSIM4 后端。

尚未支持：

- ADC transient noise、版图寄生提取和晶体管级 SAR 数字状态机。
- 多输出同时分析。
- 层次化子电路。
- 任意用户电路 SPICE 网表导入。项目内部 HSPICE 解析器当前只用于受支持的本地模型库展开。

电路描述能力的扩展通过 ``circuitopt/device_model.py``（器件模型注册表）和 ``circuitopt/ac_mna.py``（MNA 电压源 / VCCS / VCVS / CCCS / CCVS stamp 原语）实现。
