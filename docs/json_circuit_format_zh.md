# JSON 电路描述格式

[中文说明](README_zh.md) | [核心求解器概览](core_overview_zh.md) | [后续计划](futureplan.md)

## 目的

JSON 电路描述用于把电路拓扑、尺寸、偏置和分析元数据从 Python 源码中抽出来。这样换电路时优先修改 JSON，而不是修改 `core/ac_solver.py`、`core/noise_solver.py` 或 `core/transient_solver.py`。

当前 schema 文件位于：

```text
schemas/circuit.schema.json
```

当前示例文件位于：

```text
examples/single_stage.json
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
- `devices` 是 PMOS_TFT 器件列表。
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

必填。当前每个 active device 都按三端 PMOS_TFT 处理，端口顺序为 drain/gate/source。

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

## 完整示例

见：

```text
examples/single_stage.json        # 纯 PMOS_TFT
examples/resistor_load_stage.json # PMOS + 电阻负载 + 输出电容 + 电流源
```

可以这样加载并运行：

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

## 当前限制

当前 JSON 格式仍是本地求解器的电路描述，不是完整 SPICE netlist。

已支持：

- PMOS_TFT 三端器件。
- 电阻、电容、理想直流电流源（两端元件，见上文 `resistors` / `capacitors` / `current_sources`）。
- DC/AC/noise/transient 共享拓扑（电阻含热噪声）。
- 单端或差分输出。
- 固定 load capacitance。
- AC gate drive。
- transient gate waveform。
- DC 初值。

尚未支持：

- NMOS 或其他紧凑模型（本工艺无 NMOS）。
- 理想电压源、受控源、开关 / 时变元件。
- 多输出同时分析。
- 层次化子电路。
- SPICE 语法解析。

这些内容应在后续 device model registry 和更多 MNA stamp 元件完成后再扩展。
