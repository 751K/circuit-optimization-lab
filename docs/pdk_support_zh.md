# PDK 支持矩阵

[文档首页](README_zh.md) | [English](pdk_support.md)

CircuitOpt 通过逐器件模型绑定选择工艺。技术上可以在一个电路里混合模型，但实际设计通常应保持
同一工艺和一致的电源、bulk 与 corner 语义。

## 能力矩阵

| 工艺 | 模型键 | 器件后端 | DC / AC / Noise | Transient | PSS / PAC / PNoise | 外部前置条件 |
|---|---|---|---|---|---|---|
| AT4000TG | `at4000tg.pmos` | 内置校准 PMOS 模型 | 支持 | 原生 | 支持 | 无 |
| SKY130 | `sky130.nmos`、`sky130.pmos` | 随包解析卡 + 原生 Berkeley BSIM4.5 | 支持 | Numba 电路循环 + 原生 C BSIM4 BE/Gear2 | 已接入原生端口后端，需按周期拓扑验证 | 首次构建需要 C 编译器；仅生成新卡时需要外部工具 |
| FreePDK45 | `freepdk45.nmos`、`freepdk45.pmos` | 平铺模型卡加载器 + 原生 Berkeley BSIM4.5 | 支持 | Numba 电路循环 + 原生 C BSIM4 BE/Gear2 | 已接入原生端口后端，需按周期拓扑验证 | FreePDK45 模型卡；首次构建需要 C 编译器 |
| FreePDK45 oracle | `freepdk45_ngspice.nmos`、`freepdk45_ngspice.pmos` | ngspice-C 缓存网格 / 完整网表 oracle | 仅用于 oracle | 仅用于 oracle | 不是默认周期后端 | FreePDK45 模型卡和 ngspice |
| TSMC28HPC+ core | `tsmc28hpcp.nmos`、`tsmc28hpcp.pmos` | 内部 HSPICE 前端 + 原生 Berkeley BSIM4.5 | 支持 | Numba 电路循环 + 原生 C BSIM4 BE/Gear2 | 支持 | Licensed 模型文件；首次构建需要 C 编译器 |
| TSMC28HPC+ oracle | `tsmc28hpcp_ngspice.nmos`、`tsmc28hpcp_ngspice.pmos` | 显式 ngspice 对照路径 | 仅用于 oracle | 仅用于 oracle | 不是默认周期后端 | Licensed 模型文件和 ngspice |

这里的“支持”表示分析链路已经接通，不表示自动达到 foundry sign-off 等价。每个新拓扑仍应对合适的
参考仿真器做回归。

## 各工艺说明

### AT4000TG

- 未在 JSON `models` 块中列出的晶体管默认使用该工艺。
- 仅 PMOS。
- 已对仓库中的 AFE 和 chopper Cadence Spectre 参考进行校准。
- 工艺角：`typical`、`slow`、`fast`。
- 通用 `mc` 命令当前针对这套连续失配模型。

### SKY130

- 默认加载仓库随包提供的、按几何展开的 BSIM4.5 参数卡，并直接交给与 FreePDK45、
  TSMC28HPC+ 相同的进程内原生 C BSIM4 后端。
- 正常 DC、AC、noise、transient、PSS、PAC、PNoise 不启动外部仿真器。
- 适合本地优化和方法研究，不是官方 SKY130 模型的逐位替代品。
- 接受的工艺角：`tt`、`ss`、`ff`、`sf`、`fs`。
- 仓库已包含示例所需解析卡。新几何或新 corner 没有对应卡时，可在本地安装
  SKY130/ngspice 后显式调用 `circuitopt.sky130_model.extract_sky130_card()`，
  并通过 `SKY130_CARD_DIR` 指向生成目录。

### FreePDK45

- 直接解析平铺的 level-54 VTG 模型卡，并交给进程内 Berkeley BSIM4.5 内核求值。
- 模型卡声明 BSIM4 `version=4.0`。在仓库内 Berkeley 源码中，version 字段是元数据，
  不会切换另一套负载方程；单管 `Id/gm/gds`、噪声和五管 OTA AC 均以 ngspice
  做回归核对。
- 原生后端提供完整四端电流、电导、电荷、电容和相关噪声，接入 DC、AC、noise、
  transient 及周期分析。
- 固定时间网格瞬态的 Newton 迭代和矩阵盖章位于 Numba 内核中，并通过运行时
  `void *` 函数指针直接调用 C 紧凑模型；设置 `CIRCUIT_USE_NUMBA=0` 可切换到
  Python 参考路径。
- `freepdk45_ngspice.*` 与完整电路 ngspice helper 继续作为独立 oracle 保留，
  需要这些模型键时再导入 `circuitopt.freepdk45_model`；正常仿真不需要安装
  ngspice。
- 工艺角：`nom`、`tt`、`ss`、`ff`、`sf`、`fs`。
- `tt` 等价于 `nom`；`sf` 为 NMOS slow + PMOS fast，`fs` 相反。
- `circuit-opt mc` 尚未提供通用硅工艺逐器件失配语义。

### TSMC28HPC+

- 当前适配 1d8 HSPICE 模型文件中的 0.9 V `nch_mac` / `pch_mac` core wrapper。
- 内部解析器在内存中处理 `.lib`、参数、子电路、macro 和尺寸 bin。
- 原生后端提供四端电流、电导、电荷、电容和相关噪声。
- 瞬态使用与 FreePDK45 相同的 Numba 到 C BSIM4 桥。
- 工艺角：`tt`、`ss`、`ff`、`sf`、`fs`；`nom` 等价于 `tt`。
- 器件绑定 API 中温度使用开尔文。
- 当前不声称支持完整 iPDK 中的 IO、RF、SRAM、eFuse、可靠性、统计模型、版图提取或
  sign-off 检查。
- 仅在需要 `tsmc28hpcp_ngspice.*` oracle 模型键时导入
  `circuitopt.tsmc28_model`。
- 详见 [TSMC28HPC+ 原生适配](tsmc28hpcp_zh.md)。

## JSON 绑定

```json
{
  "devices": [
    {
      "name": "MN",
      "drain": "OUT",
      "gate": "IN",
      "source": "GND",
      "W": 1.0,
      "L": 0.03
    },
    {
      "name": "MP",
      "drain": "OUT",
      "gate": "IN",
      "source": "VDD",
      "W": 2.0,
      "L": 0.03
    }
  ],
  "models": {
    "MN": {"type": "tsmc28hpcp.nmos"},
    "MP": {"type": "tsmc28hpcp.pmos", "vb": 0.9}
  }
}
```

几何单位为微米。模型专用构造参数见 [JSON 电路描述格式](json_circuit_format_zh.md)
中的 `models` 字段。

## 路径解析

| 输入 | 解析顺序 |
|---|---|
| 通用 PDK 根目录 | `PDK_ROOT`，然后当前/项目虚拟环境的 `pdk/` |
| 额外 SKY130 解析卡 | `SKY130_CARD_DIR`，然后使用包内卡 |
| TSMC 模型目录 | `TSMC28_MODEL_DIR`、`TSMC28_PDK_ROOT`、项目内忽略入口、`PDK_ROOT/tsmc28hpcp` |
| ngspice | `NGSPICE_BIN`、虚拟环境约定位置、`PATH` |
| 原生模型缓存 | `CIRCUITOPT_NATIVE_MODEL_CACHE`，否则使用选定虚拟环境 |

## 不替代的流程

CircuitOpt 是本地仿真与优化框架，不替代官方 PDK 安装、原理图/版图库、DRC、LVS、寄生提取、
EM/IR、老化、可靠性、统计 sign-off 或 foundry 批准的 tapeout 流程。
