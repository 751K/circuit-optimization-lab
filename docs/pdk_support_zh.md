# PDK 支持矩阵

[文档首页](README_zh.md) | [English](pdk_support.md)

CircuitOpt 通过逐器件模型绑定选择工艺。技术上可以在一个电路里混合模型，但实际设计通常应保持
同一工艺和一致的电源、bulk 与 corner 语义。

## 能力矩阵

| 工艺 | 模型键 | 器件后端 | DC / AC / Noise | Transient | PSS / PAC / PNoise | 外部前置条件 |
|---|---|---|---|---|---|---|
| AT4000TG | `at4000tg.pmos` | 内置校准 PMOS 模型 | 支持 | 原生 | 支持 | 无 |
| SKY130 | `sky130.nmos`、`sky130.pmos` | 参数卡解析 + OpenVAF OSDI | 支持 | OSDI 后端 | 在模型提供完整端口线性化时可用，需按拓扑验证 | SKY130 PDK、用于解析的 ngspice、OpenVAF/BSIM4 VA |
| FreePDK45 | `freepdk45.nmos`、`freepdk45.pmos` | ngspice-C 表征网格 | 支持 | 完整电路 ngspice `.tran` | 没有直接 FreePDK45 周期后端 | FreePDK45 模型卡和 ngspice |
| TSMC28HPC+ core | `tsmc28hpcp.nmos`、`tsmc28hpcp.pmos` | 内部 HSPICE 前端 + 原生 Berkeley BSIM4.5 | 支持 | 原生电荷守恒 BE/Gear2 | 支持 | Licensed 模型文件；首次构建需要 C 编译器 |
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

- 先用 ngspice 展开 foundry 子电路和 bin 参数，再把扁平卡交给 OpenVAF 编译的
  BSIM4 Verilog-A，通过 OSDI 宿主求值。
- 适合本地优化和方法研究，不是官方 SKY130 模型的逐位替代品。
- 接受的工艺角：`tt`、`ss`、`ff`、`sf`、`fs`。
- 解析卡和编译产物属于缓存，不是源模型。

### FreePDK45

- 由于模型卡 BSIM4 版本与现有 OSDI BSIM 源不匹配，使用 ngspice-C 作为器件求值器。
- DC、AC、噪声使用缓存表征网格。
- 瞬态把完整受支持电路路由到 ngspice，以保留 BSIM 电荷和结电容。
- 快速 AC 网格缺少部分漏/源结电容，整机带宽应使用显式 ngspice AC oracle 复核。
- 工艺角：`nom`、`tt`、`ss`、`ff`、`sf`、`fs`。
- `circuit-opt mc` 尚未提供通用硅工艺逐器件失配语义。

### TSMC28HPC+

- 当前适配 1d8 HSPICE 模型文件中的 0.9 V `nch_mac` / `pch_mac` core wrapper。
- 内部解析器在内存中处理 `.lib`、参数、子电路、macro 和尺寸 bin。
- 原生后端提供四端电流、电导、电荷、电容和相关噪声。
- 工艺角：`tt`、`ss`、`ff`、`sf`、`fs`；`nom` 等价于 `tt`。
- 器件绑定 API 中温度使用开尔文。
- 当前不声称支持完整 iPDK 中的 IO、RF、SRAM、eFuse、可靠性、统计模型、版图提取或
  sign-off 检查。
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
| TSMC 模型目录 | `TSMC28_MODEL_DIR`、`TSMC28_PDK_ROOT`、项目内忽略入口、`PDK_ROOT/tsmc28hpcp` |
| ngspice | `NGSPICE_BIN`、虚拟环境约定位置、`PATH` |
| OpenVAF | `OPENVAF_BIN`、`OPENVAF_ROOT`、虚拟环境约定位置、`PATH` |
| 原生模型缓存 | `CIRCUITOPT_NATIVE_MODEL_CACHE`，否则使用选定虚拟环境 |

## 不替代的流程

CircuitOpt 是本地仿真与优化框架，不替代官方 PDK 安装、原理图/版图库、DRC、LVS、寄生提取、
EM/IR、老化、可靠性、统计 sign-off 或 foundry 批准的 tapeout 流程。
