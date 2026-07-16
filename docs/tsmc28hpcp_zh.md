# TSMC28HPC+ 原生工艺适配

[English](tsmc28hpcp.md) | [中文说明](tsmc28hpcp_zh.md)

> **文档状态：持续维护的适配指南。** 当前范围是下文列出的 0.9 V core MOS
> wrapper 和分析链路，不代表完整 iPDK 支持。

`tsmc28hpcp` binding 在 circuitopt 内部直接求值用户本地已有许可的 TSMC
28HPC+ 模型。默认的 `tsmc28hpcp.nmos` / `tsmc28hpcp.pmos` 不会启动
ngspice：代码自行解析 HSPICE library 闭包、foundry MOS 宏和尺寸 bin，再由
项目内置的 Berkeley BSIM4.5 原生后端完成器件求值。

当前目标是 1d8 HSPICE deck 中的 0.9 V core `nch_mac` / `pch_mac`。foundry
文件只保存在本机，不提交到 Git；这避免误上传，但不会改变 license/NDA 约束。

## 模型入口

项目内标准可迁移入口是：

```text
PDK/tsmc28hpcp/models/hspice/cln28hpcp_1d8_elk_v1d0_2p2.l
```

当前 core MOS 适配只需要这个主模型文件，不需要复制完整 iPDK delivery。
`PDK/tsmc28hpcp/models/` 已被 Git 忽略。

外部安装可以使用：

```bash
export TSMC28_MODEL_DIR=/path/to/models/hspice
# 或
export TSMC28_PDK_ROOT=/path/to/iPDK_delivery
```

解析顺序为 `TSMC28_MODEL_DIR`、`TSMC28_PDK_ROOT`、项目内标准入口，最后是
`PDK_ROOT/tsmc28hpcp`。JSON 和 Python 源码中不记录任何电脑的绝对路径。

第一次使用原生 BSIM4 时，项目会从随代码提供的 Berkeley BSIM4.5 器件源码编译
一个小型共享库，并缓存到源码树之外。因此首次运行需要本机 C 编译器；正常仿真不
需要 ngspice。

## JSON 绑定

```json
{
  "devices": [
    {"name": "MN", "drain": "OUT", "gate": "IN", "source": "GND", "W": 1.0, "L": 0.03},
    {"name": "MP", "drain": "OUT", "gate": "IN", "source": "VDD", "W": 2.0, "L": 0.03}
  ],
  "models": {
    "MN": {"type": "tsmc28hpcp.nmos"},
    "MP": {"type": "tsmc28hpcp.pmos", "vb": 0.9}
  },
  "bias": {"VDD": 0.9}
}
```

- `W`、`L` 单位为 µm。
- `NF` 原生传给 foundry macro，不在 wrapper 外重复缩放。
- PMOS bulk 通常显式设置为 core 电源 `vb=0.9`。
- 支持 `tt`、`ss`、`ff`、`sf`、`fs`；`nom` 是 `tt` 别名。
- `temperature` 单位为 K。
- 单管阈值失配通过 macro 的 `_delvto` 参数传入。

## 分析覆盖

原生路径支持：

- 非线性 DC 和完整四端工作点电流；
- AC/PAC 的四端电导与电荷线性化；
- Noise/PNoise 的四端相关白噪声和 flicker 噪声矩阵；
- 电荷守恒的 backward-Euler 与 Gear2 瞬态；
- 带解析四端 monodromy 的 PSS shooting；
- PAC 谐波转换与 PNoise 周期噪声折叠。

PNoise 会保留端口间相关性，并从模型中提取 flicker 指数，不再把所有器件都固定
假设为严格 `1/f`。

ngspice 只保留为独立 oracle。需要交叉核对时，显式使用
`tsmc28hpcp_ngspice.nmos` / `tsmc28hpcp_ngspice.pmos`，或调用
`circuitopt.ngspice_ac` 中的完整电路 helper。只有这条 oracle 路径需要
`NGSPICE_BIN`。

当前代码负责电路仿真和优化，不负责 Cadence library 导入、版图、DRC/LVS、PEX、
可靠性检查或 tapeout sign-off；这些仍应使用官方 iPDK 与 foundry 认可的工具。

## 5T OTA 验证

验证电路严格只有五只 MOS，不含 ADC/CDAC：

```bash
python experiments/tsmc28_5t_ota_compare.py \
  --output /tmp/tsmc28_5t_compare.json
pytest -q tests/test_tsmc28_5t_ota.py
```

脚本用同一份 TSMC28 5T OTA 分别运行原生后端和显式 ngspice oracle，对比五只管子的
`Id/gm/gds`、差分 AC、1 kHz 到 10 GHz 输出积分噪声，以及 2 mV 差分输入阶跃
瞬态。报告内置固定验收阈值，并给出顶层 `passed` 标志。
