# TSMC28HPC+ 本地工艺适配

[English](tsmc28hpcp.md) | [中文说明](tsmc28hpcp_zh.md)

`tsmc28hpcp` PDK binding 用本地 ngspice 调用用户已有许可的 TSMC 28HPC+ 模型。
当前适配目标是 1d8 HSPICE deck 中的 0.9V core `nch_mac` / `pch_mac`。模型文件
只保存在本机，不提交到 Git；这避免泄漏，但不会改变 foundry license/NDA 的约束。

## 统一入口

项目内标准入口是：

```text
PDK/tsmc28hpcp/models/hspice/cln28hpcp_1d8_elk_v1d0_2p2.l
```

代码从项目根目录解析该相对路径，不记录任何电脑的绝对路径。换电脑后把同一个文件放到
相同位置即可。当前 core MOS 适配只需要这个约 7.3MB 的主模型文件，不需要把整个约 160GB
的 iPDK delivery 搬进项目。`PDK/tsmc28hpcp/models/` 已加入 `.gitignore`。

也可以使用外部安装，解析优先级如下：

1. `TSMC28_MODEL_DIR`：直接指向包含主模型文件的 `models/hspice`。
2. `TSMC28_PDK_ROOT`：指向 iPDK 或 delivery 根目录。
3. 项目内标准入口 `PDK/tsmc28hpcp/`。
4. 通用入口 `PDK_ROOT/tsmc28hpcp`。

ngspice 默认从项目 uv 环境、当前虚拟环境和 `PATH` 查找，也可以显式设置：

```bash
export NGSPICE_BIN=/path/to/ngspice
```

已验证 ngspice 46。适配器使用 `-D ngbehavior=hsa` 启动，并显式展开 foundry deck 的
`setup`、工艺角、`global`、`total`、`stat` 五个 `.lib` section，原始模型文件不做修改。

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
- PMOS bulk 通常显式设为 core 电源 `vb=0.9`。
- 支持 `tt`、`ss`、`ff`、`sf`、`fs`；`nom` 是 `tt` 的别名。
- `temperature` 单位为 K，整个 ngspice 电路必须使用同一温度。
- `NF` 原生传给 `nch_mac` / `pch_mac`，不会在 wrapper 外重复缩放。
- 完整 ngspice deck 的所有 MOS 必须绑定同一个工艺适配器，不支持混合 foundry deck。

## 分析覆盖

| 分析 | TSMC28HPC+ 路径 |
|------|-----------------|
| DC / 本地 AC / 本地 noise | ngspice 单管表征网格，可缓存用于优化循环 |
| `.op` | 完整电路 ngspice，读取 `@m.x*.main[...]` 层级向量 |
| `.ac` / `.noise` | 完整电路 ngspice，直接使用原模型 deck |
| `transient()` | 自动路由到完整电荷 ngspice `.tran` |
| mismatch | full transient 实例参数 `_delvto` |
| PSS / PAC / PNoise | direct-ngspice model-card 后端尚未接入 |

本地表征网格适合大量候选的 DC/AC/noise 优化；最终带宽、结电容、开关电荷和 settling 应使用
完整 `.ac` / `.noise` / `.tran` oracle。

## 已验证

- `nch_mac` / `pch_mac` 单管 DC、gm、gds、Cgs、Cgd 表征。
- CMOS 反相器完整电荷瞬态。
- 层级 `.op` 与饱和区判断。
- 单管白噪声和 1/f 噪声系数。
- 3-bit SAR ADC：CDAC、采样开关、晶体管比较器、逐位重放和功耗回读。

```bash
pytest -q tests/test_tsmc28.py
```

真实 SAR 冒烟约需两分钟，因为每一位比较判决都会重放一次完整瞬态。没有 licensed 模型时，
纯适配器测试仍会运行，依赖模型的测试自动 skip。

## 边界

当前代码负责本地电路仿真与优化，不负责 Cadence library 导入、版图、DRC/LVS、PEX 或 tapeout
sign-off。后续进入 Cadence 时仍应使用原始 iPDK、官方 rule deck 和 foundry 规定的仿真器/版本。
