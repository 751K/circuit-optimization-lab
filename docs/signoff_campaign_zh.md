# Signoff Campaign

[文档首页](README_zh.md) | [CLI 参考](cli_reference_zh.md) |
[电路 JSON 格式](json_circuit_format_zh.md) | [English](signoff_campaign.md)

`circuit-opt signoff` 用一个显式 PVT 网格运行多个测试台。开环增益、差模与
共模环路稳定性、闭环噪声和大信号建立不能诚实地塞进同一个网表，因此 campaign
以多个独立 circuit JSON 为输入，再统一汇总 signoff。

```bash
circuit-opt signoff examples/tsmc28hpcp_mdac_ota_signoff.json \
  --workers 4 --output results/tsmc28_mdac_signoff.json
```

该示例包含 11 个测试 case，扫描
`tt/ss/ff/sf/fs x -40/27/125 degC x 0.85/0.90/0.95 V`，共 45 个
PVT 点、495 次 case 求解。覆盖开环、差模环、两个 CMFB 环、闭环输入/输出噪声、
五档 residue 和 0111 到 1000 主进位切换。

## 配置规则

`circuit` 路径必须相对 campaign 文件，绝对路径和逃出 campaign 所在目录的路径
都会被拒绝，因此换电脑或移动仓库不需要改入口地址。

每个 case 的 `overrides` 会深度合并到基础 circuit JSON：对象递归合并，数组整体
替换。随 PVT 变化的数值使用显式仿射表达式：

```json
{"$pvt": {"vdd": 0.5, "temperature_c": 0.0, "constant": 0.225}}
```

它表示 `0.5 * VDD + 0.0 * temperature_c + 0.225`。每个 PVT 点还会准确写入
模型 `section`、以 K 为单位的 MOS 温度、指定电源 bias、PMOS bulk、电压源数值
和 DC 初始猜测。

## 余量表

`--margins` 为每一条约束打印一行，而不是只给出 `worst_case` 指名的那一项最紧测量。
一条余量只剩 +0.02 的规格同样是设计约束，即使另一条规格还有 -0.5；只看最差项会让它
完全消失。

```bash
circuit-opt signoff campaign.json --margins
```

每行给出该约束最差的归一化余量与产生它的 PVT 点、观测到的最小值与最大值及其单位、
参与统计的点数，以及全部不合格的点。行按余量从紧到松排序。同样的数据可通过
`circuitopt.signoff_campaign.summarize_margins(result)` 以程序方式获取。

## 无源元件容差扫描

只在标称元件值下通过的 signoff 不算 signoff。poly 电阻的绝对容差约 20%，MOM 电容约
10%，压在刀刃上调出来的补偿网络无法通过流片。

```bash
circuit-opt signoff campaign.json --tolerance R=20,C=10
```

数值为百分比。键为 `R` 或 `C` 时扰动整个元件类别；其他键指定单个元件，例如
`--tolerance CC1=10`。含 `N` 个条目的扫描运行 `1 + 2N` 次 campaign——标称，以及每个
条目在 `1 - 容差` 与 `1 + 容差` 处各一次——并报告哪些约束在哪个方向失效。只有全部
运行都通过时结果才标记为 `robust`。

逐元件扰动与整类扰动回答的是不同问题：单独缩放一个调零电阻可以隔离出补偿网络，
而同时缩放全部电阻还会移动为它设定目标的偏置网络。

## 结果契约

每个 case 必须提供电路级 `signoff`。campaign 保存统一、带单位的 signoff
结果，不把原始大波形数组塞进汇总。模型错误、不收敛、非有限值或无效 signoff
配置都会成为 `invalid`，不会被替代值变成通过。

逐点和全局 `worst_case` 都会给出 case、corner、温度、电压、指标和归一化 margin。
状态优先级是 `invalid > fail > pass`；不同 worker 数下输出点顺序保持确定。

配置 schema 为
[`schemas/signoff_campaign.schema.json`](https://github.com/751K/circuit-optimization-lab/blob/main/schemas/signoff_campaign.schema.json)，
各测试台仍使用
[`schemas/circuit.schema.json`](https://github.com/751K/circuit-optimization-lab/blob/main/schemas/circuit.schema.json)。
