# Circuit Optimization 文档

[English](README.md) | [中文](README_zh.md)

这里是 Circuit Optimization 的维护中项目文档，按三类读者组织：

- **使用者**：安装项目、运行已有电路、编写电路 JSON。
- **PDK 适配者**：了解模型绑定、外部工具、工艺角、后端能力和精度边界。
- **开发者**：理解求解器架构、扩展点、测试体系和发布检查。

## 从这里开始

1. [安装与快速上手](getting_started_zh.md)：使用 `uv` 或标准 `venv`
   建环境，运行无 PDK 的冒烟测试，并选择可选依赖。
2. [CLI 参考手册](cli_reference.md)：分析、探索、工艺角、失配、ADC、绘图、
   数据集和本地服务命令。
3. [JSON 电路描述格式](json_circuit_format_zh.md)：电路描述的字段级维护文档。
4. [PDK 支持矩阵](pdk_support_zh.md)：模型键、前置依赖、分析覆盖、工艺角和限制。

## 理解项目细节

- [核心求解器概览](module_overview_zh.md)：数据流、器件抽象、MNA/Newton、
  瞬态积分、周期分析、优化层和服务层。
- [开发者接手指南](development.md)：仓库地图、测试策略、扩展流程和文档维护规则。
- [本地服务 API](service_api_zh.md)：FastAPI 端点、后台任务、序列化和 CLI 对应关系。
- [ngspice Oracle 辅助层](ngspice_oracles.md)：FreePDK45 与 TSMC 回归中显式使用的
  外部参考路径。

## PDK 文档

- [TSMC28HPC+ 原生适配](tsmc28hpcp_zh.md)
- [PDK 支持矩阵](pdk_support_zh.md)

Foundry 模型文件只是本地输入，不属于仓库内容，必须保持 Git 忽略，并继续受原许可协议约束。

## 设计案例

[设计案例与验证状态](design_cases.md) 会区分可复现实例、未完成 campaign 和历史工程记录。
单个设计记录说明一次具体实验，不自动代表整个版本或整个 PDK 的精度保证。

## 文档状态

| 文档 | 用途 | 维护状态 |
|---|---|---|
| [安装与快速上手](getting_started_zh.md) | 安装和第一次运行 | 持续维护 |
| [CLI 参考手册](cli_reference.md) | 当前命令行接口 | 持续维护 |
| [JSON 电路描述格式](json_circuit_format_zh.md) | 电路 schema 与示例 | 持续维护 |
| [PDK 支持矩阵](pdk_support_zh.md) | 工艺和后端能力边界 | 持续维护 |
| [核心求解器概览](module_overview_zh.md) | 实现架构 | 持续维护 |
| [本地服务 API](service_api_zh.md) | HTTP 与任务 API | 持续维护 |
| [TSMC28HPC+ 适配](tsmc28hpcp_zh.md) | Licensed 本地模型接入 | 持续维护 |
| [许可证与第三方软件](third_party_licenses.md) | MIT 范围、BSIM4 致谢和第三方源码条款 | 持续维护 |
| [运行环境与性能基准](environment_performance.md) | 带日期的性能测量和调优记录 | 参考快照 |
| [设计案例](design_cases.md) | 设计推导与实验结果 | 按案例标记 |

版本变化见仓库根目录的
[CHANGELOG](https://github.com/751K/circuit-optimization-lab/blob/main/CHANGELOG.md)。
许可证和第三方署名信息见[许可证与第三方软件](third_party_licenses.md)。

## 文档维护规则

- “持续维护”文档中的命令必须与当前 `--help` 一致。
- 能力描述必须区分本地求解器、外部 oracle 和 sign-off 工具。
- 结果表必须说明数据来源和覆盖范围；不完整 PVT 不能写成已完成 campaign。
- 路线图和已经完成的实施计划不放在面向接手者的正式文档中；长期有效的内容并入架构或
  PDK 指南。
- 当文字与实现冲突时，以 `schemas/circuit.schema.json` 和 loader 行为为准。
