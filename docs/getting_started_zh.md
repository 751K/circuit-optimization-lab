# 安装与快速上手

[文档首页](README_zh.md) | [English](getting_started.md)

本指南先在不依赖任何外部 PDK 的情况下把项目跑通。

## 环境要求

- Python 3.10 或更高版本。
- 推荐使用 `uv` 管理环境；标准 `venv` 和 `pip` 仍然支持。
- 只有首次构建原生紧凑模型后端时才需要可用的 C 编译器。
- 只有使用对应硅工艺时才需要 PDK 文件和外部工具，见
  [PDK 支持矩阵](pdk_support_zh.md)。

## 使用 uv 安装

在仓库根目录执行：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

修改项目或运行完整测试时安装开发依赖：

```bash
uv pip install -e ".[dev]"
```

常用可选依赖：

```bash
uv pip install -e ".[ml]"       # scikit-learn surrogate
uv pip install -e ".[torch]"    # 可微 PyTorch surrogate
uv pip install -e ".[plot]"     # matplotlib 绘图
uv pip install -e ".[serve]"    # FastAPI 和 uvicorn
uv pip install -e ".[parquet]"  # Parquet 数据集导出
```

## 使用标准 venv 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 验证核心安装

无源 RC 示例不需要晶体管模型或 PDK：

```bash
circuit-opt run examples/periodic_rc.json --analysis ac,noise
```

运行该 JSON 中配置的全部分析：

```bash
circuit-opt run examples/periodic_rc.json
```

模块入口与命令行脚本等价：

```bash
python -m circuitopt run examples/periodic_rc.json --analysis ac,noise
```

测试级检查：

```bash
pytest -q tests/test_cli_subcommands.py tests/test_periodic_solvers.py
```

## 运行晶体管示例

默认晶体管工艺是项目内置的 AT4000TG PMOS 模型：

```bash
circuit-opt run examples/single_stage.json --analysis ac,noise
```

其他工艺通过 JSON `models` 字段逐器件选择，不是全局仿真器开关。详见
[JSON 电路描述格式](json_circuit_format_zh.md)和
[PDK 支持矩阵](pdk_support_zh.md)。

## 常见工作流

```bash
# 设计空间探索
circuit-opt explore examples/afe_explore.json -n 200 --seed 1

# 按电路配置的工艺运行工艺角
circuit-opt corners examples/afe_explore.json

# AT4000TG 逐器件失配 Monte Carlo
circuit-opt mc examples/afe_explore.json -n 100 --seed 1

# 生成 surrogate 数据集
circuit-opt dataset examples/single_stage.json -n 500 --out results/datasets/single

# 启动可选本地 API
circuit-opt serve
```

在硅 PDK 上使用这些流程前先看 [CLI 参考手册](cli_reference.md)；不同后端的
corner 和 mismatch 覆盖并不相同。

## 路径与可迁移性

电路 JSON 不需要记录机器绝对路径。项目会依次检查显式环境变量、当前虚拟环境、
项目 `.venv` 和约定的项目内位置。

常用变量：

| 变量 | 用途 |
|---|---|
| `PDK_ROOT` | SKY130 和 FreePDK45 安装根目录 |
| `TSMC28_MODEL_DIR` | 包含受支持 TSMC HSPICE 模型文件的目录 |
| `TSMC28_PDK_ROOT` | TSMC iPDK 或交付包外层目录 |
| `NGSPICE_BIN` | ngspice 可执行文件，用于相关后端或 oracle 对照 |
| `OPENVAF_BIN` / `OPENVAF_ROOT` | OpenVAF-Reloaded 编译器或源码目录 |
| `BSIM4_VA` | 显式指定 BSIM4 Verilog-A 源文件 |
| `OSDI_CACHE_DIR` | OSDI 构建缓存 |
| `CIRCUITOPT_NATIVE_MODEL_CACHE` | 原生紧凑模型构建缓存 |

不要提交 licensed 模型、生成的模型卡、虚拟环境或仿真缓存。

## 下一步

- 编写电路：[JSON 电路描述格式](json_circuit_format_zh.md)
- 运行分析：[CLI 参考手册](cli_reference.md)
- 选择工艺：[PDK 支持矩阵](pdk_support_zh.md)
- 供其他应用调用：[本地服务 API](service_api_zh.md)
- 修改代码：[开发者接手指南](development.md)
