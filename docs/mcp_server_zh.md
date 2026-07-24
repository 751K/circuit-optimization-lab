# MCP 服务

[文档首页](README_zh.md) | [CLI 参考手册](cli_reference_zh.md) |
[English](mcp_server.md)

可选的 Model Context Protocol 服务让 LLM 客户端通过标准工具完成电路校验、仿真、
探索和签核。MCP 与本地 HTTP 服务调用同一组 application operations；数值计算仍由
Rust 求解器和原生 BSIM 后端完成，MCP 本身不包含器件方程。

## 安装与启动

```bash
uv pip install -e ".[mcp]"

# 本地 LLM 客户端推荐使用 stdio
circuit-opt mcp --transport stdio --workspace .

# 等价入口
circuit-opt-mcp --workspace .
python -m circuitopt.mcp --workspace .
```

本机 Streamable HTTP：

```bash
circuit-opt mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8342 \
  --workspace .
```

该服务没有鉴权，因此 HTTP transport 会主动拒绝非 loopback 监听地址。

## 客户端配置

客户端需要从已安装 Circuit Optimization 和 `circuitopt-core` 的环境启动：

```json
{
  "mcpServers": {
    "circuit-optimization": {
      "command": "/项目绝对路径/.venv/bin/circuit-opt-mcp",
      "args": [
        "--workspace",
        "/项目绝对路径"
      ]
    }
  }
}
```

客户端启动配置中的路径必然与机器有关；campaign 和测试台内部仍使用项目相对路径，
不会把机器路径写进电路配置。

## 工具

| 工具 | 行为 |
|---|---|
| `get_capabilities` | 返回已安装模型、分析、合法参数、工艺角和任务类型 |
| `validate_circuit` | 校验电路 JSON、分析参数和 signoff 契约 |
| `run_analysis` | 同步运行指定分析并返回有界摘要 |
| `submit_exploration` | 提交设计空间探索 |
| `submit_mismatch_mc` | 提交 mismatch Monte Carlo |
| `submit_signoff` | 提交工作区内的多测试台 PVT campaign |
| `list_jobs` | 列出后台任务 |
| `get_job` | 轮询状态，并可获取摘要或保存完整结果 |
| `cancel_job` | 请求协作式取消 |
| `inspect_signoff_result` | 按 case 和 PVT 筛选已保存的 signoff 结果 |

`run_analysis(save_result=true)` 会把完整序列化结果写到 `results/mcp/`；直接响应保留
标量指标，对长向量做压缩。`submit_signoff` 始终保存完整结果。这样既不会让协议响应
携带数 MB 波形，又不会丢掉 PVT 行或原始数据。

## Resources

- `circuitopt://capabilities`：JSON 能力快照。
- `circuitopt://workflow`：给 LLM 的简短调用顺序。

## 后台任务与取消

探索、失配 MC 和 signoff 共用进程内 `JobManager`：

```text
queued -> running -> done | failed | cancelled
```

取消是协作式的：正在运行的候选、MC 样本或 PVT 点会先结束。被取消的 signoff 会写出
带 `stopped_early: true` 的部分结果；模型失败和不收敛仍保持显式 invalid，不会被替代。

## 文件安全

- `--workspace` 是 MCP 唯一可访问的文件树。
- 工具只接受相对路径；绝对路径、`..` 和符号链接逃逸会立即失败。
- signoff campaign 必须是 JSON。
- 生成结果只能写入已被 Git 忽略的 `results/mcp/`。
- MCP 不提供任意 shell 执行或任意文件读取。

MCP 与 FastAPI 适配器共用 `circuitopt.service.operations`；两者都不实现数值逻辑或
静默求解降级。
