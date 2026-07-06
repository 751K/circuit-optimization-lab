# 本地服务 API

[项目概览](README.md) | [核心求解器概览](module_overview.md) |
[CLI 参考手册](cli_reference.md) | [English](service_api.md)

一个跑在本地的 FastAPI HTTP 层，架在与 CLI 相同的求解器栈之上。它是一层**薄适配**——
每个路由都直接转发给已有的单一事实来源（`circuit_from_dict`、`analysis_options`、
`run_analysis_suite`、`explore_from_dict`、`mismatch_mc_from_dict`），本身不带任何数值逻辑。
这是桌面 GUI 和未来 MCP server 共用的底座（见[后续开发计划](futureplan.md)）。

## 快速上手

```bash
# 安装可选的 serve extra（fastapi + uvicorn）
pip install -e ".[serve]"

# 启动服务（默认 127.0.0.1:8341，1 个任务 worker）
circuit-opt serve

# 等价的模块形式
python -m circuitopt.service
```

Swagger/OpenAPI 文档由 FastAPI 自动挂在 `http://127.0.0.1:8341/docs`（内置 UI，
不需要单独维护一份 schema 文件）。

```bash
curl http://127.0.0.1:8341/api/v1/health
# {"status":"ok","version":"0.1.0","api":"v1"}
```

### 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 绑定地址。默认仅本机可访问，见下面的"安全说明"一节。 |
| `--port` | `8341` | 监听端口。 |
| `--reload` | 关闭 | uvicorn 开发用自动重载。此模式下 `--job-workers` 不生效（reloader 通过 `circuitopt.service.app:create_app` 这个不带参数的工厂函数重新导入 app）。 |
| `--job-workers` | `1` | 后台任务（`explore`/`mc`）的线程池大小。求解是 CPU 密集型（NumPy/Numba）；只有在有空闲核心且需要并发跑多个任务时才调大。 |

所有路由都在 `/api/v1` 前缀下。

## 安全说明

这是一个**本地单用户服务**——没有鉴权、没有多租户隔离、也没有持久化（job 历史
存在内存里，重启即清）。`--host` 默认 `127.0.0.1`（仅本机回环）。传
`--host 0.0.0.0` 会把求解器（也就是任意算力）暴露到网络上，**风险自担**，
只在可信网络下这样做。

CORS 只放行 `http://localhost:<任意端口>` 和 `http://127.0.0.1:<任意端口>`
（正则 `^https?://(localhost|127\.0\.0\.1)(:\d+)?$`）——足够让本地的
Vite/Tauri 开发前端使用而不用固定端口号，除此之外任何来源都不允许从浏览器上下文调用本 API。

## 端点总览

| 方法 | 路径 | 用途 |
|------|------|------|
| `GET` | `/api/v1/health` | 存活探测 + 版本号 |
| `GET` | `/api/v1/capabilities` | 自描述：models、analyses、corners、job 种类 |
| `POST` | `/api/v1/validate` | 校验电路 JSON（恒 200） |
| `POST` | `/api/v1/solve` | 同步运行分析套件 |
| `POST` | `/api/v1/jobs/explore` | 提交设计空间探索任务 |
| `POST` | `/api/v1/jobs/mc` | 提交 mismatch Monte Carlo 任务 |
| `GET` | `/api/v1/jobs` | 列出任务（最新在前，不带 result payload） |
| `GET` | `/api/v1/jobs/{id}` | 任务状态 + （终结后）result/error |
| `DELETE` | `/api/v1/jobs/{id}` | 请求协作式取消 |
| `WS` | `/api/v1/jobs/{id}/events` | 流式推送进度，最后一条终结帧 |

## 同步端点

### `GET /api/v1/health`

```bash
curl http://127.0.0.1:8341/api/v1/health
```

```json
{"status": "ok", "version": "0.1.0", "api": "v1"}
```

### `GET /api/v1/capabilities`

GUI 下拉框的唯一事实来源——这里没有任何硬编码的编辑内容，全部反映当前构建实际支持的能力。

```bash
curl http://127.0.0.1:8341/api/v1/capabilities
```

```json
{
  "version": "0.1.0",
  "api": "v1",
  "models": {"pmos_tft": "circuitopt.pmos_tft_model.PMOS_TFT", "sky130.nmos": "...", "...": "..."},
  "analyses": {
    "ac": ["band", "corner", "freqs", "..."],
    "noise": ["...", "..."],
    "transient": ["...", "..."],
    "pss": ["...", "..."],
    "pac": ["...", "..."],
    "pnoise": ["...", "..."]
  },
  "corners": {
    "otft": ["fast", "slow", "typical"],
    "sky130": ["ff", "fs", "sf", "ss", "tt"],
    "freepdk45": ["ff", "nom", "ss"]
  },
  "jobs": ["explore", "mc"]
}
```

- `models` —— 器件模型注册表快照（`registered_models()`）：已注册的模型键 → 对应类的全限定名。
- `analyses` —— `ANALYSIS_ORDER` 中每个分析（`ac`、`noise`、`transient`、`pss`、`pac`、
  `pnoise`）各一项；值是该分析在 JSON `analyses.<name>` 块里的合法选项键排序列表
  （来自 `analysis_options.py` 的 `known_keys(name)`），客户端可据此校验或渲染表单，
  不需要重复维护一份选项注册表。
- `corners` —— 三个工艺角家族：`otft`（连续 OTFT PVT 偏移名）、`sky130` 与
  `freepdk45`（离散硅工艺角）。
- `jobs` —— 客户端可提交的后台任务种类（`explore`、`mc`）。

### `POST /api/v1/validate`

解析电路并校验其 `analyses` 块。**校验结果本身就是响应体——本端点恒返回 HTTP 200。**
错误逐项收集，不会在第一条错误处就停止，客户端可以一次性看到全部问题。

请求体：一个原始电路 JSON 对象（格式见 [JSON 电路描述格式](json_circuit_format_zh.md)）——
不套任何信封。

```bash
curl -X POST http://127.0.0.1:8341/api/v1/validate \
  -H "Content-Type: application/json" \
  -d @examples/periodic_rc.json
```

```json
{"valid": true}
```

电路有问题（缺必填字段，或 `analyses` 里某个选项键拼错）时仍返回 200：

```json
{"valid": false, "errors": ["'solved' is a required property", "..."]}
```

### `POST /api/v1/solve`

运行 `run_analysis_suite` 并返回 JSON-safe 结果——是 `circuit-opt run` 的编程等价物。

请求体：

```json
{
  "circuit": { "...": "电路 JSON 对象，见 json_circuit_format_zh.md" },
  "selected": ["ac", "noise"],
  "corner": "slow"
}
```

`circuit` 必填；`selected`（要跑的分析子集）和 `corner`（工艺角覆盖：OTFT
`typical`/`slow`/`fast`，或硅工艺角）可选。不给 `selected` 则跑电路 `analyses`
块里配置的全部分析。

```bash
curl -X POST http://127.0.0.1:8341/api/v1/solve \
  -H "Content-Type: application/json" \
  -d '{"circuit": '"$(cat examples/periodic_rc.json)"', "selected": ["ac"]}'
```

成功（`200`）：

```json
{
  "results": {
    "ac": {"Av_dc_dB": 22.90, "bw_Hz": 562.3, "response": [{"re": 1.0, "im": 0.0}, "..."]}
  },
  "elapsed_s": 0.0034
}
```

失败（`422`）——解析错误（电路结构不合法）和求解错误（如 DC 不收敛、`analyses`
选项键拼错）各带一个 `stage`，客户端可据此判断是哪个阶段失败。绝不泄漏 traceback。

```json
{"detail": {"stage": "parse", "message": "'solved' is a required property"}}
```

```json
{"detail": {"stage": "solve", "message": "unknown option(s) for 'ac': {'bogus_key'}; valid: [...]"}}
```

## 后台任务

`explore` 和 mismatch `mc` 可能跑几秒到几分钟，所以走**后台任务**模式：提交后轮询
或流式订阅进度，而不是占用一个 HTTP 请求一直等。生命周期细节见下面的
下面的"Job 状态机"一节。

### `POST /api/v1/jobs/explore`

与 `circuit-opt explore` 语义完全一致——两者都走共用入口 `explore_from_dict`，因此不会漂移。

请求体：

```json
{
  "circuit": { "...": "含 'explore' 块的电路 JSON" },
  "n": 300,
  "seed": 42,
  "corner": "slow"
}
```

只有 `circuit` 必填；`n`、`seed`、`corner` 缺省时落到 `explore_from_dict` 的默认值
（`n=200`、`seed=0`、无 corner）。

```bash
curl -i -X POST http://127.0.0.1:8341/api/v1/jobs/explore \
  -H "Content-Type: application/json" \
  -d '{"circuit": '"$(cat examples/afe_explore.json)"', "n": 300, "seed": 42}'
```

```
HTTP/1.1 202 Accepted
{"job_id": "a1b2c3d4e5f6", "kind": "explore", "status": "queued"}
```

### `POST /api/v1/jobs/mc`

与 `circuit-opt mc` 语义完全一致（共用入口 `mismatch_mc_from_dict`）。注意这里的
`corner` 是**基底工艺角**（`typical`/`slow`/`fast`），mismatch 是叠加在它上面的——
和 `jobs/explore` 的 `corner`（OTFT/硅分析工艺角）不是一回事。

```json
{
  "circuit": { "...": "电路 JSON 对象" },
  "n": 300,
  "seed": 1,
  "corner": "typical"
}
```

```bash
curl -i -X POST http://127.0.0.1:8341/api/v1/jobs/mc \
  -H "Content-Type: application/json" \
  -d '{"circuit": '"$(cat examples/afe_explore.json)"', "n": 300, "seed": 1}'
```

```
HTTP/1.1 202 Accepted
{"job_id": "f6e5d4c3b2a1", "kind": "mc", "status": "queued"}
```

### `GET /api/v1/jobs`

最新在前的状态快照列表——不带 `result`/`error` payload（即使某个 `explore`/`mc`
结果很大，列表本身仍然轻量）。

```bash
curl http://127.0.0.1:8341/api/v1/jobs
```

```json
{"jobs": [
  {"job_id": "a1b2c3d4e5f6", "kind": "explore", "status": "running",
   "created": 1751000000.0, "started": 1751000000.1, "finished": null,
   "progress": {"type": "progress", "done": 42, "total": 300, "frac": 0.14}}
]}
```

内存最多保留 **50 个任务**；超过上限时驱逐最旧的、已终结的任务（正在跑/排队中的
任务永远不会被驱逐）。服务重启会清空全部任务历史——这是一个本地、非持久化的服务。

### `GET /api/v1/jobs/{id}`

完整状态：与列表条目相同的快照，外加 `result`（一旦 `status == "done"`，或
`"cancelled"` 且有部分结果）或 `error`（一旦 `status == "failed"`）。

```bash
curl http://127.0.0.1:8341/api/v1/jobs/a1b2c3d4e5f6
```

```json
{
  "job_id": "a1b2c3d4e5f6", "kind": "explore", "status": "done",
  "created": 1751000000.0, "started": 1751000000.1, "finished": 1751000012.4,
  "progress": {"type": "progress", "done": 300, "total": 300, "frac": 1.0},
  "result": {"candidates": ["..."], "summary": {"n": 300, "feasible": 87, "pareto": 12}, "objectives": "..."}
}
```

未知 id → `404`：

```json
{"detail": {"stage": "job", "message": "unknown job 'deadbeef0000'"}}
```

### `DELETE /api/v1/jobs/{id}`

请求协作式取消。取消语义见下面的"Job 状态机"一节。

```bash
curl -X DELETE http://127.0.0.1:8341/api/v1/jobs/a1b2c3d4e5f6
```

```json
{"job_id": "a1b2c3d4e5f6", "status": "cancelling"}
```

未知 id → `404`；已终结的任务 → `409`（无需取消），两者的 detail 都是同一种
`{"stage": "job", "message": ...}` 形状：

```json
{"detail": {"stage": "job", "message": "job 'a1b2c3d4e5f6' already terminal (done)"}}
```

### `WS /api/v1/jobs/{id}/events`

为一个运行中的任务流式推送进度帧，最后恰好一条终结帧，然后关闭连接。未知任务 id
则只收到一条 error 帧后立即关闭。

```bash
# 用 websocat 或任意 WS 客户端
websocat ws://127.0.0.1:8341/api/v1/jobs/a1b2c3d4e5f6/events
```

**帧序列 —— `mc` 任务**（N 条进度帧，每条带一个滚动更新的 `partial` 统计，最后一条终结帧）：

```json
{"type": "progress", "done": 1, "total": 300, "frac": 0.0033, "partial": {"n": 1, "latched": 0, "latch_rate": 0.0, "noise_evaluated": 1}}
{"type": "progress", "done": 2, "total": 300, "frac": 0.0067, "partial": {"n": 2, "latched": 0, "latch_rate": 0.0, "noise_evaluated": 2}}
"...": "每完成一个样本一条"
{"type": "terminal", "status": "done"}
```

**帧序列 —— `explore` 任务**（形状相同，但没有 `partial`——explore 的进度只是一个百分比）：

```json
{"type": "progress", "done": 1, "total": 300, "frac": 0.0033}
"...": "每完成一个候选点一条"
{"type": "progress", "done": 300, "total": 300, "frac": 1.0}
{"type": "terminal", "status": "done"}
```

**未知任务 id：**

```json
{"type": "error", "message": "unknown job 'nope00000000'"}
```
随后连接关闭。

**失败任务**的终结帧携带与同步端点相同的 `{stage, message}` 错误形状：

```json
{"type": "terminal", "status": "failed", "error": {"stage": "solve", "message": "..."}}
```

如果客户端在任务已经结束之后才连接（队列里的事件已被 worker 线程自己排空），
仍会收到一条由任务已记录状态重建出的终结帧——晚订阅者永远不会因为等一条已经
发生过的帧而挂住。

## Job 状态机

```
queued -> running -> { done, failed, cancelled }
```

三个终结态都是最终态。`result` 在 `done` 时被填充（`cancelled` 且有部分结果时也会
填充）；`error`（一个 `{"stage", "message"}` 信封，与 422 的 detail 形状一致）在
`failed` 时被填充。

- **取消是协作式的，不是硬杀。** `DELETE /api/v1/jobs/{id}` 只是设置一个标志位；
  正在跑的那个候选点/样本总会先跑完（没有安全的办法在 NumPy 求解调用中途打断它）。
  驱动函数一旦注意到这个标志位就会停下，返回目前已经产生的结果。
- **部分结果会被保留。** 被取消任务的 `result`（以及 `result.summary`）会带
  `"stopped_early": true` 标记。实际完成的数量：`explore` 任务在
  `summary.evaluated`（`summary.n` 是最初*请求*的数量）；`mc` 任务在 `summary.n`
  （`mc` 的 `summary.n` 本来就始终是实际评估完的样本数，不管任务是否跑到底，这条
  路径上没有单独的"请求数"字段）。
- **在 `queued` 阶段就请求取消**会在任何求解器工作开始之前短路，任务直接进入 `cancelled`。
- 任务状态完全存在**内存**里；没有跨重启的持久化，最多保留 50 个任务
  （最旧的已终结任务先被驱逐——见 [`GET /api/v1/jobs`](#get-apiv1jobs)）。

## 序列化约定

求解器结果是由 NumPy 标量、ndarray、Python `complex` 数以及嵌套 dict/list 组成的——
不是严格 JSON。每个响应都会经过 `circuitopt.service.serialize.to_jsonable` 递归处理，
规则如下：

| 输入 | 输出 |
|------|------|
| `numpy` 标量（`np.float64`、`np.int64`、`np.bool_` 等） | 原生 Python 标量 |
| `numpy.ndarray` | 嵌套 Python `list`（逐元素处理，含复数/NaN 的数组也一并处理） |
| `complex`（Python 或 numpy） | `{"re": <float>, "im": <float>}` |
| `NaN`、`+Inf`、`-Inf`（裸值，或出现在复数/数组元素内部） | `null` |
| 以 `_` 开头的 dict key，或任何 callable 值 | 丢弃 |
| `bytes` | 尽力 UTF-8 解码为字符串 |
| 其余已经是 JSON 原生类型的值 | 原样透传 |

NaN/Inf → `null` 这条规则的原因是严格 JSON（RFC 8259）没有非有限浮点数的字面量；
这保证每个响应都能被标准兼容的 JSON 解析器解析（不依赖 `json.dumps` 的非标准
`NaN`/`Infinity` token）。`POST /solve` 的 `results` 对象里，值为 `None` 的分析条目
（没产出结果的分析）会被整个丢弃。

## 与 CLI 的对应关系

这里的每个端点都对应一个已有的 CLI 子命令——服务层不引入任何新的求解器行为，
只是给同一批入口套一层 HTTP 传输。各命令的完整参数参考见 [CLI 参考手册](cli_reference.md)。

| HTTP 端点 | CLI 对应 | 共用入口函数 |
|-----------|---------|-------------|
| `POST /api/v1/solve` | `circuit-opt run` | `run_analysis_suite` |
| `POST /api/v1/jobs/explore` | `circuit-opt explore` | `explore_from_dict` |
| `POST /api/v1/jobs/mc` | `circuit-opt mc` | `mismatch_mc_from_dict` |
| `POST /api/v1/validate` | （无直接 CLI 对应） | `circuit_from_dict` + `validate_analysis_cfg` |
| `GET /api/v1/capabilities` | （无直接 CLI 对应） | `registered_models`、`analysis_options.known_keys`、`device_factory.CORNERS`/`SKY130_CORNERS`、`freepdk45_model.FREEPDK45_CORNERS` |

因为两个入口调用的是同一批底层函数，一个电路在 shell 上用
`circuit-opt run/explore/mc` 跑通，通过对应的 HTTP 端点也会得到完全相同的结果
（同样的 seed → 同样的输出）。

## 另请参阅

- [JSON 电路描述格式](json_circuit_format_zh.md) —— 本 API 里每个 `circuit`
  字段遵循的 schema。
- [CLI 参考手册](cli_reference.md) —— `serve` 子命令及所有其他命令行入口。
- [核心求解器概览](module_overview_zh.md) —— 模块地图里的 `service/` 子包条目，
  以及每个端点背后调用的求解器内部实现。
- [后续开发计划](futureplan.md) —— 服务层在更大的桌面/MCP 路线图里的位置。
