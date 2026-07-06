# 后续开发计划

[English README](README.md) | [中文说明](README_zh.md) | [核心求解器概览](module_overview_zh.md)

本项目有两个定位：①面向 ML-for-EDA 的开源基础设施（数据生成器框架 + ML 方法框架）；
②可供 LLM 本地调用的模拟电路设计工具链。当前基础已经就位：三个工艺（AT4000TG OTFT 校准锚 /
SKY130 OSDI / FreePDK45 ngspice-C）、全分析栈（DC/AC/noise/transient/PSS/PAC/PNoise）、
Cadence byte-gate 5/5、数据集→代理→优化闭环、两个硅 FD-OTA 全流程设计案例、`CircuitBinding`
统一编程接口。已完成的细节见 `docs/module_overview.md`、`README.md` 和 git 历史；本文档只记录
**未来方向**。

---

## 核心架构判断

桌面客户端和 LLM 代理接口最终会共享同一个底座——**本地服务层**。`demo/server.py`
（Flask 版 AFE Tuner）已经是这个服务层的雏形；`CircuitBinding` + `run_analysis_suite`
已经是一套干净的编程接口，服务层要做的封装会很薄。策略是先把底座建好，再让桌面端和
LLM 端两个前端在其上并行生长，而不是各自造一套调用逃生通道。

---

## 方向一：本地服务层（优先级最高，先行）

把 demo 现有的 Flask 单页应用升级为通用的本地 daemon：

- **技术选型**：FastAPI + WebSocket，取代当前的同步 Flask 路由。
- **能力范围**：
  - 电路 JSON 的 CRUD（`json_circuit_format` 已经是稳定契约，直接复用）；
  - 六类分析（DC/AC/noise/transient/PSS/PAC/PNoise）按需执行；
  - `explore`/`optimize`/`dataset` 任务化——长任务通过 WebSocket 流式推送进度，
    而不是阻塞等待整批完成；
  - 结果缓存和器件表征网格（SKY130/FreePDK45 的 `.npz` 缓存）的管理接口。
- **交付物**：一个 `serve` 子命令或独立进程；一份 OpenAPI schema——这份 schema
  同时是桌面客户端和 MCP server 的调用契约，两个前端不用各自约定一遍接口。

---

## 方向二：macOS 桌面客户端（优先级高，依赖方向一）

### 壳选型

- **Tauri v2（首选）**：安装包体积小（~5MB）、用系统自带 WebView、sidecar 机制可以
  管理本地 Python daemon 的生命周期。代价是需要把 Python 服务打成二进制 sidecar，
  或者要求本机已有 Python 环境。
- **Electron（备选）**：Node 主进程可以直接拉起 Python 子进程，不需要编译 sidecar，
  工程实现更省事，代价是安装包体积比 Tauri 大一个量级。
- **原生 SwiftUI（不选）**：前端资产完全无法复用，而 `demo/static` 下已经有一套
  可直接迁移的 web 前端资产，没有理由从零写原生 UI。

参考对比：[Tauri vs Electron 2026](https://www.gethopp.app/blog/tauri-vs-electron)、
[DoltHub: Electron vs Tauri](https://www.dolthub.com/blog/2025-11-13-electron-vs-tauri/)。

### 分期交付

- **V0**：把现有 AFE Tuner 桌面化——参数旋钮实时调参 + Bode/噪声曲线展示。
- **V1**：电路编辑器（JSON 表单或拓扑视图）+ 六类分析面板 + 波形查看器。
- **V2**：`explore`/`optimize` 仪表盘（交互式 Pareto 前沿、PVT corner 表、优化
  收敛监控）+ 设计报告导出（把现有 markdown 设计报告 GUI 化）。

---

## 方向三：LLM 设计代理接口——MCP server（优先级高，工程量小，可先于方向二启动）

2026 年 MCP 已经是行业事实标准，主流模型都原生支持。数字 EDA 领域的
AutoEDA（[arXiv:2508.01012](https://arxiv.org/abs/2508.01012)）已经示范了用
MCP + 微服务控制 RTL-to-GDSII 流程——本项目要做的是**模拟电路版**：

- **Tools**：`load_circuit` / `run_analysis`（六类）/ `explore` / `optimize` /
  `build_dataset` / `train_surrogate` / `corner_sweep`。
- **Resources**：电路 JSON、分析结果、PDK 器件特性网格。
- **护栏**：评估预算上限（调用次数/时长），防止代理失控烧算力。

差异化优势：全本地运行（隐私、无许可证成本）、经过校准的真实求解器（byte-gate
背书，不是玩具仿真器）、SKY130/FreePDK45 两个 FD-OTA 案例已经证明"LLM 跑完整
设计闭环"是可行的。

MCP server 与方向一共享同一个本地服务层，本身只是一层薄适配。

---

## 方向四：ML 规模化与公共基准（持续研究线）

对标对象：AnalogGym（ICCAD'24，30 个拓扑，
[arXiv:2409.08534](https://arxiv.org/abs/2409.08534)）、
AICircuit（[arXiv:2407.18272](https://arxiv.org/abs/2407.18272)）、
OSIRIS（87k 变体数据集，[arXiv:2601.19439](https://arxiv.org/abs/2601.19439)）——
它们的共同点是依赖外部仿真器生成数据。本项目的差异化卖点是：**数据生成器本身就是
一套校准过的本地求解器**——生成速度快、逐字节可复现、零许可证成本。

计划中的方向：

1. **多拓扑电路库扩展**——OTA 家族（两级、折叠式、telescopic）、LDO、带隙基准、
   滤波器。JSON 格式已经支持任意拓扑，缺的是电路本身和对应的 explore/dataset 配置。
2. **跨拓扑数据集工厂**——统一 schema + provenance，让不同拓扑产出的数据集可以
   放进同一个训练/评测流水线，而不是每个拓扑各写一套 dataset 脚本。
3. **主动学习闭环**——用代理的不确定度驱动采样，替代当前一次性 LHS 采样；
   现有 `dataset` 流水线已经具备逐候选打标签的能力，缺的是采样策略的反馈回路。
4. **神经代理升级**——现有的 torch MLP（`circuitopt/surrogate_torch.py`）升级为
   不确定度感知（用于驱动上面的主动学习）、多任务（同时预测多个标签组）、
   跨拓扑迁移（减少新拓扑的数据需求）的模型。
5. **公开数据集 + 基准发布**——固定 split + 复现脚本，让外部研究者可以直接
   对比自己的 surrogate/优化方法，而不需要重新搭建仿真环境。

---

## 方向五：工艺与分析面扩展（按需启动）

- **新 PDK**：GF180MCU（BSIM4，OSDI 路径可以直接复用，接入方式与 SKY130 一致）、
  IHP SG13G2（开源 SiGe，需要评估其紧凑模型是否有现成的 Verilog-A/OSDI 版本）。
- **硅工艺失配 MC**：`corners`/`mc` 目前只覆盖 OTFT 的连续 PVT shift，硅侧（SKY130/
  FreePDK45）还缺逐器件 mismatch 机制，需要参考对应 PDK 的统计模型文档补齐。
- **新分析类型**：环路增益/稳定性（STB 类分析，闭环系统设计的常用指标）、THD/失真
  （大信号非线性度量，目前的分析栈只覆盖小信号指标）。
- **性能**：libngspice 进程内绑定替代当前的子进程批量表征（减少表征往返开销）、
  `explore`/`dataset` 的多进程并行（当前是单进程顺序评估候选点）。

---

## 方向六：工程化发布（伴随节奏推进）

pip 打包、CI（GitHub Actions——全套测试和 byte-gate 所需的 fixture 都已入库，
可以完整搬上 CI）、语义化版本、文档站。

---

## 优先级总排序

**方向一 → 方向三 → 方向二**：服务层是两个前端共用的底座，先做；MCP 工程量小、
且最贴合"LLM 可本地调用"的项目定位，优先于工程量更大的桌面端；桌面端随后跟进。
**方向四**是持续研究线，与工程方向并行推进，不占用发布节奏。**方向五**按需启动，
不预先排期。**方向六**伴随各方向的发布节奏推进，不单独立项。

---

## 明确不做（non-goals）

| 事项 | 原因 |
| --- | --- |
| 把求解器重写成 Rust | 已评估并拒绝：Numba 加速已经足够，重写收益不足以覆盖迁移成本 |
| 拆分 `numba_kernels.py` / `chopper.py` 成小模块 | 这两个文件是 Cadence 校准锚点，拆分带来的回归风险大于可维护性收益 |
| 全量类型注解 | 公共 API 已经注解，内部实现补全类型注解的边际收益低 |

---

## 未完成的既有里程碑

- **`evaluate` 双路径收敛为 binding-only**：`circuitopt.explore.evaluate` 目前
  `model_types=`/`device_kwargs=` 旧 kwargs 与 `binding=` 新路径并存，内部调用者
  （`explore`/`dataset`/`optimize`/`surrogate_torch`）已经全部迁移到 binding，
  旧路径仅为外部脚本兼容保留。删除旧路径是公共 API 的破坏性变更，需要等外部调用面
  稳定后再执行，并预留迁移窗口。
