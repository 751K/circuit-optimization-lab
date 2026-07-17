# Rust 核心重写方案（codex/rust-core-rewrite）

> 内部规划文档，2026-07-17，基于 v1.4.0（分支 `codex/rust-core-rewrite`）。
> 目标读者：本仓库维护者与实现 agent。行号引用以 v1.4.0 为准。

## 0. 一句话

把 v1.4.0 中由 **numba 执行**的全部求解器热路径（`numba_kernels.py` 的 ~50 个
单源 `_impl` 内核 + 其 marshalling 层）改由 **Rust crate** 执行，BSIM4.5 的
vendored Berkeley C 内核**保留 C 源**、由 Rust 在 wheel 构建期编译并以安全 FFI
包裹（替代现在的"用户机器运行时 cc + ctypes 函数指针"）；Python 保留全部编排、
CLI、service、PDK 卡解析与 oracle。最终从依赖中删除 numba。

## 1. 目标与非目标

### 1.1 目标（按优先级）

1. **单一加速路径**：消灭 numba 执行路径及其约束集——`py_impl` 双执行模式、
   `_NUMBA_GRID_ARG_GROUPS` ~90 个位置参数的 marshalling 税
   （`transient_solver.py:287-524`）、被测试钉住的内核签名契约
   （`tests/test_transient_contracts.py`）、PAC 线性化的 Python/numba 真双份
   （`pac_solver.py:426-605`）、OSDI 的 must-mirror 对
   （`numba_kernels.py:940-942`）、numba↔numpy 版本耦合
   （`pyproject.toml:31-33` 注释）。
2. **免预热 + 免运行时编译器部署**：预编译 wheel 取代"首次使用时用用户机器的
   clang/cc 编 BSIM4 dylib"（`compact_models/bsim4/native.py:78-148`）与 numba
   JIT 首调延迟/磁盘缓存。
3. **真·多线程正确性与扩展**：现状 vendor C 有 file-scope 可变全局
   （`b4v5ld.c` ~128 个），numba 内核经函数指针直呼它们且无锁——线程池下的硅
   仿真是**潜在数据竞争**（今天只是事实上单线程使用才安全）。Rust 层给出
   受控的并发模型（先全局锁保等价，后 `_Thread_local` 补丁解锁 rayon）。
4. **数值行为不变**：全部现有验收门（Cadence byte-gate、`cadence_regression`、
   物理不变量、内核等价测试的数值断言）在规定容差内继续通过。

### 1.2 非目标

- **不追求内核吞吐飞跃**。numba 即 LLVM，warm 内核已近 C 速度
  （`docs/environment_performance.md`：斩波器 numba on/off 7.5 s vs 221 s）。
  Rust 的量级收益在启动、部署、并行与维护，不在单核 FLOPS。验收目标是
  ≥0.9×（争取 ≥1×），不是 10×。
- **不重写编排层**：scipy 所有权不变——DC 的 `fsolve`(MINPACK hybrd)/有界
  `least_squares`(TRF)、续延与 gmin 阶梯（`ac_solver.py:126-234`）、pnoise 的
  `splu`/`gmres`/Woodbury（`pnoise_solver.py:455-568`）、FFT 编排全部留在
  Python。这是刻意的风险控制：DC 的根选择行为（AFE 多稳态守卫）与 Woodbury
  的逐位契约都依赖 scipy 的具体实现。
- **不动 oracle**：ngspice 全套（`ngspice_*.py`）、Cadence 校准链、OSDI 的
  oracle 定位（v1.4 起显式导入）不变。
- **不动 JSON schema / CLI 子命令集 / service 路由 / 结果字典键**（§7 兼容清单）。
- **不做 no_std / WASM / GPU**。

### 1.3 对既有"不需要 Rust"结论的回应

`docs/environment_performance.md:137`（2026-07-04）：“Rust 已无必要性论据
（仅剩多核无 GIL 批量与免预热部署两个场景，均非当前需求）。”该结论在当时
成立（其上下文是"OSDI 能进 numba 循环"的性能问题）。现在改变的不是性能事实，
而是三点：

1. **战略决策**：维护面统一为 Rust（用户 2026-07-17 决定）。numba 路径的维护
   成本不在速度而在约束集与版本耦合（§1.1-1）。
2. 那两个"非当前需求"已成为需求：PVT campaign / MC / dataset 构建要无 GIL
   多核；桌面分发（Tauri 前端 + `circuit-opt serve`）要免预热、免运行时 cc。
3. **正确性论据是新的**：vendor C 的线程不安全在 numba 方案下无干净修法
   （锁得在 jit 核外、粒度失控），Rust FFI 层是自然的修复位置。

R0 阶段在该文档追加"结论已被 v2.0 Rust 决策取代"的注记，保留原文为历史。

## 2. 现状事实（盘点结论，载有出处）

- **求解器族**：DC（scipy fsolve/TRF + 续延 + gmin 阶梯，无独立入口，内嵌于
  `ac_solve`）、AC（G+jωC 一次批量复数稠密解，`ac_solver.py:359-362`）、
  NOISE（伴随转阻抗一次批量解，`noise_solver.py:169-176`）、TRAN（BE/变步长
  BDF2，阻尼 Newton + 手写原地 GEPP ≤~12×12，步二分重试，步倍增 LTE 自适应，
  `numba_kernels.py:2289-3155`）、PSS（打靶 + 解析单值矩阵 + LM，
  `pss_solver.py:54-178,612-617`）、PAC（LTI 捷径→时域打靶→HB 伴随→FD 打靶
  四级瀑布）、PNOISE（HB 伴随 + 时域 Floquet/Woodbury 6.6×）、chopper（包装层）。
- **内核**：`numba_kernels.py` 3952 行、50 核，统一 `njit(cache=…)`，
  **无 fastmath/parallel/nogil 标注**（nogil 由 nopython 隐含）、严格 IEEE、
  指针以运行时参数传入保 `cache=True` 跨进程（`numba_kernels.py:934-936`）。
- **器件**：默认 PDK 是 **OTFT 解析模型**（`at4000tg.pmos`，内部 2 节点
  Newton，方程单源在 `_impl` 内核，参数 ABI = `NumbaParams` 16 字段）；硅 PDK
  （freepdk45/sky130/tsmc28）走 vendored Berkeley BSIM4.5 C（编译集 11 文件
  ≈16.3k 行）+ `host.c`（949 行适配层：内部节点发现、≤7 内点 Schur 消元、
  电容阵、噪声 4×4 互谱），运行时 cc 编译、hash 缓存 dylib、ctypes 加载、
  numba 内核经 `CFUNCTYPE` 全 `void*` ABI 直呼（`co_bsim4_eval_vp`）。
  已知缺陷：`co_bsim4_destroy` 泄漏 size-depend param 链；DC Newton 内每次
  eval 都跑 acLoad 算电容（host.c:659-693，纯浪费）；线程不安全（§1.1-3）。
- **并发模型**：全仓**零 multiprocessing**，只有 `ThreadPoolExecutor`
  （`sar.py:319`、`sar_mc.py:265`、`sar_explore.py:379`、`service/jobs.py:154`），
  依赖 numba nogil 放 GIL。Rust 入口必须 `allow_threads` 否则静默串行化。
- **契约面**：公共 API = `__init__` 导出 + JSON 格式 + CLI flags
  （CHANGELOG.md:13-15）；深路径 import（examples 直用 `compiled_topology` 等）；
  `--no-numba` 语义被 `tests/test_cli_numba_flag.py` 钉死（env 导入期烘焙 +
  argv 预扫，`numba_kernels.py:58-74`、`__init__.py:15-26`）；数值门 =
  CI byte-gate `python -m circuitopt.calibration --all`（dc 1e-3 V、ac 1%、
  IRN 3%、pnoise 3-5%，`calibration.py:47-56`）+ 默认跑的 `cadence_regression`
  + 物理不变量（KCL/电荷守恒 atol 1e-18/1e-24，`test_freepdk45_native.py:69-108`）。
- **Rust 既有痕迹**：`experiments/rust_device/`（PyO3 0.29、edition 2024、
  cdylib、OTFT eval/newton 镜像 spike）；`frontend/src-tauri`（Tauri 2，
  edition 2021）。`tools/version.py` 已会同步 Cargo.toml。CI 无 Rust 工具链；
  **本机（2026-07-17）cargo 不在 PATH**——R1 第一件事是装 rustup。
- **数据契约现成**：`CompiledTopology.term_arrays`/`index_array` 已把电路平铺
  为 `(kind:i64, ref:i64, value:f64)` 三元组 + 各元件平行数组
  （`compiled_topology.py:19-46`，清单见 `transient_solver.py:158-219`）——
  这就是 Rust 入口的天然 ABI，**不需要发明新的电路交换格式**。

## 3. 目标架构

```
repo/
├── rust/                          # 新增：cargo workspace（与 frontend/src-tauri 互不隶属）
│   ├── Cargo.toml                 # [workspace.package] version ← tools/version.py 同步
│   ├── vendor/bsim4v5-ngspice/    # 从 circuitopt/compact_models/bsim4/native_src/ 迁入
│   │   └── (11 个 C 编译单元 + include 树 + NOTICE + B4TERMS_OF_USE)
│   └── crates/
│       ├── co-bsim4/              # build.rs 用 cc crate 编 vendor C；host.c 的 Rust 移植
│       │   └── (create/set/setup/eval[拆 DC|full]/noise/batch + shim 符号 + 并发策略)
│       ├── co-core/               # 求解内核：device trait、OTFT、stamp、Newton、
│       │   └── (BE/BDF2 grid、adaptive、AC/noise 装配、周期族核)
│       └── co-py/                 # PyO3 (abi3-py310) + rust-numpy → Python 模块 circuitopt_core
├── circuitopt/                    # 不动结构；新增 engine 分发薄层
│   └── _engine.py                 # CIRCUIT_ENGINE 烘焙 + circuitopt_core 惰性导入 + 回退
└── (其余不变)
```

发布形态：**双发行版**。

- `circuitopt-core`（maturin 构建）：仅 Rust 扩展，顶层模块名 `circuitopt_core`，
  abi3-py310 单 wheel/平台（macOS arm64/x86_64 + manylinux x86_64/aarch64，
  对齐 native.py 现有的 macOS/Linux-only 支持面）。
- `circuit-optimization`（setuptools，现状）：过渡期不硬依赖 core（engine 缺失
  时按 §5 R6 前的默认回退 numba）；**翻转版（v2.0.0）起精确 pin
  `circuitopt-core==X.Y.Z`**，`tools/version.py` 负责双向同步与 CI check。

选双发行版而非整包转 maturin 的理由：主包保持纯 Python 可装可 `-e .`（贡献者
无 cargo 也能改编排层/文档/测试）；release 工作流增量演进；两包版本锁死由已有
version.py + `test_versioning.py` 机制守护。GitHub-Release-only 的发布方式下
双工件摩擦很小。

### 3.1 模块去向表

| 现模块 | 行数 | 去向 |
|---|---|---|
| `numba_kernels.py` 内核体 | ~3850 | → `co-core`（1:1 转写，保运算顺序）；Python 源降级为 reference engine（§4-D4） |
| `numba_kernels.py` njit 装饰/注册 | ~100 | 死 |
| `transient_solver.py` marshal/ctx (`:158-524,912-1104`) | ~700 | 塌缩为一次 PyO3 调用的结构组装 |
| `transient_solver.py` 其余（分派/回退/结果组装） | ~850 | 留 Python |
| `compiled_topology.py` | 606 | 留 Python（编译期）；flat arrays = FFI 契约 |
| `ac_mna.py` | 208 | → `co-core`（AC/noise 装配核）；Python 版留 reference |
| `ac_solver.py` / `dc_solver.py` / `noise_solver.py` 编排 | ~880 | 留 Python（scipy 所有权不变）；批量复数解可选迁 Rust（R3 尾，rayon+放 GIL） |
| `pss/pac/pnoise/chopper` 编排 | ~4600 | 留 Python |
| 周期族 jit 核（HB 块、fold、轨道线性化，`numba_kernels.py:3397-3864`） | ~470 | → `co-core`；**删** `_assemble_pac_linearization_python` 双份 |
| `pmos_tft_model.py` 方程（经 `_impl`） | — | → `co-core` device trait 实现；`NumbaParams` 16 字段为参数 ABI |
| `compact_models/bsim4/native_src/vendor` | ~16.3k C | **保留 C**，迁至 `rust/vendor/`，build.rs 编译（wheel 期，非用户运行时） |
| `compact_models/bsim4/native_src/host.c` | 949 | → `co-bsim4` Rust 移植（修 destroy 泄漏；拆 eval 省 DC 期 acLoad——先旗标保等价） |
| `compact_models/bsim4/native.py`（运行时 cc + ctypes） | 538 | 过渡期保留（engine=numba 后端），翻转后删 |
| `compact_models/bsim4/numba_transient.py` | 205 | 死 |
| `osdi_*`（host/device/transient + jit 核 ~1.2k） | ~2.7k | R4 决策：推荐 Rust 宿主化（OSDI 0.4 C ABI，杀 must-mirror）；期间维持现状 oracle |
| `ngspice_*`、`spice/`、`pdk/`、`device_factory/model` 注册层、service、CLI | — | 不动 |

### 3.2 Rust↔Python 接缝（一次定义，各分析共用）

- 入口粒度 = **整段分析**（一次 transient grid、一次 AC 批扫、一次器件网格
  评估），不是每步/每器件——把今天 numba 的边界原样上移为 PyO3 边界，杜绝
  细粒度跨语言开销。
- 输入 = `CompiledTopology` 平铺数组的 numpy 视图（rust-numpy 零拷贝只读）+
  标量参数结构（一个 `#[pyclass]` builder 或 dict→struct 一次转换）。
- 输出 = numpy 数组（波形/矩阵）+ 小结构（nfail/nretry/profile 统计），键名
  与现结果字典逐字一致。
- 全部计算入口 `py.allow_threads(...)`；内部并行用 rayon 但**逐位可复现**
  （固定分块 + 按索引顺序归并；禁 atomics 竞态归约）——`test_sar_parallel`
  的字节级确定性承诺是硬门。
- 错误 = `thiserror` 枚举 → PyErr 映射，保留现状"数值失败返回 None/回退"
  的语义（Rust 不 panic 过边界；`catch_unwind` 兜底转异常）。

## 4. 关键设计决策（D1–D10）

- **D1 打包**：双发行版（§3）。备选（整包 maturin 混合布局）记录在案，翻转
  一个大版本后若双包同步成为痛点再合并。
- **D2 引擎开关**：`CIRCUIT_ENGINE ∈ {rust, numba, python}`，沿用"导入期烘焙 +
  `__init__` argv 预扫"的既有模式（测试可用子进程断言，同 `--no-numba` 现状）。
  CLI 新增 `--engine`；`--no-numba` 变为 `--engine python` 的**弃用别名**
  （警告一个大版本再删）。过渡期默认 numba，R6 翻转默认 rust。
  numba 删除后 `{numba}` 档位消失，`CIRCUIT_USE_NUMBA` 同步弃用。
- **D3 FP 策略**：Rust 侧禁 fast-math 类优化（默认即无 FMA 收缩），转写时
  **保持源码运算顺序**；同一 IEEE 语义下多数内核可逐位对齐 numba 产物，但
  验收容差按门分级（§6），不赌逐位。
- **D4 参考实现的去留**：numba 删除后，`_impl` 的纯 Python 源**保留为
  reference engine**（tests/debug 专用，不承诺性能）。理由：零移植成本（就是
  现源码）、是内核等价测试的活 oracle、是可执行规格。代价：改方程要改两处
  ——这是仿真器该付的代价（Cadence 金门同理）。若日后仍嫌重，可降级为冻结
  golden npz 后删除；该降级不影响本方案其余部分。
- **D5 BSIM4 并发**：两步走。R2 先 `co-bsim4` 内全局 `Mutex`（与今天事实
  单线程等价，保逐位）；R2.5 对 vendor 施 `_Thread_local` 机械补丁
  （b4v5ld.c ~128 个 file-scope scratch + acld/temp/noi 同类 + host 全局），
  TSan + 单线程逐位回归 + 多线程一致性测试通过后，锁降级为 per-handle。
  补丁以独立 diff 存 `rust/vendor/patches/` 并写入 NOTICE。
- **D6 host.c 移植而非包裹**：host.c 是自有代码（非 Berkeley），移植到 Rust
  获得所有权后顺手修三件事：destroy 泄漏（走 `pSizeDependParamKnot` 链）、
  eval 拆分（DC Newton 期跳过 acLoad 电容计算，**旗标默认关**直至 R2 门过再
  开）、shim 符号（`tmalloc/SMPmakeElt/CKTmkVolt/NIintegrate/NevalSrc/…` 与
  `FILE* slogp`）以 `#[no_mangle] extern "C"` 提供。vendor C 一字不改
  （除 D5 补丁）。License 合规：B4TERMS_OF_USE + NOTICE 随 `circuitopt-core`
  wheel 分发（license-files 迁移）。
- **D7 线性代数**：稠密小矩阵（≤~24×24）沿用**手写 GEPP 的 Rust 移植**
  （与今天 `_solve_dense_neg_rhs_inplace_impl` 同算法同 pivoting，逐位可控），
  不引 faer/nalgebra 做核内解；AC 批量复数解若迁 Rust 用 faer 或手写 LU +
  rayon 按频点并行（每频点独立，无归约序问题）。scipy sparse 不替换（§1.2）。
- **D8 OSDI**：R4 里程碑决策点。推荐：`co-osdi` 宿主化（OSDI 0.4 描述符校验、
  实例内存、节点折叠、内部 Newton/Schur 全在 Rust），删 osdi jit 核与
  must-mirror；OpenVAF 编译 `.osdi` 的工作流不变。若届时性价比不足：OSDI
  oracle 回退纯 Python 解释执行（慢 ~17×但 oracle 可忍），numba 照删。
- **D9 版本管理**：`tools/version.py` 的 `synchronized_content` 增加
  `rust/Cargo.toml`（workspace.package.version）与主包对 `circuitopt-core`
  的 pin；`test_versioning.py` 同步扩展；CI `version.py check` 拒漂移。
- **D10 基线冻结**：R0 用 v1.4.0 numba 路径产出 golden 语料（器件 I/G/Q/C
  网格、AFE dc/ac/noise/tran 向量、chopper 轨道、pnoise PSD）存
  `tests/golden/engine_parity/*.npz` + 基准数字（5 个 bench + calibration
  冷/热耗时）。过渡期测试双引擎活体 A/B，numba 删除后 golden 语料接棒。

## 5. 分期计划（R0–R6，每期含验收门）

> 执行方式沿用既有分工：主 agent 出任务书（内嵌环境事实卡）+ 先写对抗性验收，
> Opus subagent 在 worktree 实现、小步 commit；403 中断按已有恢复流程。
> 每期结束主 agent 独立跑门，用户 gate 合并。

### R0 — 基线冻结与决策批准（~0.5 天）
- 本文档评审定稿;`environment_performance.md` 追加立场更新注记。
- `tools/freeze_engine_golden.py`：产出 D10 golden 语料 + 基准报告
  （bench_model/afe/sweep/chopper/periodic 各 cold/warm、calibration --all、
  默认 pytest 全套耗时），提交 `results/engine_baseline_v140.json`。
- **门**：golden 语料可重放（同机重跑逐位一致）；基准报告入库。

### R1 — 工具链与脚手架（1–2 天）
- rustup 安装（**本机现无 cargo**）；`rust/` workspace + 三 crate 骨架；
  edition 2024，MSRV 取当期稳定版；PyO3 0.29 系 + rust-numpy 配套 + abi3-py310。
- maturin 构建 `circuitopt-core` 空壳（`engine_info()` 可 import）；
  `circuitopt/_engine.py` + `CIRCUIT_ENGINE` 烘焙/argv 预扫/回退链。
- CI：test 矩阵加 rust-toolchain + Swatinem/rust-cache + `maturin develop`；
  lint 加 `cargo fmt --check` + `clippy -D warnings`；release.yml 草挂 wheel
  矩阵（先 build 不发）。D9 版本同步落地。
- **门**：三平台 CI 绿（含 wheel 构建）；`CIRCUIT_ENGINE=rust` 可 import 且
  求解自动回退 numba 并告警一次；`version.py check` 覆盖新 Cargo.toml；
  现有全量 pytest 不受影响。

### R2 — co-bsim4：BSIM4.5 Rust 化（3–5 天）
- vendor 迁移 + build.rs（cc crate，`-O2 -std=c99 -fPIC` 对齐 native.py:133-148）；
  host.c → Rust（D6）；全局 Mutex（D5 第一步）；PyO3 暴露与
  `compact_models/bsim4/abi.py` 同形的后端类；`NativeBsim4Backend` 增
  engine 分派（默认 cc 路径不变）。
- 器件级 parity 工具：三 PDK × 全角点 × 3 温度 × 偏置网格，Rust vs cc 后端
  I/G/Q/C/noise 对照。
- **门**：parity 网格 rel ≤1e-13（同一 C 模型数学，差异仅 host 层 LU——按同
  pivoting 实现预期逐位，1e-13 是保险丝）；`tests/compact_models/bsim4/` 三件套
  + `test_freepdk45_native`（守恒 atol 1e-18/1e-24）+ sky130/tsmc28 原生套件在
  `CIRCUIT_ENGINE=rust` 下绿；`test_tsmc28_5t_ota` 噪声 rel 2% 门不动摇；
  destroy 泄漏修复经 asan/重复 create-destroy RSS 曲线验证。
- **R2.5（可与 R3 并行）**：`_Thread_local` 补丁 + TSan + 单线程逐位回归。

### R3 — co-core：求解内核 Rust 化（6–10 天，最大一期）

**进度（2026-07-17，R3 完成）**：

- OTFT 器件簇已逐序移植：电流、内部二维 Newton、电容/电荷、内部 Jacobian、
  terminal token/stamp 与端口 `gm/gds`；`co-py` 同时提供标量和 GIL-free 批量入口。
- MNA 已包含线性无源、独立/受控源、支路未知量及与参考内核同 pivot/消元顺序
  的稠密 GEPP；电路 Newton、固定 BE/Gear2、二分重试/profile 与自适应 Gear2
  均由 `co-core` 执行。
- AC/noise 的复数 MNA 装配和求解已接入 Rust，包含带相位 AC 电压源及全部受控源；
  BSIM4 固定网格直接调用 `co-bsim4`，Rust 路径不再导入 `numba_transient.py`。
- PyO3 粗粒度入口已统一为 rust-numpy ABI：大数组使用只读 C-contiguous NumPy
  视图零拷贝输入，波形/矩阵直接返回 NumPy 数组，计算期间释放 GIL；非连续二维
  视图和非法拓扑映射为 Python 异常，不跨边界 panic。
- `CIRCUIT_ENGINE=rust` 已进入生产 OTFT/BSIM4 transient 与 AC/noise 路径；测试
  契约改为引擎中立。固定网格 FreePDK45 5T OTA 对照达到 rel ≤1e-12。
- PSS/PAC/PNoise 的编排与轨道初始化属于 R4。R3 中周期求解请求的 transient
  已走 Rust；分岔边缘的 OTFT PSS 初值和 `latch_screen` 暂用线程/任务局部参考
  上下文保持校准根选择，避免污染其他 Rust 求解。
- 验收结果：Rust 与 Numba 全量测试均为 510 passed / 17 skipped /
  167 deselected；周期专项 55 passed / 1 deselected；`calibration --all` 五组
  全部通过。AFE 7 次 warm 中位数相对 Numba 吞吐为 AC 1.32×、noise 1.29×、
  transient 1.22×；冷启动 AC 为 2.0%、transient 为 18.2%。`bench_model` 使用
  10000 次批内循环消除亚微秒计时噪声，21 项最差 warm 吞吐为 1.096×，首个
  OP 冷调用为 Numba 的 0.00054×，通过 ≥0.9× / ≤0.5× 性能门。

- 实现顺序（各步均已落 parity 测试）：
  ① OTFT 器件簇（currents/内部 Newton/caps/charges/terminal derivs）
  ② 三元组求值 + stamp 核 ③ 手写 GEPP ④ 电路 Newton（BE/BDF2 系数、
  阻尼/clip/stall-accept 全常数照抄 §现状清单）⑤ 固定网格 grid（二分重试）
  ⑥ 自适应 gear2（LTE/步控常数照抄 `adaptive_config.py:17-30`）
  ⑦ AC/noise 装配（`ac_mna` 对应）⑧ BSIM4 grid（调用 co-bsim4，解除 Rust
  路径对 `numba_transient.py` 的依赖）。
- Python 侧：`transient_solver` 已增加 rust 分派臂并收敛 marshal；深路径 import
  与结果键逐字保持。
- 重写被钉住的测试：`test_transient_contracts` 改为引擎中立的接缝契约；
  `test_model_kernels`/`test_compiled_topology` 保持实现与拓扑契约测试，并由 Rust、
  Numba 两次全量运行共同覆盖；新增 OTFT/LTI/transient 专项活体 parity 与 ABI 测试。
- **门**：固定网格 BE/gear2 波形 rust vs numba rel ≤1e-12（预期近逐位）；
  自适应运行为**行为门**（终值、nfail/nretry、profile 统计一致或有解释）；
  calibration byte-gate（AFE=OTFT：dc 1e-3 V / ac 1% / IRN 3%）绿；
  `test_elements/controlled_sources/vsource/afe_*/sar*` 全绿；
  bench_model/bench_afe warm ≥0.9× numba，import+首解 ≤0.5× 现状。

### R4 — 周期族内核 + OSDI 决策（5–7 天）
- HB 块装配、PSD fold、PAC 轨道线性化（含 gate1 变体）→ Rust；
  **删** Python 双份线性化器；pss/pac/pnoise 编排、scipy sparse、FFT 不动。
- OSDI：按 D8 决策执行（推荐 co-osdi 宿主化）。
- **门**：`test_periodic_solvers`（RC 解析参照）、`test_pnoise_woodbury`、
  chopper 全档（ideal/pmos/lptv/pss/pac/pnoise）+ `cadence_regression` 斩波器
  3-5% 门绿；bench_periodic/bench_chopper ≥0.9×；OSDI 决策落地后
  `test_osdi_host/transient` 在新形态下绿。

### R5 — 并行与性能收割（3–5 天）
- rayon：AC/noise 频点、MC/corner 批、dataset 构建；PyO3 `allow_threads`
  全覆盖审计；D5 锁降级（R2.5 过门后）。
- 确定性测试：同 seed 多线程逐位一致（`test_sar_parallel` 扩强）。
- **门**：8 线程 SAR MC 扩展效率 ≥0.7；bench_sweep 候选/秒 ≥2× 单线程基线；
  TSan 干净；全套默认 pytest 绿。

### R6 — 翻转与拆除（2–3 天，出 v2.0.0-rc）
- 默认 `CIRCUIT_ENGINE=rust`；主包硬依赖 pin `circuitopt-core`；
  **删除**：numba 依赖、njit 注册层、`numba_transient.py`、cc/ctypes 运行时
  编译路径（native.py 瘦身）、`CIRCUIT_USE_NUMBA`（留弃用警告一版）；
  `--no-numba` 转弃用别名。
- 文档全面更新：module_overview（EN/zh）、environment_performance 新基线、
  cli_reference、development.md 的"Changing a Solver"检查单加 Rust 条目、
  json_circuit_format 的 n_aug 注记复核、CHANGELOG 主版本条目（EN/zh）；
  release.yml 正式发双工件。
- **门**：全量默认 pytest + byte-gate + `-m ngspice_oracle`（有 ngspice 的机器）
  + `mkdocs build --strict` + `version.py check --tag` 绿；干净虚拟环境
  `pip install` 双 wheel 后无 cargo/cc 机器上跑通 quickstart + 硅例子；
  基准报告 vs R0 入库对比。

**总量级**：串行 3–5 周 agent 时间；R2 与 R3-①②③ 可并行（都只依赖 R1），
R2.5 与 R3 后段并行。以往 Opus 403 中断率纳入排期余量（worktree +
commit-early 已是既定实践）。

## 6. 验收容差总表

| 门 | 容差 | 依据 |
|---|---|---|
| 器件级 parity（R2） | rel ≤1e-13 | 同一 vendor C；host LU 同 pivoting |
| 内核级 parity（R3，固定网格） | rel ≤1e-12 | 同序 IEEE 运算，预期近逐位 |
| 自适应瞬态（R3） | 行为门：终值 rel ≤1e-9 + 步数/nfail 统计一致或有因 | 轨迹分歧属预期 |
| 物理不变量 | KCL/电荷守恒 atol 1e-18 / 1e-24 | `test_freepdk45_native.py:69-108` 现门 |
| Cadence 校准 | dc 1e-3 V、ac 增益 1%、BW 5%、IRN 3%、pac 2%、pnoise 3-5% | `calibration.py:47-56` 现门，一字不改 |
| 性能 | 内核 warm ≥0.9×；冷启动 ≤0.5×；8 线程扩展 ≥0.7 | §1.2 定位 |

## 7. 兼容性契约（重写期间逐字保持）

1. `circuitopt/__init__.py` 导出集（含 `NumbaParams`——翻转版更名议题单独走
   弃用流程，先保留别名）。
2. 深模块路径：`compiled_topology.CompiledTopology`、`ac_solver.ac_solve`、
   `transient_solver.transient`、`noise_solver.*`、`corners.*`、
   `device_factory.build_devices` 等（examples/experiments 直接 import）。
3. CLI 九子命令 + 现 flags；`--no-numba` 按 D2 弃用路径演进；exit code 语义。
4. service `/api/v1` 路由集、状态码信封、serialize 约定（complex→{re,im}、
   NaN→null）、WS 帧、`serve --port` argv（Tauri backend.rs 依赖）。
5. `schemas/circuit.schema.json` 与 `analysis_options` 的未知键硬拒。
6. 结果字典键（`Av_dc_dB`/`bw_Hz`/`irn_psd`/`nodes`/`nfail`/PSS 收敛字段/
   SAR metrics 等——`__main__.py:102-152` 与 service/测试消费）。
7. ngspice 渲染 golden decks 字节不变（`tests/golden/*.deck`）。
8. 破坏性变更全部集中在 v2.0.0：numba 删除、flag 更名、双发行版依赖。

## 8. 风险清单

| # | 风险 | 缓解 |
|---|---|---|
| 1 | DC 根选择漂移（AFE 多稳态、器件 eval ≤1e-15 差即可能翻轨） | scipy/续延/守卫全不动；R3 门含 afe/mdac/对称守卫套件；翻轨个案用现有 guards 收敛 |
| 2 | 长瞬态误差积累 → Newton 接受/拒绝翻转 → 自适应轨迹分歧 | 行为门而非逐位（§6）；固定网格保 1e-12 |
| 3 | vendor `_Thread_local` 补丁面大（~150 变量） | 机械 diff + TSan + 单线程逐位回归；补丁失败则退守全局锁（仍达成正确性目标，牺牲硅并行） |
| 4 | PyO3/rust-numpy/abi3 版本耦合 | 锁版本；abi3 单 wheel 降矩阵；spike 已验证 0.29 可用 |
| 5 | 贡献者/CI 工具链负担 | 过渡期主包纯 Python 可装；wheel 预编译；rustup 步骤入 development.md |
| 6 | 钉死测试重写量（arg-name/flag/内核名） | R3/R6 各带明确重写清单；CHANGELOG 主版本声明 |
| 7 | 双发行版版本漂移 | version.py 同步 + CI check + 精确 pin |
| 8 | agent 执行中断（Opus 403） | worktree + commit-early-often；主 agent 可随时接管重放门 |
| 9 | 周期族 scipy 留守造成"半吊子统一"观感 | 明示为刻意边界（§1.2）；R4 后热路径已全 Rust，scipy 仅编排级小矩阵/稀疏；后续按需评估 faer sparse |
| 10 | BSIM eval 拆分（省 acLoad）引入行为差 | 旗标默认关，R2 门过后单独开门验证再默认开 |

## 9. 执行注意事项（给实现期任务书）

- 本机 2026-07-17 无 cargo；R1 装 rustup 后，**所有后续 agent 任务书的环境
  事实卡必须写死 cargo/maturin 绝对路径**（沿用 `agent-brief-env-card` 惯例）。
- venv：`.venv/bin/python`（3.12.9）；默认 pytest 排除 `ngspice_oracle`；
  byte-gate 命令 `python -m circuitopt.calibration --all`。
- 器件/内核常数（阻尼 5.0、vtol 1e-8、gmin 1e-12、LTE 常数表、gear2 系数、
  stall-accept 条件等）在盘点报告与源码注释均有出处，转写时**逐条对照抄**，
  禁止"顺手改善"。
- vendor C 一字不改（除 D5 补丁与既有 b4v5noi.c 噪声 hook——后者已是本仓改动）。
