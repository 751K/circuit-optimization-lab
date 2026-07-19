# Rust 核心重写方案（codex/rust-core-rewrite）

> 内部规划文档，2026-07-17，2026-07-18 更新 R5 边界，基于 v1.4.0
>（分支 `codex/rust-core-rewrite`）。
> 目标读者：本仓库维护者与实现 agent。行号引用以 v1.4.0 为准。

## 0. 一句话

把 v1.4.0 中由 **numba 执行**的全部求解器热路径（`numba_kernels.py` 的 ~50 个
单源 `_impl` 内核 + 其 marshalling 层）改由 **Rust crate** 执行，BSIM4.5 的
vendored Berkeley C 内核**保留 C 源**、由 Rust 在 wheel 构建期编译并以安全 FFI
包裹（替代现在的"用户机器运行时 cc + ctypes 函数指针"）；SPICE 表达式、PDK
展开、候选电路构建和批量分析也进入 Rust。Python 只保留 CLI、service、配置、
优化策略、结果展示与外部 oracle，最终从依赖中删除 numba，并让生产批处理热路径
不再执行 Python 回调。

## 1. 目标与非目标

### 1.1 目标（按优先级）

1. **单一加速路径**：消灭 numba 执行路径及其约束集——`py_impl` 双执行模式、
   `_NUMBA_GRID_ARG_GROUPS` ~90 个位置参数的 marshalling 税
   （`transient_solver.py:287-524`）、被测试钉住的内核签名契约
   （`tests/test_transient_contracts.py`）、PAC 线性化的 Python/numba 真双份
   （`pac_solver.py:426-605`）、numba↔numpy 版本耦合
   （`pyproject.toml:31-33` 注释）。
2. **免预热 + 免运行时编译器部署**：预编译 wheel 取代"首次使用时用用户机器的
   clang/cc 编 BSIM4 dylib"（`compact_models/bsim4/native.py:78-148`）与 numba
   JIT 首调延迟/磁盘缓存。
3. **真·多线程正确性与扩展**：不再依赖“事实上单线程”保障 BSIM 状态安全。
   Rust 层明确区分 process-global 初始化、thread-local callback 和 per-handle
   scratch，以符号审计、TSan 与多线程逐位回归证明独立 handle 可以并行。
4. **数值行为不变**：全部现有验收门（Cadence byte-gate、`cadence_regression`、
   物理不变量、内核等价测试的数值断言）在规定容差内继续通过。
5. **粗粒度无 GIL 批处理**：Python 一次提交电路模板、候选矩阵和 seed；Rust 在
   一个 `py.detach` 区间内完成 PDK 数值展开、器件构建、DC/AC/noise/transient
   和有序归并。正式并行路径不使用 `multiprocessing` 绕开 GIL。

### 1.2 非目标

- **不追求内核吞吐飞跃**。numba 即 LLVM，warm 内核已近 C 速度
  （`docs/environment_performance.md`：斩波器 numba on/off 7.5 s vs 221 s）。
  Rust 的量级收益在启动、部署、并行与维护，不在单核 FLOPS。验收目标是
  ≥0.9×（争取 ≥1×），不是 10×。
- **不重写产品控制面**：CLI、service、JSON 配置、优化器策略、可视化和任务状态
  留在 Python。Python/scipy 求解器保留为 reference/oracle 与过渡期回退，但
  production campaign 的 DC 根选择、续延、AC/noise/transient 及批量归并进入
  Rust；否则 PDK 展开和候选编排仍会被 GIL 串行化。
- **不动外部回归 oracle**：ngspice 全套（`ngspice_*.py`）与 Cadence 校准链不变。
- **不动 JSON schema / CLI 子命令集 / service 路由 / 结果字典键**（§7 兼容清单）。
- **不做 no_std / WASM / GPU**。

### 1.3 对既有"不需要 Rust"结论的回应

`docs/environment_performance.md:137`（2026-07-04）：“Rust 已无必要性论据
（仅剩多核无 GIL 批量与免预热部署两个场景，均非当前需求）。”该结论在当时
成立（其上下文是旧 compact-model bridge 的性能问题）。现在改变的不是性能事实，
而是三点：

1. **战略决策**：维护面统一为 Rust（用户 2026-07-17 决定）。numba 路径的维护
   成本不在速度而在约束集与版本耦合（§1.1-1）。
2. 那两个"非当前需求"已成为需求：PVT campaign / MC / dataset 构建要无 GIL
   多核；桌面分发（Tauri 前端 + `circuit-opt serve`）要免预热、免运行时 cc。
3. **正确性论据是新的**：旧 numba/函数指针路径没有可审计的并发所有权边界；
   Rust FFI 层能够把全局初始化、handle 状态和 callback 生命周期显式化并测试。

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
  eval 都跑 acLoad 算电容（host.c:659-693，纯浪费）；旧函数指针路径没有可
  审计的 handle 并发所有权契约（§1.1-3）。
- **并发模型**：全仓**零 multiprocessing**，只有 `ThreadPoolExecutor`
  （`sar.py:319`、`sar_mc.py:265`、`sar_explore.py:379`、`service/jobs.py:154`），
  依赖 numba nogil 放 GIL。R5 初测确认：即使 Rust 求解入口释放 GIL，TSMC28
  候选仍在 Python 的 SPICE 表达式和 PDK 展开层串行；因此仅增加线程池不能形成
  扩展，入口必须上移到完整 campaign。
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
│       ├── co-spice/              # 表达式、deck 解析、section/subckt 展开、数值作用域
│       ├── co-pdk/                # FreePDK45/SKY130/TSMC28 适配、bin 选择、卡缓存
│       ├── co-core/               # 求解内核：device trait、OTFT、stamp、Newton、
│       │   └── (BE/BDF2、AC/noise、周期族核、compiled campaign/candidate executor)
│       └── co-py/                 # PyO3 (abi3-py310) + rust-numpy → Python 模块 circuitopt_core
├── circuitopt/                    # Python 控制面、兼容 API、reference/oracle
│   └── _engine.py                 # CIRCUIT_ENGINE 分发 + compiled campaign 薄包装
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
| `transient_solver.py` 其余（分派/回退/结果组装） | ~850 | Python 保留兼容包装；production campaign 编排 → `co-core` |
| `compiled_topology.py` | 606 | Python reference 保留；production 模板编译 → `co-core` |
| `ac_mna.py` | 208 | → `co-core`（AC/noise 装配核）；Python 版留 reference |
| `ac_solver.py` / `dc_solver.py` / `noise_solver.py` 编排 | ~880 | 标量兼容 API/reference 留 Python；production 候选执行 → `co-core` |
| `pss/pac/pnoise/chopper` 编排 | ~4600 | 控制面留 Python；进入 campaign 的数值段逐步收进 `co-core` |
| 周期族 jit 核（HB 块、fold、轨道线性化，`numba_kernels.py:3397-3864`） | ~470 | → `co-core`；生产路径删除 `_assemble_pac_linearization_python` 双份，参考实现显式改名隔离 |
| `pmos_tft_model.py` 方程（经 `_impl`） | — | → `co-core` device trait 实现；`NumbaParams` 16 字段为参数 ABI |
| `compact_models/bsim4/native_src/vendor` | ~16.3k C | **保留 C**，迁至 `rust/vendor/`，build.rs 编译（wheel 期，非用户运行时） |
| `compact_models/bsim4/native_src/host.c` | 949 | → `co-bsim4` Rust 移植（修 destroy 泄漏；拆 eval 省 DC 期 acLoad——先旗标保等价） |
| `compact_models/bsim4/native.py`（运行时 cc + ctypes） | 538 | 过渡期保留（engine=numba 后端），翻转后删 |
| `compact_models/bsim4/numba_transient.py` | 205 | 死 |
| OSDI/OpenVAF 路径（host/device/transient + jit 核） | ~2.7k | **R4 已删除**：三个模块、OpenVAF 工具链、Sky130 注册/分派、Numba 内核和专项测试均退出代码库 |
| `spice/expressions.py`、parser、elaborator | — | → `co-spice`；Python 版冻结为 differential reference |
| `pdk/` 与 `device_factory/model` | — | production 展开/构建 → `co-pdk`；Python 注册 API 保持兼容 |
| `ngspice_*`、service、CLI、优化器 | — | 留 Python；ngspice 只作显式 oracle |

### 3.2 Rust↔Python 接缝（一次定义，各分析共用）

- 入口粒度 = **完整 campaign/batch**，不是每步、每器件或每候选。Python 先创建
  一个 `CompiledCampaign`（本地 PDK 路径 + section/corner + 电路模板），随后以
  numpy 候选矩阵、分析选项和 seed 调用 `evaluate_batch`。
- 输入 = 初始化时的一次配置/路径转换 + 运行时只读 numpy 候选矩阵。PDK AST、
  数值作用域、bin 索引、模型卡和电路拓扑均由 Rust 持有，不在候选间往返 Python。
- 输出 = numpy 数组（波形/矩阵）+ 小结构（nfail/nretry/profile 统计），键名
  与现结果字典逐字一致。
- 全部计算入口 `py.detach(...)`；内部并行用 rayon 但**逐位可复现**
  （固定分块 + 按索引顺序归并；禁 atomics 竞态归约）——`test_sar_parallel`
  的字节级确定性承诺是硬门。
- 并行只使用一个受控 Rayon pool。候选数充足时并行候选、单候选内部串行频点；
  候选少而频点多时并行频点，禁止嵌套 pool 造成 oversubscription。
- 错误 = `thiserror` 枚举 → PyErr 映射，保留现状"数值失败返回 None/回退"
  的语义（Rust 不 panic 过边界；`catch_unwind` 兜底转异常）。

## 4. 关键设计决策（D1–D12）

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
  - **R6 降级执行注记**：R6 曾尝试完整删除 `_impl` 参考（golden 语料接棒）。
    实测发现 **OTFT 标量选根恢复（`rust_otft_reference_mode`）是承重路径**，删除
    后 `sc_lpf` 校准硬门失败（rust 自适应 Gear2 在该 OTFT 轨道不收敛），并连带
    打断 AFE 极端点恢复与 OTFT latch 屏；在基点 `6bda531` 用 `CIRCUIT_ENGINE=
    rust` 关掉 reference-mode 可**逐字复现**同一失败。故 R6 执行为：numba **引擎/
    JIT/依赖**移除，但 `_impl` 参考内核**保留为内部 oracle**（非用户可选引擎），
    OTFT 选根恢复继续可用。golden 语料仍在 rust 默认下重冻结、并作为引擎 parity
    的参考 oracle；`_impl` 的进一步降级需先让 rust 在这些敏感 OTFT 轨道上自洽
    （属 rust 核工作，非本次拆除范围）。
  - **R7 完成注记（`_impl` 参考彻底移除）**：R6 识别的承重路径已**移植进
    co-core**——根因定位为单一 ULP 级分歧：`_impl` 的 `Vt` 平方走 CPython
    `x ** 2`（系统 libm `pow`），rust 生产路径走 `powi(2)`（`x*x`），该 1 ULP
    差经接触电流相消与 Newton 迭代放大即改变选根。`OtftModel(...,
    reference=True)` 现承载完整参考语义（libm-pow `Vt` 平方以 `black_box`
    防 LLVM 折叠、有限差分 Jacobian 内部 Newton `_newton_internal_impl`、
    有限差分端子导数 `_terminal_derivatives_impl`（hx=1e-6）、独立电容方程
    `_capacitances_impl`），移植经 112,815 点 × 5 几何差分验证与 `_impl`
    **0 ULP** 后，`numba_kernels.py` 整文件删除，触发器更名
    `otft_reference_mode`（线程局部 ContextVar，pss/corners 包裹 + ac 重试
    照旧）。毒化探针（`_impl` 全体 raise）下校准 5/5 不变，证明恢复已零
    Python 依赖。golden 语料同期在 rust BSIM4 后端下重冻结（cc 运行时编译
    路径删除，见 D-d/R2 注记），是唯一 parity oracle。
- **D5 BSIM4 并发（R5 审计修订）**：R2 先用全局 `Mutex` 保守落地。随后对实际
  编译 archive 做 `nm` 符号审计并逐源核查，未发现计划所假设的 ~128 个可变
  file-scope scratch；导出数据仅为不可变参数表/名称/尺寸。当前实现采用 per-handle
  `Mutex`、一次性 front-end 初始化和 thread-local noise callback target。独立
  handle 并行仍须经 TSan、逐位回归和多线程一致性门，失败则退回全局锁。
- **D6 host.c 移植而非包裹**：host.c 是自有代码（非 Berkeley），移植到 Rust
  获得所有权后顺手修三件事：destroy 泄漏（走 `pSizeDependParamKnot` 链）、
  eval 拆分（DC Newton 期跳过 acLoad 电容计算，**旗标默认关**直至 R2 门过再
  开）、shim 符号（`tmalloc/SMPmakeElt/CKTmkVolt/NIintegrate/NevalSrc/…` 与
  `FILE* slogp`）以 `#[no_mangle] extern "C"` 提供。vendor C 一字不改
  （当前符号审计不要求修改 vendor）。License 合规：B4TERMS_OF_USE + NOTICE 随 `circuitopt-core`
  wheel 分发（license-files 迁移）。
- **D7 线性代数**：稠密小矩阵（≤~24×24）沿用**手写 GEPP 的 Rust 移植**
  （与今天 `_solve_dense_neg_rhs_inplace_impl` 同算法同 pivoting，逐位可控），
  不引 faer/nalgebra 做核内解；AC 批量复数解若迁 Rust 用 faer 或手写 LU +
  rayon 按频点并行（每频点独立，无归约序问题）。scipy sparse 不替换（§1.2）。
- **D8 OSDI（R4 已决策）**：完整删除，不实施 `co-osdi`，也不保留 Python
  兼容入口。FreePDK45、SKY130、TSMC28 的生产仿真统一使用 `co-bsim4`；外部
  交叉核对由现有 ngspice/Cadence oracle 承担。删除范围覆盖 host/device/transient
  模块、OpenVAF 路径解析与编译脚本、Sky130 OSDI 注册、瞬态/PSS 分派、Numba
  专用内核和测试，避免把无生产调用者且无法活体验收的 ABI 带入 Rust 架构。
- **D9 版本管理**：`tools/version.py` 的 `synchronized_content` 增加
  `rust/Cargo.toml`（workspace.package.version）与主包对 `circuitopt-core`
  的 pin；`test_versioning.py` 同步扩展；CI `version.py check` 拒漂移。
- **D10 基线冻结**：R0 用 v1.4.0 numba 路径产出 golden 语料（器件 I/G/Q/C
  网格、AFE dc/ac/noise/tran 向量、chopper 轨道、pnoise PSD）存
  `tests/golden/engine_parity/*.npz` + 基准数字（5 个 bench + calibration
  冷/热耗时）。过渡期测试双引擎活体 A/B，numba 删除后 golden 语料接棒。
- **D11 不以多进程规避 GIL**：`ProcessPoolExecutor` 不作为 production campaign
  后端。它会重复解析大型 PDK、复制模型卡/缓存、增加平台启动差异，也掩盖 Python
  热路径尚未迁移的问题。并行边界上移到 `CompiledCampaign.evaluate_batch`，由
  单进程 Rayon 调度。
- **D12 PDK 编译与保密边界**：Rust 直接读取用户本地 PDK 路径，在内存中生成
  immutable `CompiledPdk`；缓存键包含规范路径、mtime/size、section、温度和
  工艺选项。许可模型内容不写入仓库、日志、golden 或持久化缓存；测试只比较摘要、
  数值输出和用户本机临时数据。

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
- **R2.5（可与 R3 并行）**：archive/源码符号审计 + per-handle/TLS 宿主改造 +
  TSan + 单线程逐位回归；仅在审计发现真实 vendor 全局时才增加 vendor 补丁。

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

### R4 — 周期族内核 + OSDI 删除（5–7 天）

**进度（2026-07-17，R4 完成）**：

- 新增 `co-core::periodic`：HB 稠密块装配、标量 cyclostationary PSD fold、
  OTFT 轨道线性化和保留 `gate1` 状态的动态电容交叉项均由 Rust 执行，运算顺序
  对齐原 `_impl` 参考核。PyO3 入口借用只读 C-contiguous NumPy 数组并释放 GIL。
- PAC 问题采用不可变 `PeriodicLinearizationProblem`。除 OTFT 原生参数 ABI 外，
  还支持每采样点四端口 G/C 数据，使 FreePDK45、SKY130、TSMC28 的 BSIM 器件
  只在 Python 模型层求端口线性化，端口映射、驱动列和矩阵盖章统一由 Rust 完成。
- `pac_solver`/`pnoise_solver` 的 Rust 分派不静默回退；生产路径不再调用旧
  `_assemble_pac_linearization_python`。解释参考明确改名为
  `_reference_pac_linearization`，仅供 Numba/Python 参考引擎使用。
  FFT、SciPy sparse/dense HB 解及 Woodbury/Floquet 编排保持 Python 所有权。
- OSDI 按 D8 完整删除：不迁移动态宿主，也不保留 Python/Numba 兼容桥；正常
  硅工艺路径只承接原生 BSIM4 端口模型。
- 新增周期 Rust parity/ABI/非法输入测试。随机 HB 数据逐位一致，PSD fold 最大
  绝对差约 `1.42e-14`；普通 OTFT 与 gate1 轨道矩阵相对误差门为 `1e-12`；另有
  四端口 state/drive 盖章测试。删除 OSDI 后，Rust 和 Numba 全量均为
  514 passed / 8 skipped / 157 deselected；Rust 周期+chopper 专项 55 passed，
  TSMC28/SKY130 周期链 9 passed / 2 deselected，五组 `calibration --all`
  全部通过。
- 性能门（各 1 次 warm，取同机中位值）：`bench_periodic` 的 Rust/Numba 吞吐
  PSS 1.02×、PAC 1.11×、PNoise 0.99×；`bench_chopper` 五档为 1.15×–1.83×，
  均高于 0.9×。

- HB 块装配、PSD fold、PAC 轨道线性化（含 gate1 变体）→ Rust；
  **删** Python 双份线性化器；pss/pac/pnoise 编排、scipy sparse、FFT 不动。
- 删除 OSDI/OpenVAF 代码、工具链、测试和当前文档入口。
- **门**：`test_periodic_solvers`（RC 解析参照）、`test_pnoise_woodbury`、
  chopper 全档（ideal/pmos/lptv/pss/pac/pnoise）+ `cadence_regression` 斩波器
  3-5% 门绿；bench_periodic/bench_chopper ≥0.9×；仓库当前代码与使用文档
  不再包含 OSDI/OpenVAF 入口。

### R5 — 无 GIL campaign 与并行收割（8–15 天）

**R5-A：数值核与 BSIM 并发（进行中）**

- Rayon AC/noise 频点、BSIM 独立 handle batch；PyO3 `detach` 全覆盖审计。
- D5 按 archive 实际符号审计降级为 per-handle lock；补齐 TSan、ASan 和独立
  handle 逐位确定性测试。
- AC 结果复用已展开器件；BSIM terminal noise 改为“独立 handles × 全频率”
  粗粒度 batch，避免逐频点 Python/C 往返。

**R5-B：`co-spice` / `co-pdk` 编译器**

- 迁移 HSPICE 数字后缀、表达式 AST、lazy scope/循环检测、用户函数、section
  引用、subckt 参数覆盖和 model statement 展开；作用域只读共享，解析缓存线程安全。
- 实现 FreePDK45、SKY130、TSMC28 的 corner/temperature/polarity、geometry bin、
  `nf/mult/mismatch` 规则，输出 `co-bsim4` 可直接消费的 numeric card。
- Python parser/elaborator 冻结为 differential reference：合成 deck 全语法测试；
  三 PDK 在本地可用时逐参数比对，模型参数/实例参数必须逐位相同或 rel ≤1e-14。

**R5-C：compiled campaign/candidate executor**

- `CompiledCampaign` 持有 immutable PDK AST、bin 索引、电路模板、模型卡缓存和
  分析计划；一次接收候选矩阵、corner/mismatch 描述、seed 与分析集合。
- 在单个 `py.detach` 内完成器件构建、Rust DC 根选择/续延、AC、noise、transient
  及结果收集；Python 不参与候选循环，不调用 device/corner/dataset callback。
- 一个 Rayon pool 自适应选择候选级或频点级并行，按候选索引有序写回；随机量
  在调度前按 seed/索引确定，进度/取消使用原子状态但不参与数值归约。

**R5-C 完成注记（2026-07-19，已验收合并）**：引擎（device-agnostic
`co_core::campaign`，单 Rayon pool 自适应候选/频点轴、按索引有序写回）+ AFE
OTFT 与三硅 PDK 两族 evaluator。parity：硅三家 ULP 级（worst ≤4e-16，DC 同
种子逐位）；AFE 双门 = 对 cold-consistent 参考逐位 + 对 warm ≤1e-8（OTFT 内部
Newton tol=1e-12 使 gm/gds 依赖种子路径，Python 自身 warm/cold 即分叉 ~6e-8）。
workers 1/2/8 逐字节；硅 4.94×/AFE 3.88× 加速；batch 期零 Python 回调。
`mulu0→u0` fold 落地 `co_pdk::apply_mulu0_fold`（delivery 库 mulu0≡1.0，
乘法臂由单测钉死）。已裁决偏差：`band_rms` 朴素求和 vs numpy pairwise（irn
门放至 1e-11）。

**R5-C 暴露的缺口（R5-D 前置项）**：
1. **sky130 `extract_w`**：冻结 loader 支持 `reference_width_um` 钉卡（实宽
   连续偏离网点），`CompiledPdk::numeric_card` 无此参数——sky130 explore 类
   `extract_w != W` 电路接入 campaign 前必须扩 co-pdk 面并补 parity。
2. **DC 冷路径行为门**：campaign 的 Rust 电路 Newton 同种子已证逐位，但冷启动
   多猜测/守卫级联未进 parity 门——R5-D 接线 corner_table/MC 前需为冷 DC 定
   行为门（收敛率/根选择与 Python 参考一致或逐案裁决），不许静默换根。
3. **tsmc28 nmos 0-bin**（delivery 特性）：ff/sf/fs 部分几何无 bin，两侧同
   报错（Python ValueError ↔ campaign `{ok:False}`）；接线后的 corner 扫描
   要沿用"预探测同跳过"约定。

**R5-D：工作流接入与清理**

- `corner_table`、mismatch MC、SAR MC、dataset、design-space sweep 和 service job
  优先走 compiled campaign；标量 Python API 继续兼容并作为 reference/fallback。
- 删除 R5 期间实验性的候选级 `ThreadPoolExecutor`；不新增多进程 production
  分支。补充错误传播、取消、单调进度、同 seed 1/2/8 线程逐位一致测试。

- **门**：8 线程 SAR MC 扩展效率 ≥0.7；TSMC28 与 FreePDK45 `bench_sweep`
  8 线程候选/秒均 ≥2× 单线程；采样 profiler 证明 batch 计算期间无 Python PDK/
  device frame；三 PDK 卡展开 parity 过门；TSan 干净；全套默认 pytest 绿。

### R6 — 翻转与拆除（2–3 天，出 v2.0.0-rc）
- 默认 `CIRCUIT_ENGINE=rust`；主包硬依赖 pin `circuitopt-core`；
  **删除**：numba 依赖、njit 注册层、`numba_transient.py`、cc/ctypes 运行时
  编译路径（native.py 瘦身）、`CIRCUIT_USE_NUMBA`（留弃用警告一版）；
  `--no-numba` 转弃用别名。
- production PDK/solver 分派默认创建 `CompiledCampaign`；Python SPICE/PDK 展开器
  降为显式 reference/oracle，不再被普通 AC/noise/transient/corner/dataset 调用。
- 文档全面更新：module_overview（EN/zh）、environment_performance 新基线、
  cli_reference、development.md 的"Changing a Solver"检查单加 Rust 条目、
  json_circuit_format 的 n_aug 注记复核、CHANGELOG 主版本条目（EN/zh）；
  release.yml 正式发双工件。
- **门**：全量默认 pytest + byte-gate + `-m ngspice_oracle`（有 ngspice 的机器）
  + `mkdocs build --strict` + `version.py check --tag` 绿；干净虚拟环境
  `pip install` 双 wheel 后无 cargo/cc 机器上跑通 quickstart + 硅例子；
  基准报告 vs R0 入库对比。

**总量级**：串行 4–7 周 agent 时间；历史上 R2 与 R3-①②③ 可并行，当前剩余
关键路径为 R5-B → R5-C → R5-D，R5-A 的 TSan/并发审计可与 R5-B 并行。以往
Opus 403 中断率纳入排期余量（worktree + commit-early 已是既定实践）。

## 6. 验收容差总表

| 门 | 容差 | 依据 |
|---|---|---|
| 器件级 parity（R2） | rel ≤1e-13 | 同一 vendor C；host LU 同 pivoting |
| 内核级 parity（R3，固定网格） | rel ≤1e-12 | 同序 IEEE 运算，预期近逐位 |
| 自适应瞬态（R3） | 行为门：终值 rel ≤1e-9 + 步数/nfail 统计一致或有因 | 轨迹分歧属预期 |
| 物理不变量 | KCL/电荷守恒 atol 1e-18 / 1e-24 | `test_freepdk45_native.py:69-108` 现门 |
| Cadence 校准 | dc 1e-3 V、ac 增益 1%、BW 5%、IRN 3%、pac 2%、pnoise 3-5% | `calibration.py:47-56` 现门，一字不改 |
| 性能 | 内核 warm ≥0.9×；冷启动 ≤0.5×；8 线程扩展效率 ≥0.7；sweep ≥2× | §1.2、R5 定位 |

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
| 1 | DC 根选择漂移（AFE 多稳态、器件 eval ≤1e-15 差即可能翻轨） | Python/scipy 路径作 reference；Rust 逐项镜像续延/守卫；R5 门含 afe/mdac/对称守卫套件 |
| 2 | 长瞬态误差积累 → Newton 接受/拒绝翻转 → 自适应轨迹分歧 | 行为门而非逐位（§6）；固定网格保 1e-12 |
| 3 | vendor C 或 host 仍有符号审计未覆盖的数据竞争 | archive `nm` + 源审计 + TSan + 单线程逐位回归；失败退守全局锁 |
| 4 | PyO3/rust-numpy/abi3 版本耦合 | 锁版本；abi3 单 wheel 降矩阵；spike 已验证 0.29 可用 |
| 5 | 贡献者/CI 工具链负担 | 过渡期主包纯 Python 可装；wheel 预编译；rustup 步骤入 development.md |
| 6 | 钉死测试重写量（arg-name/flag/内核名） | R3/R6 各带明确重写清单；CHANGELOG 主版本声明 |
| 7 | 双发行版版本漂移 | version.py 同步 + CI check + 精确 pin |
| 8 | agent 执行中断（Opus 403） | worktree + commit-early-often；主 agent 可随时接管重放门 |
| 9 | Python/scipy 留守使 campaign 再次进入 GIL | production campaign 禁止 Python callback；scipy 仅 reference/fallback，profile gate 检查 |
| 10 | BSIM eval 拆分（省 acLoad）引入行为差 | 旗标默认关，R2 门过后单独开门验证再默认开 |
| 11 | Rust SPICE/PDK 展开与 HSPICE 方言存在语义差 | Python differential reference + 合成语法集 + 三 PDK 全参数 parity；未知语法硬失败 |
| 12 | 候选并行与频点并行嵌套造成过量线程 | 单 Rayon pool + 明确粒度选择策略；benchmark 记录实际线程数和任务粒度 |

## 9. 执行注意事项（给实现期任务书）

- 本机 2026-07-17 无 cargo；R1 装 rustup 后，**所有后续 agent 任务书的环境
  事实卡必须写死 cargo/maturin 绝对路径**（沿用 `agent-brief-env-card` 惯例）。
- venv：`.venv/bin/python`（3.12.9）；默认 pytest 排除 `ngspice_oracle`；
  byte-gate 命令 `python -m circuitopt.calibration --all`。
- 器件/内核常数（阻尼 5.0、vtol 1e-8、gmin 1e-12、LTE 常数表、gear2 系数、
  stall-accept 条件等）在盘点报告与源码注释均有出处，转写时**逐条对照抄**，
  禁止"顺手改善"。
- vendor C 一字不改（除 D5 补丁与既有 b4v5noi.c 噪声 hook——后者已是本仓改动）。
