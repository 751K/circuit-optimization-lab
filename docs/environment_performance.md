# 运行环境与性能基准

> **worker 数怎么选（`tools/workers.py`，2026-07-27）**：不同机器的最优
> `--workers` 不同，别沿用文档里在 4P+6E Apple M4 上测出的具体数字。
> `python tools/workers.py` 侦测本机拓扑（macOS `hw.perflevel*` / Intel 混合
> Linux sysfs；其余按均匀核处理）并按实测规则给出逐工作负载建议：转换级并行
> （ramp/sine/explore/transitions）用全部逻辑核——E 核承载真实负载，实测 w8 已
> 好于理想 4P；MC 在 trial 数 ≤ 4×核数时用 `workers = trials`（一 trial 一任务，
> work-stealing 抹平 P/E 不对称——16 trial 在 10 核上 w16 比 w10 *快*），更多
> trial 时用核数；单次转换无可并行。`--mc-trials N` 给具体 MC 建议；
> `--calibrate` 用编译 SAR ramp 真测饱和拐点（本机 8 s，取最快值 5% 容差内的
> 最小 worker 数，写入 `results/workers_calibration.json`）——与静态规则不一致
> 时以实测为准。
>
> **v2.0.0 基线变更（新）：numba 引擎已移除，编译 Rust 核（`circuitopt_core`，
> `CIRCUIT_ENGINE=rust`）成为唯一计算引擎。** 因此下文所有以 Numba 为前提的性能
> 指引与“确认 Numba 在跑”的步骤均为**历史记录（v1.x）**，已被 Rust 核取代：现在
> 跑性能的前提是**装好编译核**（`maturin develop --release -m rust/crates/co-py/
> Cargo.toml`，或 `pip install circuitopt-core`）。`numba_kernels.py` 已于 R7
> 整体移除（其 OTFT 选根恢复 oracle 移植进 `circuitopt_core` 的
> `OtftModel(reference=True)`）。新基线随 Rust 核演进，见
> `results/engine_baseline_v140.json` 与基准脚本 `benchmarks/`。
>
> **v2.0.0 基线数字**（详见根目录 `CHANGELOG.md` `[2.0.0]` 条目，不在此重复整段）：
> `CompiledCampaign.evaluate_batch` 单次 GIL 释放 + 单 Rayon 池，8-core 笔记本实测
> 设计空间扫描 8 workers 下 FreePDK45 **5.4×**、TSMC28 **2.2×**；SAR 失配 MC 扩展
> 效率从 GIL 束缚的 **0.13** 提升到 **0.70**，且单线程路径本身快约 **10×**；BSIM4.5
> C 在 wheel 构建期编译，冷启动首个 AC 解降到旧 JIT 路径的约 **2%**。这些数字来自
> `circuitopt-core` wheel 构建后的 Rust 核，与下文按 v1.x numba 环境测得的表格不可
> 直接比较。
>
> 原生 BSIM transient 的 MOS 批量求值使用独立 Rayon 池，默认并行度为机器可用
> 并行度与 10 的较小值。实测 12 线程会因调度开销慢于 10 线程，因此可用
> `CIRCUITOPT_BSIM_BATCH_THREADS=1..10` 显式降低并行度；该值高于 10 时仍会限制为
> 10。线程池在进程内首次使用时建立一次，之后修改环境变量不会重建。
>
> 该池是进程级共享的，因此**并行归外层所有**：当驱动本身在并发跑多次求解
> （signoff PVT 点、SAR 转换与蒙特卡洛 trial、corner/PVT 切片、探索候选），
> 其 worker 会内联求值器件批次，把核让给外层。判定与编译式 campaign 的轴策略
> 一致——仅当外层任务数不少于 worker 数时生效，单次求解或填不满机器的小批次
> 仍然使用池。实测 16 次 TSMC28 MDAC residue 瞬态：用池时单线程 4.90 s 即占满
> 7.8 核、八线程仅 4.03 s；内联时单线程 10.54 s、八线程 3.19 s（6.0 核）。
> 设置 `CIRCUITOPT_BSIM_NESTED_POOL=1` 可恢复原调度做 A/B；该开关在进程内
> 首次使用时读取一次。调度不影响数值：批次槽位相互独立写入。
>
> 原生 BSIM Gear2 transient 默认启用状态外推 predictor。固定网格使用两点线性
> 外推；adaptive 路径在三个已接受状态可用后使用变步长二阶外推，重启后先线性
> fallback，并继续在输入斜率断点禁用预测。需要进行数值回归 A/B 时，可设置
> `CIRCUITOPT_BSIM_GEAR2_PREDICTOR=0` 关闭；profile 中的
> `gear2_predictor_steps` 表示实际使用预测初值的步数。该开关按每次 solve 读取，
> 不需要重启 Python 进程。
>
> 原生 BSIM transient 现支持 Rust 内 LTE 自适应 Gear2。2026-07-25 本机
> FreePDK45 MDAC 单次热测中，默认容差与 `max_step=500 ps` 将 5 ns 轨道从
> 500 个固定 10 ps 步降为 146 个接受步（1 次拒步），热端到端时间中位数约
> `167 ms -> 56.6 ms`。该路径每个 trial 只做一次非线性 Gear2 求解，再用
> BDF3−BDF2 的 BSIM 端电荷历史 defect 做一次线性 Jacobian 投影，并由 PI
> controller 调节下一步。相对具有相同 10 ps 线性输入边沿的 0.5 ps、10,000 步
> 固定网格参考，默认容差的峰值/最终差分输出误差约为 `2.04 mV / 50 uV`；
> 把 `adaptive_reltol`/`adaptive_vabstol` 收紧至 `1e-5`/`1e-7 V` 后约为
> `0.42 mV / 12 uV`。这是特定电路的快照，不是固定 SLA。
>
> 2026-07-26 的 5 ns residue 热测进一步让 adaptive predictor 使用三点二阶
> 外推，并让 LTE 的第二个 RHS 复用收敛 Newton 的 LU：FreePDK45 中位数
> `55.6 -> 50.4 ms`、Newton `873 -> 801`；TSMC28 中位数
> `139.2 -> 106.4 ms`、Newton `2644 -> 1995`。将新轨迹插值回旧 accepted grid
> 后，全节点最大差分别为 `0.249 mV` 与 `0.137 mV`，最终输出变化分别为
> `1.5 uV` 与 `12.0 uV`。同机 TSMC28 batch 线程 sweep 的 1/4/8/10 线程中位数为
> `368/188/142/140 ms`，因此默认 10 线程上限保持不变。
>
> 同日继续优化了 BSIM 内部节点的 Schur reduction。旧实现按外部端子
> `row × column` 循环，对同一个内部矩阵执行 16 次独立分解；新实现只分解一次，
> 同时回代四个外部列。专门的位模式单测确认它与旧的逐列独立求解逐位一致。
> 同一进程 TSMC28 5 ns MDAC A/B 中，热中位数 `113.1 -> 97.4 ms`，约快 14%；
> accepted grid、41 条节点历史、5 条支路电流历史、输出及最终状态全部逐位相同。
> 此后完整器件求值也直接复用最终 floating load 已产生的电荷/电容字段，不再于
> `acLoad` 前额外执行一次 `MODEINITSMSIG` model load。TSMC28 继续由
> `97.1 -> 90.1 ms`，FreePDK45 由 `38.5 -> 36.1 ms`；两者的 transient、AC 和
> noise A/B 均逐位相同。原始 `dc` ABI 保留旧 reload 作为 oracle；设置
> `CIRCUITOPT_BSIM_REUSE_SMSIG_LOAD=0` 可让生产 `eval` 也恢复旧路径。
>
> 可设置 `CIRCUITOPT_BSIM_REUSE_FINAL_LOAD=1`，让内部节点 Newton 更新低于
> `1 pV` 后直接消费最后一次收敛线性化，不再为亚皮伏修正额外执行完整 model load。
> TSMC28 `84.8 -> 77.7 ms`、FreePDK45
> `33.7 -> 31.1 ms`，均再快约 8%；最大 MDAC 节点变化分别为 `0.36 nV` 与
> `42 pV`。不过个别 PMOS 电流/噪声点会超过公共 API 的位级 golden 契约，因此
> 该模式默认关闭；原始 `dc` ABI 无论如何都保留 reload 作为 oracle。
>
> 随后的精确版本只在内部节点解与当前 load 使用值的 IEEE-754 位模式完全相同
> （或没有内部节点）时自动跳过最终 load，并在 setup 后缓存节点布局。该条件下
> FreePDK45 SAR6 的 64-code、8 条完整节点/支路轨迹和 TSMC28 MDAC 轨迹都与旧路径
> 逐位一致。矩阵清零起初按调度选择：单次求解走 `24×24` 整帧清零，只有外层并行
> worker 才清零 `CKTmaxEqNum` 有效块。2026-07-26 重测表明整帧路径在**两种**调度
> 下都更慢，于是两条分支合并为统一的有效块清零，见下文"统一有效块清零"。设置
> `CIRCUITOPT_BSIM_FULL_FRAME_CLEAR=1` 可强制使用整帧路径。transient lease 与
> campaign worker 已独占其 handle，因此热循环会直接调用无锁内部入口；公共
> scalar/C ABI 仍保留逐 handle mutex，batch 内重复 handle 仍被归组串行。64-code
> SAR6 热运行中位数由约 `574.3 ms` 降至 `559.8 ms`，约快 2.5%，码流及完整轨迹
> 逐位不变。
>
> SAR 单 trial sweep 改为严格均衡的连续分块。旧的 ceil 分块在 64 输入、12 worker
> 时只产生 11 块，使 campaign Auto 误选串行轴，本机耗时约 `3.44 s`；修复后约
> `0.59 s`。10 worker 由约 `0.74 s` 降至 `0.57 s`。每次 bit 判决后也只重建
> 当前试探位与上一清零位的波形行。全部性能数字均在相同 64-code 码流下测得。
>
> 原生 transient 随后接入公共 BSIM handle pool。旧路径每次求解都会为全部 MOS
> 重跑 `setup/temp`；旧 cache 对同一 card 也只能保留一个可变 handle，匹配器件在
> 同一轨迹中仍会反复创建临时实例。现在每个 card 可缓存多个彼此独立的 handle，
> 一条轨迹独占租用、归还后由下一条轨迹复用；`BSIM4_DEVICE_CACHE_SIZE` 按每个 PDK
> namespace 的 handle 总数限制该池，默认 128。TSMC28 `residue_plus_fs16` 热中位数
> `124.6 -> 101.9 ms`，单 PVT 点六个 transient case `0.815 -> 0.692 s`；
> FreePDK45 `42.7 -> 38.7 ms`。相对每次新建 handle 的旧路径，插值后最大输出差
> 分别为 `0.15 uV` 与 `8.6 pV`；六 case signoff 状态和 SAR6 64-code 回归不变。
> 未提供显式 `V0` 时，native transient 现在还会用同一批已构造 device/handle
> 直接运行 Rust DC Newton，并严格检查有限值、KCL residual 和 voltage box；失败
> 才回退到原完整 `ac_solve`。这避免了重复构建设备和只为取得 `dc_op` 而执行的
> 1 Hz AC reduction。池化后的 TSMC28 单 residue 继续由
> `102.3 -> 94.4 ms`，六 case 单点由 `0.699 -> 0.645 s`；A/B 最大输出差
> 约 `38 pV`。
>
> TSMC28 MDAC signoff 的六个 transient case 进一步显式使用
> `newton_vtol=3e-8 V`；求解器全局默认值仍保持 `1e-8 V`。完整 45 PVT 点、
> 270 次 transient 的同机 A/B 为 `13.848 -> 10.247 s`（1.35 倍），没有
> case 或 PVT 状态翻转。相对默认容差，排除直接 PWL 驱动节点后的最大输出轨迹
> 差异为 `0.116 mV`，最终输出最大差异为 `18.3 uV`。此外，model/instance card
> 在构造时缓存规范化参数元组，避免每次 handle lease 重排数百个参数；600 参数
> card 的 20,000 次 key 微基准由 `350.9 ms` 降至 `1.35 ms`。后者不改变任何
> 浮点求值，并对 TSMC28、FreePDK45 和 SKY130 的公共 BSIM 路径生效。
>
> 同一组六个 MDAC case 还显式设置
> `bsim_model_bypass_tolerance=3e-9 V`，启用 BSIM4 自带的标准 device bypass；
> 该值是 `newton_vtol` 的十分之一。默认值仍为 0，配置值不得超过 Newton 容差，
> scalar/DC/AC/noise 与未显式启用的 transient 不受影响。45 点 8-worker 的
> 270 次 transient A/B 为 `12.025 -> 8.377 s`（1.43 倍），状态无变化；输出轨迹
> 最大/P99 差异为 `8.77/6.25 uV`，最终输出最大差异为 `2.74 uV`。
>
> **统一有效块清零（2026-07-26 复测）。** 三种清零方式在两种调度下同机 A/B
> （TSMC28 residue 5 ns，各取 9 次中位数，两轮反序复测）：
>
> | | 整帧 `24×24` | 有效块 | 整行块 |
> |---|---|---|---|
> | 单跑（池调度） | `85.1 / 83.6 ms` | `82.3 / 77.6 ms` | `80.2 / 78.9 ms` |
> | 外层并行（inline） | `129.0 / 127.8 ms` | `108.7 / 107.4 ms` | `112.4 / 109.1 ms` |
>
> 整帧在**两种**调度下都最慢，因此原先"单跑用整帧"的分支是错的；有效块在两种
> 调度下都不劣，整行块（一次连续 memset、字节数为整帧一半）落在本循环的运行间
> 噪声内且在 inline 下更差。于是删除按调度分岔，统一使用有效块。单个 residue
> 由 `83.9 -> 79.0 ms`（约 6%），FreePDK45 SAR6 64-code w1 由
> `3217 -> 3134 ms`（约 2.6%，码流不变），45 点 campaign 不变（它本就走该路径）。
> 45 点全部 signoff 字段与改动前逐位相同；golden 语料现在也真正覆盖该路径。

> **Newton 判据改为逐节点步长预算（2026-07-26）。** 绝对 `newton_vtol` 对所有节点
> 用同一个数，因此 0.9 V 节点被收敛到步长控制器即将接受的误差的 0.33%。实测
> 弹性（TSMC28 residue，`newton_error_fraction=0`）：`newton_vtol`
> 1e-8 / 3e-8 / 1e-7 对应每步 7.93 / 6.61 / 5.53 次 Newton 迭代——收敛是线性的，
> 每次迭代约缩 2.5–3 倍。启用 `newton_error_fraction` 后（`newton_vtol=3e-8`）：
>
> | fraction | 单跑(池) | inline | 迭代/步 | 模型求值 | 相对紧参考解的建立尾段偏差 |
> |---|---|---|---|---|---|
> | 0（关） | `78.1 ms` | `109.2 ms` | 6.61 | 89,490 | `87.6 uV` |
> | 0.03 | `64.7 ms` | `94.0 ms` | 5.42 | 73,150 | `107.1 uV` |
> | 0.1 | `58.8 ms` | `82.7 ms` | 4.58 | 61,864 | `118.9 uV` |
> | 0.3 | `50.3 ms` | `72.6 ms` | 3.88 | 52,706 | `119.1 uV` |
>
> 参考解为 `reltol=1e-7 / vabstol=1e-9 / newton_vtol=1e-9`（1269 步、263k 求值，
> 约为默认配置的 2.9 倍工作量）。误差在 0.1 之后饱和；接受步数（312–314）与
> 拒绝步数（14）在所有档位不变。作为对照，把 `newton_vtol` 由 3e-8 收紧到 1e-8
> 只把该偏差由 87.6 改到 87.0 uV，却多花 19% 求值——说明原判据的余量确实是浪费。
> manifest 采用 `0.1`：曲线拐点，且对步长容差仍保留 10 倍余量。45 点 campaign
> 由 `15.89 -> 14.62 s`，无状态变化。
>
> **campaign 的新瓶颈是被串行化的 handle 构造。** 同一次 w8 campaign 里为
> `_NativeDevice.__init__` 计时：4185 次构造、累计 `3.50 s` CPU。由于
> `NativeBsim4Backend._lock` 必须罩住构造（vendored C 的 setup 路径线程不安全），
> 这 3.50 s 是**不可并行的地板，占 14.16 s wall 的 24.7%**。这解释了为什么单个
> transient 的 1.33 倍在 campaign 上只体现为 1.09 倍：wall 里 transient 本来就
> 只占一部分。
>
> **但贵的不是 setup，是参数搬运（2026-07-26 实测）。** 拆开单次构造
> （TSMC28，339 个 model 参数）：`co_bsim4_create` `2.4 us`、
> **`set_model` × 339 = `312.7 us`（97.3%）**、`set_instance` × 18 `5.1 us`、
> **`co_bsim4_setup` 仅 `0.9 us`（0.3%）**、`destroy` `0.3 us`。所以
> "每点重建 handle 贵" 的原因从来不是 vendored setup，而是逐参数通道。
> 其中 Python 侧 `name.encode()` 只占 4%，其余在 FFI 穿越与
> `find_param` 的线性扫描里。两步修复：整卡入口 `co_bsim4_set_card`
> （339 次穿越 → 1 次）把它降到 `244 us`；把 vendor 关键字表建成每进程一次的
> 索引（原先每个参数都要扫全表做大小写不敏感比较）再降到 **`23.6 us`，共
> 13.6 倍**。campaign 的构造地板随之由 `3.50 -> 1.09 s`（占 wall 24.7% → 9.1%），
> w8 由 `14.64 -> 13.05 s`，45 点结果逐位不变。
>
> 因此 **handle 复用不是这里对症的药**：既然重建只要 23.6 us，跨 PVT 点复用能
> 省的已经不多，而每点独立 scope 是 workers>1 可复现性的来源，不值得用它换。

> **文档状态：带日期的性能快照。** 本页记录特定机器、Python/Numba 版本和缓存状态
> 下的历史测量，用于定位性能数量级，不是当前版本的固定 SLA。功能和命令以维护中
> 文档及实际基准脚本为准。

> 一句话（历史 v1.x）：**跑性能一定要用装了 Numba 的项目虚拟环境**。不含 Numba 的解释器
> 会静默回落到解释版内核，chopper 慢约 **28×**（7.5s → 221s），容易误判为“变慢了”。

## 为什么环境会决定快慢

- `CIRCUIT_USE_NUMBA=1` 只是**允许**用 Numba，并不保证真的用上。当 Numba
  `import` 不到时，代码**静默回落**到解释版 `_impl` 内核（单源化后同一份源码既是
  JIT 核也是纯 Python 核：数值内核只存在一份 `_impl`，Numba 在时 JIT、不在时即其
  `.py_func`/原始纯 Python 形式）。功能照常、结果一致，但慢一个数量级。
- 推荐先 `source .venv/bin/activate`；路径不写死，Numba/NumPy/Python 的实际版本以
  当前环境为准。

### 确认 Numba 真的在跑（历史 v1.x；该模块已随 numba 引擎移除）

```bash
# v1.x 历史命令（numba_kernels.py 已删除，仅存档）：
# CIRCUIT_USE_NUMBA=1 python -c "
# import circuitopt.numba_kernels as nk
# k = nk._transient_solve_adaptive_gear2_impl
# print('jitted:', hasattr(k, 'py_func'), type(k).__name__)"
# 期望: jitted: True CPUDispatcher   ← 已启用
# 若为   jitted: False function      ← 走的是解释版，慢 28×
```

## 实测基准（Numba 环境，`CIRCUIT_USE_NUMBA=1`，热启动）

命令：`CIRCUIT_USE_NUMBA=1 python -m circuitopt.calibration <case>`

| 用例 | 分析 | 用时（Numba 热） | 无 Numba（解释版） |
|---|---|---|---|
| `amp_design3_typical` | dc + ac + noise | ~0.8 s | ~0.6 s |
| `chopper_design3_typical` | pac + pnoise（经 PSS） | **7.5 s** | 221 s |
| `chopper_design3_fast` | pac + pnoise | **7.0 s** | — |
| `chopper_design3_slow` | pac + pnoise | **7.2 s** | — |
| chopper 三角全跑 | | **~21 s** | — |

要点：

- **chopper 支配整体耗时**，几乎全是单进程纯计算（`sys` 时间 <1s，非 I/O 等待）；
  主体是 **PSS 打靶**（Newton 求斩波周期稳态轨道），其上叠 PAC/PNoise 的 HB 折叠。
  Numba 的收益全在这里。
- **amp 本体几乎免费且 Numba 无用武之地**：线性 MNA + 一次 DC/AC/noise，没有瞬态打靶，
  numba 反因线程池初始化略慢一点。
- **冷 vs 热几乎无差**（chopper typical 冷 7.9s → 热 7.5s，仅 0.4s 抖动）：njit 用了
  `cache=True`，首次编译已落盘，冷启动不额外背首编译成本。
- `user`(~9s) > `real`(~7s)：Numba 在 PSS 轨道积分里吃到多线程并行，墙钟低于总 CPU 时间。

> 测量日期 2026-07-01（单源化收尾后）。数值全部 PASS（byte-gate 5/5）。
> 上表是**加速前**基线；下一节的两项优化把 chopper 又压掉一半。

## Chopper 求解加速（2026-07-01）

两项针对 chopper 热点的优化，都保持 `calibration --all` 5/5 PASS、pnoise IRN 数值一致：

1. **PSS 打靶用解析单值矩阵**（`solver.analytic_jacobian=true`，见三个
   `calibration/chopper_design3_*/metadata.json`）。原来打靶一步收敛却要建一个有限差分
   雅可比 = topo.n 次整周期重积分；解析单值矩阵在同一次轨道 pass 里算出，gear2 积分
   **16 → 4 次**。PSS 阶段 3.56 s → **0.96 s**。收敛到同一不动点 → 结果 bit 级一致。
2. **PNoise 时域 Floquet 伴随的 factor-once（Woodbury）**（`circuitopt/pnoise_solver.py`
   `_time_domain_pnoise_adjoint`）。原来每个噪声频点在 N·ns 的块双对角 BE 算子上做一次
   完整稀疏 `splu`（37 频 = 37 次分解）。但 `F(γ)` 逐频只有那个 ns×ns 周期角块
   `-BT[0]/γ` 变——`splu` 参考频率 `F(γ0)` **一次**，逐频用秩-ns（Woodbury）修正角块。
   **37 次完整分解 → 1 次分解 + 37 次廉价小解**，与逐频 splu bit 级一致（范数相对误差
   ~1e-13），病态频点（Floquet 共振）自动回退到新鲜 splu。PNoise 阶段 2.03 s → **0.39 s**。

**合计效果**（chopper typical，warm，PSS/PAC/PNoise 中位数）：

| 阶段 | 加速前 | 加速后 |
|---|---|---|
| PSS | 3.56 s | 0.96 s |
| PAC | 0.14 s | 0.15 s |
| PNoise | 2.03 s | 0.39 s |
| **合计（warm 纯算）** | **5.72 s** | **1.50 s（3.8×）** |

`calibration --all`（5 case 冷启动，含 Python/numba 启动固定开销）：**22.7 s → 9.1 s**
（FD+splu → 解析雅可比 → +Woodbury，逐级 22.7 → 14.7 → 9.1）。
