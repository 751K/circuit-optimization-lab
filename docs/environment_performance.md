# 运行环境与性能基准

> 一句话：**跑性能一定要用装了 Numba 的 conda `daily` 环境**。base 环境没有 Numba，
> 会静默回落到解释版内核，chopper 慢约 **28×**（7.5s → 221s），容易误判为“变慢了”。

## 为什么环境会决定快慢

- `CIRCUIT_USE_NUMBA=1` 只是**允许**用 Numba，并不保证真的用上。当 Numba
  `import` 不到时，代码**静默回落**到解释版 `_impl` 内核（单源化后同一份源码既是
  JIT 核也是纯 Python 核：数值内核只存在一份 `_impl`，Numba 在时 JIT、不在时即其
  `.py_func`/原始纯 Python 形式）。功能照常、结果一致，但慢一个数量级。
- 本机 base 解释器 `/opt/miniconda3/bin/python` **没装 Numba**；
  conda `daily` 环境（`/opt/miniconda3/envs/daily/bin/python`，Numba 0.61 / NumPy 2.1 /
  Py 3.12）有。

### 确认 Numba 真的在跑

```bash
CIRCUIT_USE_NUMBA=1 /opt/miniconda3/envs/daily/bin/python -c "
import core.numba_kernels as nk
k = nk._transient_solve_adaptive_gear2_impl
print('jitted:', hasattr(k, 'py_func'), type(k).__name__)"
# 期望: jitted: True CPUDispatcher   ← 已启用
# 若为   jitted: False function      ← 走的是解释版，慢 28×
```

## 实测基准（daily 环境，`CIRCUIT_USE_NUMBA=1`，热启动）

命令：`CIRCUIT_USE_NUMBA=1 /opt/miniconda3/envs/daily/bin/python -m core.calibration <case>`

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
2. **PNoise 时域 Floquet 伴随的 factor-once（Woodbury）**（`core/pnoise_solver.py`
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

## 硅 OSDI 求解加速（2026-07-04）

硅路径（SKY130 经 OSDI/ctypes 调 BSIM4）单次候选评估（DC+AC+noise，5T OTA）曾比
OTFT numba 路径慢 ~200×（107 ms vs 0.55 ms），~85% 耗在 Python↔ctypes 桥接而非
模型本身。七项修复（全部只动硅路径，OTFT byte-gate 5/5 不变，OTA 指标逐位一致）：

1. **散射向量化**（`osdi_host.py`）：jacobian entries 的 (row,col) 归约索引在 setup
   时算一次存成 numpy 索引数组，每次 eval 用 `np.ctypeslib.as_array` 视图 + `bincount`
   聚合，替代逐条 ctypes 访问；`OsdiSimInfo` 按 flags 缓存复用。
2. **按偏置精确键的 op 记忆化**（LRU 128）：噪声逐频重解 → 1 次;同偏置重复调用
   bit 级可复现。
3. **内部节点 Newton 热启动**：从上次收敛解出发（中位数 2 次求值/解,原 ~6），收敛后
   加一步"打磨"消路径依赖，失败回退冷启动（热启动尝试限 20 迭代）。
4. **非物理偏置早退**：电路级 DC 求根器（hybrd）信赖域探索会打出 ±千伏级节点电压，
   此类解永不收敛却烧满 100-200 次求值；两档早退（12 迭代后仍 >1e2 A;30 迭代后仍
   >1e-1 A）——真实解 <30 迭代内收敛，零风险。
5. **电容惰性计算**：DC 内环只要 Id,`with_caps=False` 跳过 reactive eval;
   ss/caps 消费方在同一 memo 条目上按需升级。
6. **器件实例 LRU + 参数卡内存 memo**（`osdi_device._shared_device`,
   `OSDI_DEVICE_CACHE_SIZE`,默认 64;`sky130_model._card_memo`）：消除每次
   `ac_solve` 重建器件时 ~0.9 ms 的 800 参数卡 ctypes 重灌与磁盘 JSON 重读。
   另将 Newton 步与 Schur 补写成 numba 核（`_osdi_newton_step_impl` /
   `_osdi_schur_impl`,遵循单源 `_impl` 惯例,numba-off 走解释版）替代小矩阵
   `np.linalg.solve` 的 LAPACK 包装开销。
7. **jitted `operating_point` 内环**（`_osdi_op_solve_impl`）：关键前置发现——
   **numba nopython 可以直接调 OSDI 的 C 函数指针**（重声明为 `c_void_p`-only 的
   `CFUNCTYPE`,以**运行时参数**传入核,不把进程相关地址烤进编译产物 →
   `cache=True` 跨进程安全,双进程验证零告警）。spike 实测单次 eval+load+散射
   0.45 µs（Python 宿主路径 3.86 µs,8.4×,结果逐位一致）。据此把整个内部
   Newton（热启动/打磨/早退/奇异救援）收进一个核,OSDI eval 核内直呼;Python
   层只剩 memo 查询、starts 编排、Schur/电容与结果打包。
   **这同时推翻了"`.osdi` 进不了 numba 循环"的旧假设——硅瞬态/斩波器
   （Phase B）可以纯 numba 做,不需要 Rust。**

| 阶段 | 单次评估 | 累计 |
|---|---|---|
| 基线 | 107 ms | — |
| +散射向量化/热启动/memo | 41 ms | 2.6× |
| +非物理偏置早退 | 26 ms | 4.1× |
| +惰性电容+卡memo | 21.6 ms | 5.0× |
| +numba Newton/Schur 核 | 10.9 ms | 9.8× |
| +jitted operating_point 内环（OSDI eval 核内直呼） | **6.0 ms** | **17.8×** |

**硅瞬态（Phase B，同日落地）**：基于同一 numba-调-OSDI 模式,
`_osdi_transient_grid_impl` 把整条固定网格 BE/BDF2 瞬态(全局折叠:外部节点+器件内部
节点一个电路级 Newton)放进单个 jitted 核——5T OTA 2000 步 gear2 **24.8 ms warm
（12.4 µs/步）**,DC-hold 漂移 <1e-9 V,ngspice 同卡同 `.osdi` oracle 对齐(τ63 一致、
终值差 1 µV)。一条 PSS 轨道 ~25 ms → 硅斩波器(PSS/PAC/PNoise)在性能上已可行。

100 候选硅 dataset 构建（含进程启动）：**1.79 s**（基线纯算即 ~11 s）。当前剩余
~6 ms 已不在器件层:~65% 是器件之上的电路级编排（fsolve 的 Python 回调、
`Id→get_Idc→_op` 包装链、`operating_point` 的 memo/dispatch 壳）。再往下的
下一杠杆是**编译版电路级 DC Newton**（类似瞬态 stamp 核）——只有需要 10 万级
硅 dataset 时才值得动,且涉及共享 DC 路径,须 silicon-only 门控以保 byte-gate。
Rust 已无必要性论据（仅剩多核无 GIL 批量与免预热部署两个场景,均非当前需求）。
