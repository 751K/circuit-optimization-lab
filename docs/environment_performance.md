# 运行环境与性能基准

> **文档状态：带日期的性能快照。** 本页记录特定机器、Python/Numba 版本和缓存状态
> 下的历史测量，用于定位性能数量级，不是当前版本的固定 SLA。功能和命令以维护中
> 文档及实际基准脚本为准。

> 一句话：**跑性能一定要用装了 Numba 的项目虚拟环境**。不含 Numba 的解释器
> 会静默回落到解释版内核，chopper 慢约 **28×**（7.5s → 221s），容易误判为“变慢了”。

## 为什么环境会决定快慢

- `CIRCUIT_USE_NUMBA=1` 只是**允许**用 Numba，并不保证真的用上。当 Numba
  `import` 不到时，代码**静默回落**到解释版 `_impl` 内核（单源化后同一份源码既是
  JIT 核也是纯 Python 核：数值内核只存在一份 `_impl`，Numba 在时 JIT、不在时即其
  `.py_func`/原始纯 Python 形式）。功能照常、结果一致，但慢一个数量级。
- 推荐先 `source .venv/bin/activate`；路径不写死，Numba/NumPy/Python 的实际版本以
  当前环境为准。

### 确认 Numba 真的在跑

```bash
CIRCUIT_USE_NUMBA=1 python -c "
import circuitopt.numba_kernels as nk
k = nk._transient_solve_adaptive_gear2_impl
print('jitted:', hasattr(k, 'py_func'), type(k).__name__)"
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
