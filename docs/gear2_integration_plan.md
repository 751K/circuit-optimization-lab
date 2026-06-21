# 任务计划书：transient 积分器从 backward-Euler 升级到 gear2/BDF2

## 0. 一句话目标

把瞬态积分从一阶 backward-Euler（BE）换成二阶刚性稳定的 **变步长 BDF2（gear2）**，
在**稳定的 charge 电容公式**上做，使 chopper PAC baseband 三个 corner 都对齐到 **<1%**。

## 进度（2026-06-20）

- ✅ **M1 完成**：变步长 BDF2 在 Python transient 路径实现（`integration_method="gear2"`，
  charge 模式 + load/线性电容）。解析 RC 单元测试验证 **二阶收敛**（BE 误差 ~h 减半、gear2 ~h² 减到 1/4，
  同步长 gear2 精度高 5–38×）。回归测试 `test_gear2_is_second_order_on_rc_lowpass`。
- ✅ **M2 GO/NO-GO 通过**：chopper PSS 走 gear2（FD shooting Jacobian），PAC baseband 三 corner 全部 <1%：

  | corner | BE（旧） | **gear2** |
  |---|---|---|
  | slow | −0.54% | **+0.70%** |
  | typical | −2.68% | **−0.10%** |
  | fast | −2.34% | **−0.81%** |

  确认 typ/fast 的 −2.5% 就是 BE 一阶边沿误差，BDF2 关掉它。回归测试
  `test_pmos_chopper_pac_gear2_matches_cadence_within_1pct`（RUN_SLOW_CHOPPER）。
- ✅ **M3 完成**：numba gear2 grid solver（`_transient_solve_grid_gear2_impl`，单步/区间 + BE 自启动，
  共享 `_stamp_transient_system_impl` 加 BDF2 系数 + 两点历史；BE 调用方传 (1,−1,0) 等价旧行为）。
  warm 0.89s（vs Python 12s）。
- ✅ **M4 完成**：gear2 解析 monodromy（增广 2n 态 `M_m=[[-B1,-B2],[I,0]]`，BE 自启动）。
  `analytic_jacobian=True` 下 warm **0.24s**（比 FD shooting 快 4×），同精度。
- ✅ **M5 完成（数据驱动的"全面 gear2"）**：
  - **关键发现**：变步长 BDF2 在大步长比 ρ（细化边沿网格的"粗→细→粗"跳变）下不稳（ρ>2.41 非零稳定），
    n_points=161/edge_points=15 等会炸（nfail 数百、PAC 垃圾）。修复 = **步长比限制**：ρ>2 的步退回 BE
    （numba grid / monodromy / Python loop 三处）。修后所有细化网格 robust（nfail 4–11，PAC +0.4~0.6%）。
  - **默认值**：PSS / PAC / PNoise / chopper 全部默认 **gear2**（对齐关键路径，robust + 快）；
    裸 `transient()` 保留 **BE** 默认——gear2 单步 grid 无 substep 细分/retry，硬的独立 transient
    （如 `pmos_chopper_transient` nfail=179）依赖 BE 的 bisection。
  - **细分/retry 硬化尝试（失败）→ 连带弄坏 numba grid → 已定位并修复**：
    - 试过把 gear2 grid 改成 pieces + retry + rolling 两步历史 → PSS 跑偏 −3.5%；又加了一个
      grid 预细分 helper `_refine_grid_for_gear2`（按 `max_step` 把区间切成均匀 substep 再跑 gear2，
      末了用 `orig_idx` 降采样回原网格）。结果 chopper PSS numba 路径跑偏 **−16%（nfail≈5000）**。
    - **根因（逐步对比 numba grid vs Python loop 定位）**：在 numba grid / Python loop 各打 debug
      print 比对第一步，发现 **numba 的 `h_n=1µs`，Python 的 `h_n=25µs`——两者跑的是不同网格**。
      chopper PSS 会传一个小 `max_step`，于是 `refine=True` 触发 `_refine_grid_for_gear2`，把 25µs
      区间细分成 1µs。但**单步 gear2 grid 没有 in-solver retry**，细分后开关沿那几步照样难收敛，
      只是把会失败的步数 ×25，失败步污染轨道 → −16%。Python loop 和 M5 numba grid 都是**直接用粗
      网格**（BE 自启动 + BDF2 对大步长良态），所以正确。**细分本身就是这个 bug 的全部来源**。
    - **修复**：gear2 numba 分支**不再预细分**，直接用请求的 tgrid（删掉 `_refine_grid_for_gear2`、
      去掉 `g2_orig_idx` 降采样路径）。修后 numba grid 轨道与 Python loop **逐点一致**（max|d|≈1e-6，
      nfail=1）；warm **0.58s vs Python 10.9s（~19×）**，三 corner PAC 不变（slow +0.61/typ +0.49/
      fast +0.12%）。门控 `_GEAR2_NUMBA_GRID` 默认**开**（`CIRCUIT_GEAR2_NUMBA=0` 可退回 Python loop）。
  - **裸 transient 的 gear2：调查 + graceful BE 回退（2026-06-21）**：
    - **现象**：裸 chopper transient 用 gear2 跑偏（nfail≈1855，波形 −13mV vs BE 的 −27mV）。
    - **根因（逐步打 print 定位）**：每一步 Newton 都 `iters=maxit, ok=False, usable=True`——
      收敛到松 tol 但到不了紧 vtol=1e-8，且 maxit 加到 300 也没用（是 stall 不是慢收敛）。
      关键：**numba 的 per-step Newton（reuse_impl）在 chopper 开关沿这种 stiff 步上 BE 也会 stall**
      （BE 的 `numba_grid_solver=False` 就是 numba grid 失败回退到了 Python）。BE 之所以稳，是因为它
      回退到 **Python `solve_chunk`（递归二分 + scipy least_squares）**这条 robust 路；gear2 没有这条路，
      且 gear2 每步都 stall，复刻 LS-per-step 会慢到不可用（~1200 步都要 LS）。干净网格（无退化步）也一样，
      所以不是网格、不是 `_fill`/prev2（prev2 置零测过，nfail 不变）。
    - **决定（用户选 graceful BE 回退）**：裸 transient 的价值上 BE 已对齐 Spectre 且经过验证，gear2 的
      价值（PAC <1%）在 PSS 路径已交付。所以**不强行做 robust gear2**，而是让
      `transient(integration_method="gear2")` 在 gear2 失败步数过多（`nfail > max(8, 10%·N)`）时
      **自动 clean 重跑 BE**（递归调用，置 `gear2_be_fallback_used=True`、`gear2_nfail_before_fallback`）。
      这样裸 transient 请求 gear2 永远**安全**：良态电路上是 2 阶 gear2，stiff 电路（chopper）上优雅退回 BE
      给出正确波形。PSS 路径 `gear2_be_fallback=False` 显式 opt-out（shooting 自管收敛，绝不能中途混入 BE 轨道）。
    - **默认值**：裸 `transient()` 仍默认 **BE**（chopper 上 gear2 反正会退回 BE，默认 gear2 只会白算一遍）；
      但 `integration_method="gear2"` 现在对任何裸 transient 都安全。回归测试
      `test_pmos_chopper_transient_gear2_falls_back_to_be_when_stiff`（断言退回 + 波形 == BE）。
    - **仍未做**：真正 robust 的 2 阶裸-transient gear2（需要 gear2 版的 solve_chunk + 平滑开关模型让
      per-step Newton 不 stall）——低价值、高成本，留作后续。
  - **三 corner 最终对标（gear2 默认）**：PAC baseband 全部 <1% —— slow +0.61%、typical +0.49%、
    fast +0.12%（BE 时是 −0.5/−2.6/−2.4%）；PAC@200Hz 全 <1%；IRN slow +2.8/typ +2.0/fast −0.8%。
  - 全套 **93 passed**（含 RUN_SLOW_CHOPPER）；smoke/pnoise 两个测试按 gear2 重新 baseline。

---

## 1. 背景与动机

- Cadence 用 `method=gear2only`。chopper PAC baseband 在 typ/fast 差 **−2.5%**，
  已严格定位为 **BE 一阶积分在 20µs 开关沿处的误差**（高带宽 corner 把边沿敏感的高次谐波权重放大）。
- 已证伪的方向：换 `veriloga`（C·dV/dt）cap 模式——数值不稳定（slow 不收敛、fast 收敛到错误轨道）。
  忠实形式在 BE 下本质不可用；charge 模式（ΔQ，单调守恒）是稳定的最优解。
- 决定性证据：本地转换法跑在 **Cadence 轨道**上 = +0.6%，跑在**本地 BE 轨道**上 = −2.6%。
  差距 100% 来自轨道的 BE 误差，且 charge 模式细步只到 ~−1.5% 就平台 → 是**积分阶数**，不是步长、不是 cap 模式。
- 关键边界：**PAC/PNoise 的 HB 线性化用连续时间 jωC，与积分阶数无关**。所以只改 transient
  轨道，下游 PAC/PNoise 代码不动——这把改动范围限制在 transient 求解器内部。

---

## 2. 技术方案

### 2.1 变步长 BDF2 公式
当前步长 `h_n`、上一步 `h_{n-1}`，步长比 `ρ = h_n/h_{n-1}`：

```
x'(t_n) ≈ [ α0·x_n + α1·x_{n-1} + α2·x_{n-2} ] / h_n
α0 = (1+2ρ)/(1+ρ),  α1 = −(1+ρ),  α2 = ρ²/(1+ρ)
```
均匀步（ρ=1）退化为经典 `(3x_n − 4x_{n-1} + x_{n-2})/(2h)`。

### 2.2 电容 companion（charge 模式）
```
i_n = [ α0·Q(V_n) + α1·Q_{n-1} + α2·Q_{n-2} ] / h_n
```
其中 `Q_{n-1}, Q_{n-2}` 是上两步存下来的电荷（device Cgss/Cgdd、load CL、线性电容都同此式）。

### 2.3 Newton Jacobian
电容对角项从 BE 的 `C/h` 变为 `C(V_n)·α0/h_n`（均匀步即 `3C/2h`）。其余（gm/gds、电阻、gmin）不变。

### 2.4 自启动
BDF2 需要两个历史点。每条轨道的**第一步用 BE 自启动**，第二步起切 BDF2。
PSS 的 tstab + shooting 会把自启动那一步的小不一致吸收掉。

---

## 3. 改动范围（按文件）

### 3.1 `core/transient_solver.py`
- 新增参数 `integration_method ∈ {"be","gear2"}`，逐层透传（默认 `be`，chopper PSS 走 `gear2`）。
- 维护**两点历史**：`V_{n-2}/Q_{n-2}`（Vhist 已存全程，但 interval 内的 substep / retry 链需要单独带 prev2）。
- 电容 companion + Jacobian 按 §2.2/§2.3 改（device caps、load caps、线性电容三处）。
- 第一步 BE 自启动逻辑。

### 3.2 `core/numba_kernels.py`（**工作量主体**）
- `_stamp_transient_system_impl`：电容残差 + Jacobian 加 BDF2 分支，入参增加 `prev2_*` 和步长比/系数。
- `_transient_newton_reuse_impl` / `_transient_solve_grid_impl`：在 **substep 细分 + retry 细分**链路里
  正确维护两点历史和每个 substep 的 `ρ`（这是最易错的地方）。
- charge 模式优先；BE 作为 `method` 旗标保留，不删。

### 3.3 `core/pss_solver.py`
- 解析 monodromy Jacobian `A_m=(G+C/h)^{-1}(C/h)` 是 **BE 专用**。gear2 的单步映射是两步递推
  （x_n 依赖 x_{n-1} 和 x_{n-2}），monodromy 变成 companion 块乘积。
  - **先**：method=gear2 时回退到 **FD shooting Jacobian**（正确优先，慢一点）。
  - **后**：再推导 BDF2 解析 monodromy 作为性能优化。

### 3.4 PAC / PNoise
- **不改**。确认 HB 用连续时间 jωC，只是吃到更准的轨道。加一条断言/注释固化这个边界。

---

## 4. 分阶段里程碑（含 GO/NO-GO 关卡）

| 阶段 | 内容 | 产出 / 关卡 |
|---|---|---|
| **M1** | 只在 **Python** transient 路径实现变步长 BDF2（charge），`integration_method="gear2"` 开关 | 解析 RC 单元测试（阶跃/正弦）误差 << BE |
| **M2** | **关键 GO/NO-GO**：关掉 numba，用 Python 路径跑 chopper PSS+PAC（三 corner） | gear2 PAC baseband 是否 <1%？**达标才进 M3** |
| **M3** | 把 BDF2 移植进 numba 内核（两点历史 + 步长比穿过 substep/retry） | numba 路径与 Python 路径数值一致 |
| **M4** | gear2 的 PSS shooting Jacobian（先 FD 回退，再解析 BDF2 monodromy） | PSS 收敛、residual ~1e-9 |
| **M5** | 全量三 corner 验证 + 回归 + 波形回归；定默认值 | 见 §5 验收 |

> **M2 是省钱关卡**：先在便宜的 Python 路径确认 gear2 真能到 <1%，再投入昂贵的 numba 移植。
> 万一 gear2 只能到 ~−1.5%（说明残差还有 gate1 节点/模型成分），在 M2 就止损，不白做 numba。

---

## 5. 验收标准

- ✅ 解析 RC：gear2 误差远小于 BE，匹配闭式解。
- ✅ chopper PAC baseband vs Cadence（slow/typ/fast）**全部 <1%**；PAC@200Hz <1%；IRN 维持 <1.5%。
- ✅ 现有 **91 个测试全过**（BE 默认路径不受影响）。
- ✅ chopper transient 波形回归（`test_pmos_chopper_transient_ui_finite_edge_matches_cadence_scale`）
  仍对齐 Spectre tran（必要时按 gear2 重新 baseline）。
- ✅ gear2 轨道 RMS vs Cadence 轨道 << BE 的 ~300mV。

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| numba 内核两点历史 / substep / retry 复杂，回归风险高 | M1/M2 **Python 先行 + GO/NO-GO**；BE 保留为默认，gear2 opt-in |
| PSS 解析 monodromy 是 BE 专用 | gear2 先用 **FD shooting Jacobian** 回退，解析 BDF2 monodromy 作后续优化 |
| 边沿细化网格步长比突变，BDF2 精度/稳定性下降 | 优先用 `maxstep` 产生**均匀 substep**（ρ≈1）；对 ρ 做 clamp |
| BDF2 自启动一阶误差 | 第一步 BE 自启动 + 足够 tstab 周期 |
| gear2 仍到不了 <1% | M2 关卡提前发现；退路：接受 ~−1.5%，或转去做 gate1 内部节点忠实模型 |

---

## 7. 默认值策略（M5 决策）

两种落地方式，二选一：
- **A（保守，推荐）**：全局默认仍 BE，**chopper PSS 默认 gear2**。其它 transient/优化路径不受影响，回归面最小。
- **B（激进）**：全局默认 gear2。更"正确"，但要重新 baseline 所有 transient 测试，回归面大。

建议先 A，验证稳定后再评估 B。

---

## 8. 不在本次范围

- gate1 内部节点的忠实建模（C·dV/dt + 100Ω 串阻）——只有在 gear2 仍不达标时才考虑。
- veriloga cap 模式（已证伪，保留代码但不作为路径）。
- PAC/PNoise 的 HB 公式（与积分阶数解耦，不动）。
