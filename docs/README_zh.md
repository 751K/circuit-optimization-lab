# 电路本地建模与优化流程

[English](README.md) | [中文说明](README_zh.md)

## 项目概述

一个**本地、免 license、对 Cadence 标定过的模拟电路仿真 + ML 驱动设计优化框架**——目标是成为芯片设计
两个转变的**基础设施**：

**1. 面向"ML 加速仿真"的开放基础设施。** 用 ML 加速芯片设计仿真是一个快速增长的方向，但这个领域至今
缺少开放的、基础性的构件：既没有标准、开放的方式来*生成大规模、物理保真的仿真数据*，也没有可复用的
*数据生成器框架*，更没有为电路量身定做的 *ML 方法框架*。本项目把整条链路做成一条开放、可复现的流水线
——以对 Cadence 标定过的求解器栈作为物理保真的 oracle → 带 provenance 的带标签数据集生成器
（[`core/dataset.py`](../core/dataset.py)）→ ML 代理 + 筛选-校验优化器（[`core/surrogate.py`](../core/surrogate.py)、
`surrogate_torch.py`、`optimize.py`）。电路进、数据出、代理训练、设计优化，全程无商业仿真器。

**2. 一条本地、可被 LLM 调用的工具链。** LLM 驱动的电路设计越来越依赖在回路里反复调用仿真器，但在多轮
对话中反复调用商业工具（要 license、在远端、慢）既昂贵又无法在本地自主完成。本项目是一条自包含的工具链，
LLM agent 可以调用它来**自主获得仿真结果、分析归纳、优化设计**——整个 DC/AC/Noise/PSS/… 回路都在本地
设备上完成，无外部往返、无 license。JSON 电路格式、`python -m core` CLI、结构化的结果字典，都是刻意做成
机器可调用、机器可读的，好让 agent 从本地 shell 里驱动整个设计闭环。

求解器栈**对 Cadence Spectre 24.1 标定**——以 AT4000TG PMOS-OTFT 心电 AFE 为标定锚点（增益/带宽/IRN
通常 <1%）——并通过同一套引擎泛化到多个工艺：AT4000TG OTFT，加上两个行业标准硅 CMOS 节点——**SKY130**
（130nm，经 OpenVAF 编译的 BSIM4 + OSDI 宿主）与 **FreePDK45**（45nm，以 ngspice-C 作为精确器件求值器）。


**具体能做：**
- **DC / AC / Noise / Transient** — 标准电路分析，无需仿真器 license。
- **PSS / PAC / PNoise** — 周期稳态 / AC / 噪声（对标 Spectre RF），用于 chopper 放大器。
- **设计探索 + ML-surrogate 优化** — 扫尺寸/偏置，造带标签数据集，训练快速 surrogate，筛选大候选池，再把入围候选送回已标定求解器校验（比暴力扫描快几个数量级）。
- **工艺角与失配** — 全局工艺角、逐器件 mismatch Monte Carlo、latch 筛查。
- **多工艺** — OTFT（AT4000TG）+ 硅 SKY130 + FreePDK45 走同一条管线；两个完整的全差分 OTA 设计案例作为报告（[SKY130](sky130_fd_ota_design.md)、[FreePDK45](freepdk45_fd_ota_design.md)）。

求解器内部实现见 [核心求解器概览](core_overview_zh.md)。

---

## 快速上手

```bash
# 1. 安装（已含 Numba —— 求解器热路径是 JIT 编译内核）
python3 -m pip install -r requirements.txt

# 2. 运行第一个电路 — 一条命令
python3 -m core examples/periodic_rc.json

# 3. 验证安装 — 跑 AFE 基准
python3 -m benchmarks.bench_afe --warm-runs 1 --skip-noise
```

上面第三条命令会对一个无源 RC 低通电路运行 AC、Noise、PSS、PAC、PNoise 分析并输出摘要。
无需编写任何 Python 代码。打印出数字即说明一切正常。之后可替换任意电路 JSON 文件，
或用 `-a ac,noise` 选择特定分析。

### 依赖

**Python（pip）。** OTFT / 分析核心只需要 `requirements.txt`（numpy、scipy；Numba 可选但推荐——
求解器热路径是 JIT 内核）。`requirements-ml.txt` 补 scikit-learn / torch 给 ML 代理层；
`requirements-demo.txt` 补 Flask 给 Web demo。以上全部无需外部仿真器、无需 license。

**外部开源工具（仅硅 PDK 需要——不是 pip 依赖）。** 默认的 AT4000TG OTFT 工艺和所有分析没有这些也能
完整运行；只有 `sky130.*` / `freepdk45.*` 模型类型需要它们，缺失时会在第一次用到时报出清晰错误。

| 工具 | 在本项目中的作用 | 用哪个环境变量指向 |
|------|-----------------|------------------|
| **[OpenVAF-Reloaded](https://github.com/751K/OpenVAF-Reloaded)** —— [OpenVAF](https://github.com/pascalkuthe/OpenVAF) 的维护 fork | 把标准 Verilog-A 紧凑模型（BSIM4 等）编译成原生 `.osdi` 共享库；**SKY130** 器件路径通过 OSDI 宿主（`core/osdi_host.py`）加载它 | `OPENVAF_ROOT` |
| **[ngspice](https://ngspice.sourceforge.io/)** | 其内置 C-BSIM4 是 **FreePDK45** 的精确器件求值器 / oracle（`core/ngspice_char.py`），也负责解析 SKY130 的 binned 参数卡 | `NGSPICE_BIN` |
| PDK 卡文件（SKY130 经 `volare`/`ciel`；FreePDK45 卡） | 工艺参数本身 | `PDK_ROOT` |

### CLI 参考

`python -m core` 使用子命令（向后兼容——无子命令时默认为 `run`）：

```bash
# ── 分析调度（默认："run"） ──
python -m core examples/periodic_rc.json                          # 运行所有已配置的分析
python -m core examples/periodic_rc.json -a ac,noise,pss          # 指定分析类型
python -m core run examples/periodic_rc.json -a ac,noise          # 显式子命令

# ── 设计空间探索 ──
python -m core examples/afe_explore.json --explore -n 500         # --explore 标志（旧用法）
python -m core explore examples/afe_explore.json -n 500 --seed 1  # 子命令

# ── 工艺角扫描 ──
python -m core corners examples/afe_explore.json                  # typ/slow/fast
python -m core corners examples/afe_explore.json --freqs-num 61

# ── Mismatch Monte Carlo ──
python -m core mc examples/afe_explore.json -n 200 --seed 1      # typical corner
python -m core mc examples/afe_explore.json --corner slow -n 500

# ── Chopper 分析 ──
python -m core chopper examples/afe_explore.json --level ideal    # 方波 LPTV
python -m core chopper examples/afe_explore.json --level pmos     # 静态相位 PMOS
python -m core chopper examples/afe_explore.json --level lptv     # PMOS 边带折叠
python -m core chopper examples/afe_explore.json --level pss      # shooting PSS
python -m core chopper examples/afe_explore.json --level pnoise   # PSS→PAC→PNoise
python -m core chopper examples/afe_explore.json --level transient

# 所有子命令通用选项：
#   --noise-band LO HI  IRN 积分带宽（默认：0.05 100.0）
#   -o PATH             结果输出到文件
#   --no-numba          关闭 Numba 加速
#   --quiet             关闭进度输出
```

### 一分钟搞懂代码结构

在看具体工作流之前，先搞清楚几个核心概念：

| 概念               | 是什么                                  | 在哪定义                                             |
| ---------------- | ------------------------------------ | ------------------------------------------------ |
| **Topology（拓扑）** | 电路结构——有哪些节点、器件如何连接、输入输出在哪            | `core/topology.py`，或从 JSON 自动生成                  |
| **Sizes（尺寸）**    | `{器件名: (W_µm, L_µm)}`——晶体管宽长         | JSON `sizes` 字段                                  |
| **NF**           | Number of fingers（晶体管并联数，等比例放大电流）    | JSON `nf` 字段，或每个器件单独指定 `devices[].NF`            |
| **Bias（偏置）**     | `{节点名: 电压}`——各 rail 节点的 DC 工作电压      | JSON `bias` 字段                                   |
| **Solver（求解器）**  | 接收 拓扑 + 尺寸 + 偏置 → 输出结果（增益、噪声、波形…）的函数 | `core/ac_solver.py`、`core/transient_solver.py` 等 |
| **Device Model（器件模型）** | 抽象接口（`TransistorModel`）——求解器面向接口调用，不依赖具体模型；通过工厂切换模型类型 | `core/device_model.py`、`core/pmos_tft_model.py` |

所有求解器的调用模式都一样：

```python
result = solver(sizes, bias, ..., topo=topology, nf=nf)
```

JSON 文件把这些输入打包在一起；`load_circuit_json()` 解包成 `CircuitSpec` 对象，
包含 `.topology`、`.sizes`、`.bias`、`.nf` 以及可选的 `.explore`。

---

## 常用工作流

以下所有示例均可直接复制运行。它们使用的都是 `examples/afe_explore.json`
中已与 Cadence 对标验证的锁定 AFE 设计。

### 1. 加载电路 + DC / AC / Noise 分析

```python
import numpy as np
from core.circuit_loader import load_circuit_json
from core.ac_solver import ac_solve
from core.noise_solver import noise_analysis, band_rms

# 从 JSON 加载电路 —— 求解器代码里不硬编码任何节点名
spec = load_circuit_json("examples/afe_explore.json")
freqs = np.logspace(-2, 4, 121)   # 0.01 Hz 到 10 kHz

# DC 工作点 + AC 增益 / 带宽
ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)
print(f"增益: {ac['Av_dc_dB']:.2f} dB,  带宽: {ac['bw_Hz']:.1f} Hz")
# → 增益: 22.89 dB,  带宽: 549.3 Hz

# 噪声分析（热噪声 + 闪烁噪声）
noise = noise_analysis(spec.sizes, spec.bias, freqs,
                       topo=spec.topology, nf=spec.nf)
irn_uv = band_rms(freqs, noise["irn_psd"], 0.05, 100.0) * 1e6
print(f"IRN (0.05–100 Hz): {irn_uv:.2f} µVrms")
# → IRN (0.05–100 Hz): 36.97 µVrms
```

### 2. 瞬态仿真

```python
from core.transient_solver import transient

# 4 ms 仿真，在 t=0.5 ms 处施加 0.5 mV 差分阶跃
t = np.linspace(0, 4e-3, 400)
vip = np.where(t >= 0.5e-3, 30.65 + 0.5e-3, 30.65)
vin = np.where(t >= 0.5e-3, 30.65 - 0.5e-3, 30.65)

# 默认：后向欧拉（BE）——鲁棒、经过充分验证
tran = transient(spec.sizes, spec.bias, t, vip, vin,
                 topo=spec.topology, nf=spec.nf)
print(f"瞬态步数: {len(t)},  失败步数: {tran['nfail']}")
# → 瞬态步数: 400,  失败步数: 0

# 可选：gear2/BDF2 —— 二阶、刚性稳定（chopper PSS/PAC/PNoise 默认使用）
tran_gear2 = transient(spec.sizes, spec.bias, t, vip, vin,
                       topo=spec.topology, nf=spec.nf,
                       integration_method="gear2")
```
Gear2（变步长 BDF2）将 PAC baseband 误差从 BE 的约 −2.5% 降到三 corner 全部 <1%。
在刚性电路（如 chopper）上，`integration_method="gear2"` 如果同时请求
`max_retry_subdivisions` / `max_step`，会留在 Numba gear2 grid 内维护两步历史并做二分 retry；
PSS/PAC/PNoise 管线默认使用快速 Numba gear2 grid 以保证精度；
裸 `transient()` 默认使用 BE。全局 transient 电容算子仍默认使用电荷守恒的 Q-stamp；
PMOS chopper PSS wrapper 单独默认 `cap_mode="average"`，用于匹配 Cadence 在高阻内部节点上的
commutation feedthrough；泛化的 stiff switched-capacitor 电路仍以 `charge` 作为更稳的默认选择。

### 3. Chopper 分析（三种精度层级）

#### 层级 1 — 理想 LPTV（最快，方波乘法器模型）

```python
from core.chopper import chopper_analysis

chop_ideal = chopper_analysis(
    spec.sizes, spec.bias, freqs, f_chop=225.0,
    topo=spec.topology, nf=spec.nf, max_harmonic=31,
    band=(0.05, 100.0))
print(f"理想 chopper: {chop_ideal['peak_dB']:.2f} dB,  "
      f"IRN: {chop_ideal['irn_uV_band']:.2f} µVrms")
```

#### 层级 2 — PMOS 开关（静态两相分析，无需 PSS）

```python
from core.chopper import pmos_chopper_analysis

pmos = pmos_chopper_analysis(
    spec.sizes, spec.bias, freqs,
    switch_size=(20000, 80), band=(0.05, 100.0))
print(f"PMOS 静态 chopper: {pmos['peak_dB']:.2f} dB,  "
      f"IRN: {pmos['irn_uV_band']:.2f} µVrms")
```

#### 层级 3 — 完整 PSS / PAC / PNoise（第一性原理，对标 Spectre）

```python
from core.chopper import (pmos_chopper_pss, pmos_chopper_pac,
                           pmos_chopper_pnoise)

# 第一步：PSS — 求解周期稳态轨道
pss = pmos_chopper_pss(
    spec.sizes, spec.bias, f_chop=225.0,
    switch_size=(5000, 30), edge_time=20e-6,
    tstab_periods=2, n_points=121)
print(f"PSS 收敛: {pss['converged']},  "
      f"残差: {pss['residual_norm']:.2e}")

# 第二步：PAC — 在 PSS 轨道上做周期 AC 增益分析
pac = pmos_chopper_pac(
    spec.sizes, spec.bias, freqs, f_chop=225.0,
    pss_result=pss)
print(f"PAC 增益: {pac['Av_dc_dB']:.2f} dB,  带宽: {pac['bw_Hz']:.1f} Hz")

# 第三步：PNoise — 周期噪声（chopper 默认 TD-adjoint，无需标定常数）
pnoise = pmos_chopper_pnoise(
    spec.sizes, spec.bias, freqs, f_chop=225.0,
    pss_result=pss, pac_result=pac, max_sideband=10,
    band=(0.05, 100.0))
print(f"PNoise IRN: {pnoise['irn_uV_band']:.2f} µVrms")
```

PSS→PAC→PNoise 三件套是 Cadence Spectre `pss` + `pac` + `pnoise` 的本地等价实现。
PAC 现在有两条一等公民路径：

- 通用默认：解析伴随谐波平衡（`method="pss_analytic_adjoint"`）。这是最通用路径，
  支持 bordered MNA 场景。
- Chopper 默认：time-domain Floquet PAC（`method="pss_time_domain"`）。它先构建一次
  单周期 monodromy，再对每个频点求一个小的 quasi-periodic 边界系统，避免大型
  `(2K+1)n` HB 矩阵。对 PMOS_TFT 周期转换，它保留每个器件的内部 `gate1` 小信号状态
  （`R_cap`、`R_cap2`、`Cgs`、`Cgd`），不再把器件逐时刻塌缩成端口
  `{gm,gds,Cgs,Cgd}`。转换线性化使用 Verilog-A 风格的 `C(V)*ddt(V)` 算子，
  与生成 PSS 轨道时使用的 transient Q/average companion 分开处理，因为 Spectre PAC
  折叠的是这个转换算子。Numba 可用且所有 PMOS 都有 `gate1` 动态时，这条 gate1
  扩维转换会走编译版 `pac_linearize_orbit_gate1` 内核。这修复了旧 slow-corner
  −1.89% chopper PAC 误差；D3 slow 回归现在已进入 Cadence 1% 门限内。

只有需要原有限差分 shooting 时才设置 `analytic=False`（每频点需 `n_state+2`
次瞬态周期）。Chopper PNoise 现在默认使用 time-domain Floquet adjoint
（`pnoise_time_domain_used=True`）：直接求稀疏周期伴随 BVP，因此转换不再受
HB 边带截断限制。旧的 K=32 HB 结果看起来已经“够舒服”，但仍留下 0.6-1.8%
IRN 误差；这条 TD 路径把这个“假舒适”揭掉了。HB PNoise 仍可用
`time_domain=False` 显式作为对照/兜底。
对 D3 / `chop_tb_d3` 官方 `slow` corner Spectre 参考，在 `f_chop=200 Hz` 下，
默认 chopper PAC 约 +0.03%，TD PNoise IRN 约 +0.02%。三 corner IRN 误差从
HB-K32 的 slow/typical/fast = +1.81% / +1.05% / +0.66%，更新为 TD adjoint 的
+0.02% / −0.00% / +0.57%。全流程仍不需要任何 Cadence 标定常数。
`pmos_chopper_pac` / `pmos_chopper_pnoise` 是 chopper 兼容包装器；任意周期拓扑可直接使用
`core.pac_solver.pac_solve` 和 `core.pnoise_solver.pnoise_solve`，输入为通用 `pss_solve`
返回的周期轨道和 `input_drive` 映射。

**JSON dispatch** — 当电路 JSON 包含 `periodic` 和 `analyses` 块时，一个调用即可运行所有分析：

```python
from core.analysis_dispatch import run_analysis_suite
from core.circuit_loader import load_circuit_json

spec = load_circuit_json("examples/periodic_rc.json")
results = run_analysis_suite(spec)
# results["pss"]、results["pac"]、results["pnoise"] 全部就绪
```

JSON dispatch 也支持同一个 opt-in PAC 开关：
`"analyses": {"pac": {"time_domain": true, "td_integration": "gear2"}}`。

### 4. 设计空间探索 / 优化

```python
from core.explore import explore
from core.circuit_loader import load_circuit_json

spec = load_circuit_json("examples/afe_explore.json")

# JSON 中的 "explore" 块定义了设计变量、约束和目标。
# explore() 采样候选点，通过求解器逐个评估，按约束过滤，返回 Pareto 前沿。
result = explore(spec.topology, spec.sizes, spec.bias, spec.nf,
                 spec.explore, n=500, method="lhs", seed=42)

print(f"候选总数: {result['n_total']},  "
      f"可行解: {result['n_feasible']},  "
      f"Pareto 最优: {len(result['pareto'])}")
# → 候选总数: 500,  可行解: 87,  Pareto 最优: 12
```

也支持命令行：

```bash
python -m core.explore examples/afe_explore.json --n 500 --seed 42
```

结果导出为 CSV 和 JSONL。JSON 中的 explore 配置指定了扫描哪些变量（器件 W/L、偏置电压）、
什么约束条件（增益 > X、IRN < Y 等），以及优化什么目标。

### 5. 工艺角与失配分析

```python
from core.corners import CORNERS, corner_table, mismatch_mc, latch_screen
import numpy as np

# 工艺角扫描 — 一个设计在 typ/slow/fast 下的指标
table = corner_table(spec.sizes, spec.bias, np.logspace(-2, 4, 121),
                     topo=spec.topology, nf=spec.nf)
for row in table:
    print(f"{row['corner']:>6s}:  增益={row['gain_peak_dB']:.2f} dB,  "
          f"BW={row['bw_Hz']:.0f} Hz,  IRN={row['irn_uV']:.2f} µVrms")
# → typical:  增益=22.89 dB,  BW=549 Hz,  IRN=36.97 µVrms
# →   slow:  增益=20.81 dB,  BW=328 Hz,  IRN=45.72 µVrms
# →   fast:  增益=24.41 dB,  BW=846 Hz,  IRN=28.40 µVrms

# 快速 latch 筛查（确定性方法，速度足够快，可嵌入搜索内循环）
rng = np.random.default_rng(0)
latch = latch_screen(spec.sizes, spec.bias, topo=spec.topology,
                     nf=spec.nf, rng=rng, k_sigma=3.0)
print(f"Latch dV: {latch['latch_dV']*1e3:.2f} mV  "
      f"({'已 latch' if latch['latched'] else '正常'})")

# 完整 mismatch Monte Carlo（较慢，用于最终验证）
mc = mismatch_mc(spec.sizes, spec.bias, np.logspace(-2, 4, 61),
                 topo=spec.topology, nf=spec.nf, n=200,
                 corner=CORNERS["typical"], seed=1)
print(f"Latch 率: {mc['latch_rate']*100:.1f}%,  "
      f"IRN: {mc['irn_mean']:.2f} ± {mc['irn_std']:.2f} µVrms")
```

### 6. ML-Surrogate 设计优化

已标定的求解器是"teacher"；一个快速 surrogate 学习它们的指标，让你能筛选比暴力 solver 扫描
多几个数量级的候选。整条链路是一串 CLI 命令——造带标签数据集、训练 surrogate、然后筛选-校验：

```bash
# 1. 从配置的 "explore" 块造一个带标签数据集（每个样本都保留，含 DC 失败样本——
#    这是训练集，不是过滤后的搜索结果）
python -m core dataset examples/single_stage.json -n 500 --out ds/run1

# 2. 训练一个指标 surrogate（梯度提升树；需要 `pip install -r requirements-ml.txt`）
python -m core.surrogate train ds/run1.npz --out ds/run1.pkl

# 3. 用 surrogate 筛选大候选池、Pareto 选择，再把入围候选送回真实的已标定求解器校验
python -m core.optimize examples/single_stage.json ds/run1.pkl -n 100000 --top-k 20
# → surrogate 在 1.64s 内筛完 100000 个设计（60,802/s）
# →   可行：89038   Pareto 前沿：19
# → 在求解器上校验 top-19，用时 0.45s（23.6 ms/design）
# →   solver 确认可行：18/19（加速比：全量 100000 跑 solver 约需 ~1,436×）
```

求解器（而不是 surrogate）永远有最终话语权：真正的设计决策只信任那个小小的、已校验的入围
名单。`surrogate train` 的 `--filter label:lo:hi` 把训练限制在感兴趣区域内（例如剔除会拖累
拟合的甩轨/collapse 设计）。完整的 dataset/surrogate/optimize 参数参考见
[CLI 参考手册 §1.7–1.8](cli_reference.md)；surrogate 路线图和已验证的精度数字见
`docs/futureplan.md` §7。

### 7. 硅 PDK（SKY130 + FreePDK45）

电路的 `models` 字段把特定器件绑到非默认 PDK——比如真实的硅 SKY130 晶体管（NMOS + PMOS，
经 OpenVAF 编译的 BSIM4，通过和 OTFT 相同的 `TransistorModel` 接口跑）。这纯粹是增量的：
没有 `models` 条目的器件仍用默认 OTFT PDK。

```json
"models": {
  "M1": {"type": "sky130.nmos", "extract_w": 24.0},
  "M3": {"type": "sky130.pmos", "vb": 1.8, "extract_w": 12.0}
}
```

硅器件走的是完全相同的 DC/AC/noise 管线——包括上面的 ML-surrogate 设计闭环和跨工艺角
（`tt`/`ss`/`ff`/`sf`/`fs`）复验：

```bash
python -m core dataset examples/sky130_5t_ota.json -n 400 --out ds/ota          # 造一个硅数据集
python -m core.surrogate train ds/ota.npz --filter gain_dB:0:60 --out ds/ota.pkl # 只在工作区训练
python -m core.optimize examples/sky130_5t_ota.json ds/ota.pkl -n 50000 --top-k 10          # 筛选+校验（typical 工艺角）
python -m core.optimize examples/sky130_5t_ota.json ds/ota.pkl -n 50000 --corner ss          # 在慢角重新校验
```

在互补 5T OTA 示例上，这条链路的筛选比 solver 快约 6000×，solver 校验确认入围的 10 个设计中
9 个可行，同一批设计在慢角（`ss`）重新校验时仍有 9 个可行。需要 SKY130 PDK + OpenVAF + ngspice
工具链（见 `docs/futureplan.md` §9）；未安装时求解器会给出清晰报错。**设计说明**：和任何单级
硅放大器一样，这个示例用的是电流镜有源负载（定 DC 工作点/增益）加电容负载（定带宽）——纯电阻
负载的共源级不代表片上设计的常见做法，扫描尺寸/偏置时也容易大范围甩轨，这正是本项目没有收录
这类示例的原因。

**FreePDK45（45nm，1.0V）** 绑定方式相同（`"freepdk45.nmos"` / `.pmos`），但用**不同的求值器**：
它的 BSIM4 卡是为 ngspice 内置 C-BSIM4 调的,所以 oracle 不是 OSDI VA 而是 **ngspice-C 本身**——
每个器件表征一次成缓存的 `(Vsb,Vds,Vgs)` 网格再插值（节点处 exact ngspice-C,µs/eval）。它同样支持
`extract_w`（快速 W 扫描）与 `temperature`（PVT），以及精确 ngspice `.noise`。完整的
[FreePDK45 FD-OTA 案例](freepdk45_fd_ota_design.md) 走了全流程——架构 → 600 样本 GBT 代理 →
筛选/校验/打磨 → 整机对 ngspice 自己的 `.ac` 交叉核对 → 27 角 PVT——最终一个全差分望远镜 OTA
落在 **58.9 dB / 119.9 MHz / 17 µW**（所有指标以 ngspice oracle 为准）。

---

## JSON 电路格式

新电路通过 JSON 定义，无需修改求解器源码。完整字段参考见
[JSON 电路描述格式](json_circuit_format_zh.md)。

快速示例 (`examples/single_stage.json`)：

```json
{
  "solved": ["OUT"],
  "rails": {"VDD": 40.0, "GND": 0.0},
  "devices": [
    {"name": "M1", "drain": "OUT", "gate": "IN", "source": "VDD",
     "W": 2000, "L": 80, "NF": 1}
  ],
  "bias": {"VDD": 40.0, "VIN": 30.0, "VB": 10.0},
  "outputs": ["OUT"],
  "input_drives": {"IN": 1.0},
  "load_caps": {"OUT": 1e-12}
}
```

主要字段说明：

| 字段                 | 必填  | 用途                                                                                                    |
| ------------------ | --- | ----------------------------------------------------------------------------------------------------- |
| `solved`           | 是   | 求解器需要求解电压的节点列表                                                                                        |
| `rails`            | 是   | 固定电压节点：`{"VDD": 40.0, "GND": 0.0, ...}`                                                               |
| `devices`          | 是   | PMOS 晶体管；无源电路可写空数组 `[]`                                                                              |
| `bias`             | 是   | 每个 rail 节点的 DC 电压：`{"VDD": 40.0, "VIN": 30.0, ...}`                                                   |
| `outputs`          | 是   | 观测增益/噪声的输出节点                                                                                          |
| `input_drives`     | —   | AC 小信号激励注入位置（驱动器件栅极）                                                                                  |
| `load_caps`        | —   | 各输出节点的负载电容 (F)：`{"OUT": 1e-12}`                                                                       |
| `resistors`        | —   | `[名称, 节点A, 节点B, 阻值]`                                                                                  |
| `capacitors`       | —   | `[名称, 节点A, 节点B, 容值]`                                                                                  |
| `current_sources`  | —   | 理想直流电流源：`[名称, nplus, nminus, 电流]`                                                                     |
| `vccs`             | —   | 压控电流源（VCCS）：`[名称, p, q, ctrl_p, ctrl_n, gm]`，``I = gm*(Vcp-Vcn)``                                    |
| `vsources`         | —   | 理想电压源（真·MNA）：`[名称, p, q, value]`，`value` 为常数 EMF 或波形 key                                              |
| `nf`               | —   | 全局 NF（fingers），作用于所有器件；可被器件自身的 `NF` 覆盖                                                                |
| `dc_guesses`       | —   | DC 收敛的初始电压猜测，复杂电路需要此字段帮助收敛                                                                            |
| `transient_inputs` | —   | 瞬态输入波形名到驱动节点的映射                                                                                       |
| `ac_drives`        | —   | 类似 `input_drives`，但驱动的是*节点*而非器件栅极（用于 testbench 前端网络）                                                  |
| `periodic`         | —   | PSS/PAC/PNoise 和周期 transient 使用的大信号周期输入描述                                                                |
| `analyses`         | —   | `run_analysis_suite()` 的分析 dispatch 配置：`ac/noise/transient/pss/pac/pnoise`                                     |
| `aliases`          | —   | 节点别名，方便工具/扫描按名称找到关键节点（如 `"VOP"`、`"VON"`）                                                              |
| `explore`          | —   | 设计空间探索配置（变量范围、约束条件、优化目标）                                                                              |

---

## 交互式 AFE Tuner

基于 Web 的实时调参工具：

```bash
python3 -m pip install -r requirements-demo.txt
python3 demo/server.py
# 浏览器打开 http://localhost:5100
```

在浏览器中调整器件 W/L 和偏置电压，实时查看增益、带宽和等价输入噪声变化。
内置预设设计（Base、Final Locked、Min Area、First Feasible）。

---

## 性能基准

五个固定性能基准，用于性能回归跟踪：

```bash
python3 -m benchmarks.bench_afe --warm-runs 3         # AC+noise+transient
python3 -m benchmarks.bench_model --warm-runs 3       # 单管微基准
python3 -m benchmarks.bench_chopper --warm-runs 3     # Chopper: 5 个分析层级
python3 -m benchmarks.bench_periodic --warm-runs 3    # 周期求解器（PSS/PAC/PNoise）
python3 -m benchmarks.bench_sweep --n-candidates 200  # 批量 explore 负载
```

设置 `CIRCUIT_USE_NUMBA=0` 可对比纯 Python 性能。Numba 内核默认使用磁盘缓存，
后续新的 Python 进程可避免大部分重复冷启动 JIT 开销；设置 `CIRCUIT_NUMBA_CACHE=0`
可关闭该缓存。MacMini M4 上 Numba 预热后的典型耗时：

| 基准                                      | 耗时      |
| --------------------------------------- | ------- |
| AC 121 点                                | ~1.5 ms |
| Noise 121 点（独立调用）                       | ~1.7 ms |
| DC+AC+Noise 121 点（复用 AC 结果）              | ~1.8 ms |
| Transient 200 步                         | ~5 ms   |
| 理想 chopper（31 次谐波）                      | ~5 ms   |
| PMOS chopper LPTV                       | ~22 ms  |
| Chopper transient（8-PMOS, 225 Hz, 2 周期，UI 尺寸） | ~0.15–0.19 s |
| Chopper PSS+PAC(HB)+PNoise（61 点，UI 尺寸，`time_domain=False`）    | ~25.6 s |
| Chopper PSS+PAC(HB)+PNoise（121 点，UI 尺寸，`time_domain=False`）   | ~48.9 s |
| Chopper PAC time-domain only（61 点，同一 PSS 轨道，含 gate1 状态） | ~1.4 s |
| Chopper PAC time-domain only（121 点，同一 PSS 轨道，含 gate1 状态） | ~1.9 s |
| 批量 sweep（200 候选, AC+noise）              | ~0.5 s  |

25.6s / 48.9s 这两行是显式 HB PAC 路径的全流程时间。对 rail-driven chopper，
`pmos_chopper_pac` 现在默认使用 time-domain 路径，PAC 不再是同量级瓶颈；只有需要
HB 对照时才设置 `time_domain=False`。

---

## 示例文件

| 文件                                  | 说明                                                |
| ----------------------------------- | ------------------------------------------------- |
| `examples/afe_explore.json`         | 锁定 10 管 AFE 设计，含尺寸、偏置、NF 和 explore 扫描配置           |
| `examples/single_stage.json`        | 最小单管共源级——新建电路的最佳起点                                |
| `examples/resistor_load_stage.json` | 带电阻负载的单管电路，演示 `resistors` 和 `current_sources` 字段         |
| `examples/periodic_rc.json`         | 无源 RC 低通，带 PSS/PAC/PNoise dispatch——最简单的端到端周期示例           |
| `examples/voltage_divider.json`     | 理想电压源（真·MNA）分压器，含电阻电容——vsource 演示                          |
| `examples/vcvs_amplifier.json`      | VCVS 放大器，线性增益 100×——受控源演示                          |
| `examples/sc_lpf.json`              | 开关电容低通滤波器（两相，单端 LPTV）                          |
| `examples/sky130_5t_ota.json`       | 硅 SKY130 互补 5T OTA——`models` 块 + 完整 dataset/surrogate/optimize 设计闭环 |
| `examples/sky130_fd_ota.json`       | 硅 SKY130 全差分望远镜 OTA + CT CMFB——[SKY130 FD-OTA 设计案例](sky130_fd_ota_design.md) |
| `examples/freepdk45_5t_ota.json`    | FreePDK45（45nm）5T OTA——ngspice-C 求值器，单端输出 |
| `examples/freepdk45_fd_ota.json`    | FreePDK45 全差分望远镜 OTA + CT CMFB——[FreePDK45 FD-OTA 设计案例](freepdk45_fd_ota_design.md) |
| `examples/afe_testbench.py`         | 完整 testbench：干电极前端（R∥C 网络）→ AFE 核心 → AC + 噪声 + 瞬态 |
| `examples/mc_mismatch.py`           | Monte Carlo mismatch 驱动：工艺角表 + 3-corner MC 图      |
| `examples/find_max_gain.py`         | PMOS 反相器最大增益扫描      |
| `examples/sweep_vin_vout.py`        | PMOS 反相器 DC VIN→VOUT 传输曲线      |
| `examples/sc_lpf.py`                | 开关电容 LPF 瞬态仿真      |

---

## 常见问题

**DC 求解不收敛。**
先用 `examples/single_stage.json`（单管，必然收敛）。对复杂电路，在 JSON
中加上 `dc_guesses`——一个近似节点电压的字典。锁定 AFE 的 JSON 里就包含了这些猜测值。

**Transient 出现 `nfail > 0`。**
部分 Newton 步失败。尝试：(a) 增加时间点 `np.linspace(0, T, more_steps)`；
(b) 收紧 `newton_vtol`（默认 `1e-8`）；(c) 启用 `fallback_least_squares=True`。
对开关电路，确保 `max_step` 小于最快的边沿时间。如果使用 `integration_method="gear2"`
并请求 `max_step` / `max_retry_subdivisions`，当前 hot path 会留在 Numba gear2 grid；
只有 Numba 拒绝 robust step 时才会落到 Python 兜底。检查 `numba_grid_solver`、
`gear2_python_retry_solver` 和 `transient_profile.failed_intervals`。

**PSS 不收敛（`converged=False`）。**
增加 `tstab_periods`（shooting 前的额外稳定周期），或降低 `max_shooting_iters`。
检查 `pss['shooting_history']`，看残差是否在下降。如果停滞，可能是轨道本身非周期——
检查所有输入波形是否周期相同、周期一致。

**PNoise 太慢。**
通用 HB 路径可减少 `max_sideband`（奇次谐波主导折叠），或降低 `n_period_samples`
（用时域分辨率换速度）。Chopper 默认 TD-adjoint 路径中，`max_sideband` 不再控制
转换截断；`n_period_samples` 小于 640 时会自动抬到 768 保证收敛。扫不同输出带宽或
重复频点时复用同一个 `pss_result`：PNoise 会按适用路径缓存 LPTV 线性化、HB block 和相同频点的
adjoint 解。安装
Numba 时，大规模 HB block 组装、噪声折叠和 gm/gds 线性化也会走编译内核。
Chopper PAC 的 gate1 转换线性化在全 PMOS gate1 拓扑下也会走编译内核；混合或暂不支持的
拓扑会回退到同一 stamp 的 Python 装配。
旧 HB PNoise 路径在 UI chopper 上约 61 点 0.55s、121 点 0.93s；现在 chopper
验证默认优先使用精度更稳的 TD 路径。

**PSS / 周期 transient 太慢。**
首先确保使用默认的 `analytic_jacobian=True` — 它将 shooting Jacobian 构建
从 `n_state` 次有限差分瞬态降为一次轨道遍历。Chopper PSS 现在默认使用
`fallback_least_squares=False`，这样完整周期会留在 Numba grid solver 内，
失败 interval 会被记录，但不会把整个周期退回 Python 重跑。只有在排查困难
收敛问题时才建议手动打开 `fallback_least_squares=True`。PMOS chopper wrapper
通常 1 个 stabilization 周期就够；继续增加主要是吞吐量取舍。

**PAC 太慢。**
普通运行不要设置 `compute_condition`。PAC condition 诊断只会在 `profile=True`、
`debug=True` 或显式 `compute_condition=True` 时计算，因为这个诊断每个频点都要
对 HB 矩阵做一次 SVD，不影响 gain/BW/noise 结果。
rail-driven chopper wrapper 已默认进入加速的 time-domain Floquet PAC。通用 JSON
电路可以用 `"time_domain": true` 显式启用。如果强制走 HB（`time_domain=false`），
PAC 每频点求解仍是主要耗时：61 点约 24–25s，121 点约 47–48s。后续 HB-only
优化应放在 factorization 复用或批量线性求解上。

**`models`/硅器件报工具链错误。**
硅 PDK 需要一套外部工具链（不是 pip 依赖）：SKY130 需 OpenVAF + ngspice + SKY130 PDK,
FreePDK45 需 ngspice + FreePDK45 卡。安装好之后把 `PDK_ROOT`/`OPENVAF_ROOT`/`NGSPICE_BIN`
指过去（见 `docs/futureplan.md` §9）。没装的话，任何用到 `sky130.*` / `freepdk45.*` 模型类型
的电路会在第一次用到时报出清晰错误；其余 PDK（默认的 AT4000TG OTFT）不受影响。依赖该工具链的
测试（`tests/test_sky130*.py`、`tests/test_freepdk45.py`、`tests/test_osdi_host.py`）在工具链
缺失时会干净地 skip。

---

## 延伸阅读

| 文档                                       | 作用                     |
| ---------------------------------------- | ---------------------- |
| [核心求解器概览](core_overview_zh.md)           | 理解每个求解器的原理、导入依赖关系和标定数据 |
| [JSON 电路描述格式](json_circuit_format_zh.md) | JSON 的字段级参考            |
| [CLI 参考手册](cli_reference.md)             | 每个子命令和参数，含 `dataset`/`surrogate`/`optimize` 和 `models`/硅工艺角选项 |
| [SKY130 FD-OTA 设计案例](sky130_fd_ota_design.md) | 130nm 全差分 OTA 全流程：架构 → 代理 → 优化 → PVT |
| [FreePDK45 FD-OTA 设计案例](freepdk45_fd_ota_design.md) | 45nm/1.0V 全差分 OTA 全流程，含整机对 ngspice `.ac` 的交叉核对 |
| [后续开发计划](futureplan.md)                  | 了解已完成、待做事项和执行路线图       |
| `tests/` 目录                              | 每个 API 调用的可运行示例，带预期输出  |
| `benchmarks/` 目录                         | 性能基线及 Numba 加速对比       |

---

## 参与贡献

欢迎提 Issue 和 PR。
