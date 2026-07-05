# CLI 参考手册

项目提供三层 CLI 入口：主分析调度、校准闭环、Web 演示。所有命令从仓库根目录执行。

## 快速索引

### 主 CLI

| 命令 | 用途 |
|------|------|
| `python -m core <circuit.json>` | 按 JSON 配置跑分析（AC/Noise/Tran/PSS/PAC/PNoise） |
| `python -m core run <circuit.json> -a ac,noise` | 显式指定分析子集 |
| `python -m core explore <circuit.json> -n 300` | 设计空间探索（LHS/随机采样 → Pareto） |
| `python -m core corners <circuit.json>` | 工艺角扫描（typ/slow/fast） |
| `python -m core mc <circuit.json> -n 300` | 逐器件 mismatch Monte Carlo |
| `python -m core chopper <circuit.json> --level pss` | Chopper 分析（7 个层级） |
| `python -m core plot [all\|transient\|bode\|afe\|chopper\|ac\|pac]` | 出图：瞬态波形 + AC/PAC Bode（PNG） |
| `python -m core dataset <circuit.json> -n 500 --out ds/run1` | 生成 surrogate 训练集（provenance + 失败样本保留） |
| `python -m core.surrogate train <ds>.npz --test <ds>.npz --out m.pkl` | 训练指标 surrogate（GBT，可选 sklearn 依赖） |
| `python -m core.optimize <circuit.json> m.pkl -n 100000 --top-k 20` | 筛选 → Pareto → solver 校验闭环 |
| `python -m core.surrogate_torch optimize <circuit.json> m.pt --verify` | 可微 surrogate 梯度设计优化（torch/MPS） |
| `python -m core dataset examples/sky130_5t_ota.json -n 400 --out ds/ota` | 硅（SKY130）设计闭环：`models` 块绑 PDK，跨工艺角 `--corner ss`（见 §1.8） |
| `python -m core.calibration --all` | Cadence 校准回归检查 |
| `python demo/server.py` | Web 前端（Flask，端口 5100） |

### 性能基准

| 命令 | 用途 |
|------|------|
| `python -m benchmarks.bench_afe` | AFE 固定负载（AC / Noise / Transient） |
| `python -m benchmarks.bench_model` | PMOS_TFT 单器件微基准（3 偏置区 × 6 操作） |
| `python -m benchmarks.bench_periodic` | 周期求解器性能（PSS/PAC/PNoise） |
| `python -m benchmarks.bench_sweep` | 批量扫描吞吐量（AC-only / AC+Noise） |
| `python -m benchmarks.bench_chopper` | Chopper 分析性能（5 种负载层级） |

### 示例 & 工具脚本

| 命令 | 用途 |
|------|------|
| `python examples/afe_testbench.py` | 干电极 AFE 全链路（AC + Noise + Transient） |
| `python -m examples.plot_transient [--afe\|--chopper]` | 瞬态波形出图（AFE 正弦放大 / chopper 斩波轨道） |
| `python -m examples.plot_bode [--ac\|--pac]` | AC/PAC 增益相位 Bode 出图 |
| `python examples/mc_mismatch.py [n] [seed]` | Mismatch MC + 直方图 |
| `python examples/find_max_gain.py` | PMOS 反相器最大增益扫描 |
| `python examples/sweep_vin_vout.py` | PMOS 反相器 DC 传输曲线 |
| `python examples/sc_lpf.py` | 开关电容 LPF 瞬态仿真 |
| `python tools/calibrate_switch.py gen/parse` | Chopper 开关 Cadence vs 本地校准 |

---

## 1. 主 CLI：`python -m core`

入口文件 `core/__main__.py`，支持 7 个子命令。默认兼容旧用法（无子命令时自动路由到 `run`）。

### 1.1 `run` — 分析调度（默认子命令）

按电路 JSON 的 `"analyses"` 块执行指定分析，或通过 `-a` 覆盖。

```bash
# 跑 JSON 中配置的全部分析
python -m core examples/periodic_rc.json

# 只跑 AC 和 Noise
python -m core examples/afe_explore.json -a ac,noise

# 显式子命令形式（等价）
python -m core run examples/afe_explore.json -a ac,noise
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `circuit` | path | (必需) | 电路 JSON 文件路径 |
| `-a`, `--analysis` | str | 全部配置项 | 逗号分隔，可选：`ac,noise,transient,pss,pac,pnoise` |
| `--corner` | str | — | 工艺角覆盖：OTFT `typical/slow/fast`，或硅 `tt/ss/ff/sf/fs`（SKY130）/ `nom/ss/ff`（FreePDK45）。硅角路由进器件卡，OTFT 角作各分析默认；不给则用 JSON 里配置的角 |
| `--noise-band LO HI` | float×2 | `0.05 100.0` | IRN 积分频带 (Hz) |
| `-o`, `--output` | path | — | 结果写出为 JSON |
| `--no-numba` | flag | — | 禁用 Numba 加速 |
| `--quiet` | flag | — | 不打印进度 |

> `run`/`explore` 都会自动绑定 `models` 块的非默认 PDK 并对多稳态 DC 播种（用第一个字典型
> `dc_guesses`）——硅配置（含 FreePDK45）经 `python -m core run` / `python -m core explore` 直接可用。

**输出示例：**

```
Running ac,noise analyses for examples/afe_explore.json
  AC:    gain=22.90 dB  BW=562.3 Hz
  Noise: IRN=38.31 µVrms  out=112.50 µVrms
```

---

### 1.2 `explore` — 设计空间探索

对电路 JSON 的 `"explore"` 块做参数采样 → AC-first 评估 → 约束过滤 → Pareto 选择。

```bash
python -m core explore examples/afe_explore.json -n 300 --seed 42
python -m core explore examples/afe_explore.json -n 200 --method random
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | path | (必需) | 含 `"explore"` 块的电路 JSON |
| `-n`, `--n` | int | `200` | 候选数量 |
| `--seed` | int | `0` | 随机种子 |
| `--method` | choices | `lhs` | 采样方法：`lhs`（拉丁超立方）或 `random` |
| `--corner` | str | — | 工艺角：OTFT `typical/slow/fast`，或硅 `tt/ss/ff/sf/fs`（SKY130）/ `nom/ss/ff`（FreePDK45） |
| `-o`, `--out`, `--output` | path | — | 输出路径前缀（生成 `<prefix>.csv` + `<prefix>.jsonl`） |
| `--no-numba` | flag | — | 禁用 Numba（仅 `python -m core explore` 子命令有此参数；独立入口 `python -m core.explore` 没有） |
| `--quiet` | flag | — | 不打印逐候选进度 |

> `explore` 的 CLI 参数由 `core/explore.py` 的 `add_cli_args(parser)` / `run_cli(args)` 定义，
> `python -m core explore` 子命令和独立入口 `python -m core.explore` 共用同一份定义，因此参数
> 表（除 `--no-numba`，只在子命令级存在）对两者一致，不会漂移。

**旧版兼容：**

```bash
# 带 --explore flag 的旧用法仍支持
python -m core examples/afe_explore.json --explore -n 300
```

---

### 1.3 `corners` — 工艺角扫描

对 typ/slow/fast 三个全局 corner 各跑一次 AC+Noise。

```bash
python -m core corners examples/afe_explore.json
python -m core corners examples/afe_explore.json --freqs-num 61 --freqs-stop 5000
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `circuit` | path | (必需) | 电路 JSON |
| `--freqs-start` | float | `0.01` | 起始频率 (Hz) |
| `--freqs-stop` | float | `10000` | 终止频率 (Hz) |
| `--freqs-num` | int | `121` | 对数间隔频点数 |
| `--noise-band LO HI` | float×2 | `0.05 100.0` | IRN 积分频带 (Hz) |
| `-o`, `--output` | path | — | 写出 CSV |
| `--no-numba` | flag | — | 禁用 Numba |
| `--quiet` | flag | — | 不打印逐 corner 输出 |

**输出示例：**

```
Corner sweep for examples/afe_explore.json
  freqs: 0.01–10000 Hz (121 pts)
  band:  0.05–100.0 Hz
  typical:  gain=22.90 dB  BW=562 Hz  IRN=38.31 µVrms
     slow:  gain=18.54 dB  BW=269 Hz  IRN=34.45 µVrms
     fast:  gain=26.47 dB  BW=980 Hz  IRN=41.25 µVrms
wrote results/corners.csv
```

---

### 1.4 `mc` — Mismatch Monte Carlo

逐器件参数失配 MC 采样 + 确定性 latch 筛查。

```bash
python -m core mc examples/afe_explore.json -n 300 --seed 1
python -m core mc examples/afe_explore.json --corner slow --quiet
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `circuit` | path | (必需) | 电路 JSON |
| `-n`, `--n` | int | `200` | MC 样本数 |
| `--seed` | int | `0` | 随机种子 |
| `--corner` | choices | `typical` | 基底 corner：`typical` / `slow` / `fast` |
| `--freqs-start` | float | `0.01` | 起始频率 (Hz) |
| `--freqs-stop` | float | `10000` | 终止频率 (Hz) |
| `--freqs-num` | int | `121` | 对数间隔频点数 |
| `--noise-band LO HI` | float×2 | `0.05 100.0` | IRN 积分频带 (Hz) |
| `-o`, `--output` | path | — | 写出 JSON |
| `--no-numba` | flag | — | 禁用 Numba |
| `--quiet` | flag | — | 不打印进度 |

**输出示例：**

```
Mismatch MC for examples/afe_explore.json
  n=300  seed=1  corner=typical
  freqs: 0.01–10000 Hz (121 pts)
  band:  0.05–100.0 Hz
  latch_rate: 2.3%
  IRN:        38.45 ± 1.82 µVrms  (P5=35.62  P95=41.71)
  gain:       22.88 ± 0.15 dB
  BW:         558.0 ± 12.0 Hz
wrote results/mc.json
```

---

### 1.5 `chopper` — Chopper 分析

8-PMOS AFE chopper 的 7 层分析，从理想方波 LPTV 到第一性原理 PSS/PAC/PNoise。

```bash
# 理想方波 chopper
python -m core chopper examples/afe_explore.json --level ideal --f-chop 225

# 硬开关瞬态
python -m core chopper examples/afe_explore.json --level transient --f-chop 225 --n-periods 8

# PSS+PAC+PNoise（第一性原理）
python -m core chopper examples/afe_explore.json --level pnoise --f-chop 225
```

**`--level` 层级（由简到精）：**

| 层级 | 说明 |
|------|------|
| `ideal` | 方波 LPTV，理想开关 |
| `pmos` | PMOS 静态相位，包含开关 Ron 调制 |
| `lptv` | PMOS 有限边沿 + 谐波边带折叠 |
| `pss` | Shooting PSS 周期稳态 |
| `pac` | PSS + PAC 边带转换增益 |
| `pnoise` | PSS + PAC + PNoise；chopper wrapper 默认使用 TD-adjoint PNoise，HB 可作为对照 |
| `transient` | 硬开关瞬态仿真 |

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `circuit` | path | (必需) | 电路 JSON |
| `--level` | choices | `ideal` | 分析层级 |
| `--f-chop` | float | `225.0` | Chopper 频率 (Hz) |
| `--switch-w` | float | `5000.0` | 开关宽度 (µm) |
| `--switch-l` | float | `30.0` | 开关长度 (µm) |
| `--edge-time` | float | `2e-5` | 时钟上升/下降时间 (s) |
| `--max-harmonic` | int | `31` | ideal/LPTV 最大谐波数 |
| `--max-sideband` | int | `10` | PNoise 边带数；TD-adjoint chopper PNoise 不再受 HB adjoint 截断限制 |
| `--tstab-periods` | int | `2` | PSS 稳定周期数 |
| `--n-points` | int | `121` | 每周期时间点数 |
| `--n-periods` | float | `8.0` | 瞬态仿真周期数 |
| `--freqs-start` | float | `0.01` | 起始频率 (Hz) |
| `--freqs-stop` | float | `10000` | 终止频率 (Hz) |
| `--freqs-num` | int | `121` | 对数间隔频点数 |
| `--noise-band LO HI` | float×2 | `0.05 100.0` | IRN 积分频带 (Hz) |
| `-o`, `--output` | path | — | 写出 JSON |
| `--no-numba` | flag | — | 禁用 Numba |
| `--quiet` | flag | — | 不打印进度 |

### 1.6 `plot` — 信号出图

把瞬态波形和 AC/PAC Bode 渲染成 PNG（默认写到 `results/`）。绘制的是已标定的 AFE/chopper
参考设计（`calibration/` 下的 case），不需要传 circuit JSON。需要 `matplotlib`（可选依赖）。

```bash
python -m core plot                      # 全部 4 张：AFE 瞬态 + chopper 轨道 + AC + PAC Bode
python -m core plot transient            # 只瞬态两张（afe + chopper）
python -m core plot bode                 # 只 Bode 两张（ac + pac）
python -m core plot afe --f0 20 --amp 1e-3        # 单张：AFE 正弦放大
python -m core plot chopper --f-chop 200 --input-diff 2e-3
python -m core plot pac --npts 121 --out-dir /tmp/plots
```

| 位置参数 `kind` | 产出 |
|------|------|
| `all`（默认） | `transient_afe.png` `transient_chopper.png` `bode_ac.png` `bode_pac.png` |
| `transient` / `bode` | 瞬态两张 / Bode 两张 |
| `afe` / `chopper` / `ac` / `pac` | 对应单张 |

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--f0` | float | `10` | AFE 瞬态正弦频率 (Hz) |
| `--amp` | float | `5e-4` | AFE 瞬态差分半幅 (V) |
| `--f-chop` | float | `225` | chopper/pac 斩波频率 (Hz) |
| `--input-diff` | float | `1e-3` | chopper 瞬态 DC 差分输入 (V) |
| `--npts` | int | 各图默认 | Bode 频点数 |
| `--out-dir` | path | `results` | 输出目录 |
| `--no-numba` | flag | — | 禁用 Numba |
| `--quiet` | flag | — | 不打印汇总行 |

> 独立脚本 `python -m examples.plot_transient` / `python -m examples.plot_bode` 提供同样的绘图，
> 参数更细（`--periods` `--case` `--fmin` `--fmax` `--ac-case` `--pac-case` 等）。

---

### 1.7 `dataset` — Surrogate 训练集生成

对电路 JSON 的 `"explore"` 块采样，把每个候选点跑过已标定的求解器，产出带 provenance 的
`(设计参数 → 指标)` 标注数据集。与 `explore` 的区别：**不做约束/Pareto 过滤**（每个样本都保留，
DC 失败样本作为分类/边界标签，不丢弃），**总是评估噪声**（每个收敛点都带完整标签），并写出 manifest
记录 schema 版本、solver commit、拓扑 hash、corner、参数范围——供下游 surrogate 判断泛化边界。
是 ML surrogate 路线（架构见 `docs/core_overview.md`，未来方向见 `docs/futureplan.md`）的数据生成前置。

```bash
python -m core dataset examples/single_stage.json -n 500 --out ds/run1
python -m core dataset examples/afe_explore.json -n 1000 --corner slow --seed 7 --out ds/slow
python -m core dataset examples/single_stage.json -n 200 --no-npz --quiet
```

产出三个文件（`--out <prefix>`）：

| 文件 | 内容 |
|------|------|
| `<prefix>.jsonl` | 每行一个样本：`{idx, design, metrics, status}`（可读、可调试；NaN→null） |
| `<prefix>.manifest.json` | provenance：schema 版本、solver commit(+dirty)、拓扑 hash、PDK、corner、采样 seed/method、变量范围、label_groups、counts |
| `<prefix>.npz` | 稠密 `X`(n×变量) / `Y`(n×标签，缺失为 NaN) + `dc_converged`/`metrics_finite` 掩码，直接喂回归器 |
| `<prefix>.parquet` | 扁平表(`--parquet`,需 pyarrow)：`design_*` 输入 + 裸标签列 + status 布尔 |

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `config` | path | (必需) | 含 `"explore"` 块的电路 JSON |
| `-n` | int | `200` | 样本数 |
| `--seed` | int | `0` | RNG 种子（同 config+seed+commit ⇒ 同数据集） |
| `--method` | lhs/random | `lhs` | 采样方法 |
| `--corner` | str | `typical` | 工艺角：OTFT `typical/slow/fast`，或硅 `tt/ss/ff/sf/fs`（SKY130）/ `nom/ss/ff`（FreePDK45） |
| `--labels` | str | `ac_noise` | 标签组(逗号分隔)：`ac_noise` / `transient` / `pss` / `pac` / `pnoise`(周期组需 `periodic` 块；`pac`/`pnoise` 另需对应 `analyses` 块) |
| `--freqs-start` | float | — | 给了 `--freqs-stop` 时，AC 网格起始十进位（log10 Hz） |
| `--freqs-stop` | float | — | 覆盖 AC 网格终止十进位（log10 Hz），如 4 = 10kHz（避免 `bw_Hz` 卡在上限） |
| `--freqs-num` | int | — | 给了 `--freqs-stop` 时的 AC 网格点数 |
| `--out` | path | — | 输出前缀（不给则只在内存计算，不落盘） |
| `--no-npz` | flag | — | 跳过 `.npz` 稠密输出 |
| `--parquet` | flag | — | 额外写 `.parquet`（需可选依赖 pyarrow） |
| `--no-numba` | flag | — | 禁用 Numba（仅 `python -m core dataset` 子命令有此参数；独立入口 `python -m core.dataset` 没有） |
| `--quiet` | flag | — | 不打印进度 |

> `dataset` 的 `--corner` 帮助文案与 `run`/`explore` 一致：OTFT `typical/slow/fast`，
> 或硅 `tt/ss/ff/sf/fs`（SKY130）/ `nom/ss/ff`（FreePDK45）。
>
> `dataset` 的 CLI 参数由 `core/dataset.py` 的 `add_cli_args(parser)` / `run_cli(args)` 定义，
> `python -m core dataset` 子命令和独立入口 `python -m core.dataset` 共用同一份定义，因此参数表
> （除 `--no-numba`，只在子命令级存在）对两者一致，不会漂移。

**标签组**（`--labels`，schema 1.2）：

- `ac_noise`（默认）：`gain_dB` `gain_peak_dB` `bw_Hz` `irn_uV` `power_uW` `area`
- `transient`：`out_pp` `out_mean` `out_rms` `slew_rate` `final_value` — **激励无关**的波形特征，
  复用配置里已验证的 `periodic` transient（无 `periodic` 块则报错；不假设阶跃语义，故不给 settling/overshoot）
- `pss`：`pss_converged` `pss_residual` `pss_iters` `pss_out_pp` `pss_out_mean` — 周期稳态质量 + 轨道输出特征
  （需 `periodic` 块）。`pss_converged`(1/0) 是可信标志；diverged 样本保留标签由它区分。
  **相位裕度**（AC 环路指标）和 **settling**（阶跃响应）不属于 PSS，不在此组。
- `pac`：`pac_gain` `pac_gain_dB` `pac_bw_Hz` — LPTV 小信号传递：最低分析频点的基带转换增益（斩波的
  解调增益）+ PAC 网格内的 −3dB 角频率（带内未跌 3dB 则为 null）。**需配置带 `analyses.pac` 块**。
- `pnoise`：`pnoise_out_uV` `pnoise_irn_uV` — 折叠周期噪声在 `analyses.pnoise.band` 上的积分输出噪声
  与（经 PAC 0 阶边带增益折算的）等效输入噪声——斩波 AFE 的核心指标。**需配置带 `analyses.pnoise` 块**。

`pss`/`pac`/`pnoise` 三组每候选走一次 `run_analysis_suite`：配置 `analyses` 块里**已验证的求解设置**
（gear2/tstab/容差、`time_domain`、drive、band）原样生效，PSS 轨道只算一次共享，PNoise 复用 PAC 增益；
数据集级 corner 覆盖 `analyses` 块内的 corner，保证一行标签同属一个工艺点。硬开关电路（斩波/SC）的
`analyses.pac`/`analyses.pnoise` 记得 `time_domain: true`（HB 边带截断在方波电导上收敛很慢）。
`transient` 组保持直连 `periodic` 上下文。周期组比纯 `ac_noise` 慢；按需
`--labels ac_noise,transient,pss` 或（硅斩波）`--labels pss,pac,pnoise`，例如：

```bash
python -m core dataset examples/sky130_chopper.json --labels pss,pac,pnoise -n 200 --out chopper_ds
```

**设计轴（`explore.variables` 目标语法）**：除 `DEV.W/.L/.NF`（器件尺寸）和裸 bias key 外，dataset
还支持三类扩展轴（manifest 每个变量记 `kind`）：

- `<CapName>.C` — 具名电容（`capacitors` 里带 `name` 的项）容值，扫 load（`structural`，逐候选重建电路）
- `<ResName>.R` — 具名电阻（`resistors` 里带 `name` 的项）阻值，扫无源负载（`structural`，逐候选重建电路）
- `periodic.frequency` — 周期激励的时钟频率，扫 clock（`structural`，只影响 periodic 标签）
- `pvt0` / `pbeta0` — 连续全局工艺偏移（`corner`），逐候选路由进 `evaluate(corner=...)`。**采样它 = 全局
  process MC**，让一个 surrogate 覆盖连续 PVT 空间并内插到任意工艺点（manifest `corner="sampled"`）。
  离散 corner 签核用 `--corner`；连续统计/良率用 `pvt0/pbeta0` 轴。

`structural` 轴（cap/resistor/clock）在 `dataset` **和** `optimize` 校验阶段都逐候选重建电路
（共享 `dataset.candidate_circuit()`），扫到的无源值在最终 solver 校验里也生效。

### 1.8 硅 PDK 器件绑定 `models` 块 + 硅设计闭环

配置里的 **`models` 块**把某个器件绑到非默认 PDK 模型（如硅 SKY130），其余器件仍用默认 OTFT
（纯增量，OTFT 数值 byte-identical）。`type` 是模型注册键，其余键透传给器件构造：

```json
"models": {
  "M1": {"type": "sky130.nmos", "extract_w": 24.0},
  "M3": {"type": "sky130.pmos", "vb": 1.8, "extract_w": 12.0}
}
```

- `extract_w`（µm）在参考 W 处解析一次卡片、实际 W 交给紧凑模型缩放 → ~2 ms/eval、平滑，扫 W 不触发逐候选 ngspice。
  FreePDK45 上是"参考 W 表征一次网格 + 线性缩放实际 W"（<0.7% vs 逐 W 真卡）。
- **硅工艺角**用 `--corner`：SKY130 用 `tt|ss|ff|sf|fs`,FreePDK45 用 `nom|ss|ff`;路由进硅器件的
  `corner=` 卡片（与 OTFT 的连续 PVT `pvt0/pbeta0` 分开）。
- **`run`/`dataset` 会自动播种 DC 并透传 `models`**：`run_analysis_suite` 的 AC/noise 分支用配置里
  第一个 `dc_guesses`（字典形式）作为 `x0_guess`、并绑定 `model_types`/`device_kwargs`——多稳态硅
  电路（如带 CMFB 的全差分 OTA）因此能落在物理分支上；OTFT 配置（无 `models`、回调式种子）行为不变。

**FreePDK45**（`"freepdk45.nmos"` / `.pmos"`）绑定相同,但求值器是 ngspice-C（非 OSDI VA）。`temperature`
（开尔文）可做 PVT 温度轴（按温度重表征卡）。

完整闭环（`examples/sky130_5t_ota.json`，SKY130 互补 5T OTA）：

```bash
python -m core dataset examples/sky130_5t_ota.json -n 400 --out ds/ota          # 造硅数据集
python -m core.surrogate train ds/ota.npz --filter gain_dB:0:60 --out ota.pkl   # 训练（--filter 只学工作区，剔除甩轨角）
python -m core.optimize examples/sky130_5t_ota.json ota.pkl -n 50000 --top-k 10 # 筛选→Pareto→solver 校验
python -m core.optimize examples/sky130_5t_ota.json ota.pkl -n 50000 --corner ss # 跨工艺角复验（慢角）
```

需外置盘的 OpenVAF/ngspice/SKY130 工具链（见 `silicon-pdk-openvaf` 记忆）。工作区 surrogate 精度
（gain/power/bw/irn/area 中位误差 ≈1%）；筛选比 solver 快约 6000×，solver 校验保证入围设计真实可行。
两个全差分 OTA 完整设计案例见 [SKY130 FD-OTA](sky130_fd_ota_design.md)、
[FreePDK45 FD-OTA](freepdk45_fd_ota_design.md)（FreePDK45 案例含整机对 ngspice `.ac` 的交叉核对）。

---

## 2. 校准 CLI：`python -m core.calibration`

Cadence/Spectre 校准闭环检查。入口文件 `core/calibration.py`。

```bash
# 全部 5 个 case
python -m core.calibration --all

# 单个 case
python -m core.calibration calibration/amp_design3_typical/

# 只跑指定分析
python -m core.calibration calibration/chopper_design3_typical/ --analyses pac,pnoise

# CI 友好（JSON 输出 + 非零退出码 = 失败）
python -m core.calibration --all --json

# 放宽 3 倍容差（调试用）
python -m core.calibration --all --relaxed
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cases` | path(s) | `calibration` | case 目录或包含多个 case 的父目录 |
| `--all` | flag | — | 扫描 `calibration/` 下所有含 `metadata.json` 的子目录 |
| `--analyses` | str | 全部配置 | 逗号分隔的分析子集，如 `ac,noise` |
| `--json` | flag | — | JSON 格式输出（适合 CI 解析） |
| `--relaxed` | flag | — | 容差放大 3 倍 |

**输出示例：**

```
[PASS] amp_design3_typical  (Spectre 24.1.0.078 @ 10:32:02 PM, Sun Jun 21, 2026)
  ok dc:
      [ok] VOP                local=29.08 ref=29.08  Δ=-6.1e-05 (tol 1e-03)
      [ok] VON                local=29.08 ref=29.08  Δ=-6.099e-05 (tol 1e-03)
  ok ac:
      [ok] gain_dc_dB         local=22.9 ref=22.89  Δ=+0.00% (tol 1%)
      [ok] bw_Hz              local=562.3 ref=551.4  Δ=+1.99% (tol 5%)
  ok noise:
      [ok] irn_uVrms          local=38.31 ref=38.31  Δ=+0.00% (tol 3%)
```

**退出码：** 全部 PASS 返回 0，任一 case 失败返回 1。

---

## 3. Web 演示：`python demo/server.py`

Flask Web 前端 + REST API，用于交互式 AFE 调谐。

```bash
python demo/server.py
# AFE Tuner Server starting at http://localhost:5100
```

浏览器打开 `http://localhost:5100`，提供：

- 预设加载 / 手动调参
- AC 增益–带宽实时更新
- 噪声瀑布图
- 瞬态波形
- Pareto 探索结果可视化

---

## 4. 性能基准：`python -m benchmarks.*`

基准脚本位于 `benchmarks/`，均支持 `--warm-runs`（预热轮数）、`--json`（JSON 输出）和 `CIRCUIT_USE_NUMBA=0` 环境变量切纯 Python 对比。

### 4.1 `bench_afe` — AFE 固定负载

```bash
python -m benchmarks.bench_afe
python -m benchmarks.bench_afe --warm-runs 5 --skip-tran
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--warm-runs` | int | `3` | 预热轮数（cold = 首轮含 JIT） |
| `--json` | flag | — | JSON 格式输出 |
| `--skip-noise` | flag | — | 跳过噪声分析 |
| `--skip-tran` | flag | — | 跳过瞬态分析 |

**负载：**

- `ac121` — 121 频点 AC 求解
- `noise121` — 121 频点噪声分析
- `tran200` — 200 时间点瞬态阶跃响应

---

### 4.2 `bench_model` — PMOS_TFT 微基准

```bash
python -m benchmarks.bench_model --warm-runs 3
CIRCUIT_USE_NUMBA=0 python -m benchmarks.bench_model --warm-runs 3 --json
```

**参数：** 同 `bench_afe`（仅 `--warm-runs`、`--json`）。

**负载（3 偏置区 × 6 操作）：**

| 偏置区 | (Vs, Vd, Vg) |
|--------|---------------|
| saturation | (40, 0, 20) |
| subthreshold | (40, 0, 38) |
| linear | (40, 35, 20) |

每区测试：OP 求解、`get_Idc`、电容、Idc+电容组合、噪声 PSD、`get_os`。

---

### 4.3 `bench_periodic` — 周期求解器性能

```bash
python -m benchmarks.bench_periodic --warm-runs 3
CIRCUIT_USE_NUMBA=0 python -m benchmarks.bench_periodic --warm-runs 3 --json
```

**参数：** 同 `bench_afe`（仅 `--warm-runs`、`--json`）。

**负载：**

- `pss` — Shooting PSS 周期稳态求解
- `pac` — 周期 AC（PSS 基础上的频域 HB + 时域 Floquet）
- `pnoise` — 周期噪声（HB adjoint + TD adjoint）

---

### 4.4 `bench_sweep` — 批量扫描吞吐量

```bash
python -m benchmarks.bench_sweep --n-candidates 100 --warm-runs 3
python -m benchmarks.bench_sweep --n-candidates 200 --seed 42 --json
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--warm-runs` | int | `3` | 预热轮数 |
| `--n-candidates` | int | `50` | 候选数量 |
| `--seed` | int | `0` | RNG 种子 |
| `--json` | flag | — | JSON 格式输出 |

**负载：**

- `ac_only` — N 候选纯 AC 求解（模拟快速预筛）
- `ac_noise` — N 候选 AC+Noise（模拟完整评估）

报告每候选耗时 (ms) 和每秒吞吐量 (cand/s)。

---

### 4.5 `bench_chopper` — Chopper 分析性能

```bash
python -m benchmarks.bench_chopper --warm-runs 3
python -m benchmarks.bench_chopper --skip-tran --json
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--warm-runs` | int | `3` | 预热轮数 |
| `--json` | flag | — | JSON 格式输出 |
| `--skip-tran` | flag | — | 跳过瞬态（最慢） |

**负载（5 层级，f_chop=225 Hz，锁定设计尺寸）：**

| 层级 | 说明 |
|------|------|
| `harmonics` | 有限边沿谐波系数（纯数学） |
| `ideal` | 理想 LPTV（频域边带折叠，无开关） |
| `pmos_static` | PMOS 静态相位（8 开关 + AFE） |
| `pmos_lptv` | 准静态 PMOS 边带折叠 |
| `pmos_tran` | 硬开关瞬态（最重） |

---

## 5. 示例脚本：`python examples/*.py`

### 5.1 `afe_testbench.py` — 干电极 AFE 全链路

```bash
python examples/afe_testbench.py
```

无参数。跑干电极 AFE（前端 AC 耦合 + AFE 核心）的 AC（带通）、Noise（0.05–100 Hz IRN，含电阻贡献）和 Transient（10 Hz 差分正弦波增益验证）。

---

### 5.2 `mc_mismatch.py` — Mismatch MC + 直方图

```bash
python examples/mc_mismatch.py
python examples/mc_mismatch.py 500 42       # n=500, seed=42
```

**位置参数：** `[n_samples]`（默认 300）、`[seed]`（默认 0）。

三个工艺角各跑一次 MC，打印增益/带宽/IRN 统计和 latch 率，保存直方图到 `results/mc_mismatch.png`。

---

### 5.3 `find_max_gain.py` — 最大增益扫描

```bash
python examples/find_max_gain.py
```

无参数。对 PMOS 反相器放大器扫描多个 W/L 组合的 VIN，找到每个组合的最大小信号增益。绘制传输曲线，保存到 `results/max_gain_analysis.png`。

---

### 5.4 `sweep_vin_vout.py` — DC 传输曲线

```bash
python examples/sweep_vin_vout.py
```

无参数。PMOS 反相器放大器的完整 VIN→VOUT DC 扫描，跨多个 W/L 组合计算数值小信号增益。保存到 `results/vin_vout_sweep.png`。

---

### 5.5 `sc_lpf.py` — 开关电容 LPF 瞬态仿真

```bash
python examples/sc_lpf.py
```

无参数。两相开关电容低通滤波器全瞬态仿真，使用 PMOS 开关 + 理想 vsource 时钟驱动。

---

## 6. 工具脚本：`python tools/*.py`

### 6.1 `calibrate_switch.py` — Chopper 开关校准

```bash
# 生成 Spectre 网表 + 运行脚本到 /tmp/sw_cal
python tools/calibrate_switch.py gen

# 解析 PSF 结果并与本地模型对比
python tools/calibrate_switch.py parse
```

**子命令：**

| 子命令 | 说明 |
|--------|------|
| `gen` | 写入 Spectre 网表和运行脚本到 `/tmp/sw_cal` |
| `parse` | 从 `/tmp/sw_cal_out` 解析 PSF，对比 Ron/gm/交叠电容 |

针对 chopper 开关 PMOS_TFT 5000/30 导通状态的 Cadence vs 本地校准。

---

## 7. 独立模块入口（开发/调试用）

以下入口仅供开发调试，参数硬编码，非通用 CLI：

```bash
python -m core.ac_solver          # 硬编码尺寸跑一次 AC
python -m core.noise_solver       # 硬编码尺寸跑一次 Noise
python -m core.transient_solver   # 硬编码尺寸跑一次 Transient
python -m core.pmos_tft_model     # 打印单管 Id / Cgss / Cgdd / 噪声 PSD
```

---

## 8. 通用约定

### 频率参数

多个子命令共享 `--freqs-*` 参数族：

```
--freqs-start 0.01 --freqs-stop 10000 --freqs-num 121
```

生成对数均匀间隔的频点数组 `np.logspace(log10(start), log10(stop), num)`。默认 0.01–10kHz 121 点，对应 Cadence `dec=20` 的 AC/Noise 标准网格。

### 噪声积分频带

`--noise-band LO HI` 控制 IRN（等价输入噪声）和输出噪声的积分区间。默认 0.05–100 Hz，匹配 ECG AFE 应用频带。子命令内部调用 `band_rms(freqs, psd, lo, hi)` 做梯形积分。

### 输出

- `-o result.json` — 分析结果写出为 JSON
- `-o results/run1` — explore 写出 `results/run1.csv` + `results/run1.jsonl`，corners 写出 `results/run1`（CSV）
- 目录不存在时自动创建

### Numba

默认启用 Numba JIT 加速。`--no-numba` 强制走纯 Python 路径（调试 / 基准对比用）。环境变量 `CIRCUIT_USE_NUMBA=0` 等效。

> ⚠️ `CIRCUIT_USE_NUMBA=1` 只是**允许**用 Numba；解释器里 import 不到 Numba 时会**静默回落**到解释版内核（结果一致但 chopper 慢约 28×）。跑性能必须用装了 Numba 的 conda `daily` 环境，详见 [`environment_performance.md`](environment_performance.md)（含实测基准与验证方法）。

> **生效机制与限制**：`numba_kernels` 的 `USE_NUMBA` 标志在该模块首次 import 时烙死为常量，事后设环境变量是静默 no-op。`--no-numba` 之所以能生效，是因为 `core/__init__.py` 在其求解器 import（会连带传递 import `numba_kernels`）之前，先对 `sys.argv` 预扫该 flag 并设好 `CIRCUIT_USE_NUMBA=0`；在 `python -m core …` 下 `__init__` 先于 `__main__.py` 执行，所以命令行用法（如上）总能生效。但如果从 Python 内部直接调用 `core.__main__.main()`（跳过了 `python -m core` 的正常 import 顺序，或在那之前已 import 了某个求解器模块），预扫可能被绕过——这种情况下 `_assert_numba_flag()` 守卫会检测到 `--no-numba` 请求但 Numba 仍激活，直接抛 `SystemExit` 响亮报错，而不是静默地假装关闭了 Numba。

### 退出码

全部返回 0 表示成功，非零表示失败。`python -m core.calibration --all` 任一 case 失败则退出码为 1，适合 CI 集成。

---

## 9. CI 集成示例

### GitHub Actions 片段

```yaml
- name: calibration regression
  run: python -m core.calibration --all --json

- name: core analysis smoke test
  run: |
    python -m core run examples/periodic_rc.json -a ac,noise,pss
    python -m core corners examples/afe_explore.json --quiet

- name: mismatch MC (spot check)
  run: python -m core mc examples/afe_explore.json -n 50 --seed 1 --quiet
```

### pre-commit 快速检查

```bash
# 跑 amp 校准 + 周期分析冒烟测试，任一步失败则阻止 commit
python -m core.calibration calibration/amp_design3_typical/ && \
python -m core run examples/periodic_rc.json -a pss --quiet
```

---

## 10. ML Surrogate + 优化闭环：`python -m core.surrogate` / `core.optimize`

ML 代理层（可选依赖 `scikit-learn`：`pip install -r requirements-ml.txt`）。求解器仍是真值，
surrogate 只做快速筛选。完整链路：`dataset`（生成）→ `surrogate train`（训练）→ `optimize`（筛选 + 校验）。

### 10.1 `surrogate` — 训练 / 预测指标代理

每个标签一个 GBT（`HistGradientBoostingRegressor`）；宽动态范围标签自动 log 空间。

```bash
# 训练 + held-out 评估 + 存模型
python -m core.surrogate train results/datasets/afe/afe_typical_train.npz \
    --test results/datasets/afe/afe_typical_test.npz --out results/models/afe.pkl
# 单点预测（设计向量按变量顺序）
python -m core.surrogate predict results/models/afe.pkl --x 65000,70,3500,30.5,10.0
```

| 子命令 | 参数 | 说明 |
|---|---|---|
| `train` | `<train.npz> [--test] [--out] [--max-iter]` | 拟合 + 评估（median / P95 相对误差 + R²） |
| `predict` | `<model.pkl> --x <csv>` | 预测一个设计的全部标签 |

辅助 API：`surrogate.filter_rows`（region-of-interest 过滤）、`surrogate.load_multi_corner`
（多 corner 堆叠，corner → `pvt0/pbeta0` 特征）。

### 10.2 `optimize` — 筛选 → Pareto → 校验闭环

用 surrogate 快筛大候选池，取约束内 Pareto 前沿，top-K 回本地 solver 校验（约束 / 目标读自配置的 `explore` 块）。

```bash
python -m core.optimize examples/afe_explore.json results/models/afe_typical.pkl -n 100000 --top-k 20
```

输出：筛选速率、可行 / Pareto 数、top-K 的 surrogate-vs-solver 误差 + solver 确认可行数 + 加速比。
典型：10 万候选 ~2.4s 筛完（**~1900× 于 solver**），shortlist 误差 **<1.5%**，solver 确认可行 10/12。

| 参数 | 默认 | 说明 |
|---|---|---|
| `config` `surrogate` | (必需) | 配置 JSON + 训练好的 `.pkl` |
| `-n` | `100000` | 候选池大小 |
| `--top-k` | `20` | 回 solver 校验的 Pareto 点数 |
| `--no-verify` | — | 只筛不校验（只返回 Pareto 候选） |

> ⚠️ `optimize` 的校验频率网格默认 0.01 Hz–10 kHz，需与 surrogate 训练用的网格一致（否则 `bw` 不可比）。

### 10.3 `surrogate_torch` — 可微 surrogate + 梯度设计优化

PyTorch MLP（可选 `torch`；Apple Silicon 自动用 MPS）。比 GBT 更平滑，把 GBT 最硬的 `bw` 也做到
p95 <2%。**可微** ⇒ 可直接对设计向量做梯度优化（几百步），而不是随机筛 10 万点。

```bash
# 训练（在有 torch 的环境，如 conda mps）
python -m core.surrogate_torch train results/datasets/afe/afe_typical_train.npz \
    --test results/datasets/afe/afe_typical_test.npz --out results/models/afe_torch.pt
# 梯度优化一个设计（minimize 目标 s.t. 约束，读自 explore 块）+ solver 校验
python -m core.surrogate_torch optimize examples/afe_explore.json results/models/afe_torch.pt \
    --steps 600 --verify
```

`optimize` 在归一化 [0,1] 设计空间做投影梯度下降（约束软惩罚），返回优化后的设计 + surrogate 指标；
`--verify` 回 solver 确认。典型：area −34% / power −41%，仍满足 gain/bw/irn，solver 校验 <1%。

> torch 与 scipy 求解器可分处不同 conda 环境：torch surrogate 只吃数据集 `.npz`（不需要 solver）。
> 若某环境的 torch 因 numpy ABI 不匹配坏了，用专门的 torch/MPS 环境训练（solver/数据仍在原环境）。
