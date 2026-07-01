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
| `python examples/mc_mismatch.py [n] [seed]` | Mismatch MC + 直方图 |
| `python examples/find_max_gain.py` | PMOS 反相器最大增益扫描 |
| `python examples/sweep_vin_vout.py` | PMOS 反相器 DC 传输曲线 |
| `python examples/sc_lpf.py` | 开关电容 LPF 瞬态仿真 |
| `python tools/calibrate_switch.py gen/parse` | Chopper 开关 Cadence vs 本地校准 |

---

## 1. 主 CLI：`python -m core`

入口文件 `core/__main__.py`，支持 5 个子命令。默认兼容旧用法（无子命令时自动路由到 `run`）。

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
| `--noise-band LO HI` | float×2 | `0.05 100.0` | IRN 积分频带 (Hz) |
| `-o`, `--output` | path | — | 结果写出为 JSON |
| `--no-numba` | flag | — | 禁用 Numba 加速 |
| `--quiet` | flag | — | 不打印进度 |

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
| `circuit` | path | (必需) | 含 `"explore"` 块的电路 JSON |
| `-n`, `--n` | int | `200` | 候选数量 |
| `--seed` | int | `0` | 随机种子 |
| `--method` | choices | `lhs` | 采样方法：`lhs`（拉丁超立方）或 `random` |
| `-o`, `--output` | path | — | 输出路径前缀（生成 `<prefix>.csv` + `<prefix>.jsonl`） |
| `--no-numba` | flag | — | 禁用 Numba |
| `--quiet` | flag | — | 不打印逐候选进度 |

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
