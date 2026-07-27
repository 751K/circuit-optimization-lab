# CLI 参考手册

[文档首页](README_zh.md) | [安装与快速上手](getting_started_zh.md) |
[JSON 电路格式](json_circuit_format_zh.md) | [English](cli_reference.md)

本文只记录当前公开命令。以实际 `--help` 输出为最终依据：

```bash
circuit-opt --help
python -m circuitopt --help
```

两个入口等价。下文统一使用 `circuit-opt`。

## 命令总览

| 命令 | 用途 |
|---|---|
| `run` | 按 JSON 配置运行 AC、noise、transient、PSS、PAC、PNoise |
| `signoff` | 在显式 PVT 网格上运行多个 signoff 测试台 |
| `verify-engine` | 在内部引擎与 ngspice 之间交叉校验单个 signoff case |
| `explore` | 从 `explore` 块采样、求解、筛约束并生成 Pareto 前沿 |
| `corners` | AT4000TG 的 `typical/slow/fast` 固定工艺角扫描 |
| `mc` | AT4000TG 逐器件 mismatch Monte Carlo |
| `chopper` | AFE chopper 的理想、静态、LPTV、PSS、PAC、PNoise 和瞬态流程 |
| `adc` | SAR ADC 单次转换、静态扫描、正弦动态、失配 MC 和设计探索 |
| `plot` | 生成内置 AFE/chopper 波形和 Bode 图 |
| `dataset` | 生成带 provenance 的 surrogate 数据集 |
| `serve` | 启动可选 FastAPI 本地服务 |
| `mcp` | 启动可选 Model Context Protocol 服务 |

无子命令的旧写法仍会自动路由到 `run`：

```bash
circuit-opt examples/periodic_rc.json
```

新文档和脚本应显式写 `run`。

## `run`

```bash
circuit-opt run CIRCUIT.json [options]
```

常用示例：

```bash
# 运行 JSON analyses 块中的全部分析
circuit-opt run examples/periodic_rc.json

# 只运行 AC 和 noise
circuit-opt run examples/periodic_rc.json --analysis ac,noise

# 覆盖工艺角并输出 JSON
circuit-opt run examples/tsmc28hpcp_5t_ota.json \
  --analysis ac,noise --corner ss --output results/tsmc28_ss.json
```

参数：

| 参数 | 说明 |
|---|---|
| `-a`, `--analysis` | 逗号分隔的分析子集：`ac,noise,transient,pss,pac,pnoise` |
| `--corner` | 覆盖工艺角；AT4000TG 为 `typical/slow/fast`，硅工艺使用各自支持的 corner |
| `--noise-band LO HI` | CLI 汇总中的噪声积分带宽，默认 `0.05 100.0` Hz |
| `-o`, `--output` | 把结果写成 JSON |
| `--engine {rust}` | 计算引擎；v2.0.0 起仅 `rust`（省略即默认 `rust`） |
| `--no-numba` | **已在 v2.0.0 移除（会报错）**：numba 引擎不再存在，改用 `--engine rust` |
| `--quiet` | 关闭进度和摘要输出 |

`run` 的具体数值选项来自 JSON 顶层 `analyses` 块。字段见
[JSON 电路格式](json_circuit_format_zh.md)。

## `signoff`

```bash
circuit-opt signoff CAMPAIGN.json [--workers N] [--margins] [--tolerance SPEC] [--output RESULT.json]
```

该命令运行 PVT 笛卡尔积中的全部测试台，保留 invalid case，并输出顺序确定的逐点
及全局最差点信息。配置和结果契约见
[Signoff Campaign](signoff_campaign_zh.md)。

| 参数 | 作用 |
|---|---|
| `--margins` | 打印逐约束余量表（实测范围、最紧角点、失败点数），按最紧优先排序。内置的 `worst_case` 只给出单个测量量；余量表才能显示**其他哪些**规格也逼近极限。 |
| `--tolerance SPEC` | 按元件容差扰动无源器件后重跑，报告哪条约束先破。`SPEC` 为逗号分隔的 `NAME=PERCENT`，`NAME` 可以是器件类（`R`、`C`）或单个器件名（`CC1`）：`--tolerance R=20,C=10`。只在标称值下成立的签核不算签核。 |

## `verify-engine`

```bash
circuit-opt verify-engine CAMPAIGN.json --case NAME --point CORNER/TEMP/SUPPLY
                          [--nodes A,B] [--tolerance-mv MV] [--top N]
```

把同一个 signoff case 分别交给内部 BSIM4 引擎与 ngspice 模型卡路径求解，逐节点
对比轨迹。判定基于**稳定后**偏差：快速跃变期间，两个求解器仅仅步点位置不同就能
产生几十 mV 的差，而真正的电路层面分歧在窗口末尾依然存在。默认排除由激励驱动的
节点以及跨越输入不连续点的采样——这两类比较的都是波形与其自身。

当某个设计结果看起来是**形态**错误而不是略有偏差时用它，那正是单引擎工具链无法
诊断的失效模式。

## `explore`

```bash
circuit-opt explore CONFIG.json [options]
```

配置文件必须是完整电路 JSON，并包含 `explore` 块。

```bash
circuit-opt explore examples/afe_explore.json -n 500 --seed 1
circuit-opt explore examples/sky130_5t_ota.json -n 200 --corner ss --workers 8
circuit-opt explore examples/tsmc28hpcp_5t_ota.json -n 200 --corner ff
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `-n`, `--n` | `200` | 候选数量 |
| `--seed` | `0` | 随机种子 |
| `--method` | `lhs` | `lhs` 或 `random` |
| `--corner` | 无覆盖 | 求解时使用的工艺角 |
| `--workers` | `1` | 满足条件的 Rust `CompiledCampaign` batch 所用 Rayon worker 数 |
| `-o`, `--out`, `--output` | 无 | 输出前缀，写 `<prefix>.csv` 和 `<prefix>.jsonl` |
| `--quiet` | 关闭 | 不打印逐候选进度 |
| `--engine {rust}` | `rust` | v2.0.0 起仅 `rust`（省略即默认 `rust`） |
| `--no-numba` | — | v2.0.0 移除（报错）：numba 引擎不再存在 |

可探索变量、约束和目标见 JSON 文档的 `explore` 字段。

## `corners`

```bash
circuit-opt corners CIRCUIT.json [options]
```

当前实现调用 `circuitopt.corners.corner_table`，固定扫描 AT4000TG 的
`typical/slow/fast`。它不是通用硅 PVT campaign 驱动器。硅工艺多测试台 PVT
应使用 `signoff`；单电路单 corner 可使用 `run --corner`。

```bash
circuit-opt corners examples/afe_explore.json \
  --freqs-start 0.01 --freqs-stop 10000 --freqs-num 121 \
  --noise-band 0.05 100 --output results/afe_corners.csv
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--freqs-start` | `0.01` | 起始频率，Hz |
| `--freqs-stop` | `10000` | 终止频率，Hz |
| `--freqs-num` | `121` | 对数频率点数 |
| `--noise-band LO HI` | `0.05 100.0` | IRN 积分带宽 |
| `-o`, `--output` | 无 | CSV 输出 |
| `--workers` | `1` | 并行 corner worker 数（`ThreadPoolExecutor`，每次求解各自释放 GIL）；只有 3 个 corner，故超过 3 无收益 |
| `--engine {rust}` | `rust` | v2.0.0 起仅 `rust`（省略即默认 `rust`） |
| `--no-numba` | — | v2.0.0 移除（报错）：numba 引擎不再存在 |
| `--quiet` | 关闭 | 关闭逐 corner 输出 |

## `mc`

```bash
circuit-opt mc CIRCUIT.json [options]
```

当前通用 `mc` 使用 AT4000TG 的 `mvt0`/`mbeta0` 连续失配模型和 AFE latch
判据。它不是通用 foundry mismatch engine。

```bash
circuit-opt mc examples/afe_explore.json \
  -n 300 --seed 1 --corner slow --output results/afe_mc.json
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `-n`, `--n` | `200` | MC 样本数 |
| `--seed` | `0` | 随机种子 |
| `--workers` | `1` | 并行 MC worker 数（`ThreadPoolExecutor`，每次求解各自释放 GIL）；mismatch 抽样在调度前预先抽好，结果与 worker 数无关、逐字节确定 |
| `--corner` | `typical` | `typical`、`slow` 或 `fast` |
| `--freqs-start/stop/num` | `0.01/10000/121` | AC/noise 网格 |
| `--noise-band LO HI` | `0.05 100.0` | IRN 积分带宽 |
| `-o`, `--output` | 无 | JSON 汇总 |
| `--engine {rust}` | `rust` | v2.0.0 起仅 `rust`（省略即默认 `rust`） |
| `--no-numba` | — | v2.0.0 移除（报错）：numba 引擎不再存在 |
| `--quiet` | 关闭 | 关闭进度输出 |

SAR ADC 有独立的 `adc --mc` 流程，其失配语义来自 JSON `adc.mismatch`。

## `chopper`

```bash
circuit-opt chopper CIRCUIT.json --level LEVEL [options]
```

`LEVEL`：

| Level | 含义 |
|---|---|
| `ideal` | 理想方波 LPTV |
| `pmos` | PMOS 开关静态相位 |
| `lptv` | PMOS 边带折叠 |
| `pss` | Shooting PSS |
| `pac` | PSS 轨道上的 PAC |
| `pnoise` | PSS/PAC 后的周期噪声 |
| `transient` | 硬开关瞬态 |

```bash
circuit-opt chopper examples/afe_explore.json --level ideal
circuit-opt chopper examples/afe_explore.json --level pnoise \
  --f-chop 225 --max-sideband 10
circuit-opt chopper examples/afe_explore.json --level transient \
  --n-periods 8 --n-points 121
```

主要参数：

| 参数 | 默认值 |
|---|---:|
| `--f-chop` | `225` Hz |
| `--switch-w` / `--switch-l` | `5000` / `30` µm |
| `--edge-time` | `20e-6` s |
| `--max-harmonic` | `31` |
| `--max-sideband` | `10` |
| `--tstab-periods` | `2` |
| `--n-points` | `121` |
| `--n-periods` | `8` |
| `--freqs-start/stop/num` | `0.01/10000/121` |
| `--noise-band LO HI` | `0.05 100.0` Hz |
| `-o`, `--output` | 把结果写成 JSON |
| `--engine {rust}` | v2.0.0 起仅 `rust`（省略即默认 `rust`） |
| `--no-numba` | v2.0.0 移除（报错）：numba 引擎不再存在 |
| `--quiet` | 关闭摘要输出 |

这个命令是项目 AFE chopper 包装层，不是任意 JSON 周期电路的唯一入口。通用周期电路
优先在 JSON `periodic` 和 `analyses` 中配置，然后使用 `run`。

## `adc`

```bash
circuit-opt adc CIRCUIT.json MODE [options]
```

模式互斥：

```bash
# 单次转换
circuit-opt adc examples/freepdk45_sar3.json --vin 0.7

# 静态 ramp、DNL、INL 和 missing code
circuit-opt adc examples/freepdk45_sar6.json --sweep 64 --workers 8

# 相干正弦、SNDR、SFDR 和 ENOB
circuit-opt adc examples/freepdk45_sar6.json \
  --sine 128 --tone-bin 13 --sample-rate 10e6 --workers 8

# 使用 adc.mismatch 配置做 MC
circuit-opt adc examples/freepdk45_sar6.json \
  --mc 32 --seed 1 --workers 8

# 设计空间探索
circuit-opt adc examples/freepdk45_sar6.json \
  --explore examples/freepdk45_sar6_explore.json -n 20 --workers 4

# 终验 DNL：在二进制 major carry 处二分定位物理跃变电压
# （0.05 LSB 分辨率，~O(log 1/tol) 次转换）
circuit-opt adc examples/freepdk45_sar6.json --transitions --workers 8
```

主要参数：

| 参数 | 说明 |
|---|---|
| `--vin VIN` | 单次转换；不指定模式时默认以 0.5 V 运行一次 |
| `--sweep N` | N 个均匀 ramp 输入。小于 `2**n_bits` 即**子采样**：该密度下不存在跃变 DNL/INL，指标与 `--plot` 图切换为带符号码误差（`max_abs_code_err`；每个样本处\|INL\| 与其相差不超过 0.5 LSB）——12-bit 量级分辨率的筛查模式 |
| `--sine N` | N 个相干正弦样本 |
| `--mc N` | N 次逐器件失配 MC |
| `--sweep-points M` | 仅 mc：把每个 trial 的码中心扫描子采样到 M 点（覆盖 `adc.mismatch.sweep_points`）；良率改由 `code_err_threshold` 判定而非 DNL/INL 上限 |
| `--transitions [CODES]` | 锁步二分定位物理码界电压并报告其 DNL/INL。默认 `carries` = 每个二进制 major carry 两侧的 DNL bin 加失调跃变；也可给逗号分隔码表。每轮把全部未收敛探针合成一次编译 batch，`--workers` 跨跃变并行。收敛到 `--tol-lsb`（默认 0.05——比全码 ramp 的 ±0.5 LSB 量化细 10 倍），初始括号 `--bracket-lsb`（默认 2，筛查失真时自动放宽全量程）。12-bit 终验模式：~34 跃变 × ~9 探针 ≈ 310 次转换，而非 4096 |
| `--explore CONFIG` | 独立 SAR-explore 配置 JSON，跑 ADC 设计空间探索；与 `--vin`/`--sweep`/`--sine`/`--mc` 互斥 |
| `--tone-bin` | 相干输入 FFT bin，默认 3 |
| `--sample-rate` | 结果中报告的采样率，默认 10 MHz |
| `--amplitude` | 正弦峰值，默认 `0.45*vref` |
| `--offset` | 正弦直流偏置，默认 `0.5*vref` |
| `--corner` | 当前 ADC CLI 接受 `nom/ss/ff` |
| `-n`, `--n` | `--explore` 模式下的候选数量，默认 `50` |
| `--seed` | `--explore` 模式下的随机种子，默认 `0` |
| `--workers` | 独立最终瞬态、MC trial，或探索模式下每个 candidate 自己转换 sweep 的并发数（candidate 串行执行；被替换的外层线程池设计只能到 1.8x，内核自身并行可达 4.7x）。原生 BSIM 的单次转换、sweep、正弦测试、MC 和探索都使用编译式 Rust SAR 续算内核完成 bit 判决；随后每个输入只运行一次最终瞬态来报告波形和功耗。MC 在单个 Rayon 池中批处理 trial。不支持的拓扑或不完整 DC seed 会显式回退 Python replay。 |
| `--plot [DIR]` | 输出对应 PNG，需要 `plot` extra |
| | 不确定本机该给多少 `--workers`？`python tools/workers.py` 侦测拓扑并按工作负载给建议；`--calibrate` 实测饱和拐点 |
| `--waveforms` | 录制完整逐转换波形（`t`、`input_waveforms`、`transient`）并写入 `--output`。不带它时 `-o` 只落码流、bits、判决轨迹与功耗/指标标量——64 码 sweep 从数十 MB 降到 KB 量级——且 `--sweep` 完全跳过轨迹录制。必须与 `-o` 同用；`--mc`/`--explore` 拒绝该选项（其行本就不含波形）。 |
| `--csv` / `--jsonl` | ADC explore 输出 |
| `-o`, `--output` | JSON 结果（默认码流/指标档；见 `--waveforms`） |

闭环 bit 判决状态机在 Rust 续算内核中运行；Python 负责工作流编排以及最终结果和功耗组装。
比较器、CDAC 和开关仍由晶体管级瞬态计算。当前流程不等价于完整晶体管级数字 SAR 控制器。

## `plot`

```bash
circuit-opt plot [all|transient|bode|afe|chopper|ac|pac] [options]
```

该命令绘制项目内置 AFE/chopper 示例，不读取任意电路 JSON。

```bash
uv pip install -e ".[plot]"
circuit-opt plot bode --npts 121 --out-dir results
circuit-opt plot chopper --f-chop 225 --input-diff 1e-3
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--f0` | `10` Hz | AFE 瞬态正弦频率 |
| `--amp` | `5e-4` V | AFE 瞬态差分半幅值 |
| `--f-chop` | `225` Hz | chopper/pac 图使用的 chopper 频率 |
| `--input-diff` | `1e-3` V | chopper 瞬态直流差分输入 |
| `--npts` | 按图各自默认 | Bode 频率点数 |
| `--out-dir` | `results` | 输出目录 |
| `--engine {rust}` | `rust` | v2.0.0 起仅 `rust`（省略即默认 `rust`） |
| `--no-numba` | — | v2.0.0 移除（报错）：numba 引擎不再存在 |
| `--quiet` | 关闭 | 关闭摘要输出 |

## `dataset`

```bash
circuit-opt dataset CONFIG.json [options]
```

输入必须是带 `explore` 块的完整电路 JSON。每个样本保留设计变量、标签、失败状态和
provenance。

```bash
circuit-opt dataset examples/single_stage.json \
  -n 500 --seed 1 --labels ac_noise --out results/datasets/single

circuit-opt dataset examples/sky130_chopper.json \
  -n 200 --labels pss,pac,pnoise --out results/datasets/sky_chopper
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `-n`, `--n` | `200` | 样本数 |
| `--seed` | `0` | 随机种子 |
| `--workers` | `1` | 并行 candidate worker 数（`ThreadPoolExecutor`，每次求解各自释放 GIL） |
| `--method` | `lhs` | `lhs` 或 `random` |
| `--corner` | `typical` | 求解 corner；硅工艺可传自己的 corner |
| `--labels` | `ac_noise` | `ac_noise,transient,pss,pac,pnoise` 的组合 |
| `--freqs-start` | `-2` | AC 起始 decade |
| `--freqs-stop` | 无覆盖 | AC 终止 decade |
| `--freqs-num` | `101` | AC 点数 |
| `--out` | 自动 | 输出前缀 |
| `--no-npz` | 关闭 | 不写 dense NPZ |
| `--parquet` | 关闭 | 额外写 Parquet，需要 `parquet` extra |
| `--quiet` | 关闭 | 关闭进度 |
| `--engine {rust}` | `rust` | v2.0.0 起仅 `rust`（省略即默认 `rust`） |
| `--no-numba` | — | v2.0.0 移除（报错）：numba 引擎不再存在 |

## Surrogate 与优化

这些是独立模块入口，不是主 CLI 子命令：

```bash
uv pip install -e ".[ml]"

python -m circuitopt.surrogate train \
  results/datasets/single.npz --out results/models/single.pkl

python -m circuitopt.surrogate predict \
  results/models/single.pkl --x 2000,1500,25

python -m circuitopt.optimize \
  examples/single_stage.json results/models/single.pkl \
  --n-screen 100000 --top-k 20
```

PyTorch 版本：

```bash
uv pip install -e ".[torch]"
python -m circuitopt.surrogate_torch --help
```

Surrogate 只用于筛选或梯度搜索，最终可行性应回到物理求解器复核。

## `serve`

```bash
uv pip install -e ".[serve]"
circuit-opt serve
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8341` | TCP 端口 |
| `--reload` | 关闭 | uvicorn 开发自动重载 |
| `--job-workers` | `1` | `explore`/`mc` 后台 worker 数 |

`0.0.0.0` 会把无鉴权服务暴露到网络，不应作为默认配置。完整协议见
[本地服务 API](service_api_zh.md)。

## `mcp`

```bash
uv pip install -e ".[mcp]"
circuit-opt mcp --transport stdio --workspace .
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--transport` | `stdio` | `stdio` 或 `streamable-http` |
| `--workspace` | `.` | MCP 工具唯一可访问的文件树 |
| `--host` | `127.0.0.1` | Streamable HTTP 监听地址，只允许 loopback |
| `--port` | `8342` | Streamable HTTP 端口 |
| `--job-workers` | `1` | 并发 explore/MC/signoff 后台任务数 |

等价独立入口为 `circuit-opt-mcp`。工具、resources 和客户端配置见
[MCP 服务](mcp_server_zh.md)。

## 模拟设计工具

`tools/` 下的独立工具，服务于 signoff campaign **外面**的那层循环。

```bash
# 尺寸迭代：在内存里覆盖生成器常量，跑全 PVT 网格，
# 打印逐规格通过数以及每条失败约束背后的角点清单。
python tools/design_iterate.py run --generator tsmc28_mdac_ota_gen \
    --manifest examples/tsmc28hpcp_mdac_ota_signoff.json CC=850e-15 SZ:M11=139.286/0.30

# 根因定位：某个测量量在全网格上的分布、按每条 PVT 轴分组的均值，
# 以及有符号误差的最优公共修调。
python tools/design_iterate.py map --generator tsmc28_mdac_ota_gen \
    --manifest examples/tsmc28hpcp_mdac_ota_signoff.json \
    --case residue_plus_fs16 --measurement checkpoint_error --scale 1e3 --unit mV --limit 20

# 单点轨迹解剖。
python tools/design_iterate.py trace --generator tsmc28_mdac_ota_gen \
    --manifest examples/tsmc28hpcp_mdac_ota_signoff.json \
    --case residue_plus_fs16 --point ff/27/0.85 --nodes OUTP,OUTN,CTRL2

# 无源物料清单与硅面积：DUT 与测试台件由 deck 成员关系推导，
# 并给出无源/有源面积比。
python tools/passive_bom.py --generator tsmc28_mdac_ota_gen --exclude CL1,CL2 --top 8
```

`map` 的分组是值得记住的根因判别器：跨度落在温度轴上说明是器件级失调，落在电源
轴上说明是参考或电平位移不匹配。

## 校准与基准

校准回归：

```bash
python -m circuitopt.calibration --all
python -m circuitopt.calibration --all --json
python -m circuitopt.calibration calibration/amp_design3_typical/ --analyses ac,noise
```

性能基准：

```bash
python -m benchmarks.bench_afe --warm-runs 3
python -m benchmarks.bench_model --warm-runs 3
python -m benchmarks.bench_periodic --warm-runs 3
python -m benchmarks.bench_chopper --warm-runs 3
python -m benchmarks.bench_sweep --n-candidates 200
```

性能数字受 Python、Numba、CPU、冷启动和缓存状态影响。历史测量见
[运行环境与性能基准](environment_performance.md)。

## 退出码与输出

- 成功返回 0，参数错误、缺文件、分析失败或校准不通过返回非零。
- `run` 输出 JSON；`corners` 输出 CSV；`explore` 输出 CSV/JSONL；
  `dataset` 输出 JSONL/manifest/NPZ，可选 Parquet。
- 数值结果中的频率单位为 Hz，时间单位为秒，电压为 V，电流为 A。
- 噪声积分结果必须同时记录积分带宽。
