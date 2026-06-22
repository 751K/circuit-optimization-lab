# Cadence 校准闭环 — 实现方案

> **实现状态（2026-06-21）：✅ 已落地。** `core/psf.py`（通用 PSFASCII 解析 + provenance）、
> `core/cadence_netlist.py`（从仓库拓扑生成 amp/chopper netlist）、`core/calibration.py`
> （对比引擎 + CLI）、`calibration/`（amp design#3 + chopper typ/slow/fast，全量新拉 Spectre
> 24.1.0.078）、`tests/test_calibration.py` 均已就绪。运行 `python -m core.calibration --all`。
>
> **首次闭环结果：全部 PASS。** amp DC/AC/noise 精确到机器精度（gain +0.00 dB、IRN +0.0%）；
> chopper PAC/PNoise 三 corner 均在 ~1–2%（typ +1.11%/+0.18%、slow −0.88%/+1.92%、fast
> +1.07%/−0.26%）。**关键经验**：本地 chopper 必须复刻已验证的求解配置（gear2 PSS 轨道 +
> `switch_size`/`edge_time`/`output_filter`/settling，存于 `metadata.json` 的 `circuit`/`solver`
> 块）——裸默认调用会把增益错报 >10%。下文为原始设计方案。

[English version below](#english-version)

## 背景

`docs/futureplan.md` 第 1 条：「建立一键式自动校准流水线：本地求解器跑完 → 自动对比 Cadence PSF/CSV 参考数据 → 输出差异表 → CI 可集成」。这是当前最高优先级事项。

现状：
- Cadence/Spectre 参考数据散落在 `/tmp/chopper_pss_verify/`、`/tmp/ver3/`、`/tmp/corner_*/` 下，无 provenance（不知道哪次代码、哪个 Spectre 版本生成的）
- 对比逻辑内嵌在 `core/noise_solver.py`、`core/ac_solver.py`、`core/chopper.py` 的 docstring 注释里，不可复用
- PSF 解析仅有 `tools/calibrate_switch.py` 中的 ad-hoc 实现，只支持 DC sweep 和简单 AC
- 每次改动后手动对比、不可重复、跟不上迭代速度

## 总体架构

```
calibration/
  ├── README.md                    # 目录约定
  ├── amp_design3_typical/         # 测试台 1: 放大器（非 chopper）
  │   ├── metadata.json            # provenance: netlist, Spectre 版本, 时间戳, corner
  │   ├── dcOp.dc                  # Cadence PSFASCII 输出
  │   ├── ac.ac
  │   └── noiseAnal.noise
  ├── chopper_design3_typical/     # 测试台 2a: chopper — typical
  │   ├── metadata.json
  │   ├── pss.td.pss
  │   ├── pac.pac
  │   └── pnoise.pnoise
  ├── chopper_design3_slow/        # 测试台 2b: chopper — slow
  │   └── ...
  ├── chopper_design3_fast/        # 测试台 2c: chopper — fast
  │   └── ...
  └── switch_5000_30/              # 测试台 3: 开关单管 on-state
      ├── metadata.json
      ├── ronvds.dc
      ├── ronvgs.dc
      └── capac.ac

core/
  ├── psf.py                       # [新增] 通用 PSFASCII 解析器
  └── calibration.py               # [新增] 校准对比引擎

tests/
  └── test_calibration.py          # [新增] 校准回归测试
```

## 分步实现

### 第 1 步：生成可信参考数据（服务器端）

**目标**：在 TU/e `flex` Cadence Spectre 服务器上重新跑三套测试台，导出 PSFASCII，拉回本地归档到 `calibration/`。

每套数据的 provenance 记录在 `metadata.json` 中：

```json
{
  "case": "amp_design3_typical",
  "description": "Design #3 amplifier — no chopper — DC / AC / Noise",
  "cadence": {
    "version": "24.1.0.078",
    "date": "2026-06-21T12:00:00",
    "server": "flex",
    "netlist": "chop_tb_d3/.../input.scs",
    "netlist_sha256": "abc123...",
    "corner": "typical",
    "analyses": ["dc", "ac", "noise"]
  },
  "circuit": {
    "topology": "afe_differential",
    "sizes": {"M6": [2264,78], "M7": [61365,61], ...},
    "bias": {"VDD": 40.0, "VCM": 31.38, "VB": 10.60, "VC": 16.47},
    "nf": 1
  },
  "reference_files": {
    "dc": "dcOp.dc",
    "ac": "ac.ac",
    "noise": "noiseAnal.noise"
  },
  "tolerances": {
    "dc_v_atol": 1e-3,
    "ac_gain_rtol": 0.01,
    "ac_bw_rtol": 0.05,
    "noise_irn_rtol": 0.03
  }
}
```

**三套测试台**：

| # | 测试台 | Cadence 分析 | 覆盖的本地 solver | 来源 |
|---|--------|-------------|------------------|------|
| 1 | 放大器 (非 chopper), typical | DC / AC / Noise | `ac_solve`, `noise_analysis` | 已有 netlist（`/tmp/ver3/` 同款） |
| 2a | Chopper, typical | PSS / PAC / PNoise | `pss_solve`, `pac_solve`, `pnoise_solve` | 官方 `chop_tb_d3` ADE netlist |
| 2b | Chopper, slow | PSS / PAC / PNoise | 同上 + corner | 同上，换 `monte.scs section=slow` |
| 2c | Chopper, fast | PSS / PAC / PNoise | 同上 + corner | 同上，换 `monte.scs section=fast` |
| 3 | 开关单管 (5000/30), typical | DC sweep / AC | `create_device("pmos_tft")` → `get_ss_params` | `tools/calibrate_switch.py` 已有 netlist |

**工作流**（通过 `cadence-server-verify` skill）：

```
1. 为每个测试台 gen netlist + runner → scp 到 flex
2. ssh flex 跑 spectre -format psfascii
3. pull PSF 文件回本地 → 放到 calibration/<case>/
4. 手写 metadata.json（provenance 在第一版手动填）
```

### 第 2 步：通用 PSF 解析器 (`core/psf.py`)

**目标**：将 `tools/calibrate_switch.py` 中的 ad-hoc PSFASCII 解析提炼为通用模块，支持 Cadence 全部常用分析类型。

PSFASCII 格式规则（已从现有文件中确认）：

```
HEADER           # 元数据 key-value 对
"key" "value"
...
TYPE             # 信号类型声明
"SignalName" FLOAT DOUBLE PROP(...)
...
VALUE            # 数据块开始
"freq" 1.0e-2
"vop" (-7.007  -8.370e-06)
...
END              # 数据块结束
```

**模块接口**：

```python
# core/psf.py  (~200 行)

def parse_psf_header(path: str) -> dict:
    """解析 PSFASCII HEADER 段 → {key: value}"""

def parse_psf_dc(path: str) -> dict[str, float]:
    """解析 DC 工作点 → {signal_name: voltage_or_current}"""

def parse_psf_ac(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """解析 AC 扫频 → (freqs, {signal: complex_array})"""

def parse_psf_noise(path: str) -> tuple[np.ndarray, dict[str, tuple]]:
    """解析 noise 分析 → (freqs, {device: (total, thermal, flicker)})"""

def parse_psf_tran(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """解析瞬态波形 → (time, {signal: real_array})"""

def parse_psf_pac(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """解析 PAC 扫频 → (freqs, {signal: complex_array})  [与 AC 格式一致]"""

def parse_psf_pnoise(path: str) -> tuple[np.ndarray, dict]:
    """解析 PNoise → (freqs, {signal_or_device: ...})  [与 noise 格式一致]"""
```

**与 `calibrate_switch.py` 的关系**：
- `core/psf.py` → 通用解析器，不依赖任何 solver
- `tools/calibrate_switch.py` → 保留；可以改为 `from core.psf import parse_psf_dc, parse_psf_ac`，或保持独立不破坏已有 workflow

### 第 3 步：校准对比引擎 (`core/calibration.py`)

**目标**：加载参考数据 + 跑本地 solver → 逐指标对比 → 输出结构化报告。

**流程图**：

```
metadata.json
     │
     ├──→ load_reference_files() → Cadence 参考数据集
     │
     └──→ run_local_solvers()   → 本地 solver 结果集
                  │
                  ▼
           compare_*() 逐分析对比
                  │
                  ▼
           CalibrationReport
           {
             "case": "amp_design3_typical",
             "results": {
               "dc":  {"vop": {local, ref, delta, pass}, ...},
               "ac":  {"gain_dc_dB": {local, ref, delta, pass},
                       "bw_Hz": {local, ref, delta_pct, pass},
                       "gain_curve_max_delta_dB": ...},
               "noise": {"irn_uVrms": {local, ref, delta_pct, pass},
                         "out_psd_max_delta_pct": ...},
               ...
             },
             "overall_pass": true/false
           }
```

**对比指标定义**：

| 分析 | 指标 | 判断方式 | 默认容差 |
|------|------|---------|---------|
| DC | 各节点电压 | `\|local - ref\| < atol` | `atol=1e-3 V` |
| AC | DC 增益 (dB) | `\|Δ\| < rtol` | `rtol=0.01` (1%) |
| AC | -3dB 带宽 | `\|Δ\|/ref < rtol` | `rtol=0.05` (5%) |
| AC | 全频带增益曲线 | `max\|local_gain - ref_gain\| < atol_dB` | `atol_dB=0.1` |
| Noise | IRN RMS (指定 band) | `\|Δ\|/ref < rtol` | `rtol=0.03` (3%) |
| Noise | 输出噪声 PSD 曲线 | `max\|local - ref\|/ref < rtol` | `rtol=0.05` |
| Tran/PSS | 轨道波形 | `max\|local(t) - ref(t)\|` + RMS 差 | `rtol=0.05` |
| PAC | Baseband 转换增益 | `\|local\| - \|ref\|| / \|ref\| < rtol` | `rtol=0.02` (2%) |
| PAC | @f_chop 增益 | 同上 | `rtol=0.02` |
| PNoise | 输出噪声 PSD | `max\|local - ref\|/ref < rtol` | `rtol=0.05` |
| PNoise | IRN RMS (指定 band) | `\|Δ\|/ref < rtol` | `rtol=0.03` (3%) |

所有容差可在 `metadata.json` 的 `tolerances` 字段中 per-case 覆盖。

**前端函数签名**：

```python
# core/calibration.py  (~300 行)

def load_reference(case_dir: str | Path) -> dict:
    """加载 metadata.json + 所有关联的 PSF 参考文件"""

def run_local(metadata: dict) -> dict:
    """根据 metadata 中的 circuit + analyses 配置，跑全部本地 solver"""

def compare_dc(local: dict, ref: dict, tol: dict) -> dict:
def compare_ac(local: dict, ref: dict, tol: dict) -> dict:
def compare_noise(local: dict, ref: dict, tol: dict) -> dict:
def compare_pss(local: dict, ref: dict, tol: dict) -> dict:
def compare_pac(local: dict, ref: dict, tol: dict) -> dict:
def compare_pnoise(local: dict, ref: dict, tol: dict) -> dict:

def run_calibration(
    case_dir: str | Path,
    *,
    analyses: list[str] | None = None,   # 可只跑部分分析
    tol_overrides: dict | None = None,    # 运行时容差覆盖
    relaxed: bool = False,                # 宽松容差模式 (×3)
) -> dict:
    """主入口：加载参考 → 跑本地 → 逐项对比 → 输出报告"""

def format_report(report: dict, fmt: str = "text") -> str:
    """格式化为 text / markdown / json"""
```

### 第 4 步：CLI 入口

```bash
# 单个用例
python -m core.calibration calibration/amp_design3_typical/

# 批量（所有 calibration/ 下子目录）
python -m core.calibration --all

# 只跑指定分析
python -m core.calibration calibration/chopper_design3_typical/ --analyses pss,pac

# JSON 输出（CI 友好）
python -m core.calibration calibration/amp_design3_typical/ --json

# 宽松容差（快速检查，不阻塞开发）
python -m core.calibration calibration/amp_design3_typical/ --relaxed

# 文本报告（人类可读）
python -m core.calibration calibration/amp_design3_typical/
```

### 第 5 步：CI 集成

新增 `tests/test_calibration.py`：

```python
import pytest
from core.calibration import run_calibration

CASES = [
    "calibration/amp_design3_typical",
    "calibration/chopper_design3_typical",
    "calibration/switch_5000_30",
]

@pytest.mark.parametrize("case_dir", CASES)
def test_calibration_ac_noise(case_dir):
    """AC + Noise vs Cadence — 快速核心检查"""
    report = run_calibration(case_dir, analyses=["ac", "noise"])
    assert report["results"]["ac"]["pass"], f"AC mismatch: {report['results']['ac']}"
    assert report["results"]["noise"]["pass"], f"Noise mismatch: {report['results']['noise']}"


@pytest.mark.slow
@pytest.mark.parametrize("case_dir", ["calibration/chopper_design3_typical"])
def test_calibration_chopper_full(case_dir):
    """Chopper PSS/PAC/PNoise vs Cadence — 完整三件套（慢）"""
    report = run_calibration(case_dir, analyses=["pss", "pac", "pnoise"])
    assert report["overall_pass"]


@pytest.mark.slow
@pytest.mark.parametrize("corner", ["slow", "typical", "fast"])
def test_calibration_chopper_corners(corner):
    """Chopper 三工艺角 PAC/PNoise vs Cadence"""
    report = run_calibration(f"calibration/chopper_design3_{corner}/",
                             analyses=["pac", "pnoise"])
    assert report["overall_pass"]
```

## 风险与约束

| 风险 | 缓解 |
|------|------|
| **PSF noise 格式复杂**：noise 贡献是 `(total thermal flicker)` 三元组 | 先解析 `/tmp/ver3/noiseAnal.noise` 验证解析器正确性，已知格式 |
| **PSS 瞬态波形网格对齐**：Cadence 和本地 `tgrid` 不同 | 对比前用 `np.interp` 将 reference 重采样到本地网格 |
| **Chopper PAC/PNoise 包装器 vs 通用 solver 路径不同** | 校准脚本默认走 chopper 包装器（`pmos_chopper_pac/pnoise`），因为 Cadence 参考数据来自 chopper 测试台 |
| **容差设定可能过严/过松** | 初始值基于已知对标结果；每个 `metadata.json` 可覆盖；CI 初期用 `--relaxed` 模式 |
| **没有二进制 PSF 解析** | Cadence 统一用 `-format psfascii`，不做 `psfbin` |
| **参考数据 provenance 关键** | `metadata.json` 强制填写 Cadence 版本 + 时间戳 + netlist hash |

## 与现有代码的关系

- **`tools/calibrate_switch.py`**：保留。`gen`/`parse` workflow 仍可用于单管快速检查。`core/psf.py` 提取其解析逻辑为通用模块，此工具可以改为 import `core.psf`，也可以保持独立。
- **`benchmarks/`**：互补，不做改动。benchmark 测量性能，calibration 测量精度。
- **`core/corners.py` line 17**：明确声明「Cadence/Spectre comparison should live in dedicated verification scripts」——`core/calibration.py` 就是这个 dedicated script。
- **`core/chopper.py` 的两个经验常数(换向相位 24.93°、噪声 PSD scale 1.0355)**：已于 2026-06-22 **retire**。它们只是给快速 `pmos_chopper_lptv_analysis`(一阶 quasi-static 近似)打的补丁;真正无常数的一等公民是谐波平衡路径 `pmos_chopper_pss`→`pmos_chopper_pac`/`pmos_chopper_pnoise`(`core/calibration.py` 校验的就是它)。`lptv_analysis` 现诚实返回一阶估计(增益偏低约 10%)。

## 工作量估计

| 模块 | 估计行数 | 难度 | 备注 |
|------|---------|------|------|
| 生成参考数据（服务器端） | — | 中 | 三套测试台，已知 netlist，主要是跑和拉 |
| `core/psf.py` | ~200 | 中 | PSFASCII 规则性强，体力活 |
| `calibration/` 数据目录 | ~10 个 metadata JSON | 低 | 填表 |
| `core/calibration.py` | ~300 | 中 | 关键在统一 solver 输出 key 到对比指标 |
| CLI 入口 | ~50 | 低 | argparse + dispatch |
| `tests/test_calibration.py` | ~100 | 低 | 参数化 pytest |
| 文档 | — | 低 | 本文件即核心文档 |
| **合计** | **~650 行代码 + 数据** | | |

---

# English Version

## Cadence Calibration Closed Loop — Implementation Plan

### Background

Item #1 from `docs/futureplan.md`: build a one-click automated calibration pipeline: local solver → auto-compare against Cadence PSF/CSV → difference report → CI-integrable. This is the highest-priority item.

Current state:
- Cadence reference data is scattered under `/tmp/` with no provenance
- Comparison logic is embedded in docstring comments across solver files
- PSF parsing exists only as ad-hoc code in `tools/calibrate_switch.py`
- Every model/solver change requires manual re-comparison

### Architecture

```
calibration/         # Reference data with full provenance
core/psf.py          # Generic PSFASCII parser
core/calibration.py  # Comparison engine
tests/test_calibration.py  # CI regression tests
```

### Implementation Steps

1. **Generate trusted reference data** — run 3 testbenches on the TU/e `flex` Cadence server, export PSFASCII, pull back with `metadata.json` recording Spectre version, netlist hash, and timestamp
2. **`core/psf.py`** — generic PSFASCII parser supporting DC, AC, noise, transient, PSS, PAC, PNoise (~200 lines)
3. **`core/calibration.py`** — comparison engine: load reference + run local solvers → per-metric comparison with configurable tolerances → structured report (~300 lines)
4. **CLI** — `python -m core.calibration <case_dir> --json`
5. **CI** — `tests/test_calibration.py` with parameterized cases, `@pytest.mark.slow` for PSS/PAC/PNoise

### Tolerances (defaults, overridable per-case)

| Analysis | Metric | Default |
|----------|--------|---------|
| DC | node voltages | `atol=1e-3 V` |
| AC | gain | `rtol=1%` |
| AC | bandwidth | `rtol=5%` |
| Noise | IRN RMS | `rtol=3%` |
| PAC | baseband gain | `rtol=2%` |
| PNoise | IRN RMS | `rtol=3%` |

### Effort

~650 lines of new code + reference data migration. 1–2 days.
