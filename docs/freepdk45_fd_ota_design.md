# FreePDK45 全差分 OTA 设计记录

> **文档状态：可复现设计快照。** 本文保留架构、尺寸和 ngspice 交叉核对结果。
> FreePDK45 快速 AC 网格缺少部分结电容，整机带宽应以显式 ngspice AC oracle
> 复核；本文不是版图后 sign-off 报告。

用本仓库工具链在 **FreePDK45(45 nm, 1.0 V)** 上从零走完的设计案例——器件由
**ngspice-C BSIM4**(FreePDK45 的 oracle,经缓存特性化网格 `circuitopt.ngspice_device`)精确
求值,配合 dataset/surrogate/optimize 的 ML 筛选管线。测试台:
[examples/freepdk45_fd_ota.json](https://github.com/751K/circuit-optimization-lab/blob/main/examples/freepdk45_fd_ota.json)(其中尺寸即最终优化设计,
直接可复现)。方法沿用 SKY130 案例([docs/sky130_fd_ota_design.md](sky130_fd_ota_design.md)),
但工艺换成用户的目标工艺 FreePDK45,电压从 1.8 V 收紧到 1.0 V。

## 1. 设计需求与测试台

| 需求 | 实现 | 结果 |
|---|---|---|
| 差分输入/输出 | VINP/VINN → OUTP/OUTN 全差分信号路径 | ✅ |
| 真实源阻抗 | Rs = 1 kΩ/边 | ✅ |
| 真实负载电容 | CL = 0.2 pF/边(45 nm 片上下一级 + 布线量级) | ✅ |
| 输入 AC 耦合 + 输入 CM 设置 | Cc = 2 pF 耦合,RB = 1 MΩ 上拉到 VCMI = 0.65 V(高通角 ~80 kHz) | ✅ |
| 输出 CMFB | 连续时间双差分对 CMFB(见 §2.2) | ✅ |
| 输出 CM 最优 + 可调 ≥50 mV | VCM_REF 编程;OCM 随 VCM_REF 0.45→0.56 跟踪 475→587 mV,**范围 >110 mV** | ✅ |
| 低频增益 > 40 dB | **58.9 dB**(通带,ngspice 全 OTA 交叉核对;PVT 全 27 角 ≥42.8 dB) | ✅ |
| UGBW ≥ 0.1 GHz | **119.9 MHz**(ngspice oracle;恒 gm 偏置下 PVT 全角 ~119–121 MHz) | ✅ |
| 功耗 | **17.1 µW @ 1.0 V**(见 §4;100 MHz 处地板 ~14 µW) | — |

> **头条数字取 ngspice(oracle)值。** 求解器网格 AC 模型只带 Cgs/Cgd,不含漏/源结电容
> (Cdb/Csb),故 `ac_solve` 少算输出节点负载、UGBW 偏乐观 ~8%(读 130 MHz)。用完整 OTA 网表
> 直接对 ngspice `.ac` 核对(见 §4.5)后,以 oracle 的 119.9 MHz 为准,并给设计留了 ~20% 裕量。

## 2. 架构综合

### 2.1 OTA 架构:全差分单级望远镜级联(telescopic cascode)。为什么?

先做单管特性化(`circuitopt.device_model.create_transistor` 直接探 ngspice-C BSIM4)定标:

- FreePDK45 NMOS Vth ≈ 0.38 V、PMOS |Vth| ≈ 0.47 V(1.0 V 轨);
- **单管本征增益 gm/gds 在 L = 0.1 µm 仅 ~28–38 dB**,L = 0.2 µm 才升到 ~42–48 dB——
  所以**朴素 5 管/电流镜 OTA 达不到稳健的 40 dB,必须级联**;
- gm/Id 效率甜点 ≈ 15–20,对应 Vdsat ≈ 0.1–0.13 V。

选型逻辑(功耗效率是这道题的核心矛盾:UGBW = gm1/(2π·CL) 定死 gm1 ≥ 2π·10⁸·0.2p ≈ 126 µS):

- **单级**:负载电容本身就是补偿电容——无第二级静态电流、无 Miller 补偿、无 RHP 零点代价。
  全部尾电流都换成输入 gm。
- **望远镜而非折叠级联**:折叠要第二条支路(≈2× 电流);望远镜把级联管叠进同一支路复用电流,
  是"每 µA 换 gm"效率最高的 ≥40 dB 拓扑——与"最低功耗"这一核心诉求一致。
- **1.0 V 头room 够不够?** 5 管叠(尾+输入+N级联+P级联+P负载)≈ 5 × 0.12 V ≈ 0.6 V Vdsat,
  1.0 V 下留 ~0.4 V 输出窗,Vdd−10% = 0.9 V 下 ~0.3 V。45 nm 速度饱和使 Vdsat 低(~0.1 V),
  加上输入对工作在弱-中反型,窗口够用(实测输出 CM 窗 ~0.43–0.59 V,增益 >40 dB)。**折叠级联是
  更宽头room 的备选**(输入 CM 可到轨、更耐低 Vdd),但它多一条电流支路,功耗劣势明显;在
  1.0 V 且以最低功耗为纲时,望远镜 + 恒 gm 偏置(见 §5)是更优解。
- **弱反型输入对**(gm/Id ≈ 20):126 µS 只要 ~6 µA/边。
- 增益预算:A0 = gm1·(gm3·ro3·ro1 ∥ gm5·ro5·ro7),级联把输出阻抗抬到数十 MΩ,实测 57.7 dB,
  对 40 dB 规格留 17 dB 裕量。

### 2.2 CMFB 方案:连续时间双差分对 + 电流镜。为什么?

- **不用电阻平均检测**:级联输出阻抗数十 MΩ,检测电阻要 >100 MΩ 才不拉垮增益,不现实;
  双差分对检测只加栅电容负载。
- **不用开关电容 CMFB**:本电路无时钟(连续时间 AC 耦合前端),引时钟得不偿失(纹波/馈通)。
- 工作方式:两个 NMOS 检测对(MS1/MS2 测 OUTP vs VCM_REF、MS3/MS4 测 OUTN vs VCM_REF),
  参考侧漏极汇入二极管接法 PMOS(CTRL 节点),镜像到负载管 M7/M8。CM ↑ → 参考侧电流 ↓ →
  CTRL ↑ → 负载电流 ↓ → CM ↓(负反馈)。
- **1.0 V 头room 细节**:检测对尾节点(CMT)只坐在 ~0.17 V(OUTP≈0.53 − Vgs≈0.36),
  尾管弱反型 Vdsat ~0.06 V 下仍饱和——1.0 V 下能工作的关键(1.8 V 的 SKY130 版尾节点在 ~0.45 V)。
- **CM 精度**:+26 mV 系统偏移(CMFB 环增益有限 + 镜像残差),随 VCM_REF 1:1 跟踪,全 27 PVT
  角 OCM 稳定跟 0.5·VDD(偏移恒定),可调性规格本身覆盖此偏移。

## 3. 手算 → 目标设计参数(gm/Id 法)

| 参数 | 手算目标 | 仿真(最终,ngspice) |
|---|---|---|
| UGBW | 100 MHz(+裕量) | 119.9 MHz |
| gm1 | 126 µS | 167 µS |
| 每边支路电流 | 6.3 µA | ~8.5 µA |
| gm/Id(输入对) | ~20(弱反型) | ~20 |
| A0 | ≥40 dB | 58.9 dB |
| PM | ~90°(单极点) | 84.2° |
| 功耗 | ~15 µW | 17.1 µW |

偏置轨(1.0 V):VCMI = 0.65(输入对 Vgs)、VCM_REF = 0.50(≈中轨,最优输出 CM)、VB_N = 0.44
(尾电流,含 UGBW 裕量)、VB_CN = 0.814(N 级联栅,把输入对 Vds 顶到 ~0.13 V 饱和)、
VB_CP = 0.35(P 级联栅)。输入对 L = 0.15 µm(比其余核心管 L = 0.1 长——见 §5 的 PVT 增益裕量
教训);负载/尾 L = 0.2。

## 4. 功耗优化:能压到多低?

**流程**(全用 ngspice-C 真值,`extract_w` 参考 W 特性化使 W 扫描成纯网格插值):
teacher dataset(LHS 600 样本 × 7 设计轴,50 s 生成,600/600 收敛)→ HistGBT surrogate →
6 万候选粗筛 → 40 候选真解校验 → 坐标下降打磨 → **逐 W 真卡终检**。

**代理模型生成与选择**(留出 20% held-out,GBT vs 线性基线):

| 标签 | GBT R² | GBT MAE | 线性 R² | 线性 MAE |
|---|---|---|---|---|
| gain (dB) | 0.918 | 0.57 | 0.625 | 1.30 |
| UGBW (MHz) | 0.992 | 3.9 | 0.961 | 8.8 |
| power (µW) | 0.991 | 0.63 | 0.948 | 1.61 |
| OCM (mV) | 0.985 | 2.0 | 0.960 | 3.4 |

GBT 在功耗/UGBW/增益上全面胜出 → 选 GBT。(PM 恒 ~90°、方差极小,R² 无意义、MAE 仅 1.9°。)
筛出的 40 个候选中 **29 个真解一次过**(surrogate 保真度高);extract_w 筛选(13.9 µW)与逐 W
真卡终检(14.0 µW)吻合到 <1 %。

**功耗地板 ≈ 14.0 µW**(UGBW 恰好压到 ngspice 100 MHz 处);**交付设计取 17.1 µW**,把 UGBW
抬到 ngspice 119.9 MHz 留 ~20% 裕量(见 §4.5:求解器 UGBW 偏乐观 ~8%,不留裕量则 PVT 会滑到
规格线以下)。

**功耗-带宽 Pareto**(可行域内每 UGBW 档最低功耗,≈ 0.16 µW/MHz):

| UGBW | 最低功耗 |
|---|---|
| 100 MHz | **12.9 µW** |
| 150 MHz | 19.8 µW |
| 200 MHz | 27.7 µW |
| 300 MHz | 45.6 µW |

这直接回答"boost BW 同时兼顾功耗效率":单级望远镜里 UGBW ∝ gm1 ∝ I(弱反型),功耗几乎与
UGBW 线性,每加 100 MHz 约 +16 µW(此 Pareto 用 ac_solve 的 UGBW;§4.5 校正后约 +18 µW/100MHz)。

**功耗地板讨论**:UGBW 约束下主体电流下限 = 2·gm1/(gm/Id)max ≈ 2·126µS/23 ≈ 11 µA(11 µW);
CMFB + 镜像支路再加 ~3 µA。**~14 µW 是 UGBW 压到 ngspice 100 MHz 的实际地板**;交付 17 µW 是为
UGBW 留 20% 裕量。再压空间在:CL 减半(规格允许则主体电流直接减半)、CMFB 缩(以 CM 精度/环速换)。

### 4.5 独立验证:全 OTA 对 ngspice `.ac`

代理/优化都跑在 `ac_solve` 上,而 `ac_solve` 只是单器件精确的 ngspice-C——环路级别是否也对?把
**完整 FD-OTA 网表**(含 CMFB 环、AC 耦合前端、Rs/CL,ngspice `.nodeset` 用同一 DC 种子)丢进
**ngspice 自己的 `.ac`**,对差分 H(f) 取增益/UGBW/PM:

| 指标 | ac_solve | ngspice `.ac` | 差 |
|---|---|---|---|
| 通带增益 | 58.8 dB | **58.9 dB** | +0.16 dB |
| PM | 91.4° | **84.2°** | −7.3° |
| UGBW | 130.4 MHz | **119.9 MHz** | **−8.0%** |

- **增益/PM 基本逐位对上**(<0.2 dB / <8°)。
- **UGBW 系统性偏乐观 ~8%**:求解器的网格 AC 模型只放 Cgs/Cgd,**不含漏/源结电容 Cdb/Csb**
  (与 [[freepdk45-ngspice-eval]] 记的"Cgd 饱和归零"同源),少算输出节点负载 → UGBW 偏高、PM 偏高。
- **处理**:头条数字全部改取 ngspice 值;设计把 UGBW 顶到 ngspice 119.9 MHz(≈20% 裕量),使
  恒 gm PVT 全角在 oracle 下仍 >100 MHz。此偏差已固化为回归测试 `test_fd_ota_ac_matches_ngspice`
  (增益 <1 dB、PM <10°、UGBW <12%)。

## 5. PVT 仿真(P: nom/ss/ff × V: 0.9/1.0/1.1 V × T: 30/60/90 °C,27 点)

温度经 BSIM4 温度方程逐器件生效(ngspice `.options temp`,卡按角/温度重特性化并缓存);
VDD 参考轨(VCM_REF、VB_CP、PMOS 体)随电源缩放,接地参考轨(VB_N、VB_CN、VCMI)不变。

**关键对照——偏置方案决定 PVT 鲁棒性**(与 SKY130 案例同一教训):

| 方案 | 增益范围 | UGBW 范围(ac_solve) | 功耗范围 |
|---|---|---|---|
| 固定 VB_N(电压偏置) | 42–62 dB | **52–188 MHz** ❌ | 7–28 µW |
| 恒 gm 偏置(逐点解 VB_N 使 gm1 = 167 µS) | **42.8–60.6 dB** ✅ | **129–131 MHz** ✅ | 15.3–23.1 µW |

UGBW 为 ac_solve 值;按 §4.5 校 ~−8% 得 ngspice ~119–121 MHz,全角仍 >100 MHz。

- 固定电压偏置在高温/慢角**尾电流塌缩**:90 °C 迁移率降 → Id 掉 ~30%,UGBW 从 130 掉到
  ~80 MHz(nom)、**ss 角更低——远不达标**(增益因输入对 L=0.15 的裕量仍保住 ≥42 dB)。
- **恒 gm 偏置**(片上即经典恒 gm 电流基准做的事,本报告用逐点二分 VB_N 仿真其行为)把
  UGBW 钉在 129–131 MHz(ngspice ~120),增益全角 ≥42.8 dB(最差 ff/0.9 V/90 °C)。恒 gm 下
  功耗随温度上升(90 °C 比 30 °C 贵 ~20%:gm ∝ I/T,保 gm 必须加流)——这是物理,不是缺陷。
- **一个 PVT 增益裕量教训**:最初输入对 L = 0.1 时,ff/0.9 V/90 °C 角增益只有 36.3 dB(欠 40)。
  加长**级联/负载** L 反而更糟(扰乱 0.9 V 下 CMFB 电流平衡,OCM 下坠、级联出饱和);正确的杠杆
  是加长**输入对** L(0.1→0.15):望远镜里 N 侧输出阻抗 = gm3·ro3·**ro1**,抬 ro1 直接增益 +5 dB
  且输入对在底部、不动输出 CM 平衡 → 最差角回到 41.7 dB。
- PM 全角 87–94°;输出 CM 被 CMFB 钉在 0.5·VDD(+26 mV 恒定偏移)全 PVT 稳定。

## 复现

```bash
# 单点验证(需要 PDK_ROOT 指向 FreePDK45 卡 + ngspice)
python -m circuitopt run examples/freepdk45_fd_ota.json -a ac,noise
```

流程脚本(dataset→surrogate→screen→verify→polish、Pareto、27 角 PVT)见本会话 scratchpad
(`fp45_flow.py` / `fp45_opt.py` / `fp45_pvt.py`)。注意事项(硅 + 多稳态电路,沿用 SKY130 经验):

- 14 节点 CMFB 拓扑存在简并 DC 稳态,**一切求解都要喂 `dc_guesses` 种子**;扫 VDD 时 P/CTRL/OUT
  这些 VDD 参考种子节点要随 VDD 平移。
- dataset/筛选阶段给器件加 `extract_w=1.0`(参考 W 特性化 + 线性 W 缩放,单器件误差 ~0.7%),
  免逐候选 ngspice 重表征;**终选设计务必逐 W 真卡复核**。
- FreePDK45 器件评估器是 ngspice-C(非 SKY130 的 OSDI VA)——见
  [freepdk45-ngspice-eval 记忆](https://github.com/751K/circuit-optimization-lab/blob/main/README.md) 与 `circuitopt/ngspice_char.py` 头注。
