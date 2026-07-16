# FreePDK45 6-bit 差分 SAR ADC 设计记录

> **文档状态：可复现实验记录。** 本文展示晶体管级 CDAC/比较器与 Python SAR
> 状态机的联合流程。它不是完整晶体管级数字控制、瞬态噪声或版图后 ADC
> sign-off 流程。本文结果由旧的完整电路 ngspice 瞬态流程生成；当前默认
> `freepdk45.*` 已切换为原生 BSIM4，尚未声明本文全部 SAR 指标已在新后端重跑。

用本仓库工具链在 **FreePDK45(45 nm, 1.0 V)** 上从零走完的第二个设计案例——一个 **6-bit
全差分共模翻转 SAR ADC**,核心是一个**时钟同步 StrongARM 动态锁存比较器**。器件由
**ngspice-C BSIM4**(FreePDK45 的 oracle)在全电荷 `.tran` 中精确求值:`circuitopt` 为每一次
物理比较判决重放一遍 ngspice 瞬态,Python 只负责按判决更新后续 CDAC 控制。测试台即
[examples/freepdk45_sar6.json](https://github.com/751K/circuit-optimization-lab/blob/main/examples/freepdk45_sar6.json)
(尺寸/电容即最终设计,直接可复现)。它是 3-bit 静态比较器示例
[examples/freepdk45_sar3.json](https://github.com/751K/circuit-optimization-lab/blob/main/examples/freepdk45_sar3.json)
的自然升级:分辨率 3→6 bit、比较器从静态 5T 换成时钟动态锁存。

## 1. 设计需求与实测结果

| 需求 | 实现 | 结果(nom, ngspice 实跑) |
|---|---|---|
| 分辨率 | 6-bit 全差分 SAR,双二进制加权 CDAC | ✅ |
| 参考 / LSB | Vref = 1.0 V 差分,LSB = Vref/64 ≈ **15.6 mV** 差分 | ✅ |
| 采样电容 / kT/C | 单位电容 2 fF,每边总 64·Cu = **128 fF**(kT/C 噪声 ~180 µV ≪ LSB/2) | ✅ |
| 比较器 | **时钟同步 StrongARM 动态锁存**(经 `adc.clock` 选通,见 §2.3) | ✅ |
| 转换速率 | 采样 5 ns + 7×10 ns → 75 ns/次 → **13.3 MS/s** | ✅ |
| 静态线性度 | 64 码心全部正确,**max\|DNL\| = 0 / max\|INL\| = 0 LSB**(理想电容+理想驱动) | ✅ |
| 动态(相干正弦) | 128 样本/13 周期:**SNDR 36.9 dB / ENOB 5.84 bit / SFDR 44.1 dB** | ✅ |
| 功耗 / 能量 | **~7.2 µW/次**,**~0.54 pJ/次**(含比较器+CDAC+时钟切换) | — |
| 角鲁棒性 | ss / ff 全 64 码心仍正确,DNL/INL = 0 | ✅ |
| 失配良率 | 逐器件 Vth + CDAC 电容失配 MC(见 §4.4) | 见下 |

> **验收线**:6-bit 设计必须在 nom 角把全部 64 个码心正确转换。本报告所有头条数字均由本地
> ngspice 实跑得到(命令见 §5),非手算/外插。理想电容 + 理想 CDAC 驱动下静态 DNL/INL 恒为 0
> ——真正的线性度退化只来自失配,故 §4.4 的 MC 才是本设计静态精度的诚实评估。

## 2. 架构综合

### 2.1 CDAC:共模翻转的二进制加权电容阵列。为什么?

- **共模翻转(common-mode switching)**沿用 3-bit 示例的成熟方案:采样相把两侧顶板经 CMOS
  传输门接到 VCM、底板接输入差分;保持相底板回到共模基线,顶板浮起并携带 −Vin 的差分;每个 bit
  试探把被测电容底板在 P 侧推到 Vref、N 侧拉到 0,比较器判 TOPP vs TOPN 的符号。相较单端
  "充/放到 Vref"的经典结构,它天然差分、共模稳定、对参考馈通与开关电荷注入的一阶抵消更好。
- **单位电容 Cu = 2 fF**:兼顾 kT/C 与匹配。每边总电容 = (32+16+8+4+2+1+1)·Cu = 64·Cu = 128 fF
  (MSB 32Cu = 半阵列,dummy 1Cu 补足 2ⁿ·Cu 使 MSB 恰为半量程)。kT/C 噪声 √(kT/C) ≈
  √(4.14e-21/128f) ≈ **180 µV**,远小于 LSB/2 = 7.8 mV;Cu 取更小(如 1 fF)噪声升到 254 µV
  仍够,但 2 fF 给匹配留裕量(见 §4.4 的 σ_cu 假设)。
- **理想电容 + 理想底板驱动**:CDAC 电容为理想线性元件、底板由理想 PWL 电压源驱动,所以电荷重
  分布几乎瞬时(仅受比较器输入栅容影响),bit_period 主要由比较器再生时间决定,而非 CDAC 建立
  (见 §3 时序)。这是一处抽象——真实设计需研究开关 Ron·C 建立与底板驱动器非线性(§6 局限)。

### 2.2 判决抽取约束:harness 只在 `decision_time` 读一个节点电压

`run_sar_conversion` 的物理判决机制是:在 `decision_time = sample_end + (bit+1)·bit_period`
处对 `comparator_node` 的瞬态电压做线性插值,与一个**静态** `comparator_threshold` 比较,
`high_means_clear` 决定"高电平=清零还是保留"。而 `sar_input_waveforms` 只会生成采样/各 bit/
dummy 的 CDAC 控制波形;`render_freepdk45_transient_netlist` 的 `waveform()` 对**未生成的波形
键直接报错**——所以不能凭空在网表里引用一个 harness 不产生的时钟键。这就是设计动态比较器的核心
约束:要么用连续偏置的静态比较器(3-bit 示例的 5T),要么**扩展 machinery 让它生成时钟**。

### 2.3 比较器:时钟同步 StrongARM 动态锁存。为什么 + 如何满足约束?

选 **StrongARM 锁存**而非静态 5T:6-bit LSB 差分仅 15.6 mV,需要一个**无静态失调放大、再生增益
趋于无穷**的判决器。StrongARM 在评估相通过正反馈把任意小的输入差分再生到轨,nom 无失配时完全对称
→ 理论上可分辨任意小差分(实测码心处差分 ≥ 半 LSB,轻松分辨);它还是**动态**的——复位相零静态
电流,只在选通时耗电,契合 SAR 低功耗。

**如何在判决抽取约束下驱动它——新增 `adc.clock`(向后兼容的 machinery 扩展)**:

- 观察:harness 每个 bit 都从 t=0 **重放**整条瞬态,且只在**被试 bit** 的 `decision_time` 读一次
  比较器。所以我只需要一路**固定的、逐 bit 的选通模式**(与 `trial_index`、与已定判决无关),
  它在每个 `decision_time` 附近拉高即可服务所有重放。
- 实现:`_sar_config` 解析可选 `adc.clock` 块;`sar_input_waveforms` 生成一路选通波形键(`clk`):
  静息在 `low`(复位),在每个 `decision_time` 前 `eval_before` 拉高到 `high`(评估)、
  `decision_time` 后 `reset_hold` 复位。该键经 `transient_inputs` 驱动 StrongARM 的时钟尾管与
  输出/中间节点复位管的栅。
- **向后兼容**:无 `clock` 块 → `cfg["clock"]=None` → 不生成 `clk` 键。3-bit 静态示例不引用 `clk`,
  渲染出**字节级一致**的网表、码值不变(回归测试 `test_clock_config_present_for_sar6_absent_for_sar3`
  / `test_clock_waveform_absent_and_keys_stable_for_sar3` 固化)。
- 时序约束(schema + `_clock_config` 强制):`eval_before` 必须 `< bit_period/2 − edge_time`,
  保证选通拉高时被测电容早已切换、锁存器采到已建立的差分;`eval_before + reset_hold < bit_period`
  保证相邻选通不重叠。默认 `eval_before = 0.3·bit_period`、`reset_hold = 0.1·bit_period`。

**StrongARM 拓扑**(全 nMOS 输入、时钟尾管、pMOS 输出预充):

| 器件 | 角色 | d / g / s | 尺寸 W/L (µm) |
|---|---|---|---|
| MTAIL | 时钟尾管(nMOS) | TAIL / clk / GND | 8 / 0.2 |
| MIP / MIN | 输入对(nMOS) | DIP,DIN / TOPP,TOPN / TAIL | 6 / 0.2 |
| MLNP / MLNN | 交叉耦合锁存(nMOS) | OUTP,OUTN / OUTN,OUTP / DIP,DIN | 3 / 0.1 |
| MLPP / MLPN | 交叉耦合锁存(pMOS) | OUTP,OUTN / OUTN,OUTP / VDD | 4 / 0.1 |
| MRP1..4 | 复位预充(pMOS,栅=clk) | OUTP,OUTN,DIP,DIN / clk / VDD | 1.5 / 0.05 |

- **复位相(clk=low)**:MTAIL 关、MRP1..4 开 → OUTP/OUTN/DIP/DIN 全预充到 VDD,尾节点浮低,
  锁存无电流。**t=0 的 DC 工作点因此是确定的**(无正反馈简并态),ngspice `.op` 稳定收敛。
- **评估相(clk=high)**:MTAIL 开拉低 TAIL,输入对按 TOPP/TOPN 差分放电 DIP/DIN,交叉耦合再生,
  OUTP/OUTN 分裂到轨。
- **判决极性**:输入对 gate=TOPP/TOPN,当 TOPP>TOPN 时 MIP 放电更快、OUTP 先落,故 **OUTN 保持高**。
  读 `comparator_node = OUTN`、阈值 0.5、`high_means_clear=true` —— 与 3-bit 示例的 `vout` 极性
  一致(TOPP>TOPN ⟺ 输出高 ⟺ 清零),保证 SAR 环为负反馈、码随 Vin 单调。

## 3. 时序与尺寸

| 参数 | 值 | 理由 |
|---|---|---|
| sample_end | 5 ns | 采样相;理想开关/源,采样瞬时,取整数便于时序 |
| bit_period | 10 ns | 每 bit 一相;CDAC 瞬时建立,主要给比较器再生+选通留裕量 |
| edge_time | 0.2 ns | 控制沿/最大步长;与 3-bit 示例一致 |
| clk eval_before | 3 ns (=0.3·T) | `decision_time` 前 3 ns 拉高;此时被测电容早已切换(< T/2−edge) |
| clk reset_hold | 1 ns (=0.1·T) | 判决后 1 ns 复位,下一 bit 前重新预充 |
| 转换周期 | 5 + 7×10 = **75 ns** | sample_end + (n_bits+1)·bit_period → **13.3 MS/s** |

**比较器尺寸取向**:输入对 MIP/MIN 取 **W/L = 6/0.2**(面积 1.2 µm²)以压低失配失调——这是 6-bit
SAR 的主导误差源;尾管 8/0.2 供再生电流;锁存管中等尺寸求快再生;复位管小(开关)。nom 无失配时任何
合理尺寸都能分辨码心差分,尺寸主要服务 §4.4 的失配良率。CDAC 传输门开关沿用 3-bit 的 W/L=1/0.05。

## 4. 实测结果(全部由本地 ngspice 实跑,命令见 §5)

### 4.1 nom 码心静态扫描(全 64 码)

- 全部 64 个码心 → 正确码 `0..63`,**无缺码**;**max\|DNL\| = 0.000 LSB**,**max\|INL\| = 0.000 LSB**。
- 理想电容 + 理想 CDAC 驱动 + nom 无失配比较器 ⇒ 转移完美,DNL/INL 精确为 0。这既是设计正确性的
  证明,也说明真正的线性度退化只能来自失配(§4.4)或电容失配(未在 nom 体现)。

### 4.2 角扫描(ss / ff,全 64 码)

| 角 | 码心正确 | 缺码 | max\|DNL\| | max\|INL\| |
|---|---|---|---|---|
| nom | 64/64 | 0 | 0.000 | 0.000 |
| ss | 64/64 | 0 | 0.000 | 0.000 |
| ff | 64/64 | 0 | 0.000 | 0.000 |

StrongARM 在 ss/ff 下再生速度变化,但 3 ns 评估窗远大于再生时间常数(~ps 级),码心处差分 ≥ 半 LSB
被稳健分辨 → 全角零缺码。(这与 OTA 案例不同:那里角鲁棒性靠恒 gm 偏置;SAR 比较器是判决器、只需
符号正确,再生窗有巨大裕量,故对角不敏感。)

### 4.3 动态测试(相干正弦,FFT)

| 记录 | 幅度/偏置 | SNDR | SNR | SFDR | ENOB |
|---|---|---|---|---|---|
| 128 样本 / 13 周期 | 0.49·Vref / 0.5·Vref | **36.93 dB** | 37.27 dB | 44.07 dB | **5.84 bit** |

理想 6-bit 满量程 SNDR = 6.02·6 + 1.76 = 37.9 dB(ENOB 6.0)。本设计 −0.2 dB 幅度下量化极限
≈ 5.97 bit,实测 5.84 bit —— 本质是量化受限,接近理想 6-bit,差值来自码心采样与边缘码效应。

### 4.4 逐器件失配蒙特卡洛(n = 20)

**失配假设**(FreePDK45-ish,document 在 `adc.mismatch`):A_Vth ~ 3–5 mV·µm(45 nm 量级),
取参考面积 w0·l0 = 1·0.05 µm² 处 σ_vth0 = 5 mV(nMOS)/ 6 mV(pMOS),逐器件按 Pelgrom 面积律
`σ = σ_vth0 / √(W·L / (w0·l0))` 缩放——输入对 6/0.2 面积大,失调被平均;CDAC 单位电容相对失配
σ_cu = 1 %/√(C/Cu)(Cu=2 fF)。良率门限 ±0.5 LSB。

**(a) 计划失配(σ_vth0 = 5/6 mV nMOS/pMOS, σ_cu = 1 %, seed=1, n=20)**:

| 指标 | mean | std | worst |
|---|---|---|---|
| max\|DNL\| (LSB) | 0.000 | 0.000 | 0.000 |
| max\|INL\| (LSB) | 0.000 | 0.000 | 0.000 |
| 首次跳变失调 (LSB) | 0.000 | 0.000 | 0.000 |
| 缺码数 | 0.0 | — | 0.0 |

- 单调率 **100 %**,对 ±0.5 LSB 的**良率 = 100 %(20/20)**。**全零是诚实的裕量结果**,不是 MC 未
  生效:码心输入距每个判决边界半 LSB(7.8 mV 差分),而输入对(6/0.2,面积 1.2 µm²)的差分失调
  σ = √2·σ_vth0/√(W·L/(w0·l0)) = √2·5 mV/√24 ≈ **1.44 mV**,要翻转一个码心判决需 ~5.4σ;CDAC
  MSB 电容相对失配 σ = 1 %/√32 ≈ 0.18 %(权重误差 ~0.055 LSB)也远不足以跨过半 LSB 裕量。故在
  这组合理 σ 下设计有充裕裕量。**注意**:码心 MC 的粒度是半 LSB——它只捕捉大到能翻转码心判决的
  误差,是保守的良率下界,不等于亚 LSB 的 DNL 分布。

**(b) 应力失配(σ_vth0 = 30 mV, σ_cu = 4 %, seed=3, n=12)——验证 MC 确实响应并定位失效边界**:

| 指标 | mean | worst | 良率 / 单调率 |
|---|---|---|---|
| max\|DNL\| (LSB) | 0.92 | 2.0 | 良率 **16.7 %**(2/12) |
| max\|INL\| (LSB) | 1.00 | 2.0 | 单调率 100 % |
| 首次跳变失调 (LSB) | 0.83 | 2.0 | 缺码 mean 2.0 / worst 7 |

- 把 σ 推到 6× 后 DNL/INL/失调随之抬头(见表),证明 `delvto` 注入 + 电容扰动确实进入闭环判决。
- 主导误差:比较器输入对 Vth 失配 → 全局失调(首次跳变偏移)+ 逐码判决抖动;CDAC 电容失配 →
  MSB 附近 DNL 尖峰。要在应力 σ 下保良率需再放大输入对 / 加大电容(功耗/面积权衡)。

### 4.5 功耗 / 能量

- nom 码心扫描平均 **总功耗 ≈ 7.19 µW/次**(VDD 供电 + CDAC 底板/时钟切换的理想源功率之和)。
- 单次转换能量 = 7.19 µW × 75 ns ≈ **0.54 pJ/次**;FoM(能量/2^ENOB)≈ 0.54 pJ / 2^5.84 ≈
  **9.4 fJ/conv-step**。
- StrongARM 复位相零静态电流,功耗主要是每 bit 的一次再生 + CDAC 开关能量。

### 4.6 设计空间探索(可选流程件)

`examples/freepdk45_sar6_explore.json` 是一份可跑的探索配置:变量为比较器输入对 W(`W:MIP/MIN`,
3–8 µm)、LSB 单位电容组(`C:C0P/C0N`,1.6–2.4 fF)、采样开关 W(`W:MSNP..`,0.5–2 µm);约束
`missing_codes ≤ 0` + `monotonic ≥ 1`,目标 `min max|DNL|` + `min power_uw`,`sweep_points=64`
(全码心,DNL/缺码为真值;每候选 64 次转换,候选间由 `--workers` 并行)。

一次 `n=8, seed=0, workers=8` 实跑(~13 min,512 次转换):**8/8 收敛、8/8 可行、Pareto 1 个**
——全部候选 64 码全对、`max|DNL| = max|INL| = 0`(理想电容下改比较器 W/开关 W 不动转移曲线,
单位电容组只动 2×Cu/128 fF 的权重、不足以在码心粒度产生误差)⇒ Pareto 沿功耗轴展开:功耗随
比较器输入对 W 单调,最低 **6.58 µW / 0.49 pJ**(候选 #6:W=3.4 µm, Cu=2.19 fF, 开关
W=1.54 µm),最高 7.51 µW(W=7.5 µm)。设计取 W=6/0.2 而非最低功耗端,是把 ~0.6 µW 换给
§4.4 的失配裕量(输入对面积 ×1.8 → 失调 σ ÷1.3)——这个权衡正是 nom 探索看不见、只有失配
MC 能看见的轴。

## 5. 复现

```bash
# 当前默认原生转换需要 FreePDK45 卡和首次构建用 C 编译器；
# 复现本文历史 oracle 结果时还需要 ngspice
circuit-opt adc examples/freepdk45_sar6.json --vin 0.7109375        # native reference: code 44

# 全 64 码心静态扫描(并行);角用 --corner ss/ff
circuit-opt adc examples/freepdk45_sar6.json --sweep 64 --workers 8
circuit-opt adc examples/freepdk45_sar6.json --sweep 64 --workers 8 --corner ss

# 相干正弦动态(SNDR/ENOB/SFDR):128 样本、13 周期、0.49·Vref 幅度
circuit-opt adc examples/freepdk45_sar6.json --sine 128 --tone-bin 13 --amplitude 0.49 --workers 8

# 设计空间探索
circuit-opt adc examples/freepdk45_sar6.json --explore examples/freepdk45_sar6_explore.json -n 8 --workers 8
```

Python API(失配 MC 与本报告各表同源):

```python
import numpy as np
from circuitopt.circuit_loader import load_circuit_json
from circuitopt.sar import run_sar_sweep, run_sar_signal
from circuitopt.sar_mc import sar_mismatch_mc

spec = load_circuit_json("examples/freepdk45_sar6.json")
vin = (np.arange(64) + 0.5) / 64.0
sweep = run_sar_sweep(spec, vin, workers=8)                 # §4.1 nom；corner="ss"/"ff" → §4.2

phase = 2 * np.pi * 13 * np.arange(128) / 128               # §4.3 相干正弦
sine = np.clip(0.5 + 0.49 * np.sin(phase), 0.0, 1.0)
sig = run_sar_signal(spec, sine, 13.3e6, fundamental_bin=13, workers=8)

mc = sar_mismatch_mc(spec, n=20, seed=1, workers=8)         # §4.4(a)
mc_stress = sar_mismatch_mc(spec, n=12, seed=3, workers=8,  # §4.4(b)
    config={"sigma_vth0": 0.030, "sigma_vth0_pmos": 0.030, "sigma_cu": 0.04})
```

## 6. 局限与诚实边界

- **理想 CDAC 电容与理想底板驱动**:电容为线性理想元件、底板由理想 PWL 电压源驱动,故电荷重分布
  瞬时、nom 静态 DNL/INL 恒为 0。这隐藏了:开关 Ron·C 建立不足、底板驱动器有限输出摆率/非线性、
  参考 buffer 压降。真实 SAR 的 bit_period 需按 (n_bit·τ_settle) 定,本设计的 10 ns 是比较器/时序
  裕量而非建立限。
- **判决抽取抽象**:harness 在 `decision_time` 读一个连续节点电压 vs 静态阈值,**不建模比较器
  亚稳态**(极小差分下再生不完成)、**回踢噪声**(kickback:输入对栅经 Cgd 对 TOPP/TOPN 的电荷注入
  会扰动 CDAC 顶板,本抽象里比较器输入是理想栅、回踢仅体现为栅容,未研究其对相邻判决的影响)、
  以及**判决相互作用**。码心输入远离判决边界,故这些在本验收下不触发,但边界码/满速下需专门研究。
- **无 T/H 非线性研究**:采样开关为理想 CMOS 传输门(3-bit 同款),未扫 Ron 随输入的调制、
  电荷注入/时钟馈通对采样精度的影响(差分共模翻转对其一阶抵消,但二阶残差未量化)。
- **无瞬态噪声**:kT/C 与比较器噪声只做了解析估算(§2.1),`.tran` 不含随机噪声源,故 SNDR 是
  纯量化+失配极限,不含热噪声地板。
- **失配 MC 规模**:n=20 给趋势而非尾部良率;σ 假设为 45 nm 量级的合理取值而非某次流片的实测
  A_Vth,报告已显式记录假设(§4.4)。
- **数字控制理想**:SAR 逐次逼近逻辑在 Python 里,未做晶体管级的 SAR 寄存器/时序逻辑与其功耗。
