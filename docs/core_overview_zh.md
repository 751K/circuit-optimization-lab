# 核心求解器概览

[项目概览](README.md) | [中文说明](README_zh.md)

本文介绍当前 `core/` 求解器栈。代码是 AT4000TG OTFT ECG AFE 求解器的紧凑本地实现，已针对 Cadence/Spectre 行为进行校准。它是更广泛的本地电路优化流程的第一个具体后端。

## 覆盖范围

当前求解器栈覆盖：

- DC 工作点求解。
- AC 小信号增益与带宽分析。
- 噪声分析，包括闪烁噪声和热噪声。
- 瞬态响应仿真。
- 周期稳态（PSS）shooting 求解。
- PSS 辅助的 PAC（周期 AC），支持解析伴随谐波平衡（默认）、可选 time-domain
  Floquet shooting 加速路径，或有限差分 shooting。
- 谐波平衡 PNoise（周期噪声），含循环平稳噪声折叠。
- 工艺角与逐器件 mismatch 扰动。
- 面向 Cadence/Spectre 的验证，涵盖工作点、AC、噪声、瞬态、PSS、PAC 和 PNoise 行为。

实现刻意保持小而自包含。目前 `core/` 下有 22 个 Python 文件（含 `__init__.py`、CLI 入口 `__main__.py`、
校准/PSF/Cadence 网表辅助模块和主求解器栈）。

## 文件结构

```text
core/
  topology.py          电路拓扑单一事实来源。
  compiled_topology.py 运行态拓扑/index/stamp 元数据编译层。
  circuit_loader.py    JSON 电路描述加载器。
  device_model.py      TransistorModel ABC + NumbaParams + 模型工厂/注册表 + PDK/极性分层。
  pmos_tft_model.py    AT4000TG PMOS-OTFT 紧凑模型实现。
  numba_kernels.py     可选 Numba 加速标量内核。
  ac_mna.py            MNA stamp 原语。
  ac_solver.py         DC 工作点与 AC 小信号求解器。
  noise_solver.py      噪声传播与等价输入噪声分析。
  transient_solver.py  时域瞬态求解器。
  pss_solver.py        基于 transient shooting 的 PSS 求解器。
  pac_solver.py        通用 PSS 辅助 PAC 求解器。
  pnoise_solver.py     通用谐波平衡 PNoise 求解器。
  analysis_dispatch.py JSON 分析配置 dispatch 入口。
  psf.py               Spectre PSFASCII 参考数据解析器。
  calibration.py       本地结果与 Cadence 参考的校准/比较工具。
  cadence_netlist.py   用于验证的 Spectre 网表生成工具。
  chopper.py           理想与 PMOS 开关差分 chopper 分析。
  explore.py           设计空间探索 / 优化驱动。
  corners.py           工艺角、mismatch MC 与 latch 检测。
```

## 导入关系

```text
topology.py          <- 无内部依赖
compiled_topology.py <- 无内部依赖；运行时消费 Topology 风格对象
circuit_loader.py    <- topology
numba_kernels.py     <- 无内部依赖；运行时可选 numba
device_model.py      <- 无内部依赖（仅 abc、dataclasses）
pmos_tft_model.py    <- 可选 numba_kernels、device_model
ac_mna.py            <- 无内部依赖
ac_solver.py         <- topology, compiled_topology, ac_mna, device_model
noise_solver.py      <- ac_solver, compiled_topology, topology, ac_mna, device_model
transient_solver.py  <- ac_solver, compiled_topology, topology, device_model
pss_solver.py        <- ac_solver, ac_mna, device_model, topology, transient_solver
pac_solver.py        <- ac_mna, ac_solver, device_model, transient_solver
pnoise_solver.py     <- ac_solver, noise_solver, pac_solver, device_model, ac_mna
analysis_dispatch.py <- ac_solver, noise_solver, transient_solver, pss_solver, pac_solver, pnoise_solver, circuit_loader
psf.py               <- 无内部依赖
calibration.py       <- ac_solver, noise_solver, chopper, psf, circuit_loader
cadence_netlist.py   <- circuit_loader, topology
chopper.py           <- noise_solver, pss_solver, pac_solver, pnoise_solver, device_model, topology
explore.py           <- ac_solver, noise_solver, device_model, topology, circuit_loader
corners.py           <- ac_solver, noise_solver, topology
```

## 主要组件

### `pmos_tft_model.py`

实现了 AT4000TG PMOS-OTFT 紧凑模型的 Python 版本。提供：

- 通过 `get_Idc` 计算端电流。
- 通过 `get_noise_psd` 计算漏极电流噪声 PSD。
- 通过 `get_capacitances` 计算偏置相关的端电容。
- 通过 `g_area` 计算几何面积。
- 工艺和 mismatch 参数，如 `pvt0`、`mvt0`、`pbeta0` 和 `mbeta0`。
- 带热启动的内部节点工作点求解。
- 安装 Numba 时自动对热点标量内核启用加速；设置 `CIRCUIT_USE_NUMBA=0`
  可强制关闭；设置 `CIRCUIT_NUMBA_CACHE=0` 可关闭默认启用的磁盘 JIT 缓存。

AC 和噪声分析时，求解器通过有限差分 `get_Idc` 提取端 `gm` 和 `gds`，与电路求解器使用的端行为保持一致。

`PMOS_TFT` 继承 :class:`~device_model.TransistorModel`，即所有求解器消费的抽象基类。
它还提供 `get_numba_params()` 供瞬态求解器编译内循环使用，
以及 Numba 加速的 `get_ss_params()` 覆盖方法。

### `device_model.py`

定义抽象器件模型接口，将求解器与具体晶体管实现解耦：

- **`TransistorModel` (ABC)** — 七个抽象方法（`get_Idc`、`get_op`、`get_capacitances`、`get_capacitance_charges_from_op`、`get_capacitance_branch_terms_from_op`、`get_noise_psd`、`get_numba_params`）；`get_ss_params` 提供有限差分默认实现，子类可覆盖。
- **`NumbaParams` (frozen dataclass)** — 16 个标量参数，瞬态求解器每个器件提取一次，传入 Numba 加速内核。
- **`register_model()` / `create_device()` + PDK/极性分层** — 工厂 + 注册表。每个 `(pdk, polarity)` 以结构化键 `"<pdk>.<polarity>"`（如 `"at4000tg.pmos"`）注册；`register_pdk()` 把一个工艺的各极性归组并标记默认。求解器文件调用 `create_device(get_default_model_type(), …)`（单一切换点）而非硬编码模型名，新增工艺或 `nmos` 极性只需一次 `register_pdk`、不改任何求解器。`"pmos_tft"` 保留为向后兼容别名。通用元件（电阻/电容/理想 V/I/受控源）是与工艺无关的拓扑原语，**不在**此注册表中，故每个 PDK 零改动复用。

### `topology.py`

将电路拓扑定义为单一事实来源。拓扑包含晶体管列表、被求解节点列表、rail/bias 节点、输出、AC 输入驱动、负载电容、瞬态输入映射、DC 初值猜测和 DC 别名。求解器运行态元数据均从这个拓扑派生，而不是在各个求解器中分别手写。

除了晶体管之外，还承载无源/源元件——`resistors`（a-b，阻值 R 欧姆）、`capacitors`（a-b，容值 C 法拉）、`isources`（理想直流电流源，I 从 nplus 流向 nminus）、`vccs`（压控电流源：p、q、ctrl_p、ctrl_n、gm）、`vcvs`（压控电压源：p、q、cp、cn、mu → Vp−Vq=μ(Vcp−Vcn)）、`cccs`（流控电流源：p、q、ctrl_name、beta → Iout=β·Ictrl）、`ccvs`（流控电压源：p、q、ctrl_name、gamma → Vp−Vq=γ·Ictrl）和 `vsources`（理想电压源，真·MNA：p、q、value）。每个 vsource/VCVS/CCVS 新增一个支路电流未知量和一行约束，系统从 `n` 增长到 `n_aug = n + m`。这些通用于全部分析：电阻支路电流和电流源注入进入 DC KCL；电阻在 AC/噪声中按 `1/R` stamp，电容按 `jωC` stamp，VCCS 按 ``gm*(Vcp-Vcn)`` stamp，VCVS/CCVS/vsource 按 bordered ``[[Y,B],[B^T,0]]`` 块 stamp 含各自约束行，CCCS 按 KCL 行耦合 stamp；电阻贡献热噪声 `4kT/R`（所有受控源和理想电压源无噪声）；瞬态加入电导、电容伴随模型、恒定/VCCS/CCCS 源电流以及带约束方程的 VCVS/CCVS/vsource 支路电流。电流源在小信号 AC 系统中视为开路。CCCS 和 CCVS 支持级联：可控制任何 vsource/VCVS/CCVS 的支路电流。这些都不影响晶体管模型相关逻辑。

默认拓扑是 `AFE_TOPO`，一个 10 管全差分 AFE 核心，包含尾电流器件、输入对、输出级和交叉耦合正反馈电平移位器件。

### `compiled_topology.py`

从声明式 `Topology` 以及当前 bias/input 上下文构建运行态 plan。它会把节点名一次性解析成紧凑 terminal token，并为 DC、AC/噪声和 transient 暴露共享元数据：

- solved-node index 和 rail 数值；
- 每个器件的 drain/gate/source terminal token；
- 电阻、电容、电流源、VCCS、VCVS、CCCS 和 CCVS 的 stamp 元数据；
- AC/噪声使用的 `("n", idx)` / `("v", value)` 端表；
- transient input 与 `node_inputs` 映射。

这样 AC、noise 和 transient 使用同一套 indexing/stamping 约定，同时仍能保持 JSON 电路替换能力。

### `circuit_loader.py`

加载 JSON 电路描述并返回 `CircuitSpec`，包含：

- `topology`
- `sizes`
- `bias`
- `nf`

这使得可以通过 JSON 文件（如 `examples/single_stage.json`）添加新电路，而无需修改求解器源码。

### `numba_kernels.py`

为纯标量热点路径提供可选 Numba 内核。该模块可在未安装 Numba 时安全导入。安装
Numba 时默认自动启用；如需强制走纯 Python 路径，设置：

```bash
CIRCUIT_USE_NUMBA=0
```

Numba 编译结果默认会写入磁盘缓存，因此后续新的 Python 进程可以复用已编译内核，
避免再次支付完整冷启动 JIT 开销。如只想关闭缓存，设置：

```bash
CIRCUIT_NUMBA_CACHE=0
```

`core.explore` 和 `core.corners` 仍会默认把该变量设为 `1`，因为设计空间探索、
corner sweep 和 mismatch MC 都是长任务；普通 solver 路径现在也会在 Numba 可用时
自动使用加速内核。

目前加速路径包括 PMOS 电流计算、内部节点 Newton 迭代、偏置相关电容计算、
AC/PNoise 小信号参数端导数、PNoise HB block 组装和噪声折叠，
以及 transient Newton 内循环：拓扑 token 查值、PMOS 工作点求解、residual/Jacobian
stamp 和小规模稠密 Newton 线性求解。稠密 Newton 求解使用原地 `A*x = -R`
路径，避免每次迭代里不必要的数组拷贝。如果 compiled 路径处理不了某一步，
`transient_solver.py` 会回退到原 Python Newton / full-Jacobian / least-squares 路径。

### `ac_mna.py`

提供小信号求解器使用的底层 MNA stamp 原语：

- 导纳 stamp。
- VCCS、VCVS、CCCS、CCVS stamp。
- 理想电压源 stamp（bordered MNA）。
- MOS 小信号 stamp。

### `ac_solver.py`

求解 DC 工作点和 AC 响应：

- `ac_solve(sizes, bias, freqs, corner=None, x0_guess=None, topo=AFE_TOPO, nf=None)`
- 使用 `scipy.fsolve` 求解 DC 节点方程。
- 返回增益、带宽、节点工作点以及提取的小信号参数。
- 同时支持全局工艺角和逐器件 mismatch 映射。
- 使用拓扑元数据确定输出、负载电容和 AC 输入驱动。

DC 求解包含物理支路选择、对称工作点和 rail 有界节点解的鲁棒性处理。

### `noise_solver.py`

在与 AC 分析相同的拓扑派生 MNA 系统上执行噪声传播。每个晶体管漏极电流噪声源注入到漏源之间，传播到配置的输出，并除以信号增益得到等价输入噪声。

噪声流程支持与 AC 求解器相同的拓扑派生端映射和 corner/mismatch 参数传递。
当 JSON dispatch 中 AC 与 noise 使用同一频点网格时，noise 会直接复用前面
AC 的 `dc_op`、小信号参数和增益，避免重复 DC/AC 求解；频点不同则至少复用
`dc_op` 作为 warm seed。

### `chopper.py`

计算 AFE 周围不同 chopper 版本的 gain、带宽和基带噪声：

- `chopper_analysis(...)` 是理想同步差分 chopper 模型，把八开关换向器看作输入
  和输出端的 +/-1 方波乘法器，再用奇次谐波系数把边带 gain/noise 折回基带。
  这是描述理想 chopping 与 flicker noise 搬移的 LPTV 频域路径。
- `build_afe_pmos_chopper(...)` 会在 AFE 输入/输出端口周围插入 8 个真实
  `PMOS_TFT` pass switch。
- `pmos_chopper_analysis(...)` 对这个 PMOS 开关拓扑分别运行静态 A/B 相 AC
  和 noise，并对两相平均；结果包含 switch Ron 负载、非线性电容和 PMOS
  switch 自身噪声。
- `finite_edge_clock_pair(...)` 与 `finite_edge_chopper_harmonics(...)` 建模有限
  clock edge 和 break-before-make dead time 对 chopper 谱线权重的影响。
- `pmos_chopper_lptv_analysis(...)` 用这些有限边沿谐波权重折叠 PMOS-switch
  sideband response/noise。这是一个快速的**一阶** quasi-static 估计，会低估基带
  增益约 10%(漏掉了高阶 LPTV 变换)。要 Cadence 级的增益/噪声请用无经验常数的
  谐波平衡路径(`pmos_chopper_pss` → `pmos_chopper_pac`/`pmos_chopper_pnoise`)。
  原来那两个 Cadence 拟合常数(换向相位/噪声 PSD scale)已retire;
  `conversion_phase_rad`/`periodic_noise_psd_scale` 仍保留为手动可调参数。
- `pmos_chopper_transient(...)` 用有限边沿 clock 驱动八 PMOS 拓扑。默认 clock
  采用 Spectre `type=pulse` 语义（`delay=T/2`、`width=T/2`、有限 `rise/fall`）；
  旧的居中相位波形仍可通过 `clock_style="phase"` 使用，适合 dead-time 实验。
  clock feedthrough 来自 PDK `Cgss/Cgdd * ddt()` 项以及 PDK Verilog-A 中长期有效的
  `R_cap2` gate-leak 分支，均由 transient solver stamp；可选 charge injection
  脉冲由同一套 PDK 电容公式估算，并作为时变电流源注入。这个 helper 会在 clock
  边沿附近自动加密内部时间网格，对 8 个双向 pass switch 使用 signed terminal
  current，并收紧残差容差，避免慢 common-mode 电荷平衡被忽略。
- `pmos_chopper_pss(...)` 把同一 hard-switched 八 PMOS 拓扑接入通用 shooting
  PSS 求解器，返回给定 clock 周期上的周期稳态轨道。这是后续原生 PAC/PNoise
  的本地工作点基础。该 wrapper 默认使用 `cap_mode="average"` 生成轨道，即
  `0.5*(C_n+C_{n-1})*dV` 的梯形电容离散，用来匹配 Cadence 在 chopper 高阻内部节点上的
  commutation feedthrough；通用 transient/PSS 仍默认使用电荷守恒 Q-stamp。
- `pmos_chopper_pac(...)` 是 chopper 兼容包装器，内部调用通用
  `core.pac_solver.pac_solve(...)`。通用 solver 仍默认使用解析伴随 HB；chopper
  wrapper 默认使用 time-domain Floquet PAC：在时域构建一次单周期 monodromy，
  然后每频点求一个小的 quasi-periodic 边界系统（`method="pss_time_domain"`）。
  对 PMOS_TFT 周期转换，它会保留每个器件的内部 `gate1` 小信号状态（`R_cap`、
  `R_cap2`、`Cgs`、`Cgd`），不再逐时刻塌缩成端口 `{gm,gds,Cgs,Cgd}`。
  周期转换线性化使用 Spectre PAC 折叠的 Verilog-A 风格 `C(V)*ddt(V)` 算子，
  不必等同于生成大信号 PSS 轨道时的 transient companion。Numba 可用且所有 PMOS
  都暴露 `gate1` 网络时，gate1 扩维 PAC 线性化由编译版
  `pac_linearize_orbit_gate1` 内核装配；混合拓扑回退到 Python 装配。设置
  `time_domain=False` 可跑解析伴随 HB 对照路径；设置 `analytic=False` 可回退到原有限差分
  shooting 路径。静态 PSS 会自动退化为普通 `ac_solve` fast path，不跑 PAC transient。
- `pmos_chopper_pnoise(...)` 是 chopper 兼容包装器，内部调用通用
  `core.pnoise_solver.pnoise_solve(...)`。通用 PNoise 在 PSS 轨道上使用谐波平衡：
  沿轨道 N 点采样 → 时变小信号 G(t)/C(t) → FFT 到频域 → 组装 `nb×nb`
  块矩阵 `Y[kr,kc] = G_{kr-kc} + jω·C_{kr-kc}` → 每基带频率一次伴随求解得到
  传递阻抗 Z_{j,k} → 循环平稳器件/电阻噪声折叠 `S_out = Σ_j Σ_k |Z_{j,k}|² S_j`。
  与 `pmos_chopper_lptv_analysis` 不同，它无需 Cadence 标定常数，已是第一性原理解。

PMOS-switch sideband 路径最初使用 `pmos_chopper_lptv_analysis` 配合 Cadence 标定
常数验证通过。原生 `pmos_chopper_pac` 和 `pmos_chopper_pnoise` 现在已替代这些依赖
标定的路径，提供第一性原理的周期小信号和噪声求解。瞬态 finite-edge 路径已与
Spectre `tran` 对齐。对 D3 / `chop_tb_d3` 官方 `slow` corner PSS/PAC/PNoise
参考，默认 time-domain PAC 的 baseband 和 200 Hz 增益误差均 <1%。原生 PNoise
也使用同一套扩展 PMOS `gate1` 的 HB 转换模型，仍是第一性原理求解。

### `pss_solver.py`

在现有 transient 引擎之上做 shooting PSS：

- `pss_solve(sizes, bias, period, topo=..., tgrid=..., inputs=..., node_inputs=...)`
- 用 `transient(...)` 积分一个周期，并求解 `x(T)-x(0)=0`。
- 默认用 DC 工作点作为初值；可先跑若干 stabilization 周期再进入 shooting。
- **Shooting Jacobian：**默认（`analytic_jacobian=True`）直接在收敛轨迹上一次性
  遍历构建：采样每步的小信号 G(t)/C(t) stamp，按后向欧拉离散化形成
  A_m = (G_m + C_m/h)^{-1} · (C_m/h)，累积 monodromy 矩阵 Φ = ∏ A_m，
  得到 Jacobian = Φ - I — O(1) 遍历替代 `n_state` 次有限差分瞬态。失败时
  自动回退到有限差分。设置 `analytic_jacobian=False` 可强制使用原有限差分路径。
- 首轮 Jacobian 构建后（无论解析还是 FD），默认用 Broyden secant update 复用
  Jacobian，减少多轮 shooting 时的重复构建。每一步仍用真实一周期 transient
  重新计算 residual，精度判据不变；可用 `jacobian_reuse=False` 恢复每轮重建，
  或设置 `jacobian_rebuild_interval` 周期性重建。
- 返回一周期轨迹、`x0`、`x_end`、残差向量/范数、收敛标志和迭代历史。
  结果中还包含 `shooting_period_runs`、`shooting_jacobian_evals` 和
  `shooting_jacobian_reuses` 等性能计数器。历史记录中可查看所用的 Jacobian
  类型（`"analytic_monodromy"` 或 `"finite_difference"`）。
- PMOS chopper wrapper 默认使用 Numba-grid transient 路径
  （`fallback_least_squares=False`），并在为 PAC/PNoise 内部构造 PSS 轨道时
  使用 1 个 stabilization 周期和 chopper 专用 `cap_mode="average"` 轨道。这样
  residual/nfail 收敛判据不变，同时避免每个周期退回 Python fallback 重跑。
- chopper PSS 自动初值会缓存同一 bare AFE 尺寸/bias/corner 的 DC seed；重复分析
  同一候选时只复用初始猜测，仍会重新跑真实 shooting residual，因此不改变精度判据。

这一步输出的 PSS 周期轨道可以直接供通用 `pac_solve` 和 `pnoise_solve` 使用；
`pmos_chopper_pac` / `pmos_chopper_pnoise` 只是把 chopper 的差分输入 drive
映射成通用求解器需要的 `input_drive={"vip": 0.5, "vin": -0.5}`。

### `pac_solver.py`

- `pac_solve(sizes, bias, freqs, pss_result=..., input_drive=...)`
- 与具体电路无关，只要求 PSS 结果包含 `topology`、`t`、`nodes`、`x0`、`x_end`、
  `output` 以及周期输入波形元数据。
- `input_drive` 是小信号复幅值映射，例如差分输入为 `{"vip": 0.5, "vin": -0.5}`，
  单端输入为 `{"vin": 1.0}`。
- 四条性能路径，按优先级依次尝试：
  1. **LTI fast path** — 静态 PSS 轨道直接退化为普通 `ac_solve`。
  2. **Time-domain Floquet PAC**（`time_domain=True`；chopper wrapper 默认）— 在均匀轨道网格上
     采样周期性 G(t)/C(t) 和输入耦合，先构建一次与频率无关的 monodromy，
     再对每个频点求 `(exp(jωT)I - Ψ)x0 = g`。它避免 HB 边带截断和大型
     `(2K+1)n` 转换矩阵。PMOS_TFT 器件会在周期转换中扩展内部 `gate1`
     小信号状态；全 PMOS gate1 拓扑走 Numba 装配内核。不支持 mixed/bordered/vsource
     驱动时返回 `None` 并继续尝试下一条路径。
  3. **解析伴随**（通用默认，`analytic=True`）— 沿 PSS 轨道采样周期性 G(t)/C(t)
     和输入耦合列 G_in(t)/C_in(t)，FFT 到谐波系数，构建谐波平衡转换矩阵
     Y_HB(f)，每频率一次伴随线性求解得到 sideband-0 增益。O(1) 求解，零额外
     瞬态运行。由 `n_period_samples`（时域分辨率）和 `max_sideband`（边带数）控制。
  4. **有限差分 shooting**（`analytic=False`）— 有限差分状态转移矩阵 Φ 和
     复输入扰动，求解 `(Φ-γI)dx0=-b`。每频点需 `n_state+2` 次瞬态运行。
     结果缓存在 `pss_result` 上供重复调用。
- 结果中包含 `pac_period_runs`、`pac_state_cache_hit`、`pac_input_cache_hits`
  和 `pac_td_setup_time_s` 等计数器，以及 `method` 字段
  （`"pss_time_domain"`、`"pss_analytic_adjoint"` 或 `"pss_fd_shooting"`）。
  PAC condition 诊断默认关闭，只会在 `profile=True`、`debug=True` 或显式
  `compute_condition=True` 时启用；该诊断每频点做一次 SVD，不影响 gain/BW/noise。

### `pnoise_solver.py`

- `pnoise_solve(sizes, bias, freqs, pss_result=..., fundamental=...)`
- 基于通用 `Topology` 的器件、电阻、电容 stamp；PMOS 器件噪声沿 PSS 轨道采样，
  电阻热噪声按 stationary source 折叠。
- 静态 PSS 轨道会直接走普通 `noise_analysis` 的 LTI fast path；真正的 LPTV
  路径会把采样 `G(t)/C(t)`、HB block 和相同频点的 adjoint 解缓存到
  `pss_result`。
- HB adjoint 求解支持 `hb_solver="auto" | "dense" | "sparse" | "iterative"`。
  默认小矩阵继续走 dense BLAS/LAPACK；HB 规模变大且非常稀疏时切到 SciPy sparse
  direct。强制 `iterative` 时使用按谐波对角块 LU 的 block-Jacobi 预条件 GMRES，
  若不收敛会回退 sparse direct，避免损失精度。
- Numba 可用时，大规模 LPTV PNoise 会使用编译版 HB block 组装和
  `freq × source × sideband²` 噪声折叠；`get_ss_params()` 也会用编译端导数
  计算 gm/gds，在小电流或 kink 附近自动回退原有限差分。全 PMOS `gate1` PAC
  转换装配也会在可用时走编译内核。Numba/Rust 这类编译实现主要能加速矩阵填充和
  噪声折叠循环；HB 线性求解本身主要由 BLAS/LAPACK、SuperLU 或 GMRES 决定，
  不是 Python 循环开销主导。
- 如果未显式传入 `gains` 或 `pac_result`，可传入同一套 `input_drive`，函数会调用
  通用 `pac_solve` 得到输入参考所需的 PAC 增益。

### `transient_solver.py`

使用后向欧拉（默认）或变步长 BDF2/gear2 积分求解拓扑定义系统的时域响应：

- `transient(sizes, bias, tgrid, vip=None, vin=None, nf=None, V0=None, topo=AFE_TOPO, inputs=None, node_inputs=None, integration_method="be")`
- 支持传统的 AFE `vip/vin` 输入，也支持通过 `topo.transient_inputs` 驱动的通用 `inputs={name: waveform}`。
- `node_inputs={node: input_key}` 在某个（rail）节点上驱动波形——用于前端 testbench，其激励在源节点注入并通过无源网络传播，而非直接驱动器件栅极。
- `current_inputs=[{"p": node_a, "q": node_b, "input": key}]` stamp 一个时变
  理想电流源，方向为 `p -> q`；PMOS chopper helper 用它注入 charge-injection 脉冲。
- `max_step`、`max_retry_subdivisions`、`fallback_full_jacobian` 和
  `fallback_least_squares` 用于 switched
  transient 步的受控细分和有界 fallback 求解。
- 包含拓扑定义的负载电容（及电容元件），加上电阻和理想电流源支路。
- 在牛顿迭代期间重新计算非线性电容，并包含 PDK Verilog-A 使用的 PMOS
  `R_cap2` 源/漏到 gate 的泄漏支路。
- 支持 `signed_devices`，用于双向 pass switch。默认 AFE 路径保持与已校准
  DC/AC/noise 求解器一致的 `abs(Idc)` 约定；开关器件则可在源漏电压反向时保留
  物理 drain-current 符号。
- 使用来自 `ac_solve` 的 DC 工作点作为默认初始条件。
- Numba 可用时使用 transient Newton compiled kernel。该路径在一个内循环里完成
  PMOS 工作点/电容计算、residual/Jacobian stamp 和稠密 Newton step 求解。
  线性求解会原地覆盖临时 Jacobian/residual 数组；Python substep loop 会复用
  上一段已插值的输入作为下一段起点，避免重复插值。
- 对 PSS 常用的非 robust 模式（`fallback_least_squares=False` 且
  `fallback_full_jacobian=False`），compiled grid solver 会留在 Numba 内跑完整周期。
  失败 substep 会记为失败 interval，并从最后接受的状态继续，这与非抛错 Python
  transient 行为一致，但不会把整个周期退回 Python 重跑。
- robust fallback 模式仍会回到 Python 路径，以便只在明确请求时使用 least-squares
  或完整有限差分 Jacobian 恢复。
- 使用 PMOS 内部节点的隐式微分加快瞬态 Jacobian 计算，并回退到有限差分。

**Gear2/BDF2 积分**（`integration_method="gear2"`）：瞬态求解器也支持变步长 BDF2（二阶、刚性稳定）。关键特性：

- 使用稳定的 charge 模式电容伴随模型 `i_n = (α0·Q_n + α1·Q_{n-1} +
  α2·Q_{n-2})/h_n`，与 BE 的 `(Q_n − Q_{n-1})/h` 相同形式但使用两步历史。
  另有 per-call 电容模式 override；PMOS chopper PSS wrapper 用 `average` 算子匹配
  Spectre feedthrough，通用 stiff 电路仍保持 `charge` 默认。
- 步长比限制 ρ≤2 保证非均匀网格上的零稳定性。
- 每个 interval 第一步用 BE 自启动。
- 编译版 Numba gear2 grid 求解器（`_transient_solve_grid_gear2_impl`）处理
  PSS/PAC/PNoise 的周期轨道，也处理 raw transient 的 `max_step`、`flat_max_step`
  和 `max_retry_subdivisions`；解析 gear2 monodromy（增广 2n 态）供给 PSS shooting Jacobian。
- 裸 `transient(integration_method="gear2")` 仍是显式 opt-in；当调用方请求
  `max_retry_subdivisions` 或 `max_step` 的 robust 行为时，会留在 Numba gear2 grid；
  grid 在每个 accepted internal substep 后更新 rolling 两步 BDF2 历史，并在失败时按
  `2**max_retry_subdivisions` 做固定二分 retry。Python gear2 `solve_chunk` 只保留为
  Numba 拒绝 robust step 时的兜底。
- Chopper PSS/PAC/PNoise 默认使用 gear2——PAC baseband 误差从 BE 的 −2.5%
  （typ/fast）改善到三 corner 全部 <1%。
- 裸 `transient()` 默认仍保留 BE，以保持既有 raw transient 回归和一阶阻尼语义；
  默认 BE hard-switched chopper transient 的热路径也已全 Numba 化，正常不再触发
  Python tail 或 SciPy `least_squares`。

### 前端激励（`ac_drives`）

对于 testbench，小信号 AC 激励可以通过 `Topology.ac_drives`（如 `{"VINP": +0.5, "VINN": -0.5}`）施加在节点上，而非器件栅极。驱动通过前端无源网络传播到（现在作为被求解节点的）放大器输入端，增益按差分激励归一化。噪声分析中这些驱动被视为 AC 地（输入端无信号）。`examples/afe_testbench.py` 在 AFE 核心之前构造了干电极 + AC 耦合前端（R_EL∥C_EL、C_AC 串联、R_AC 到 VCM），并运行 AC（带通约 0.05 Hz–几百 Hz）、等价输入噪声（含 R_EL/R_AC 热噪声）和带内瞬态。由于 AC 耦合输入使裸 AFE DC 多稳态，testbench 从鲁棒的裸 AFE 工作点（`dc_seed`）作为种子启动 DC 求解。

### `explore.py`

建立在 AC 和噪声求解器之上的设计空间探索/优化驱动——即项目名称所指的"优化"。给定一个电路及 `explore` 配置（带范围的设计变量、可行性约束和一个或多个目标），它对候选方案进行采样，通过求解器评估每个候选，按约束过滤，并 Pareto 选择权衡前沿。

- `explore(topo, base_sizes, base_bias, nf, cfg, n=, seed=, method=, corner=)`——运行一次扫描。
  `corner` 对每次评估施加工艺偏移（如 `CORNERS["slow"]`），实现在不修改配置的情况下进行 corner 感知搜索。
- `evaluate(topo, sizes, bias, nf, freqs, band, x0_guess=None, corner=None)`——单候选求解器评估，
  新增可选的 corner/mismatch 参数。在 `explore` 中评价流程为 AC-first：先计算
  gain/BW/power/area，非噪声约束失败的候选会立即淘汰；只有幸存候选的约束或目标
  需要 `irn_uV` 时才运行 `noise_analysis`。
- `load_explore_json(path)`——从完整电路 JSON 中读取 `explore` 块。拓扑、器件尺寸、
  偏置和可选 NF 都走同一条 JSON 路径；探索层不再接受旧的 `builtin_topology` 配置。
- 采样方式为 `lhs`（拉丁超立方）或 `random`，使用带种子的 RNG 保证可重复性。
- 指标：`gain_dB`、`bw_Hz`、`irn_uV`、`power_uW`（顶 rail 供电电流 × rail 电压）和 `area`（各器件 `g_area` 之和）。
- 变量的 `targets` 可以同时驱动多个键值，保持匹配对（M7=M8, …）一致，使 AFE 的对称 DC 续流保持在物理支路上。
- 结果导出为 CSV 和 JSONL；CLI 运行 `python -m core.explore <config.json>`。

示例配置：`examples/afe_explore.json` 和 `examples/single_stage.json`，二者都是带
`explore` 块的完整电路 JSON。

### `corners.py`

工艺角和鲁棒性工作的单一事实来源——这些内容原本会在每次扫描中重复推导：

- `CORNERS`——全局工艺偏移（`typical` / `slow` / `fast`，按 `pvt0`/`pbeta0` 表示），来源于 PDK 的 monte.scs 段落；如 slow = `{"pvt0": -0.2259, "pbeta0": -0.54}`。
- `mismatch_corner(rng, devices, base)`——在工艺角基础上叠加逐器件随机 `mvt0`/`mbeta0`。
- `metrics(...)`——单设计单 corner → `gain_peak_dB`、`bw_Hz`、`irn_uV` 和 `latch_dV`（DC 工作点的 `|out+ - out-|`；大值 ⇒ 交叉耦合正反馈已 latch）。
- `corner_table(...)`——typ/slow/fast 三个 corner 的指标汇总。
- `latch_screen(...)`——确定性最坏情况 latch 筛查：对每个对称对在所有符号组合上施加 ±kσ 推开，返回最大输出失衡。单次固定 kick 存在假阴性（latch 的符号模式依赖于设计），因此筛查遍历所有模式；计算开销足够低，可在搜索内部代替完整 MC 使用。它只需要 DC/AC 工作点和 latch 失衡，因此会跳过噪声。
- `mismatch_mc(...)`——单个 corner 上的逐器件 mismatch MC，从名义工作点播种；返回各指标数组、latch 掩码以及汇总（latch 率 + 未 latch 样本的 mean/std/P5/P95）。每个样本先跑 AC/latch，只有进入最终噪声统计的未 latch 样本才计算 IRN。

`ac_solve` / `noise_analysis` 接受相同的 `corner` 参数（扁平的工艺 dict 或逐器件 mismatch 映射）。驱动脚本 `examples/mc_mismatch.py` 将其封装为 corner 表 + 3-corner MC 图。

## 快速示例

```python
import numpy as np

from core.ac_solver import ac_solve
from core.noise_solver import noise_analysis, band_rms
from core.transient_solver import transient

sizes = {
    "M6": (2264, 78),
    "M7": (61365, 61),
    "M8": (61365, 61),
    "M9": (3175, 468),
    "M10": (3175, 468),
    "M11": (465, 66),
    "M12": (894, 85),
    "M13": (894, 85),
    "M14": (5224, 46),
    "M15": (5224, 46),
}

bias = {
    "VDD": 40.0,
    "VCM": 30.65,
    "VB": 9.84,
    "VC": 16.0,
}

freqs = np.logspace(-2, 4, 121)

ac = ac_solve(sizes, bias, freqs)
noise = noise_analysis(sizes, bias, freqs)
irn_uv = band_rms(freqs, noise["irn_psd"], 0.05, 100) * 1e6

t = np.linspace(0, 4e-3, 400)
vip = np.where(t >= 0.5e-3, bias["VCM"] + 0.5e-3, bias["VCM"])
vin = np.where(t >= 0.5e-3, bias["VCM"] - 0.5e-3, bias["VCM"])
tran = transient(sizes, bias, t, vip, vin)
```

## JSON 电路示例

新电路可以从 JSON 加载。字段级格式见 [JSON 电路描述格式](json_circuit_format_zh.md)。

```python
import numpy as np

from core.circuit_loader import load_circuit_json
from core.ac_solver import ac_solve
from core.transient_solver import transient

spec = load_circuit_json("examples/single_stage.json")
freqs = np.logspace(0, 4, 121)

ac = ac_solve(spec.sizes, spec.bias, freqs, topo=spec.topology, nf=spec.nf)

t = np.linspace(0, 1e-3, 100)
vin = np.full_like(t, spec.bias["VIN"])
tran = transient(spec.sizes, spec.bias, t, topo=spec.topology,
                 nf=spec.nf, inputs={"vin": vin})
```

## 基准测试

`benchmarks/` 下有四个基准：

```bash
# 全 AFE 基准（ac121 / noise121 / tran200）
python3 -m benchmarks.bench_afe --warm-runs 3
CIRCUIT_USE_NUMBA=0 python3 -m benchmarks.bench_afe --warm-runs 3

# 单管 PMOS_TFT 微基准（7 个热路径操作 × 3 个偏置工作区）
python3 -m benchmarks.bench_model --warm-runs 3
CIRCUIT_USE_NUMBA=0 python3 -m benchmarks.bench_model --warm-runs 3

# Chopper 分析基准（harmonics / ideal / pmos_static / pmos_lptv / pmos_tran）
python3 -m benchmarks.bench_chopper --warm-runs 3
python3 -m benchmarks.bench_chopper --skip-tran --warm-runs 3

# 批量 sweep 基准（N × AC / AC+noise，模拟 explore 层负载）
python3 -m benchmarks.bench_sweep --n-candidates 200 --warm-runs 3
```

`bench_afe.py` 报告三种全 AFE 负载的 cold/warm 耗时。`bench_model.py` 测量单管
操作的性能（DC OP、Idc、电容、噪声 PSD、Cadence 指标），覆盖饱和区、亚阈值区和
线性区三个偏置点。`bench_chopper.py` 覆盖五个 chopper 分析层级，使用 f_chop=225 Hz
——从快速的有限边沿谐波计算（~1 ms），到理想 LPTV 折叠、PMOS 静态相位、
准静态 PMOS 边带折叠，以及最重的 hard-switched PMOS chopper 瞬态。`bench_sweep.py`
测量 N 个随机扰动候选的 AC / AC+noise 批量吞吐量，模拟 explore 层的逐候选
评估负载。默认运行在 Numba 可用时启用加速；`CIRCUIT_USE_NUMBA=0` 可用于纯 Python
对比。

旧 UI chopper 全流程瓶颈是通用 HB PAC frequency solve：显式
`PSS+PAC(HB)+PNoise`（`time_domain=False`）61 点约 25.6s（PSS≈0.35s、
PAC≈24.7s、PNoise≈0.55s），121 点约 48.9s（PSS≈0.44s、PAC≈47.6s、
PNoise≈0.93s）。当前默认 chopper time-domain PAC 保留 PMOS `gate1` 状态，
Numba 可用时使用 gate1 转换装配内核；同一 PSS 轨道上 61 点约 1.4s。非 chopper AFE 的 `DC+AC+Noise` 121 点在复用
AC 结果时约 1.8ms。

## 校准状态

当前核心已针对 AT4000TG AFE 用例在 Cadence Spectre 24.1 上完成校准。原始项目中观察到的吻合度包括：

- 典型和 corner AC 行为增益误差约 0.01 dB 以内。
- 已验证场景中等价输入噪声误差在百分之几以内。
- 逐器件 mismatch Monte Carlo 的均值和标准差与 Cadence 趋势一致。
- 瞬态阶跃和正弦响应与 Cadence `tran` 行为高度吻合。
- PMOS 八开关 chopper finite-edge transient 已按 UI 锁定尺寸、`f_chop=225 Hz`、
  switch `W/L=5000/30`、`rise/fall=20 us` 与 Spectre `tran` 对齐；默认
  `edge_time/10` 内部步长下，最后一周期输出均值约 `-10.76 mV`
  （Spectre `-10.62 mV`），输出 `21.11 mVpp`（Spectre `21.46 mVpp`），
  输入 common-mode 摆幅 `5.14 Vpp`（Spectre `5.43 Vpp`），且 `nfail=0`。
- PMOS 八开关 chopper PSS/PAC/PNoise 已按同一 UI 锁定案例与
  `pmos_chopper_lptv_analysis(...)` 对齐：gain `21.370 dB` 对 Spectre
  `21.369 dB`，带宽 `738.6 Hz` 对 `721.9 Hz`，IRN `12.592 µVrms` 对
  `12.591 µVrms`。
- 原生 `pmos_chopper_pac` 和 `pmos_chopper_pnoise`（第一性原理，无标定常数）已
  与 D3 / `chop_tb_d3` 官方 `slow` corner Spectre PSS/PAC/PNoise 对齐：
  `f_chop=200 Hz` 时默认 time-domain PAC 的 baseband 和 200 Hz 增益误差均 <1%。
  PNoise 仍是第一性原理 HB，并已使用同一套扩展 PMOS `gate1` 转换模型。
- 最终锁定设计约 22.9 dB 增益、549 Hz 带宽、37 µVrms 等价输入噪声。

上述数据描述当前的 AT4000TG 验证案例。后续 PDK 或拓扑应针对其各自的仿真器参考重新进行校准。
