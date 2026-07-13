# 核心求解器概览

[项目概览](README.md) | [中文说明](README_zh.md)

本文介绍当前 `circuitopt/` 求解器栈。代码是 AT4000TG OTFT ECG AFE 求解器的紧凑本地实现，已针对 Cadence/Spectre 行为进行校准。它是更广泛的本地电路优化流程的第一个具体后端。

## 覆盖范围

当前求解器栈覆盖：

- DC 工作点求解。
- AC 小信号增益与带宽分析。
- 噪声分析，包括闪烁噪声和热噪声。
- 瞬态响应仿真。
- 周期稳态（PSS）shooting 求解。
- PSS 辅助的 PAC（周期 AC），支持解析伴随谐波平衡（默认）、可选 time-domain
  Floquet shooting 加速路径，或有限差分 shooting。
- 周期 PNoise，支持谐波平衡路径和 time-domain Floquet-adjoint 路径，并含循环平稳噪声折叠。
- 工艺角与逐器件 mismatch 扰动。
- 面向 Cadence/Spectre 的验证，涵盖工作点、AC、噪声、瞬态、PSS、PAC 和 PNoise 行为。

实现刻意保持小而自包含，包含 `__init__.py`、CLI 入口 `__main__.py`、校准/PSF/Cadence 网表辅助模块、共享诊断/profiling
模块、主求解器栈、一套 ML surrogate 层（数据集构建、surrogate 训练、筛选-校验优化器）、接入同一套
`TransistorModel` 接口的三个硅 PDK——SKY130（OpenVAF/OSDI）、FreePDK45（ngspice-C）与
TSMC28HPC+（可迁移 ngspice 工艺适配器）——
以及架在整个栈之上的可选本地 HTTP 服务层（`circuitopt/service/`）。

## 文件结构

```text
circuitopt/
  topology.py          电路拓扑单一事实来源。
  compiled_topology.py 运行态拓扑/index/stamp 元数据编译层。
  circuit_loader.py    JSON 电路描述加载器。
  device_model.py      TransistorModel ABC + NumbaParams + 模型工厂/注册表 + PDK/极性分层。
  device_factory.py    器件构建/解析层（build_devices、get_ss_params）+ corner 路由（OTFT CORNERS、
                        硅工艺 apply_silicon_corner）。leaf 模块：只依赖 device_model。
  pmos_tft_model.py    AT4000TG PMOS-OTFT 紧凑模型实现。
  numba_kernels.py     可选 Numba 加速标量内核。
  ac_mna.py            MNA stamp 原语。
  ac_solver.py         纯 AC 小信号求解器：DC 工作点 + AC 响应（ac_solve）。
  dc_solver.py         DC 求解 fallback（有界最小二乘）+ AFE 专用对称 DC seeding/续流启发式。
  noise_solver.py      噪声传播与等价输入噪声分析。
  transient_solver.py  时域瞬态求解器。
  transient_profile.py 瞬态/chopper 求解器共享分析计数器槽位。
  pss_solver.py        基于 transient shooting 的 PSS 求解器。
  pac_solver.py        通用 PSS 辅助 PAC 求解器。
  pnoise_solver.py     通用 PNoise 求解器（HB + TD adjoint）。
  adaptive_config.py   自适应步进配置共享类型和工具函数。
  analysis_dispatch.py JSON 分析配置 dispatch 入口。
  analysis_options.py  面向 JSON dispatch 的中央求解器选项注册表。
  diagnostics.py       线程安全的求解器回退观察器（计数器 + 日志）。
  psf.py               Spectre PSFASCII 参考数据解析器。
  calibration.py       本地结果与 Cadence 参考的校准/比较工具。
  cadence_netlist.py   用于验证的 Spectre 网表生成工具。
  chopper.py           理想与 PMOS 开关差分 chopper 分析。
  explore.py           设计空间探索 / 优化驱动。
  corners.py           工艺角、mismatch MC 与 latch 检测。
  dataset.py           带 provenance、保留失败样本的 surrogate 训练数据集构建器。
  surrogate.py         基线指标 surrogate（GBT，可选 scikit-learn）+ 感兴趣区域过滤。
  surrogate_torch.py   可微 surrogate（torch/MPS）+ 基于梯度的设计优化。
  optimize.py          surrogate 筛选 / Pareto 选择 / solver 校验闭环。
  osdi_host.py         OSDI 0.4 ctypes 宿主——加载编译好的 Verilog-A（.osdi）模型，单器件 DC/AC/noise 求值。
  osdi_device.py       OSDI 宿主紧凑模型的 TransistorModel 适配器（把任意 OSDI PDK 桥接进求解器栈）。
  osdi_transient.py    OSDI 器件的后向欧拉瞬态（在 numba 循环之外）。
  sky130_model.py      SKY130 nfet/pfet PDK：BSIM4 参数卡提取（经 ngspice）+ PDK 注册。
  ngspice_char.py      model-card 求值器：批量 ngspice .dc/.noise 表征 → 缓存 (Vsb,Vds,Vgs) 网格。
  ngspice_device.py    基于缓存 ngspice 网格的 TransistorModel（插值 Id/gm/gds/caps/noise；extract_w + 温度）。
  ngspice_process.py   工艺适配协议：deck 前导、器件语法、op 向量、仿真器参数。
  freepdk45_model.py   FreePDK45 nmos/pmos PDK：角卡绑定 + PDK 注册（ngspice-C 求值器）。
  tsmc28_model.py      TSMC28HPC+ core nmos/pmos：nch_mac/pch_mac + HSPICE library 闭包。
  service/             可选的本地 FastAPI HTTP 服务层（`serve` extra）——见下文。
    __init__.py        只重导出 CLI 胶水代码；从不 import fastapi（import circuitopt 保持无 fastapi 依赖）。
    app.py             create_app() —— /api/v1 路由（health/capabilities/validate/solve/jobs/*）；薄适配，无数值逻辑。
    jobs.py            JobManager —— 进程内线程池后台任务（explore/mc），带进度队列 + 协作式取消。
    serialize.py        to_jsonable()/serialize_results() —— numpy/complex/NaN → 严格 JSON 的转换约定。
    cli.py              add_cli_args()/run_cli() —— 共享的 `serve` 子命令参数定义（延迟导入 fastapi/uvicorn）。
```

## 导入关系

```text
topology.py          <- 无内部依赖
compiled_topology.py <- 无内部依赖；运行时消费 Topology 风格对象
circuit_loader.py    <- topology
numba_kernels.py     <- 无内部依赖；运行时可选 numba
device_model.py      <- 无内部依赖（仅 abc、dataclasses）
device_factory.py    <- 仅 device_model（leaf 器件层；不 import 任何 solver/workflow 模块）
pmos_tft_model.py    <- 可选 numba_kernels、device_model
ac_mna.py            <- 无内部依赖
ac_solver.py         <- device_factory, dc_solver, topology, compiled_topology, diagnostics
dc_solver.py         <- device_factory, topology, diagnostics
noise_solver.py      <- device_model, ac_mna, ac_solver, device_factory, topology, compiled_topology, diagnostics
transient_solver.py  <- adaptive_config, topology, ac_solver, device_factory, transient_profile, compiled_topology, numba_kernels, diagnostics；按需延迟 import osdi_transient（OSDI 器件路由）
transient_profile.py <- 无内部依赖（计数器槽位常量）
pss_solver.py        <- ac_mna, ac_solver, device_factory, adaptive_config, topology, transient_solver, diagnostics
pac_solver.py        <- ac_mna, ac_solver, device_factory, numba_kernels, topology, transient_solver, diagnostics
pnoise_solver.py     <- ac_mna, device_factory, noise_solver, numba_kernels, pac_solver, diagnostics
adaptive_config.py   <- 无内部依赖（仅 dataclass）
analysis_dispatch.py <- ac_solver, noise_solver, transient_solver, pss_solver, pac_solver, pnoise_solver, circuit_loader, analysis_options
analysis_options.py  <- 无内部依赖（注册表）
diagnostics.py       <- 无内部依赖（线程安全计数器）
psf.py               <- 无内部依赖
calibration.py       <- psf, ac_solver, adaptive_config, noise_solver
cadence_netlist.py   <- circuit_loader, topology
chopper.py           <- ac_solver, dc_solver, device_factory, adaptive_config, device_model, noise_solver, pac_solver, pnoise_solver, pss_solver, topology, transient_solver
explore.py           <- ac_solver, device_factory, device_model, noise_solver, circuit_loader, diagnostics
corners.py           <- ac_solver, device_factory, noise_solver, topology, diagnostics
dataset.py           <- diagnostics, circuit_loader, corners, device_model, device_factory, explore, transient_solver
surrogate.py         <- 无内部依赖；运行时可选 scikit-learn/joblib
surrogate_torch.py   <- dataset（仅 CLI）；运行时可选 torch
optimize.py          <- surrogate, circuit_loader, dataset, explore
osdi_host.py         <- 无内部依赖；仅 ctypes + numpy
osdi_device.py       <- device_model, osdi_host（延迟导入）
osdi_transient.py    <- diagnostics, numba_kernels, osdi_host, compiled_topology（通过 OsdiDevice 通用接口调用；不 import transient_solver）
sky130_model.py      <- device_model, osdi_device
ngspice_char.py      <- 无内部依赖；ngspice 子进程 + numpy
ngspice_device.py    <- device_model, ngspice_char；运行期可选 scipy
ngspice_process.py   <- device_model
freepdk45_model.py   <- device_model, ngspice_device
tsmc28_model.py      <- device_model, ngspice_device, ngspice_process, toolchain
service/app.py       <- analysis_dispatch, analysis_options, circuit_loader, device_factory, device_model,
                        freepdk45_model, service/jobs, service/serialize；import 时可选 fastapi/pydantic
service/jobs.py      <- explore, corners, service/serialize；不依赖 fastapi（纯 threading/queue）
service/serialize.py <- 无内部依赖；仅 numpy
service/cli.py       <- service/app（延迟导入）；运行期可选 uvicorn
```

`service/` 子包是一个纯*消费方* leaf——没有任何模块反过来 import 它，`circuitopt/__init__.py`
也从不 import 它，所以即使装了 `serve` extra，裸 `import circuitopt` 仍然不依赖 fastapi。

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
- **后端能力类属性** — 通用求解器按*能力*分派，而非具体后端类型（不再 `isinstance(dev, OsdiDevice)`）。
  `HAS_TERMINAL_LINEARIZATION`（默认 `False`）标记该模型是否暴露周期 PAC/PNoise 线性化用的完整
  quasi-static 4×4 terminal `(G, C)` stamp（`get_terminal_linearization`）；`OsdiDevice` 覆写为 `True`。
  `TRANSIENT_BACKEND`（默认 `None`，即走通用 OTFT numba 瞬态路径）指定要路由的专用积分器名称；
  `OsdiDevice` 设为 `"osdi"`，`transient_solver.py` 读取它并路由到 `circuitopt.osdi_transient.transient_osdi`。
- **`register_model()` / `create_device()` + PDK/极性分层** — 工厂 + 注册表。每个 `(pdk, polarity)` 以结构化键 `"<pdk>.<polarity>"`（如 `"at4000tg.pmos"`）注册；`register_pdk()` 把一个工艺的各极性归组并标记默认。求解器文件调用 `create_device(get_default_model_type(), …)`（单一切换点）而非硬编码模型名，新增工艺或 `nmos` 极性只需一次 `register_pdk`、不改任何求解器。`"pmos_tft"` 保留为向后兼容别名。`get_model_class(model_type)` 是公开只读的注册表访问器，让求解器无需 import 具体后端类即可读取模型的能力标志。`registered_models()` 返回整个注册表的只读快照 `{model_type: "module.QualName"}`（按插入顺序），供需要*枚举*而非查单条的调用者使用——服务层 `GET /api/v1/capabilities` 用它列出全部可选模型键。通用元件（电阻/电容/理想 V/I/受控源）是与工艺无关的拓扑原语，**不在**此注册表中，故每个 PDK 零改动复用。`register_model()` 重复注册时仍会*覆盖*旧条目（有意替换——例如测试打桩——继续静默生效）；但真正的冲突——不同类（按 `__module__.__qualname__` 判断）抢占已被占用的名字，比如两个 PDK 模块争同一别名——现在会在覆盖前发出 `RuntimeWarning`；对*同一个*类的重复 import 或 `importlib.reload` 仍保持静默。

### `device_factory.py`

leaf 器件构建层：只依赖 `device_model`（不 import 任何 solver 或 workflow 模块），因此所有求解器都
能放心 import 它而不引入循环依赖。

- **`build_devices(sizes, *, nf=None, corner=None, topo, model_types=None, device_kwargs=None)`** /
  **`get_ss_params(...)`** — 把求解器已有的逐器件输入（sizes、NF、corner、`model_types`、
  `device_kwargs`）转换成具体的 `TransistorModel` 实例；原样从 `ac_solver.py` 迁入。
  `dev_corner`/`dev_nf`/`is_per_device_corner` 是背后的逐器件 corner/NF 解析辅助函数。
- **`CORNERS`** — OTFT 连续 PVT 全局工艺偏移 dict（`typical`/`slow`/`fast`，按 `pvt0`/`pbeta0`），
  从 `corners.py` 迁入此处；`corners.py` 现在从这里 import 它，而不是自己定义。
- **`SKY130_CORNERS`** / **`SILICON_CORNERS`** / **`apply_silicon_corner(model_types, device_kwargs,
  corner)`** — 硅离散 corner 路由，把一个 corner 名（SKY130 的 `tt`/`ss`/`ff`/`sf`/`fs`，加上
  FreePDK45 的 `nom`）stamp 到解析好的硅器件卡上；从 `explore.py` 迁入此处；`explore.py` 和
  `dataset.py` 现在从这里 import 它，而不是自己定义。
- **`CircuitBinding`** — frozen dataclass，把每个求解器过去逐个手工穿透的六个逐电路输入打包成一体：
  `topo` / `model_types` / `device_kwargs` / `nf` / `corner` / `dc_seed`。它的存在是为了钉死一类 bug：
  向求解器传参时漏掉 `model_types`/`device_kwargs`，电路会静默退回默认 OTFT PDK。现在调用方传一次
  `binding=` 即可，不再重新铺设这一整簇参数。用 `CircuitSpec.binding()` 构造（见 `circuit_loader.py`）。
  `binding.build(sizes)` 为这些 sizes 生成 `{name: TransistorModel}`；`binding.at_corner(corner)` 返回
  一个路由到该 corner 的 binding——硅 corner 经 `apply_silicon_corner` 烘焙进 `device_kwargs`（并清空
  solver corner），OTFT corner 留在 `binding.corner` 上，`None` 原样返回 `self`。**解析优先级**（经
  `resolve_binding`）：显式传给求解器的非 `None` keyword 永远优先；否则由 binding 字段提供默认
  （`binding.dc_seed` 兜底 `x0_guess`）；`binding=None` 时与旧 kwargs 路径逐字节一致。六个求解器入口
  （`ac_solve`/`noise_analysis`/`transient`/`pss_solve`/`pac_solve`/`pnoise_solve`）都接受 `binding=`，
  内部工作流——`analysis_dispatch.run_analysis_suite`、`explore`、`dataset`、`optimize`——统一穿一个
  binding，而不是逐分支重新转发 model 簇。

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

它还承载两个小的编组辅助函数 `term_arrays()`（把 `(kind, ref_or_value)` terminal token 拆成并行的
`kind`/`ref`/`value` int/float 数组）和 `index_array()`（把可选整数 index 打包成 int64 数组，`None`
→ `-1`）。`transient_solver.py` 的 raw-transient 编组和 `osdi_transient.py` 的 OSDI transient 编组都
基于这两个辅助函数构建同一套 stamp-ready 数组，所以它们和拓扑 token 放在一起，而不是每个后端各自
复制一份。

### `circuit_loader.py`

加载 JSON 电路描述并返回 `CircuitSpec`，包含：

- `topology`
- `sizes`
- `bias`
- `nf`

这使得可以通过 JSON 文件（如 `examples/single_stage.json`）添加新电路，而无需修改求解器源码。`CircuitSpec.binding()` 把 spec 的 `topology`、`model_types`、`device_kwargs`、`nf` 以及默认 DC 初值（其第一个 dict 型 `dc_guess`）打包成一个 `CircuitBinding`（见 `device_factory.py`），这样工作流可以向求解器传 `binding=`，而不必逐个穿透这一整簇参数。

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

solver 路径在 Numba 可用时会自动使用加速内核；不再有任何模块在 import 时设置
`CIRCUIT_USE_NUMBA=1`（想关掉设 `CIRCUIT_USE_NUMBA=0`）。

该开关**在 import 时烙死**：`USE_NUMBA`/`NUMBA_AVAILABLE` 是 `numba_kernels`
首次被 import 时一次性算好的常量，事后再设环境变量是静默 no-op。`circuitopt/__init__.py`
在其求解器 import（会连带把 `numba_kernels` 传递 import 进来）之前，先对 `sys.argv`
预扫 `--no-numba` 并设 `CIRCUIT_USE_NUMBA=0`——在 `python -m circuitopt …` 下这个
`__init__` 先于 `__main__.py` 执行，故 CLI flag 能生效。`__main__.py` 每个子命令
handler 随后都调用 `_assert_numba_flag(args)`：若请求了 `--no-numba` 但
`numba_kernels.USE_NUMBA` 仍为 `True`（例如某处已先 import 了求解器模块，绕过了
预扫），则抛 `SystemExit`，把"预扫被绕过"从静默无效变成响亮报错。从 Python 代码
（非 CLI）里禁用 Numba，必须在 `import circuitopt` **之前**设 `CIRCUIT_USE_NUMBA=0`。

目前加速路径包括 PMOS 电流计算、内部节点 Newton 迭代、偏置相关电容计算、
AC/PNoise 小信号参数端导数、PNoise HB block 组装和噪声折叠，
以及 transient Newton 内循环：拓扑 token 查值、PMOS 工作点求解、residual/Jacobian
stamp 和小规模稠密 Newton 线性求解。稠密 Newton 求解使用原地 `A*x = -R`
路径，避免每次迭代里不必要的数组拷贝。如果 compiled 路径处理不了某一步，
`transient_solver.py` 会回退到原 Python Newton / full-Jacobian / least-squares 路径。

### `analysis_options.py` / `analysis_dispatch.py`

`analysis_options.py` 是中央的逐分析选项注册表，`analysis_dispatch.py` 的
`run_analysis_suite`（以及 JSON schema 回归测试）都从它派生，故 solver kwargs/
默认值/schema 不会互相静默漂移。`validate_analysis_cfg(analysis, cfg)` 拒绝
JSON `analyses` 块里的残留键：`known_keys(analysis)` 把该 solver 的选项注册表
和 `DISPATCH_KEYS`（`run_analysis_suite` 直接从 `cfg` 里读、而不转发进
`solver_kwargs` 的少数键——如 `ac`/`noise` 的 `freqs`/`corner`/`band`，
`transient` 的 `signed_devices`）取并集；任何键落在这个并集之外都会抛
`ValueError`，点名该分析名、出错的键、以及排序后的合法键列表。这把一次拼写
错误（比如把 `max_sideband` 拼成 `max_sidebands`）从"静默用默认值跑掉"变成
立即报错。

`analysis_dispatch.py` 里的 `ANALYSIS_ORDER = ("ac", "noise", "transient", "pss",
"pac", "pnoise")` 是权威的分析名元组与执行顺序；`run_analysis_suite` 遍历它，
服务层 `GET /api/v1/capabilities` 的 `analyses` 映射也遍历同一个元组来构建，
两处列出的分析名因此天然一致，不需要另外维护一份硬编码列表。

### `psf.py`

`provenance(path)["fundamental"]` 读取 PSF HEADER 里 `"fundamental
frequency"` 键（若某个非标准写入器只给出裸 `"fundamental"` 拼法则回退到它）——
周期性分析（PAC/PNoise/PSS）会报出真实驱动频率；DC/AC/noise/tran 两个键都没有，
读回 `None`。`parse_noise(path)` 的逐器件噪声数组是**ragged 的**：列数跟随每个
器件在 TYPE 段声明的 struct 宽度（MOSFET 的 `(flicker, thermal, total)` struct
宽度为 3；电阻的 `(rn, total)` struct 宽度为 2），故调用者要取*最后一列*
（`[:, -1]`）拿总量，且要先查 `.shape[1]` 再去切某个具体字段——不能假设宽度
总是 3。

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

DC 求解包含物理支路选择、对称工作点和 rail 有界节点解的鲁棒性处理——实现在 `dc_solver.py` 里，
由这里调用。`ac_solver.py` 本身现在是纯 AC 小信号模块：`ac_solve` 加上带宽辅助函数
`bw_from_gain`；不再承载器件工厂或 DC seeding 代码。

### `dc_solver.py`

DC 工作点求解支持代码——以前内嵌在 `ac_solver.py` 里，现在拆出来，因为它混杂着两类不同的关注点：

- **`bounded_least_squares_dc(...)`** / **`dc_residual_ok(...)`** — 通用的最后手段 DC 求解：当主
  Newton/`fsolve` 路径不收敛时使用的有界最小二乘 fallback，以及门控它的残差判定。
- **`symmetric_seed(...)`** / **`symmetric_continuation(...)`** / **`is_afe_topology(...)`** /
  **`is_pairwise_symmetric_afe(...)`** / **`_AFE_SYMMETRIC_PAIRS`** — 只针对 AFE 拓扑的电路专用
  seeding 启发式，**不是**通用求解器逻辑。它为这一个电路选出物理上（匹配 Spectre）的对称上电支路。
  单独放一个模块能让 `ac_solver.py` 的通用求解保持不含逐电路分支。

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
  `circuitopt.pac_solver.pac_solve(...)`。通用 solver 仍默认使用解析伴随 HB；chopper
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
  `circuitopt.pnoise_solver.pnoise_solve(...)`。chopper 验证默认使用 time-domain
  Floquet adjoint：直接求稀疏周期伴随 BVP，再复用现有循环平稳器件/电阻噪声折叠。
  这去掉了 HB adjoint 边带截断误差。谐波平衡 PNoise 路径仍可用
  `time_domain=False` 显式调用：沿轨道 N 点采样 → 时变小信号 G(t)/C(t) → FFT
  到频域 → 组装 `nb×nb` 块矩阵 `Y[kr,kc] = G_{kr-kc} + jω·C_{kr-kc}` →
  每基带频率一次伴随求解得到传递阻抗 Z_{j,k} → 噪声折叠到 baseband 输出。
  与 `pmos_chopper_lptv_analysis` 不同，两条路径都无需 Cadence 标定常数。

PMOS-switch sideband 路径最初使用 `pmos_chopper_lptv_analysis` 配合 Cadence 标定
常数验证通过。原生 `pmos_chopper_pac` 和 `pmos_chopper_pnoise` 现在已替代这些依赖
标定的路径，提供第一性原理的周期小信号和噪声求解。瞬态 finite-edge 路径已与
Spectre `tran` 对齐。对 D3 / `chop_tb_d3` 官方 `slow` corner PSS/PAC/PNoise
参考，默认 time-domain PAC 约 +0.03%，原生 TD PNoise IRN 约 +0.02%。旧 HB-K32
PNoise IRN 三 corner 误差为 slow/typical/fast = +1.81% / +1.05% / +0.66%；
TD adjoint 后为 +0.02% / −0.00% / +0.57%。这把此前由边带截断造成的“假舒适”
彻底揭掉，同时仍是第一性原理求解。

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
  `pss_result`（HB 路径适用时）。
- 设置 `time_domain=True` 时，PNoise 会用稀疏 Floquet 伴随 BVP 替代 K 截断的 HB
  adjoint 求解（结果字段 `pnoise_time_domain_used=True`）。这条路径在转换边带上无截断；
  剩余误差主要来自时域网格离散，因此默认式 `n_period_samples < 640` 会自动抬到 768。
  这条路径返回 `method="pss_time_domain_floquet_adjoint"`；HB 兜底路径返回
  `method="pss_harmonic_balance_conversion_matrix"`。
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

- `transient(sizes, bias, tgrid, vip=None, vin=None, nf=None, V0=None, topo=AFE_TOPO, inputs=None, node_inputs=None, integration_method="be", adaptive=False)`
- 支持传统的 AFE `vip/vin` 输入，也支持通过 `topo.transient_inputs` 驱动的通用 `inputs={name: waveform}`。
- `node_inputs={node: input_key}` 在某个（rail）节点上驱动波形——用于前端 testbench，其激励在源节点注入并通过无源网络传播，而非直接驱动器件栅极。
- `current_inputs=[{"p": node_a, "q": node_b, "input": key}]` stamp 一个时变
  理想电流源，方向为 `p -> q`；PMOS chopper helper 用它注入 charge-injection 脉冲。
- `max_step`、`max_retry_subdivisions`、`fallback_full_jacobian` 和
  `fallback_least_squares` 用于 switched
  transient 步的受控细分和有界 fallback 求解。
- `cap_mode` / `cap_mode_id` 是 per-call 电容算子 override；生产路径只支持
  `charge`/id 0 和 `average`/id 1。`None` 使用环境默认 `charge`，chopper PSS
  显式传 `average` 匹配 Cadence feedthrough。该 override 只影响
  transient/PSS 轨道，不影响 PAC/PNoise conversion 线性化。
- `adaptive=True` 是 opt-in 的 LTE-controlled gear2 路径：传入的 `tgrid` 作为输入采样网格
  和 `[t0, tstop]` 边界，返回自选的非均匀 accepted grid。它只允许配合
  `integration_method="gear2"`。公开 API 仍兼容 `adaptive_reltol`、
  `adaptive_vabstol`、`adaptive_iabstol`、`adaptive_max_steps` 和
  `adaptive_h0`；内部会统一归一化成 `AdaptiveConfig`，供
  transient/PSS/chopper 共用。PSS 额外从同一个 config 读取
  `adaptive_freeze_factor`。
  adaptive LTE policy 的常量和 Python helper 位于 `circuitopt/adaptive_config.py`；
  Numba 为性能保留 compiled mirror，并由测试校验两边一致。Newton 失败的
  reject 现在会缩小候选步长，零误差 accepted step 才允许放大。
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
- adaptive gear2 使用 step-doubling LTE 估计；PSS 会在接近收敛时冻结 accepted grid，
  再用该固定 grid 生成最终 orbit/monodromy。Numba 覆盖 `n_aug == n` 的 adaptive
  gear2；含理想电压源支路未知量的拓扑会回退到 Python adaptive loop。
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
- 结果导出为 CSV 和 JSONL；CLI 运行 `python -m circuitopt.explore <config.json>`。
- 硅 corner 路由（`SKY130_CORNERS`/`SILICON_CORNERS`/`apply_silicon_corner`）现在在
  `device_factory.py` 里；`explore.py` 从那里 import，而不是自己定义。
- `add_cli_args(parser)` / `run_cli(args)` 是 CLI 参数定义的单一来源——`python -m circuitopt explore`
  子命令和独立的 `python -m circuitopt.explore` 入口都调用同一对函数，两个入口不会再互相漂移
  （见 [`cli_reference.md`](cli_reference.md)）。
- `explore_from_dict(data, n=, seed=, method=, corner=, progress=None, should_stop=None)`——
  `explore` 子命令和服务层 `POST /api/v1/jobs/explore` 共用的单一入口：解析 `explore` 块、
  绑定硅 `models`、再调用 `explore()`。`progress(done, total)` / `should_stop()` 是可选钩子，
  原样透传给 `explore()`，供需要实时进度或协作式取消的调用者使用（如服务层的后台任务
  管理器，见 `service/jobs.py`）；两者默认 `None`，此时行为与加钩子之前逐字节一致。提前
  停止时 `results["stopped_early"]` 和 `results["summary"]["stopped_early"]` 为 `True`，
  `summary["evaluated"]` 记录实际跑完的候选数（`summary["n"]` 仍是最初请求的数量）。

示例配置：`examples/afe_explore.json` 和 `examples/single_stage.json`，二者都是带
`explore` 块的完整电路 JSON。

### `corners.py`

工艺角和鲁棒性*工作*的单一事实来源——这些内容原本会在每次扫描中重复推导。`CORNERS` 数据本身（全局
工艺偏移 `typical` / `slow` / `fast`，按 `pvt0`/`pbeta0` 表示，来源于 PDK 的 monte.scs 段落；如
slow = `{"pvt0": -0.2259, "pbeta0": -0.54}`）现在放在共享的 leaf 器件层 `device_factory.py` 里；
`corners.py` 从那里 import 它（`from .device_factory import CORNERS`），所以既有的
`from circuitopt.corners import CORNERS` 调用点无需改动。`corners.py` 里保留的是建立在它之上的
mismatch/latch 相关工作：

- `mismatch_corner(rng, devices, base)`——在工艺角基础上叠加逐器件随机 `mvt0`/`mbeta0`。
- `metrics(...)`——单设计单 corner → `gain_peak_dB`、`bw_Hz`、`irn_uV` 和 `latch_dV`（DC 工作点的 `|out+ - out-|`；大值 ⇒ 交叉耦合正反馈已 latch）。
- `corner_table(...)`——typ/slow/fast 三个 corner 的指标汇总。
- `latch_screen(...)`——确定性最坏情况 latch 筛查：对每个对称对在所有符号组合上施加 ±kσ 推开，返回最大输出失衡。单次固定 kick 存在假阴性（latch 的符号模式依赖于设计），因此筛查遍历所有模式；计算开销足够低，可在搜索内部代替完整 MC 使用。它只需要 DC/AC 工作点和 latch 失衡，因此会跳过噪声。
- `mismatch_mc(...)`——单个 corner 上的逐器件 mismatch MC，从名义工作点播种；返回各指标数组、latch 掩码以及汇总（latch 率 + 未 latch 样本的 mean/std/P5/P95）。每个样本先跑 AC/latch，只有进入最终噪声统计的未 latch 样本才计算 IRN。支持可选的 `progress(i, n, partial)` / `should_stop()` 钩子（默认 `None`，结果与加钩子之前逐字节一致）：`progress` 在每个样本完成后触发，带一份轻量的滚动汇总；`should_stop` 在每个样本开始前检查，若返回 `True` 则提前结束，在顶层和 `summary` 里都加上 `"stopped_early": True`——实际评估完的数量就是 `summary["n"]`（这条路径没有单独的"请求数"字段；与 `explore` 的 `summary["evaluated"]` 不同，`mismatch_mc` 的 `summary["n"]` 本来就始终反映实际跑过的样本数）。
- `mismatch_mc_from_dict(data, n=, seed=, corner="typical", progress=None, should_stop=None)`——
  `mc` 子命令和服务层 `POST /api/v1/jobs/mc` 共用的入口；解析电路后调用 `mismatch_mc()`，
  两个入口因此不会漂移。

`ac_solve` / `noise_analysis` 接受相同的 `corner` 参数（扁平的工艺 dict 或逐器件 mismatch 映射）。驱动脚本 `examples/mc_mismatch.py` 将其封装为 corner 表 + 3-corner MC 图。

### ML surrogate 层（`dataset.py` / `surrogate.py` / `surrogate_torch.py` / `optimize.py`）

把已验证的求解器变成一条完整的 **造数据集 → 训练 surrogate → 优化 → 校验** 闭环。求解器全程仍是
唯一的 ground truth；surrogate 只加速大候选池的**筛选**环节。

- **`dataset.py`** ——采样/评估方式与 `explore.py` 相同，但**不做**约束/Pareto 过滤，且**总是**评估
  噪声，所以每个样本（含 DC 失败样本）都成为一条带标签的训练行。写出 `.jsonl`（可读的逐行样本）+
  `.manifest.json`（provenance：schema 版本、solver git commit(+dirty)、拓扑 hash、PDK、`models` 绑定、
  corner、采样 seed/method、变量范围——供下游拒绝域外样本）+ `.npz`（稠密 `X`/`Y` 矩阵，缺失标签为
  NaN）+ 可选 `.parquet`。**标签组**（`--labels`，在默认 `ac_noise` 之外可选加）：`transient`（复用配置
  已验证的 `periodic` 瞬态得到的、激励无关的波形特征）、`pss`（周期稳态质量 + 轨道输出）、`pac`（基带
  转换增益 + PAC 网格内 −3dB 角）和 `pnoise`（带内积分输出/等效输入周期噪声——斩波的核心指标）。
  `pss`/`pac`/`pnoise` 三组每候选共享一次 `run_analysis_suite` 调用，配置 `analyses` 块里已验证的求解
  设置（`time_domain`、drive、band、打靶容差）原样生效；`pac`/`pnoise` 要求配置带对应 `analyses` 块。
  **设计轴语法**在 `DEV.W/.L/.NF`/bias 之外还支持：`<Cap>.C` / `<Res>.R`（具名无源器件值——`structural`，通过
  `candidate_circuit()` 逐候选重建电路）、`periodic.frequency`（clock）、`pvt0`/`pbeta0`（连续全局工艺
  偏移——采样它就把离散 corner 扫描变成一个连续 PVT 训练轴）。和 `explore.py` 一样，`dataset.py` 的硅
  corner 路由（`SKY130_CORNERS`、`apply_silicon_corner`）也从 `device_factory.py` import，而不是自己
  定义；并暴露同一对 `add_cli_args(parser)` / `run_cli(args)`，使 `python -m circuitopt dataset` 子命令与
  独立的 `python -m circuitopt.dataset` 入口共用一份参数定义。
- **`surrogate.py`** ——`HistGradientBoostingRegressor`（可选 `scikit-learn` 依赖）逐标签独立训练，对跨
  多个数量级的标签（如 IRN）自动用 log-space 拟合。`filter_rows()` / CLI `--filter label:lo:hi` 把训练
  限制在感兴趣区域内（例如剔除甩轨/collapse 的极端设计——它们的极端标签会拖累平方误差拟合，反正也会被
  约束筛掉）。`score()` 逐标签报告 median/P95 相对误差和 R²。
- **`surrogate_torch.py`** ——可微 MLP surrogate（可选 `torch` 依赖；Apple Silicon 上支持 MPS），用于
  带约束惩罚的多目标梯度优化，附带 `--verify` 收尾接回求解器。
- **`optimize.py`** ——筛选-校验闭环的落地：用 surrogate 对大候选池做预测（µs/candidate），取约束下的
  Pareto 前沿，再把 top-K 送回真实的已标定求解器复核。用 `dataset.candidate_circuit()`/
  `split_variables()` 保证每种变量（含结构化的电容/电阻/时钟轴）在校验阶段都生效，而不只是 size/bias。

一条关键的经验教训：**没有一个 surrogate 能同时是精确的"感兴趣区域"
模型又是懂"失败区域"的好筛选器**——用 `--filter` 只在工作区训练能拿到最紧的指标精度，但会让 surrogate
对甩轨设计一无所知，筛选阶段就分辨不出它们。screen-and-verify 架构正是为容忍这个矛盾设计的：可行性
的最终话语权在求解器，不在 surrogate。

### 硅 PDK / OSDI 层（`osdi_host.py` / `osdi_device.py` / `osdi_transient.py` / `sky130_model.py`）

把**第二套行业标准器件物理模型（BSIM4）**接入 AT4000TG OTFT 模型所用的同一个 `TransistorModel`
接口——所以任何 bulk-BSIM4 PDK（目前是 SKY130）都跑在同一套 DC/AC/noise 求解器引擎里，且是纯增量的
（`default=False`；OTFT PDK 不受影响，数值 byte-identical）。

- **`osdi_host.py`** ——**OSDI 0.4 ABI** 的 ctypes 宿主，这是 [OpenVAF](https://github.com/pascalkuthe/OpenVAF)
  把 Verilog-A 紧凑模型编译成的、仿真器无关的 C 接口（`.osdi`，原生共享库）。`load_osdi()` 内省
  descriptor（节点/参数/opvar），带 struct 尺寸自检；`Device` 通过 ABI 的 `access()` 设置模型/实例参数、
  复现仿真器侧的节点合并、跑内部节点 Newton（对 DC 悬空的内部节点做 gmin 正则化），暴露
  `operating_point()`（Id/gm/gds/gmb/电容通过对内部节点做 Schur 补得到——BSIM4 不暴露任何 opvar，
  所以小信号量全部来自 Jacobian）和 `noise_psd()`。这是一个**单器件** DC/AC/noise 求值器；电路级的
  MNA/Newton 仍由现有的 `ac_solver`/`noise_solver` 负责。
- **`osdi_device.py`** ——`OsdiDevice(TransistorModel)` 包装一个 `Device`，实现
  `get_Idc`/`get_ss_params`/`get_capacitances`/`get_noise_psd`。`TransistorModel.kcl_sign`
  （默认 +1，即 source-high——匹配 PMOS/OTFT）让 `ac_solve` 的 DC KCL 也能支持 NMOS（source-low，
  `kcl_sign=-1`），且不改变 OTFT 路径（byte-identical：`1.0 * abs(x) == abs(x)`）。`OsdiDevice`
  覆写基类的能力类属性：`HAS_TERMINAL_LINEARIZATION = True`（提供 `get_terminal_linearization`）和
  `TRANSIENT_BACKEND = "osdi"`。三个瞬态专用的 ABC 钩子仍会抛 `NotImplementedError`——`.osdi`
  不能进 numba 瞬态循环。
- **`osdi_transient.py`** ——`transient_osdi(sizes, bias, tgrid, ...)` 是电路级入口：一个固定步长
  后向欧拉积分器，每步直接调用 OSDI 宿主（在 numba 循环之外），建立在更底层的单器件辅助函数
  `cs_transient()` 上，是一个基础性（非全保真度）的硅瞬态实现。`transient_solver.py` 里的
  `transient()` 会检查器件的 `TRANSIENT_BACKEND` 类属性，若为 `"osdi"` 就延迟 import 并路由到
  `transient_osdi`——单向依赖。`osdi_transient.py` 自身不再 import `transient_solver.py`，
  两个模块不再构成循环 import（这次拆分之前是的）。
- **`sky130_model.py`** ——`Sky130Nfet`/`Sky130Pfet(OsdiDevice)` + `register_pdk("sky130", ...)`。
  SKY130 的 binned BSIM4 子电路（63 个 bin，2000+ 个 `.param` 表达式）**让 ngspice 去解析**：实例化子
  电路、跑一次 `op`、`showmod` 拿到完全展开的扁平参数卡（731 个参数），缓存到 `data/pdk/sky130/*.json`，
  喂给 OpenVAF 编译的 `bsim4va`。`EXTRACT_W`/`extract_w`：在参考宽度处解析一次卡片，让 `bsim4va` 缩放
  实际 W——设计扫描时避免逐候选起一个 ngspice 子进程（改为 ~2ms/eval）。Oracle：**加载同一个 `.osdi` 的
  本地 ngspice**——因为求解器和 oracle 跑的是同一个编译好的模型，正确性是 *model==oracle*，与
  SKY130-vs-VA 的 BSIM4 版本差异无关（SKY130 的 ngspice 内置模型是 4.5，VA 源码是 4.8——这是一个真实的
  130nm 工艺，不是 SkyWater 逐字节对齐的 sign-off 模型，但对优化器泛化而言这是正确的取舍）。

### 第三个 PDK：FreePDK45（`ngspice_char.py` / `ngspice_device.py` / `freepdk45_model.py`）

FreePDK45（45nm，1.0V，用户目标工艺）接入**同一个** `TransistorModel` 接口，但用**不同的求值器**。
FreePDK45 的 BSIM4 卡声明 `version = 4.0`,而我们 OpenVAF 编的 BSIM4.8 VA 没有版本开关,在这些 45nm 卡上
算出 ~30% 不同的 I-V(与版本无关——已用改卡验证),所以 OSDI 宿主复现不了 FreePDK45 的预期行为。它的
oracle 因此是 **ngspice-C 本身**:`ngspice_char.characterize()` 按 `(model, W, L, corner, temp)` 跑一次批
量 `.dc` 扫（~0.03s/1000 点）成缓存的 Id/gm/gds/Cgs/Cgd 网格,`NgspiceDevice` 插值它（µs/eval,节点处即
exact ngspice-C）。已验证 model==oracle:单器件 op 逐位对 ngspice `.op`,5T OTA 过 `ac_solve` 与 ngspice
自己的 `.ac` 差 0.05dB/0.3%,输出噪声在 ngspice `.noise` 的 ~5% 内。噪声也是精确 ngspice-C
（`characterize_noise` 逐偏置跑 `.noise`,CCVS 跨阻→漏噪 PSD,拟合 S_id=A+B/f,log 空间插值）。
快速网格负责 DC+AC+noise；`transient()` 遇到 FreePDK45 会路由到 `ngspice_transient.py`，生成完整
四端网表并直接跑 ngspice `.tran`，因此保留 BSIM4 端口电荷和 Cdb/Csb。PSS/PAC/PNoise 尚未接入该后端。
网格 AC 模型带 Cgs/Cgd 但**不含**漏/源结电容 Cdb/Csb,故整机 `ac_solve`
的 UGBW 比 ngspice 自己的 `.ac` 偏高 ~8%（增益/PM 对到 <0.2dB / <8°）——头条数字取 ngspice 值并留裕量。

- **`ngspice_char.py`** ——批量表征器。`characterize()` 按 Vsb 切片扫 `.dc vg vd`（op-vars
  `@m[id/gm/gds/cgs/cgd]`）;`characterize_noise()` 逐偏置跑一次 `.noise`。两者都接受 `temp_c`
  （`.options temp`,进缓存键——27°C 保持无标签故标称缓存不失效）,按工艺缓存到
  `data/pdk/freepdk45/` 或 `data/pdk/tsmc28hpcp/`。
  两者都经由共享的 `_run_ngspice()` 帮手让失败显式化：ngspice 非零 returncode，或 returncode
  为 0 但输出文件缺失（静默失败），都会抛出携带 ngspice stderr 尾部的 `RuntimeError`——而不是让
  临时 deck/输出文件（现在建在 `tempfile.TemporaryDirectory()` 里，不再是裸 `mktemp`）悄悄丢失，
  最终在下游 `numpy.loadtxt` 里报一个看不出原因的 `FileNotFoundError`。
- **`ngspice_device.py`** ——`NgspiceDevice(TransistorModel)` 插值网格（`scipy.RegularGridInterpolator`,
  **线性**——cubic 在 ~1e-17 电容数组上返回 0）。`extract_w`:在参考 W 处表征一次,线性缩放实际 W（BSIM4
  近似正比 W,<0.7% vs 逐 W 真卡）,使 dataset/优化器的 W 扫描变纯插值;`temperature`（开尔文 kwarg）→
  `temp_c` 物理重表征。Cgs/Cgd 取自 ngspice C 矩阵;瞬态钩子 `NotImplementedError`。
- **`ngspice_transient.py`** ——完整电路 `.tran` 后端：MOS/R/C/独立源/受控源/PWL、corner/温度、
  节点与电源电流回读；统一 `transient()` 自动路由，网格对象本身无需伪造电荷 companion。
- **`freepdk45_model.py`** ——`Fp45Nfet`/`Fp45Pfet(NgspiceDevice)` + `register_pdk("freepdk45", ...)`。
  `corner` 支持 nom/tt/ss/ff/sf/fs，并按极性选择卡目录。`FREEPDK45_CORNERS`
  是合法 corner 名的公开元组；服务层 `GET /api/v1/capabilities` 直接读取它
  （连同 `device_factory.SKY130_CORNERS`、`device_factory.CORNERS`），而不是硬编码三个工艺角家族。

### TSMC28HPC+ 工艺适配器（`ngspice_process.py` / `tsmc28_model.py`）

`NgspiceProcessAdapter` 把 foundry 特有网表语义与 circuitopt 拓扑/求解器隔离。适配器负责 model-card
前导、MOS 实例语法、层级工作点向量、corner 校验、缓存命名空间和额外 ngspice 启动参数。
FreePDK45 保留原有扁平卡路径；TSMC28HPC+ 走适配器路径，源码中不嵌入任何模型参数。

`Tsmc28HpcpAdapter` 面向 licensed 1d8 HSPICE deck 里的 0.9V `nch_mac` / `pch_mac` core wrapper。
它显式展开 `setup`、工艺角、`global`、`total`、`stat` 五个 `.lib` section，并使用
`-D ngbehavior=hsa` 启动 ngspice。`NF` 在 foundry wrapper 内原生表征；层级
`@m.x*.main[...]` 向量提供 Id/gm/gds/电容及完整电路 `.op` 数据。完整 `.tran`、`.ac`、
`.noise`、`.op` 共用同一适配器和模型 deck。

默认可迁移模型入口为
`PDK/tsmc28hpcp/models/hspice/cln28hpcp_1d8_elk_v1d0_2p2.l`，且被 Git 忽略。
解析优先级为 `TSMC28_MODEL_DIR`、`TSMC28_PDK_ROOT`、项目内入口、
`PDK_ROOT/tsmc28hpcp`。详见 [TSMC28HPC+ 适配说明](tsmc28hpcp.md)。

`circuitopt/circuit_loader.py` 的可选 `models` 块（`{"M1": {"type": "sky130.nmos", ...}}` 或
`"freepdk45.nmos"` / `"tsmc28hpcp.nmos"`）把 JSON 电路里的特定器件绑到非默认 PDK，
所以一个混合 OTFT+硅（或全硅）电路只是配置
问题——见 [JSON 电路格式](json_circuit_format_zh.md)。两个完整的全差分 OTA 设计流程案例:
[SKY130 FD-OTA](sky130_fd_ota_design.md)、[FreePDK45 FD-OTA](freepdk45_fd_ota_design.md)。

### 本地服务层（`service/app.py` / `jobs.py` / `serialize.py` / `cli.py`）

架在整个求解器栈之上的**可选**本地 FastAPI HTTP 层，由 `serve` extra 门控
（`pip install -e ".[serve]"`）。它是一层薄适配——每个路由都直接转发给已有的单一事实来源，
本身不带任何数值逻辑——是未来桌面 GUI 或 MCP server 共用的底座（见
[后续开发计划](futureplan.md)）。完整端点参考见 [本地服务 API](service_api_zh.md)。

- **`app.py`** —— `create_app(job_workers=1) -> FastAPI` 构建 `/api/v1` app：`GET
  health`/`capabilities`、`POST validate`/`solve`（同步，直接调用
  `circuit_from_dict`/`validate_analysis_cfg`/`run_analysis_suite`），以及由 `jobs.JobManager`
  支撑的 `jobs/*` 后台任务路由（`POST jobs/explore`/`jobs/mc`、`GET jobs`/`jobs/{id}`、
  `DELETE jobs/{id}`、`WS jobs/{id}/events`）。CORS 只放行 `localhost`/`127.0.0.1` 任意端口。
  `pydantic` 请求模型（`SolveRequest`/`ExploreJobRequest`/`McJobRequest`）把 `circuit` 当作不透明
  `dict` 透传——电路 schema 的单一事实来源始终是 `circuit_from_dict`，这里不重新描述一遍。
- **`jobs.py`** —— `JobManager`/`Job`：为两个长任务驱动函数（`explore_from_dict`、
  `mismatch_mc_from_dict`）建立的进程内 `ThreadPoolExecutor` 后台任务表。状态机
  `queued -> running -> {done, failed, cancelled}`；进度被推到逐 job 的 `queue.Queue`
  （由 WebSocket 路由消费）,同时缓存在 `Job.progress` 供轮询读取；取消操作设置一个
  `threading.Event`，作为 `should_stop` 回调传给核心驱动函数（协作式——正在跑的候选点/样本
  总会先跑完）。内存最多保留 `MAX_JOBS`（50）个任务，优先驱逐最旧的已终结任务。不 import
  `fastapi`——纯 threading/queue 管道代码，可独立单测。
- **`serialize.py`** —— `to_jsonable()`/`serialize_results()`：numpy/complex/NaN → 严格 JSON
  的转换约定，被所有响应共用（同步端点和 job/WebSocket payload 都走它）。NaN/±Inf → `null`，
  `complex` → `{"re", "im"}`，`numpy.ndarray` → 嵌套 `list`，`_` 前缀的 dict key 和 callable 被丢弃。
- **`cli.py`** —— `add_cli_args(parser)`/`run_cli(args)`，`serve` 子命令参数定义
  （`--host`/`--port`/`--reload`/`--job-workers`）的单一来源，被 `circuit-opt serve` 子命令和独立
  的 `python -m circuitopt.service` 入口共用，与 `explore`/`dataset` 的单一来源 CLI 模式一致。
  `run_cli` 内部延迟 import `fastapi`/`uvicorn`，所以 `circuitopt/__main__.py`（为了注册子命令会
  提前 import `circuitopt.service`）不需要装了 `serve` extra 才能正常工作。

## 快速示例

```python
import numpy as np

from circuitopt.ac_solver import ac_solve
from circuitopt.noise_solver import noise_analysis, band_rms
from circuitopt.transient_solver import transient

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

from circuitopt.circuit_loader import load_circuit_json
from circuitopt.ac_solver import ac_solve
from circuitopt.transient_solver import transient

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
  `f_chop=200 Hz` 时默认 time-domain PAC 约 +0.03%，TD-adjoint PNoise IRN 约 +0.02%。
  slow/typical/fast 三 corner 的旧 HB-K32 IRN 误差为 +1.81% / +1.05% / +0.66%，
  TD PNoise 后为 +0.02% / −0.00% / +0.57%。
- SC-LPF calibration 现在显式默认 `gear2 + adaptive + cap_mode="average"`，
  输入网格补 clock edge 断点，并用 `pnoise_n_period_samples=512` /
  `pnoise_max_sideband=20` 保证噪声采样。对入库 Spectre SC-LPF 参考，当前
  PASS：PAC 增益约 −0.32%、带宽 +1.07%、输出噪声 +2.82%。
- 最终锁定设计约 22.9 dB 增益、549 Hz 带宽、37 µVrms 等价输入噪声。

上述数据描述当前的 AT4000TG 验证案例。后续 PDK 或拓扑应针对其各自的仿真器参考重新进行校准。
