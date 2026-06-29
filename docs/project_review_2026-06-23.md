# 项目全景分析 — 2026-06-23

## 一、当前状态总览

```
工作树: 21 文件已修改 (+358/-140)，1 新文件未入库 (test_device_model.py)
测试:   155 条，146 pass，9 skip (slow chopper)，0 fail — 4m42s
校准:   5 case / 3 电路拓扑，对标 Cadence Spectre 24.1 全 PASS
```

### 已提交的最新里程碑（HEAD: `0a2ae8a`）

| 提交 | 日期 | 内容 |
|------|------|------|
| `0a2ae8a` | 06-22 | 修复bug，扩展支持 — 校准数据入库、受控源补齐、SC-LPF 闭环 |
| `2801a09` | 06-21 | 支持理想电压源（真·MNA），device_model 初版 |
| `28cf967` | 06-21 | 完善文档和 CLI 入口 |
| `9083c69` | 06-21 | 引入 Gear2 积分引擎，对齐 Cadence |
| `39b1b59` | 06-20 | 优化性能与精度 |

---

## 二、工作树未提交内容

这是一批**自洽、已测试、已写文档**的改动，分为四个主题，应该作为一次提交落地：

### 2.1 PDK / 极性注册层（11 个文件）

`device_model.py` (+120 行) 新增 `PDK` 类、`register_pdk()`、`create_transistor()`、`transistor_type()`、`get_pdk()`、`get_default_pdk()`、`list_pdks()`。

`pmos_tft_model.py` 从 `register_model("pmos_tft", ...)` 切换为：
```python
register_pdk("at4000tg", {"pmos": PMOS_TFT}, default=True,
             aliases={"pmos_tft": "pmos"})
```

8 个 solver 文件全部从硬编码 `create_device("pmos_tft", ...)` 迁移为 `create_device(get_default_model_type(), ...)` —— 单一开关点，新增工艺或 NMOS 极性无需改任何 solver。

`core/__init__.py` 新增导出：`PDK`, `create_transistor`, `register_pdk`, `get_default_pdk`, `get_pdk`, `list_pdks`, `transistor_type`。

`tests/test_device_model.py`（新文件，untracked）— 6 条测试，覆盖所有新 API + 向后兼容 + 错误路径，全部 PASS。

### 2.2 Cadence 标定常数退役（chopper.py + test_chopper.py）

删除两个仅用于一阶快速路径的经验修补常数：
- `_CADENCE_PMOS_CHOPPER_CONVERSION_PHASE_RAD` = 24.93°
- `_CADENCE_PMOS_CHOPPER_PERIODIC_NOISE_PSD_SCALE` = 1.0355

`pmos_chopper_lptv_analysis` 移除 `cadence_calibrated` 参数，默认返回原始一阶边带求和（增益偏低 ~10%）。Cadence 级精度现独家由无常数谐波平衡路径（`pmos_chopper_pss` → `pmos_chopper_pac`/`pmos_chopper_pnoise`）提供，由 `core/calibration.py` 验证。

`test_pmos_chopper_lptv_ui_matches_spectre_pss_scale` 替换为 `test_pmos_chopper_lptv_is_first_order_underestimate`，验证常数已退役、增益为诚实一阶估计。

### 2.3 解析伴随 PAC 的 vsource drive 修复（pac_solver.py + test_vsource.py）

**这是一个关键 bug 修复。** 当小信号激励在 true-MNA 电压源上时（如 SC-LPF），`_analytic_adjoint_pac` 因无驱动节点而返回 `None`，退回对 x0 敏感的有限差分 shooting。对刚性 τ≫T 开关电容电路，0.003V 的 gear2-vs-BE 轨道差异经近奇异 (I−Φ)⁻¹ 放大为 **24× 虚假增益**。

修复：将 vsource 小信号驱动耦合进 bordered HB 的支路约束行（baseband kr=0），使 PAC **与积分方法无关**（gear2 ≡ BE ≡ 1.006），且比旧 FD-BE 更准。

`test_sc_lpf_pac_is_integration_method_independent` 守卫此修复。

### 2.4 Dispatch 层 `integration_method` 转发（analysis_dispatch.py + test_json_circuit.py）

`_PSS_KWARGS` 和 `_TRANSIENT_KWARGS` 各加 `"integration_method"` 键，JSON 中 `analyses.pss.integration_method` / `analyses.transient.integration_method` 可选 `"gear2"` / `"be"`。`test_dispatch_forwards_integration_method` 守卫。

---

## 三、测试覆盖分析

### 3.1 整体统计

```
测试文件:   16 个 (15 committed + 1 untracked)
测试函数:   155 条
断言数:     ~515 条
通过/跳过:  146 pass / 9 skip (慢测需 RUN_SLOW_CHOPPER=1)
失败:       0
```

### 3.2 覆盖热力图

| 模块 | 测试文件 | 测试数 | 评价 |
|------|---------|--------|------|
| **chopper.py** | test_chopper.py | 24 | ★★★ 优秀 — 谐波权重、LPTV、PSS/PAC/PNoise vs Cadence |
| **受控源(VCVS/CCCS/CCVS)** | test_controlled_sources.py | 22 | ★★★ DC/AC/Noise/瞬态/JSON 全覆盖 |
| **理想电压源** | test_vsource.py | 20 | ★★★ DC/AC/瞬态/PSS/PAC/PNoise/SC-LPF |
| **基本元件(R/C/Isrc/VCCS)** | test_elements.py | 16 | ★★★ 全分析路径 |
| **explore.py** | test_explore.py | 11 | ★★☆ Pareto/采样/约束过滤 |
| **numba_kernels.py** | test_model_kernels.py | 11 | ★★☆ Numba vs Python 一致性 |
| **periodic solvers** | test_periodic_solvers.py | 10 | ★★☆ 通用 PAC/PNoise/PSS |
| **calibration.py** | test_calibration.py | 7 | ★★☆ 数值回归（4 条慢测 skip） |
| **corners.py** | test_corners.py | 7 | ★★☆ corner/mismatch/latch |
| **pss_solver.py** | test_pss_solver.py | 5 | ★★☆ 收敛/Broyden/种子复用 |
| **compiled_topology.py** | test_compiled_topology.py | 3 | ★☆☆ 基本功能 |
| **device_model.py** | test_device_model.py | 6 | ★★☆ PDK 注册/别名/错误路径 |

### 3.3 覆盖缺口

| 优先级 | 模块 | 问题 |
|--------|------|------|
| 🔴 **HIGH** | **`cadence_netlist.py`** | **零测试。** 该模块生成对标参考网表，bug 会导致静默错误的 Cadence 对比 |
| 🔴 **HIGH** | **`ac_mna.py`** | **零单元测试。** MNA stamp 原语是所有频域求解器的基础，仅通过集成测试间接覆盖 |
| 🟡 MEDIUM | **`psf.py`** | 仅 1 条断言（检查 provenance 存在），无数值精度验证 |
| 🟡 MEDIUM | **`pmos_tft_model.py`** | 900 行，仅 numba-kernel 对比测试。gm/gds/电容/噪声 vs 偏置从未独立验证 |
| 🟡 MEDIUM | **`analysis_dispatch.py`** | ~15 个 helper 函数无单元测试，仅顶层 `run_analysis_suite` 有烟雾测试 |
| 🟢 LOW | **`transient_solver.py`** | 66KB，通过集成测试间接覆盖，无独立测试文件 |
| 🟢 LOW | **`noise_solver.py`** | 通过无源网络测试覆盖，无独立测试文件 |

---

## 四、代码健康度审计

### 🔴 CRITICAL

| 问题 | 位置 | 影响 |
|------|------|------|
| **50+ 处裸 `except Exception` 静默吞错** | 全部 solver 文件 | 模型发散时返回零电流/gm，产生"看起来合理但完全错误"的电路解 |
| ~~`_bw_from_gain` 重复定义~~ | ~~`chopper.py` + `pac_solver.py`~~ | ✅ 已合并至 `ac_solver.py`（2026-06-23） |
| ~~`_nfval` 重复定义~~ | ~~`pss/pac/pnoise` 各一份~~ | ✅ 已统一为 `ac_solver._dev_nf`（2026-06-23） |

### 🟡 HIGH

| 问题 | 位置 |
|------|------|
| `transient()` 函数 ~1220 行，含 9 个嵌套函数 | `transient_solver.py` |
| `pac_solve()` ~800 行 | `pac_solver.py` |
| ~~器件创建 dict comprehension 重复 ×7~~ | ~~全部 solver~~ | ✅ 已提取为 `ac_solver.build_devices()`（2026-06-23） |
| ~~21 个文件各自写 `try/except ImportError` 样板，~80 行冗余~~ | ~~全部 core/*.py~~ | ✅ 已统一为包内相对导入，examples/tools 移除 `sys.path.insert`（2026-06-29；`demo/server.py` 保留） |

### 🟢 MEDIUM/LOW

- 各处散落魔术数字（`1e-12` gmin, `1e-9` dB floor, `4000` maxfev, `1e-3` FD step 等）
- 约半数私有函数缺少 docstring
- solver 返回 dict 格式不统一
- `psf.py` 使用已弃用的 `StopIteration` 模式

---

## 五、校准对标覆盖

### 5.1 已有对标

| Case | 电路 | Corner | 分析 | 对标结果 |
|------|------|--------|------|---------|
| `amp_design3_typical` | AFE 放大器 | typical | DC, AC, Noise | gain +0.00dB / IRN +0.0% |
| `chopper_design3_typical` | 8-PMOS chopper | typical | PAC, PNoise | TD PAC <1% / TD IRN −0.00% |
| `chopper_design3_slow` | 8-PMOS chopper | slow | PAC, PNoise | TD PAC +0.03% / TD IRN +0.02% |
| `chopper_design3_fast` | 8-PMOS chopper | fast | PAC, PNoise | TD PAC <1% / TD IRN +0.57% |
| `sc_lpf` | SC 低通 | typical | PAC, PNoise | gain −1.4% / BW +0.9% / 噪声 +1.4% |

### 5.2 覆盖缺口

```
❌ 无瞬态对标          — compare_tran() 不存在
❌ 无 PSS 稳态对比      — sc_lpf 有 pss.td.pss 参考文件但从未加载
❌ Amp 只有 typical    — 无 slow/fast corner
❌ Chopper 仅 f_chop=225Hz — 无其他斩波频率
❌ 仅标量指标           — 只比对 DC gain/BW/IRN，不比对全频谱形状
❌ 周期对标默认 CI skip — 需 RUN_SLOW_CHOPPER=1（9 条中的 7 条）
```

### 5.3 PSF 解析器状态

所有分析类型的 PSF 解析器均已实现：`parse_dc`, `parse_ac`, `parse_noise`, `parse_tran`, `parse_pac`, `parse_pnoise`。但 `parse_tran`/`parse_pss` 和 `parse_dc_sweep` 无消费者 —— 代码存在但从未在对标中使用。

---

## 六、架构总结

### 优点

- **求解器栈完整且通用** — DC → AC → Noise → Transient → PSS → PAC → PNoise，全部拓扑驱动、JSON 可配
- **Cadence 对标扎实** — 三个电路类型、五个 case、Spectre 24.1 参考数据随码入库，一键回归
- **PDK 抽象到位** — `TransistorModel` ABC + 注册表 + 极性感知，新增工艺/NMOS 不改 solver
- **元件类型全面** — R/C/Idc/Vsource/VCCS/VCVS/CCCS/CCVS 全分析路径覆盖
- **Numba 加速全覆盖** — 热路径全部 JIT 编译

### 技术债

- **静默吞错是最大风险** — 50+ 处 `except Exception` 可能掩盖模型发散
- **重复代码已清理** — 器件创建 × 7、`_bw_from_gain` × 2、`_nfval` × 3 已于 2026-06-23 统一（见第五节）。import 样板 × 21 仍待处理。
- **测试盲点** — MNA 原语、网表生成器、PMOS 模型物理均无独立测试

---

## 七、下一步路线图

### 第 0 步：提交工作树 🔴 立即

当前 358 行改动已是完整功能弧线，自洽、已测试、已写文档。搁置越久越容易和后续改动冲突。建议单次提交，涵盖四个主题。

### 第 1 步：清理 CRITICAL 代码债 ✅ 已完成（2026-06-23）

| 动作 | 工作量 | 结果 |
|------|--------|------|
| **提取共享 `build_devices()` 工厂** | 小 | ✅ 新增 `ac_solver.build_devices()`，5 个 solver 改用，4 个不再依赖 `device_model` |
| **合并 `_bw_from_gain`** | 小 | ✅ 合并到 `ac_solver.py`，chopper + pac_solver 改为 import |
| **统一用 `_dev_nf` 替代三个 `_nfval`** | 小 | ✅ pss/pac/pnoise 的本地 `_nfval` 全部删除 |

净减 ~60 行重复代码，146 测试零回归。

### 第 2 步：补校准缺口 🟡 本周/下周

| 动作 | 工作量 | 理由 |
|------|--------|------|
| **加 amp slow/fast corner 对标** | 小 | 已有 `gen_amp_netlist`，加两个 metadata.json 即可，补齐 DC/AC/Noise 工艺角验证 |
| **给 `cadence_netlist.py` 加 `gen_sc_lpf_netlist`** | 小 | SC-LPF 参考数据已有但无法从代码重新生成网表 |
| **加 `test_cadence_netlist.py`** | 中 | 网表生成器零测试是目前最大的盲点 |

### 第 3 步：加固测试 🟢 后续

| 动作 | 理由 |
|------|------|
| **`ac_mna.py` 单元测试** — 每个 stamp 函数独立验证 | MNA 原语是所有频域求解器的数学基础，一处符号错误影响全部 |
| **`pmos_tft_model.py` 独立测试** — gm/gds/电容/噪声 vs 偏置 | 900 行模型只有 numba-kernel 对比，模型物理从未独立验证 |
| **瞬态对标** — 加 `compare_tran()` + 一个 chopper case | 瞬态是 chopper 验证的最后一环，已有 transient solver 和 Cadence 参考数据 |
| **加 amp slow/fast PSF 参考数据** | 扩展现有 `gen_amp_netlist` 跑两个 corner |

### 第 4 步：功能扩展 🟢 按需启动

| 项目 | 触发条件 |
|------|---------|
| JSON `"model"` 字段（消费层） | 注册机制已就绪，等实际需要 NMOS 或多 PDK 时再做 |
| 贝叶斯优化 / NSGA-II | 当前 LHS + 随机采样对百级扫描够用 |
| transient 性能深化 | 等成为瓶颈再动 |
| Rust/Cython 编译后端 | 等千级/万级 sweep 需求出现 |

---

## 八、不做的事项（维持 Future Plan 决定）

| 事项 | 原因 |
|------|------|
| 全局 Verilog-A/average 电容模式 | 不全局切换。通用 transient/PSS 仍以 charge Q-stamp 为默认；PMOS chopper PSS 单独用 `cap_mode="average"` 对齐 Cadence feedthrough，PAC/PNoise conversion 另行使用 Spectre PAC 的 `C(V)*ddt(V)` 小信号折叠。 |
| GPU 加速 | PMOS + MNA 矩阵 ≤20×20 对 GPU 无优势 |
| Sign-off 级仿真器认证 | 项目定位是设计探索工具，不做 Spectre 替代品 |
| 大规模 CI/CD | 单人研究项目，手动 pytest 足够 |
| 互感和耦合电感 | 当前应用场景不需要 |
