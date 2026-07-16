# Circuit Optimization

[![CI](https://github.com/751K/circuit-optimization-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/751K/circuit-optimization-lab/actions/workflows/ci.yml)
[![Docs](https://github.com/751K/circuit-optimization-lab/actions/workflows/docs.yml/badge.svg)](https://751k.github.io/circuit-optimization-lab/)
[![Version](https://img.shields.io/badge/version-v1.3.0-blue)](CHANGELOG.md)

A local analog-circuit simulation and optimization framework with DC, AC, noise,
transient, PSS, PAC, PNoise, process-corner, mismatch, dataset, surrogate-model,
and design-space exploration workflows.

这是一个本地模拟电路仿真与优化框架，覆盖 DC、AC、噪声、瞬态、PSS、PAC、
PNoise、工艺角、失配、数据集生成、代理模型和设计空间搜索。

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

circuit-opt run examples/periodic_rc.json --analysis ac,noise
```

## Documentation

- [English documentation](docs/README.md)
- [中文文档](docs/README_zh.md)
- [CLI reference](docs/cli_reference.md)
- [Circuit JSON format](docs/json_circuit_format.md)
- [Core solver architecture](docs/module_overview.md)
- [TSMC28HPC+ adapter](docs/tsmc28hpcp.md)
- [Service API](docs/service_api.md)
- [Licenses and third-party notices](THIRD_PARTY_NOTICES.md)
- [Changelog](CHANGELOG.md)

## License

CircuitOpt's original code is licensed under the [MIT License](LICENSE).
Vendored BSIM4 and ngspice compatibility sources retain their original
copyright notices and terms; see
[Third-Party Notices / 第三方软件声明](THIRD_PARTY_NOTICES.md).
