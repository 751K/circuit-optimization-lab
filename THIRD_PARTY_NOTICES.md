# Third-Party Notices / 第三方软件声明

CircuitOpt's original source code is licensed under the
[MIT License](LICENSE). This repository also redistributes third-party source
code under its original copyright notices, licenses, and terms. The MIT
license does not replace or override those terms.

CircuitOpt 的原创源代码采用 [MIT License](LICENSE)。本仓库同时再分发了一些
第三方源代码；这些代码继续适用其原始版权声明、许可证和使用条款，CircuitOpt
的 MIT 许可证不会替代或覆盖这些条款。

## UC Berkeley BSIM4.5.0

**Component / 组件**

`circuitopt/compact_models/bsim4/native_src/vendor/bsim4v5/`

**Source / 来源**

The University of California, Berkeley BSIM4.5.0 implementation, imported from
the ngspice source tree at commit
`032b1c32c4dbad45ff132bcfac1dbecadbd8abb0`.

从 ngspice 源码树提交
`032b1c32c4dbad45ff132bcfac1dbecadbd8abb0` 引入的 University of
California, Berkeley BSIM4.5.0 实现。

**Copyright and acknowledgment / 版权与致谢**

Copyright 2005 Regents of the University of California. The source headers
credit Weidong Liu, Xuemei (Jane) Xi, Mohan Dunga, Ali Niknejad, and Chenming
Hu, with Professor Chenming Hu named as project director. CircuitOpt
acknowledges the UC Berkeley BSIM Research Group for developing BSIM4.

版权所有 2005 Regents of the University of California。源码头部列出的主要
作者包括 Weidong Liu、Xuemei (Jane) Xi、Mohan Dunga、Ali Niknejad 和
Chenming Hu，项目负责人为 Chenming Hu 教授。CircuitOpt 在此致谢开发 BSIM4
的 UC Berkeley BSIM Research Group。

**Terms / 条款**

The authoritative terms are retained in
[`B4TERMS_OF_USE`](circuitopt/compact_models/bsim4/native_src/vendor/bsim4v5/B4TERMS_OF_USE).
They include acknowledgment, copyright-retention, redistribution, and export
requirements. Those terms govern the vendored BSIM source.

权威条款完整保留在
[`B4TERMS_OF_USE`](circuitopt/compact_models/bsim4/native_src/vendor/bsim4v5/B4TERMS_OF_USE)
中，其中包括致谢、保留版权声明、再分发和出口限制等要求。仓库内的 BSIM
源代码以该文件为准。

## ngspice Compatibility Sources / ngspice 兼容源码

**Components / 组件**

- `circuitopt/compact_models/bsim4/native_src/vendor/include/ngspice/`
- `circuitopt/compact_models/bsim4/native_src/vendor/support/devsup.c`

**Source and terms / 来源与条款**

These compatibility declarations and device-support routines were imported
from the same ngspice revision. ngspice is generally distributed under the
modified BSD license, while individual files retain their own copyright,
credit, permission, and warranty notices. Those file-level notices are
authoritative and must be preserved when the files are copied or modified.

这些兼容声明和器件支持例程来自同一个 ngspice 修订版本。ngspice 整体通常采用
modified BSD license，但其中部分文件带有各自的版权、作者、许可和免责声明。
再复制或修改这些文件时，必须保留各文件中的原始声明，并以文件内声明为准。

The imported headers and support code credit contributors including the
Regents of the University of California, Thomas L. Quarles, Kenneth S.
Kundert, Holger Vogt, Emmanuel Rouat, and other ngspice and SPICE contributors.
Their inclusion here does not imply that those authors contributed directly to
the CircuitOpt Git repository.

引入的头文件和支持代码中包括 Regents of the University of California、
Thomas L. Quarles、Kenneth S. Kundert、Holger Vogt、Emmanuel Rouat，以及
其他 ngspice 和 SPICE 贡献者的署名。这些第三方代码的存在不表示相关作者直接
参与了 CircuitOpt Git 仓库的开发。

## CircuitOpt Adapter Code / CircuitOpt 适配代码

`circuitopt/compact_models/bsim4/native_src/host.c` and the Python integration
around it are CircuitOpt code licensed under MIT. The localized extension in
`b4v5noi.c` exposes the final physical noise-source densities to the adapter;
the BSIM4 noise equations remain unchanged.

`circuitopt/compact_models/bsim4/native_src/host.c` 及其外围 Python 集成属于
采用 MIT 许可证的 CircuitOpt 代码。`b4v5noi.c` 中的局部扩展仅将最终物理噪声
源密度暴露给适配层，BSIM4 噪声方程本身未被修改。

Implementation provenance and modification details are recorded in
[`native_src/NOTICE.md`](circuitopt/compact_models/bsim4/native_src/NOTICE.md).

实现来源和修改细节记录在
[`native_src/NOTICE.md`](circuitopt/compact_models/bsim4/native_src/NOTICE.md)。

## Foundry PDK Files / 代工厂 PDK 文件

Licensed foundry model files, including TSMC model payloads, are not part of
this repository or its Python distributions. Users must obtain and use them
under their own foundry agreements, licenses, and NDA obligations.

受许可保护的代工厂模型文件（包括 TSMC 模型数据）不属于本仓库或其 Python
分发包。用户必须根据自己的代工厂协议、许可证和 NDA 义务获取并使用这些文件。
