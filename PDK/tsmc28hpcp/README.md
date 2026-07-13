# TSMC28HPC+ Local Model Entry

Place the licensed HSPICE model at:

```text
PDK/tsmc28hpcp/models/hspice/cln28hpcp_1d8_elk_v1d0_2p2.l
```

The `models/` directory is ignored by Git. Circuitopt resolves this path relative
to the repository root, so the project can be moved without changing source code.
`TSMC28_MODEL_DIR` and `TSMC28_PDK_ROOT` can override it when using a shared PDK
installation.

Do not commit foundry model files or other licensed delivery content.
