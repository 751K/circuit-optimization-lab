# Native BSIM4.5 Source Notice

The compact-model equations under `vendor/bsim4v5` are the University of
California, Berkeley BSIM4.5.0 implementation distributed by the ngspice
project. The imported revision is ngspice commit
`032b1c32c4dbad45ff132bcfac1dbecadbd8abb0`.

CircuitOpt acknowledges the UC Berkeley BSIM Research Group that developed
BSIM4. The original terms are reproduced in
`vendor/bsim4v5/B4TERMS_OF_USE`.

The compatibility declarations under `vendor/include/ngspice` and the device
support file under `vendor/support` come from the same ngspice revision. They
are compiled only to host the BSIM4.5 device equations. CircuitOpt does not
link libngspice and does not use ngspice's parser, analyses, matrix solver, or
executable.

`host.c` is CircuitOpt's simulator-neutral adapter. It supplies a private dense
device matrix, solves compact-model internal nodes, and exports four-terminal
residual, conductance, charge, and capacitance data to Python.

`b4v5noi.c` has one localized CircuitOpt extension that reports the final
physical-source densities to `host.c` before circuit-level noise transfer. The
BSIM4 noise equations themselves are unchanged.
