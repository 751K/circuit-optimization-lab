/*
 * bindgen input for the co-bsim4 crate.
 *
 * This mirrors the include set of
 * circuitopt/compact_models/bsim4/native_src/host.c exactly, so bindgen
 * produces Rust struct layouts and function signatures that are bit-for-bit
 * compatible with the vendored Berkeley BSIM4.5 C that build.rs compiles.
 *
 * The vendor tree itself is never modified; this header lives in the crate.
 */
#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/noisedef.h"
#include "ngspice/sperror.h"
#include "ngspice/tskdefs.h"
#include "bsim4v5def.h"
#include "bsim4v5ext.h"
#include "bsim4v5init.h"
