#ifndef CIRCUITOPT_BSIM4_CONFIG_H
#define CIRCUITOPT_BSIM4_CONFIG_H

/*
 * Windows / MSVC feature configuration for the vendored ngspice headers.
 *
 * This file shadows `vendor/include/ngspice/config.h` *only* on the MSVC target:
 * `co-bsim4/build.rs` prepends this directory to the C and libclang include
 * paths when `CARGO_CFG_TARGET_ENV == "msvc"`. On macOS/Linux this directory is
 * never on the include path, so the vendored config.h is used verbatim and those
 * builds are bit-for-bit unchanged. The vendored Berkeley/ngspice tree itself is
 * never modified.
 *
 * Why a different macro set is required on MSVC: the vendored ngspice headers
 * already provide `_MSC_VER` branches that map POSIX calls to the CRT
 * (<io.h>/<direct.h>/<process.h>, strdup->_strdup, isnan->_isnan, ...), and
 * every `__attribute__(...)` is `#ifdef __GNUC__`-guarded. The *only* thing that
 * forces the non-existent POSIX headers (<unistd.h>, <strings.h>, <dirent.h>,
 * <sys/time.h>) and functions (bcopy, bzero) into an MSVC compile is the
 * vendored config.h unconditionally advertising HAVE_UNISTD_H / HAVE_STRINGS_H /
 * HAVE_DIRENT_H / HAVE_SYS_TIME_H / HAVE_BCOPY / HAVE_BZERO. Those are omitted
 * here; only features cl.exe actually provides are declared.
 *
 * PROVISIONAL: this set was derived by static analysis of the vendored headers.
 * It has NOT been compiled with cl.exe from the authoring environment (macOS).
 * The first Windows CI runner is the point of truth — see the "Windows runner
 * verification checklist" accompanying this change. Tune the macros below if the
 * MSVC compile reports a missing/failed feature.
 */

#define HAVE_STDINT_H 1
#define HAVE_STDBOOL_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_MATH_H 1
#define HAVE_ISNAN 1
#define HAVE_ISFINITE 1

#endif
